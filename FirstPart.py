import random
import sys
import time
import matplotlib.pyplot as plt
import numpy as np

from MyMathHelper import *
from typing import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication
from PyQt5 import uic, QtGui, QtWidgets, QtCore
from UiFiles.MainWindow import Ui_MainWindow
from PyQt5.QtCore import *

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras


class Node:
    def __init__(self, val: str):
        self.value: str = str(val)
        self.right: Node = None
        self.left: Node = None
        self.height = -1
        self.parent: Node = None

    @staticmethod
    def is_leaf(node):
        return node.right is None and node.left is None

    @staticmethod
    def node_at_index_and_height(root, index, height=1):
        queue: List[Node] = [root]
        counter = -1
        front = None
        while queue:
            front = queue.pop(0)
            if front.height >= height:
                counter += 1
            if counter == index:
                return front
            if front.left is not None:
                queue.append(front.left)
            if front.right is not None:
                queue.append(front.right)

    @staticmethod
    def count_nodes_at_height(root, height):
        queue: List[Node] = [root]
        c = 0
        while queue:
            front = queue.pop(0)
            if front.height >= height:
                c += 1
            if front.left is not None:
                queue.append(front.left)
            if front.right is not None:
                queue.append(front.right)
        return c

    @staticmethod
    def calc_heights_at_node(node):
        if node is None:
            return -1
        if Node.is_leaf(node):
            node.height = 0
            return 0
        h = max(Node.calc_heights_at_node(node.left), Node.calc_heights_at_node(node.right)) + 1
        node.height = h
        return h


class TreeHelper:
    def __init__(self):
        self.root = None
        self.my_helper_ops: Dict[str, Operator] = MathHelper().get_operations_dict()
        self.operations_str = MathHelper().operations_str
        self.inorder_str = ""
        self.post_order_list = []
        self.post_order_numsofone = 0

    @staticmethod
    def is_digit(inp: str, has_variable=False, minus_is_underline=False):
        minus = "-"
        if minus_is_underline:
            minus = "_"
        if inp in ["X", "E", "P"] and has_variable:
            return True
        if inp.startswith(minus):
            inp = inp.replace(minus, "", 1)
        inp = inp.replace("e-", "").replace("e+", "")
        return inp.replace(".", "", 1).isdigit()

    def make_token_list_form_exp(self, exp):
        exp = exp.replace(" ", "")
        exp = exp.upper()
        exp = exp.replace("SIN", "S").replace("COS", "C").replace("÷", "/").replace("Ππ", "P").replace("Π", "P") \
            .replace("−", "-").replace("×", "*")
        tokens = []
        token = ""

        error = False
        for item in exp:
            if item in ["S", "C", "/", "*", "+", "-", "^", "E", "P", "(", ")", "X"]:
                if token != "" and self.is_digit(token, False, True):
                    tokens.append(token)
                token = ""
                tokens.append(item)
                continue
            if item == "_":
                token = "_"
                continue
            if item == "." or item.isdigit():
                token += item
                continue
            error = True

        if token != "" and self.is_digit(token, False, True):
            tokens.append(token)

        if error:
            raise Exception(f"Exception in parsing => {exp}")

        new_tokens = []
        for item in tokens:
            new_tokens.append(item.replace("_", "-"))

        return new_tokens

    def make_RPN_from_tokens(self, tokens: list):
        error = False
        token_list = tokens.copy()
        rpn_stack = []
        rpn_queue = []

        my_helper_ops: Dict[str, Operator] = MathHelper().get_operations_dict()

        def has_to_pop_item(my_op: Operator):
            if len(rpn_stack) == 0:
                return False
            term1 = False
            term2 = False
            term3 = False
            item = rpn_stack[len(rpn_stack) - 1]
            if item in my_helper_ops.keys():
                if my_helper_ops[item].operator_type == OperatorType.Function:
                    term1 = True
                elif my_helper_ops[item].precedence > my_op.precedence:
                    term2 = True
                elif my_helper_ops[item].precedence == my_op.precedence and my_op.left_associative:
                    term3 = True

            term4 = (item != "(")
            return (term1 or term2 or term3) and term4

        for token in token_list:
            if self.is_digit(token, True):
                rpn_queue.append(token)
            elif token in my_helper_ops.keys():
                if my_helper_ops[token].operator_type == OperatorType.Function:
                    rpn_stack.append(token)
                else:
                    while has_to_pop_item(my_helper_ops[token]):
                        rpn_queue.append(rpn_stack.pop())
                    rpn_stack.append(token)
            elif token == "(":
                rpn_stack.append(token)
            elif token == ")":
                while len(rpn_stack) > 0 and rpn_stack[len(rpn_stack) - 1] != "(":
                    rpn_queue.append(rpn_stack.pop())
                if len(rpn_stack) == 0:
                    error = True
                elif rpn_stack[len(rpn_stack) - 1] == "(":
                    rpn_stack.pop()

        while len(rpn_stack) > 0:
            if rpn_stack[len(rpn_stack) - 1] in ["(", ")"]:
                error = True
            rpn_queue.append(rpn_stack.pop())

        if error:
            raise Exception(f"Exception in parsing => {tokens}")
        return rpn_queue

    def make_tree_from_RPN(self, rpn_queue: list) -> Node:
        my_helper_ops: Dict[str, Operator] = MathHelper().get_operations_dict()
        tree_stack = []
        for item in rpn_queue:
            n = Node(item)
            if not self.is_digit(item, True):
                if my_helper_ops[item].operator_type == OperatorType.Function:
                    n.right = tree_stack.pop()
                    n.right.parent = n
                else:
                    n.right = tree_stack.pop()
                    n.right.parent = n
                    n.left = tree_stack.pop()
                    n.left.parent = n
            tree_stack.append(n)
        return tree_stack.pop()

    def make_RPN_from_tree(self, root: Node) -> Tuple[int, list]:
        self.post_order_list = []
        self.post_order_numsofone = 0
        self.__post_order_iteration(root)
        return self.post_order_numsofone, self.post_order_list.copy()

    def make_tree_from_expression(self, exp):
        token_list = self.make_token_list_form_exp(exp)
        rpn = self.make_RPN_from_tokens(token_list)
        return self.make_tree_from_RPN(rpn)

    def copy_tree(self, root: Node) -> Node:
        _, rpn = self.make_RPN_from_tree(root)
        n = self.make_tree_from_RPN(rpn)
        return n

    def __post_order_iteration(self, root: Node):
        if root is None:
            return
        self.__post_order_iteration(root.left)
        self.__post_order_iteration(root.right)
        self.post_order_list.append(root.value)
        if self.is_digit(root.value, True):
            self.post_order_numsofone += 1

    def print_inorder_tree(self, root: Node):
        self.inorder_str = ""
        self.__inorder_tree_iteration(root)
        for i in self.operations_str.keys():
            self.inorder_str = self.inorder_str.replace(i, self.operations_str[i])
        self.inorder_str = self.inorder_str.replace("~", "-")
        print(self.inorder_str)

    def __inorder_tree_iteration(self, root: Node):
        if root is None:
            return
        need_paran = False
        is_func = False
        if root.value in self.my_helper_ops.keys():
            tmp1 = self.my_helper_ops[root.value]
            if root.parent is not None:
                tmp2 = self.my_helper_ops[root.parent.value]
                if (tmp1.precedence < tmp2.precedence) \
                        or (tmp1.precedence == tmp2.precedence and tmp1.left_associative is False) \
                        or (tmp1.precedence == tmp2.precedence and root.parent.right == root):
                    need_paran = True
            if tmp1.operator_type == OperatorType.Function:
                is_func = True
        if need_paran:
            self.inorder_str += "("

        self.__inorder_tree_iteration(root.left)
        value = root.value
        if self.is_digit(value):
            value = str(round(float(value), 3))
        if value != "-":
            value = value.replace("-", "~")
        self.inorder_str += value
        if is_func:
            self.inorder_str += "("
        self.__inorder_tree_iteration(root.right)
        if is_func:
            self.inorder_str += ")"

        if need_paran:
            self.inorder_str += ")"

    def print_beautiful_tree(self, root: Node):
        if root is None:
            return
        print(root.value)
        self.__print_subtree(root, "")
        print()

    def __print_subtree(self, root: Node, prefix):
        if root is None:
            return
        has_left = root.left is not None
        has_right = root.right is not None
        if not (has_left or has_right):
            return
        print(prefix, end="")
        print("├── " if (has_left and has_right) else "", end="")
        print("└── " if ((not has_left) and has_right) else "", end="")

        if has_right:
            print_strand = (
                    has_left and has_right and ((root.right.right is not None) or (root.right.left is not None)))
            new_prefix = prefix + ("│   " if print_strand else "    ")
            print(root.right.value)
            self.__print_subtree(root.right, new_prefix)

        if has_left:
            if has_right:
                print(prefix, end="")
            print("└── ", end="")
            print(root.left.value)
            self.__print_subtree(root.left, prefix + "    ")

    def calculate_tree_given_x(self, root: Node, x: float):
        if root is None:
            return None
        if root.left is None and root.right is None:
            if root.value == "X":
                return x
            elif root.value == "P":
                return math.pi
            elif root.value == "E":
                return math.e
            return root.value
        input1 = self.calculate_tree_given_x(root.left, x)
        input2 = self.calculate_tree_given_x(root.right, x)
        if input1 is not None:
            input1 = float(input1)
        if input2 is not None:
            input2 = float(input2)
        return self.my_helper_ops[root.value].solve(input1, input2)

    def plot_trees(self, roots: List[Node], start=-100, end=100, step=1.0):
        index = 1
        for root in roots:
            x = []
            y = []
            i = start
            while i <= end:
                try:
                    tmp = (self.calculate_tree_given_x(root, i))
                except Exception as e:
                    pass
                else:
                    if tmp != float('-inf'):
                        x.append(i)
                        y.append(tmp)
                i += step
            plt.plot(x, y, label="Line " + str(index))
            index += 1
        plt.legend()
        plt.show()


class PlotValuesObject:
    def __init__(self, x_list1, y_list1, x_list2, y_list2, x_list3, y_list3):
        self.x_list1 = x_list1
        self.y_list1 = y_list1
        self.x_list2 = x_list2
        self.y_list2 = y_list2
        self.x_list3 = x_list3
        self.y_list3 = y_list3


class CustomSignals(QObject):
    printStatus = pyqtSignal(int, str, bool)  # status_number, text, clear
    prgValue = pyqtSignal(int, int)
    plotSignal = pyqtSignal(object)


class Checker:

    @staticmethod
    def is_digit(inp: str, has_variable=False, minus_is_underline=False):
        minus = "-"
        if minus_is_underline:
            minus = "_"
        if inp in ["X", "E", "P"] and has_variable:
            return True
        if inp.startswith(minus):
            inp = inp.replace(minus, "", 1)
        inp = inp.replace("e-", "").replace("e+", "")
        return inp.replace(".", "", 1).isdigit()

    @staticmethod
    def isDigitValues(values: List[str], checkPos):
        for value in values:
            if not Checker.is_digit(value):
                return False
            if float(value) <= 0.0 and checkPos:
                return False
        return True

    @staticmethod
    def isTupleDigitValues(values: List[str], checkPos):
        for value in values:
            if value.strip() != "":
                nums = value.split(",")
                for item in nums:
                    if not Checker.is_digit(item.strip()):
                        return False
                    if float(item.strip()) <= 0.0 and checkPos:
                        return False
        return True

    @staticmethod
    def convertToNumbersFromTuple(inpt: str, isInt):
        result = []
        for item in inpt.split(","):
            if item != "":
                if isInt:
                    result.append(int(item.strip()))
                else:
                    result.append(float(item.strip()))
        return result


class FirstPart:
    def __init__(self):
        self.signals = CustomSignals()
        self.threadpool = QThreadPool()
        self.myWorker = None
        self.helper = TreeHelper()
        self.model = None

    def startLearning(self, funcExpr, trainDom, trainStep, testDom, testStep, epochs, batch, layers, noiseMean, noiseSd):
        if self.myWorker is not None:
            self.printToStatus("\nAnother process is running, Try Again\n")
            return None

        digitValues = [trainStep, testStep, epochs, batch]
        tupleValues = [trainDom, testDom, layers]

        if funcExpr == "" or not Checker.isDigitValues(digitValues, True) or not Checker.isTupleDigitValues(tupleValues, False):
            self.printToStatus("Input types error, Try Again\n")
            return None
        try:
            tree = self.helper.make_tree_from_expression(funcExpr)
        except:
            self.printToStatus("Cannot parse your function expression, Try Again\n")
            return None

        noiseM = 0
        noiseS = 0
        if Checker.is_digit(noiseMean):
            noiseM = float(noiseMean)
        if Checker.is_digit(noiseSd):
            noiseS = float(noiseSd)

        domainStart, domainEnd = Checker.convertToNumbersFromTuple(trainDom, False)
        testStart, testEnd = Checker.convertToNumbersFromTuple(testDom, False)
        layers = Checker.convertToNumbersFromTuple(layers, True)

        myWorker = AiWorker(self, tree, domainStart, domainEnd, float(trainStep),
                            testStart, testEnd, float(testStep), int(epochs), int(batch), layers, noiseM, noiseS)
        self.threadpool.start(myWorker)
        self.myWorker = myWorker

    def printToStatus(self, str, clear=False):
        self.signals.printStatus.emit(1, str, clear)

    def changePrgVal(self, value):
        self.signals.prgValue.emit(1, value)

    def workerFinished(self, model, x_list1, y_list1, x_list2, y_list2, x_list3, y_list3):
        self.myWorker = None
        self.model = model
        self.signals.plotSignal.emit(PlotValuesObject(x_list1, y_list1, x_list2, y_list2, x_list3, y_list3))

    def stopLearning(self):
        if self.myWorker is not None:
            self.myWorker.stop()


class AiWorker(QRunnable):
    def __init__(self, firstPart: FirstPart, tree, domStart, domEnd, step, testStart, testEnd, testStep, epochs, batch, layers, noiseMean, noiseSd):
        super(AiWorker, self).__init__()
        self.firstPart = firstPart
        self.tree = tree
        self.domStart = domStart
        self.domEnd = domEnd
        self.step = step
        self.testStart = testStart
        self.testEnd = testEnd
        self.testStep = testStep

        self.noiseMean = noiseMean
        self.noiseSd = max(noiseSd, 0.0)

        self.layers = layers
        self.epochs = epochs  # 15
        self.batchSize = batch  # 32
        self.callBack = CustomCallback(self)

        self.tolearn = 0

        self.large = np.power(2, 63, dtype=np.float)

    def stop(self):
        self.callBack.stop = True

    @pyqtSlot()
    def run(self):

        self.firstPart.printToStatus("Start generating train points and test points ...\n")

        train_x, train_y = self.loadDatasFromFuntion(self.tree, self.domStart, self.domEnd, self.step, self.noiseMean, self.noiseSd)
        test_x, test_y = self.loadDatasFromFuntion(self.tree, self.testStart, self.testEnd, self.testStep, self.noiseMean, self.noiseSd)

        activation = keras.activations.relu
        optimizer = keras.optimizers.RMSprop(0.001)
        loss = 'mse'
        metrics = ['mae', 'mse']

        self.firstPart.printToStatus(f"Building and compile model with following params :\n \t'activation'={activation.__name__}, ")
        self.firstPart.printToStatus(f"\n\t'optimizer'={optimizer.__class__.__name__}, \n\t'loss'='{loss}', 'metrics'={metrics} ...\n")

        # 512,512,512
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(1,)))
        for item in self.layers:
            model.add(keras.layers.Dense(item, activation=activation))
        model.add(keras.layers.Dense(1))

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        self.firstPart.printToStatus(f"Model Compiling Finished\nModel Summary:\n\n{short_model_summary}\n\n")
        self.firstPart.printToStatus(f"Start learning {len(train_x)} points with 'epochs'={self.epochs}, 'batch_size'={self.batchSize} ...\n")

        self.tolearn = self.epochs * math.ceil(len(train_x) / float(self.batchSize))
        model.fit(train_x, train_y, epochs=self.epochs, batch_size=self.batchSize, callbacks=[self.callBack])

        if self.callBack.stop:
            self.firstPart.printToStatus("Model training stopped by user request\n")

        self.updateProgress(self.tolearn)

        self.firstPart.printToStatus(f"\nStart predicting for {len(test_x)} points ...\n")

        test_eval = model.evaluate(test_x, test_y, return_dict=True)
        predicted_y = model.predict(test_x)
        self.firstPart.printToStatus(
            f"\nModel predicting finished with mae={round(test_eval['mae'], 2)} and mse = {round(test_eval['mse'], 2)} for TestData\n")

        self.firstPart.printToStatus(f"\nStart plotting correct function and learned function ...\n")

        self.firstPart.workerFinished(model, test_x, test_y, test_x, predicted_y, train_x, train_y)

    def updateProgress(self, value):
        self.firstPart.changePrgVal(min(100, math.ceil(value / float(self.tolearn) * 100.0)))

    def epochsFinished(self, epoch, logs):
        self.firstPart.printToStatus(f"\nEpoch {epoch} finished ==>\n{logs}\n")

    def trainFinished(self, logs):
        self.firstPart.printToStatus(
            f"\nModel learning finished with mae={round(logs['mae'], 2)} and mse = {round(logs['mse'], 2)} for TrainData\n")

    def loadDatasFromFuntion(self, tree, start, end, step, noiseMean, noiseSd):
        x_points = np.arange(start, end, step, dtype=np.float).tolist()

        data_x = []
        data_y = []

        for point in x_points:
            try:
                y = self.firstPart.helper.calculate_tree_given_x(tree, point)
            except:
                pass
            else:
                y = float(y)
                if y != float('-inf') and -self.large < y < self.large:
                    data_x.append(point)
                    data_y.append(y)

        data_x = np.array(data_x, dtype=np.float)
        data_y = np.array(data_y, dtype=np.float)

        noise = np.random.normal(noiseMean, noiseSd, data_y.shape)
        data_y = data_y + noise

        return data_x, data_y


class CustomCallback(keras.callbacks.Callback):

    def __init__(self, worker: AiWorker):
        super().__init__()
        self.worker = worker
        self.started = 0
        self.stop = False

    def stopTraining(self):
        self.stop = True

    def on_epoch_end(self, epoch, logs=None):
        if self.stop:
            self.model.stop_training = True
        self.worker.epochsFinished(epoch, logs)

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        self.worker.trainFinished(logs)

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        self.started += 1
        self.worker.updateProgress(self.started)
