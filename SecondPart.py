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
from FirstPart import Checker

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
    variables = ["X", "Y", "E", "P"]

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
        if inp in TreeHelper.variables and has_variable:
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
            if item in (["S", "C", "/", "*", "+", "-", "^", "(", ")"] + TreeHelper.variables):
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

    def calculate_tree_given_values(self, root: Node, values: List[float]):
        if root is None:
            return None
        if root.left is None and root.right is None:
            if root.value == "X":
                return values[0]
            elif root.value == "Y":
                return values[1]
            elif root.value == "P":
                return math.pi
            elif root.value == "E":
                return math.e
            return root.value
        input1 = self.calculate_tree_given_values(root.left, values)
        input2 = self.calculate_tree_given_values(root.right, values)
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
    def __init__(self, x_list1, y_list1, z_list1, x_list2, y_list2, z_list2):
        self.x_list1 = x_list1
        self.y_list1 = y_list1
        self.z_list1 = z_list1
        self.x_list2 = x_list2
        self.y_list2 = y_list2
        self.z_list2 = z_list2


class CustomSignals(QObject):
    printStatus = pyqtSignal(int, str, bool)  # status_number, text, clear
    prgValue = pyqtSignal(int, int)
    plotSignal = pyqtSignal(object)


class SecondPart:
    def __init__(self):
        self.signals = CustomSignals()
        self.threadpool = QThreadPool()
        self.myWorker = None
        self.helper = TreeHelper()
        self.model = None

    def startLearning(self, funcExpr, domainStart, domainEnd, step, testStart, testEnd, testStep, epochs, batch, layers):
        if self.myWorker is not None:
            self.printToStatus("\nAnother process is running, Try Again\n")
            return None

        digitValues = [step, testStep, epochs, batch]
        tupleValues = [domainStart, domainEnd, layers, testStart, testEnd]

        if funcExpr == "" or not Checker.isDigitValues(digitValues, True) or not Checker.isTupleDigitValues(tupleValues, False):
            self.printToStatus("Input types error, Try Again\n")
            return None

        try:
            tree = self.helper.make_tree_from_expression(funcExpr)
        except:
            self.printToStatus("Cannot parse your function expression, Try Again\n")
            return None

        layers = Checker.convertToNumbersFromTuple(layers, True)

        myWorker = AiWorker(self, tree,
                            Checker.convertToNumbersFromTuple(domainStart, False),
                            Checker.convertToNumbersFromTuple(domainEnd, False),
                            float(step),
                            Checker.convertToNumbersFromTuple(testStart, False),
                            Checker.convertToNumbersFromTuple(testEnd, False),
                            float(testStep),
                            int(epochs), int(batch), layers)
        self.threadpool.start(myWorker)
        self.myWorker = myWorker

    def printToStatus(self, str, clear=False):
        self.signals.printStatus.emit(2, str, clear)

    def changePrgVal(self, value):
        self.signals.prgValue.emit(2, value)

    def workerFinished(self, model, shape1, inputs1, outputs1, shape2, inputs2, outputs2):
        self.myWorker = None
        self.model = model

        inputs1 = inputs1.flatten("F")
        tmp1 = np.split(inputs1, 2)
        X1 = (tmp1[0].reshape(shape1[0], shape1[1]))
        Y1 = (tmp1[1].reshape(shape1[0], shape1[1]))
        Z1 = outputs1.reshape(X1.shape)
        inputs2 = inputs2.flatten("F")
        tmp2 = np.split(inputs2, 2)
        X2 = (tmp2[0].reshape(shape2[0], shape2[1]))
        Y2 = (tmp2[1].reshape(shape2[0], shape2[1]))
        Z2 = outputs2.reshape(X2.shape)
        self.signals.plotSignal.emit(PlotValuesObject(X1, Y1, Z1, X2, Y2, Z2))

    def stopLearning(self):
        if self.myWorker is not None:
            self.myWorker.stop()


class AiWorker(QRunnable):
    def __init__(self, secondPart: SecondPart, tree, domStart, domEnd, step, testStart, testEnd, testStep, epochs, batch, layers):
        super(AiWorker, self).__init__()
        self.secondPart = secondPart
        self.tree = tree
        self.domStart = domStart
        self.domEnd = domEnd
        self.testStart = testStart
        self.testEnd = testEnd
        self.step = step
        self.testStep = testStep

        self.layers = layers
        self.epochs = epochs  # 2
        self.batchSize = batch  # 64
        self.callBack = CustomCallback(self)

        self.tolearn = 0

        self.large = np.power(2, 63, dtype=np.float)

    def stop(self):
        self.callBack.stop = True

    @pyqtSlot()
    def run(self):

        self.secondPart.printToStatus("Start generating train points and test points ...\n")

        train_shape, train_input, train_output = self.loadDatasFromFuntion(self.tree, self.domStart, self.domEnd, self.step)
        test_shape, test_input, test_output = self.loadDatasFromFuntion(self.tree, self.testStart, self.testEnd, self.testStep)

        activation = keras.activations.relu
        optimizer = keras.optimizers.RMSprop(0.001)
        loss = 'mse'
        metrics = ['mae', 'mse']

        self.secondPart.printToStatus(f"Building and compile model with following params :\n \t'activation'={activation.__name__}, ")
        self.secondPart.printToStatus(f"\n\t'optimizer'={optimizer.__class__.__name__}, \n\t'loss'='{loss}', 'metrics'={metrics} ...\n")

        # 1024,1024,1024,1024
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(2,)))
        for item in self.layers:
            model.add(keras.layers.Dense(item, activation=activation))
        model.add(keras.layers.Dense(1))

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        self.secondPart.printToStatus(f"Model Compiling Finished\nModel Summary:\n\n{short_model_summary}\n\n")
        self.secondPart.printToStatus(f"Start learning {len(train_input)} points with 'epochs'={self.epochs}, 'batch_size'={self.batchSize} ...\n")

        self.tolearn = self.epochs * math.ceil(len(train_input) / float(self.batchSize))
        model.fit(train_input, train_output, epochs=self.epochs, batch_size=self.batchSize, callbacks=[self.callBack])

        if self.callBack.stop:
            self.secondPart.printToStatus("Model training stopped by user request\n")
        self.updateProgress(self.tolearn)

        self.secondPart.printToStatus(f"\nStart predicting for {len(test_input)} points ...\n")

        test_eval = model.evaluate(test_input, test_output, return_dict=True)
        predicted_z = model.predict(test_input)
        self.secondPart.printToStatus(
            f"\nModel predicting finished with mae={round(test_eval['mae'], 2)} and mse = {round(test_eval['mse'], 2)} for TestData\n")

        predicted_z = predicted_z.reshape(test_shape)

        self.secondPart.printToStatus(f"\nStart plotting correct function and learned function ...\n")

        self.secondPart.workerFinished(None, test_shape, test_input, test_output, test_shape, test_input, predicted_z)

    def updateProgress(self, value):
        self.secondPart.changePrgVal(min(100, math.ceil(value / float(self.tolearn) * 100.0)))

    def epochsFinished(self, epoch, logs):
        self.secondPart.printToStatus(f"\nEpoch {epoch} finished ==>\n{logs}\n")

    def trainFinished(self, logs):
        self.secondPart.printToStatus(
            f"\nModel learning finished with mae={round(logs['mae'], 2)} and mse = {round(logs['mse'], 2)} for TrainData\n")

    def loadDatasFromFuntion(self, tree, start, end, step):
        x_points = np.arange(start[0], end[0], step, dtype=np.float)
        y_points = np.arange(start[1], end[1], step, dtype=np.float)

        data_shape = (x_points.size, y_points.size)

        data_inputs = []
        data_output = []

        for i in range(x_points.size):
            for j in range(y_points.size):
                try:
                    z = self.secondPart.helper.calculate_tree_given_values(tree, [x_points[i], y_points[j]])
                except:
                    pass
                else:
                    z = float(z)
                    if z != float('-inf') and -self.large < z < self.large:
                        data_inputs.append([x_points[i], y_points[j]])
                        data_output.append(z)

        data_inputs = np.array(data_inputs, dtype=np.float)
        data_output = np.array(data_output, dtype=np.float)

        return data_shape, data_inputs, data_output


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
