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
from FirstPart import Checker
from PyQt5.QtCore import *

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras


class PlotValuesObject:
    def __init__(self, x_list1, y_list1, x_list2, y_list2):
        self.x_list1 = x_list1
        self.y_list1 = y_list1
        self.x_list2 = x_list2
        self.y_list2 = y_list2


class CustomSignals(QObject):
    printStatus = pyqtSignal(int, str, bool)  # status_number, text, clear
    prgValue = pyqtSignal(int, int)
    plotSignal = pyqtSignal(object)


class ThirdPart:
    def __init__(self):
        self.signals = CustomSignals()
        self.threadpool = QThreadPool()
        self.myWorker = None
        self.model = None

    def startLearning(self, points, testStart, testEnd, testStep, normalFactor, spaceFactor, epochs, batch, layers):
        if self.myWorker is not None:
            self.printToStatus("\nAnother process is running, Try Again\n")
            return None

        digitValues = [testStart, testEnd]
        digitValuesPos = [testStep, normalFactor, spaceFactor, epochs, batch]

        if points == "" or not Checker.isDigitValues(digitValues, False) \
                or not Checker.isDigitValues(digitValuesPos, True) or not Checker.isTupleDigitValues([layers], True):
            self.printToStatus("Input types error, Try Again\n")
            return None

        layers = Checker.convertToNumbersFromTuple(layers, True)

        myWorker = AiWorker(self, points, float(testStart), float(testEnd), float(testStep),
                            float(normalFactor), int(spaceFactor), int(epochs), int(batch),
                            layers)
        self.threadpool.start(myWorker)
        self.myWorker = myWorker

    def printToStatus(self, str, clear=False):
        self.signals.printStatus.emit(3, str, clear)

    def changePrgVal(self, value):
        self.signals.prgValue.emit(3, value)

    def workerFinished(self, model, plotObject):
        self.myWorker = None
        self.model = model
        if plotObject is not None:
            self.signals.plotSignal.emit(plotObject)

    def stopLearning(self):
        if self.myWorker is not None:
            self.myWorker.stop()


class AiWorker(QRunnable):
    def __init__(self, thirdPart: ThirdPart, points, testStart, testEnd, testStep, normalFactor, spaceFactor, epochs, batch, layers):
        super(AiWorker, self).__init__()
        self.thirdPart = thirdPart
        self.points = points
        self.testStart = testStart / normalFactor
        self.testEnd = testEnd / normalFactor
        self.testStep = testStep / normalFactor

        self.normalizeFactor = normalFactor  # 20.0
        self.lineSpaceFactor = spaceFactor  # 6000
        self.layers = layers
        self.epochs = epochs  # max(1, int(epochs))
        self.batchSize = batch  # 128
        self.callBack = CustomCallback(self)

        self.tolearn = 0

    def stop(self):
        self.callBack.stop = True

    def generateValidPoints(self, inpt: str):
        try:
            xpoints = []
            ypoints = []
            for item in inpt.split("\n"):
                if item.strip() != "":
                    x = item.split(",")[0].strip()
                    y = item.split(",")[1].strip()
                    xpoints.append(float(x))
                    ypoints.append(float(y))
            xpoints = np.array(xpoints, dtype=np.float)
            ypoints = np.array(ypoints, dtype=np.float)
            if xpoints.size > 0 and xpoints.size == ypoints.size:
                xpoints = xpoints / self.normalizeFactor
                ypoints = ypoints / self.normalizeFactor
                newx = np.array([])
                newy = np.array([])
                for i in range(xpoints.size - 1):
                    newx = np.append(newx, np.linspace(xpoints[i], xpoints[i + 1], self.lineSpaceFactor, endpoint=False))
                    newy = np.append(newy, np.linspace(ypoints[i], ypoints[i + 1], self.lineSpaceFactor, endpoint=False))
                newx = np.append(newx, xpoints[xpoints.size - 1])
                newy = np.append(newy, ypoints[ypoints.size - 1])
                return newx, newy
        except:
            return None, None
        return None, None

    @pyqtSlot()
    def run(self):
        self.thirdPart.printToStatus("Start generating actual points and test points ...\n")

        train_x, train_y = self.generateValidPoints(self.points)

        if train_x is None:
            self.thirdPart.printToStatus(f"\nCannot load points data, Try again\n")
            self.thirdPart.workerFinished(None, None)
            return None


        test_x = np.arange(self.testStart, self.testEnd, self.testStep, dtype=np.float)

        activation = keras.activations.relu
        optimizer = keras.optimizers.RMSprop(0.001)
        loss = 'mse'
        metrics = ['mae', 'mse']

        self.thirdPart.printToStatus(f"Building and compile model with following params :\n \t'activation'={activation.__name__}, ")
        self.thirdPart.printToStatus(f"\n\t'optimizer'={optimizer.__class__.__name__}, \n\t'loss'='{loss}', 'metrics'={metrics} ...\n")

        # 2048,2048,2048
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(1,)))
        for item in self.layers:
            model.add(keras.layers.Dense(item, activation=activation))
        model.add(keras.layers.Dense(1))
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        self.thirdPart.printToStatus(f"Model Compiling Finished\nModel Summary:\n\n{short_model_summary}\n\n")
        self.thirdPart.printToStatus(f"Start learning {len(train_x)} points with 'epochs'={self.epochs}, 'batch_size'={self.batchSize} ...\n")

        self.tolearn = self.epochs * math.ceil(len(train_x) / float(self.batchSize))
        model.fit(train_x, train_y, epochs=self.epochs, batch_size=self.batchSize, callbacks=[self.callBack])

        if self.callBack.stop:
            self.thirdPart.printToStatus("Model training stopped by user request\n")
        self.updateProgress(self.tolearn)

        self.thirdPart.printToStatus(f"\nStart predicting for {len(test_x)} points ...\n")

        predicted_y = model.predict(test_x)
        self.thirdPart.printToStatus("\nModel predicting finished\n")

        predicted_y = predicted_y.flatten()

        self.thirdPart.printToStatus(f"\nStart plotting correct function and learned function ...\n")

        self.thirdPart.workerFinished(None, PlotValuesObject(train_x, train_y, test_x, predicted_y))

    def updateProgress(self, value):
        self.thirdPart.changePrgVal(min(100, math.ceil(value / float(self.tolearn) * 100.0)))

    def epochsFinished(self, epoch, logs):
        self.thirdPart.printToStatus(f"\nEpoch {epoch} finished ==>\n{logs}\n")

    def trainFinished(self, logs):
        self.thirdPart.printToStatus(
            f"\nModel learning finished with mae={round(logs['mae'], 2)} and mse = {round(logs['mse'], 2)} for TrainData\n")


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
