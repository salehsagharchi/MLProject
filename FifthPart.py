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
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from PIL import Image


class PlotValuesObject:
    def __init__(self, baseImages, noisyImages, cancelledImages):
        self.baseImages = baseImages
        self.noisyImages = noisyImages
        self.cancelledImages = cancelledImages


class CustomSignals(QObject):
    printStatus = pyqtSignal(int, str, bool)  # status_number, text, clear
    prgValue = pyqtSignal(int, int)
    plotSignal = pyqtSignal(object)


class FifthPart:
    def __init__(self):
        self.signals = CustomSignals()
        self.threadpool = QThreadPool()
        self.myWorker = None
        self.model = None

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

    def startLearning(self, address, shape, noiseVar, sampleNum, epoch, batch, layers):
        if self.myWorker is not None:
            self.printToStatus("\nAnother process is running, Try Again\n")
            return None

        try:
            shapex = int(shape.strip().split(",")[0].strip())
            assert shapex >= 1
            shapey = int(shape.strip().split(",")[1].strip())
            assert shapey >= 1

            assert (address != "" and self.is_digit(batch) and self.is_digit(epoch) and self.is_digit(noiseVar)
                    and self.is_digit(sampleNum) and Checker.isTupleDigitValues([layers], True))
        except:
            self.printToStatus("Input types error, Try Again\n")
            return None

        layers = Checker.convertToNumbersFromTuple(layers, True)
        myWorker = AiWorker(self, address, batch, epoch, shapex, shapey, noiseVar, sampleNum, layers)
        self.threadpool.start(myWorker)
        self.myWorker = myWorker

    def printToStatus(self, str, clear=False):
        self.signals.printStatus.emit(5, str, clear)

    def changePrgVal(self, value):
        self.signals.prgValue.emit(5, value)

    def workerFinished(self, model, plotObject):
        self.myWorker = None
        self.model = model
        if plotObject is not None:
            self.signals.plotSignal.emit(plotObject)

    def stopLearning(self):
        if self.myWorker is not None:
            self.myWorker.stop()


class AiWorker(QRunnable):
    def __init__(self, fifthPart: FifthPart, address, batch, epoch, shapex, shapey, noiseVar, sampleNum, layers):
        super(AiWorker, self).__init__()
        self.fifthPart = fifthPart
        self.address = address
        self.shapex = shapex
        self.shapey = shapey
        self.noiseVar = float(noiseVar)
        self.sampleNum = int(sampleNum)
        self.epochs = max(1, int(epoch))
        self.batchSize = max(1, int(batch))
        self.callBack = CustomCallback(self)
        self.layers = layers

        self.tolearn = 0

    def stop(self):
        self.callBack.stop = True

    @pyqtSlot()
    def run(self):

        self.fifthPart.printToStatus("Start reading image files and prepare datas ...\n")

        train_x, train_y, test_x, test_y = self.loadDatasFromImages(self.address, self.shapex, self.shapey, self.noiseVar)

        if train_x is None:
            self.fifthPart.printToStatus(f"\nCannot load images datas try again\n")
            self.fifthPart.workerFinished(None, None)
            return None

        activation = keras.activations.relu
        optimizer = 'adam'
        loss = tf.keras.losses.binary_crossentropy
        metrics = ['mse']

        self.fifthPart.printToStatus(f"Building and compile model with following params :\n \t'activation'={activation.__name__}, ")
        self.fifthPart.printToStatus(
            f"\n\t'optimizer'={optimizer}, \n\t'loss'='{loss.__name__}', 'metrics'={metrics} ...\n")

        # 256

        model = keras.Sequential()
        model.add(keras.layers.Flatten(input_shape=(self.shapex, self.shapey)))
        for item in self.layers:
            model.add(keras.layers.Dense(item, activation=activation))
        model.add(keras.layers.Dense(self.shapex * self.shapey, activation=keras.activations.sigmoid))
        model.add(keras.layers.Reshape((self.shapex, self.shapey), input_shape=(self.shapex * self.shapey,)))

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        self.fifthPart.printToStatus(f"Model Compiling Finished\nModel Summary:\n\n{short_model_summary}\n\n")
        self.fifthPart.printToStatus(f"Start learning {len(train_x)} images with 'epochs'={self.epochs}, 'batch_size'={self.batchSize} ...\n")

        self.tolearn = self.epochs * math.ceil(len(train_x) / float(self.batchSize))
        model.fit(train_x, train_y, epochs=self.epochs, batch_size=self.batchSize, callbacks=[self.callBack])

        if self.callBack.stop:
            self.fifthPart.printToStatus("Model training stopped by user request\n")

        self.updateProgress(self.tolearn)

        self.fifthPart.printToStatus(f"\nStart predicting for {len(test_x)} images ...\n")

        test_eval = model.evaluate(test_x, test_y, return_dict=True)
        predicted_images = model.predict(test_x)
        self.fifthPart.printToStatus(
            f"\nModel predicting finished with mse={round(test_eval['mse'], 3)} for TestData\n")

        self.fifthPart.printToStatus(f"\nStart drawing {self.sampleNum} random images of test images ...\n")

        indices = random.sample(range(0, test_x.shape[0]), self.sampleNum)

        baseImageArr = np.take(test_y, indices, axis=0)
        baseImages = []
        for item in baseImageArr:
            img = np.dstack((item, item, item))
            img = img * 255
            im = array_to_img(img)
            baseImages.append(self.pil2pixmap(im))

        noisyImageArr = np.take(test_x, indices, axis=0)
        noisyImages = []
        for item in noisyImageArr:
            img = np.dstack((item, item, item))
            img = img * 255
            im = array_to_img(img)
            noisyImages.append(self.pil2pixmap(im))

        cancelledImageArr = np.take(predicted_images, indices, axis=0)
        cancelledImages = []
        for item in cancelledImageArr:
            img = np.dstack((item, item, item))
            img = img * 255
            im = array_to_img(img)
            cancelledImages.append(self.pil2pixmap(im))

        self.fifthPart.workerFinished(model, PlotValuesObject(baseImages, noisyImages, cancelledImages))

    def pil2pixmap(self, im):
        if im.mode == "RGB":
            r, g, b = im.split()
            im = Image.merge("RGB", (b, g, r))
        elif im.mode == "RGBA":
            r, g, b, a = im.split()
            im = Image.merge("RGBA", (b, g, r, a))
        elif im.mode == "L":
            im = im.convert("RGBA")
        im2 = im.convert("RGBA")
        data = im2.tobytes("raw", "RGBA")
        qim = QtGui.QImage(data, im.size[0], im.size[1], QtGui.QImage.Format_ARGB32)
        pixmap = QtGui.QPixmap.fromImage(qim)
        pixmap5 = pixmap.scaled(48, 48)
        return pixmap5

    def updateProgress(self, value):
        self.fifthPart.changePrgVal(min(100, math.ceil(value / float(self.tolearn) * 100.0)))

    def epochsFinished(self, epoch, logs):
        self.fifthPart.printToStatus(f"\nEpoch {epoch} finished ==>\n{logs}\n")

    def trainFinished(self, logs):
        self.fifthPart.printToStatus(
            f"\nModel learning finished with mse={round(logs['mse'], 3)} for TrainData\n")

    def loadDatasFromImages(self, address, shapex, shapey, noiseVar):
        try:
            train_path = os.path.join(address, "train")
            test_path = os.path.join(address, "test")
            train_files = [os.path.join(train_path, f) for f in os.listdir(train_path) if f.endswith(".jpg")]
            test_files = [os.path.join(test_path, f) for f in os.listdir(test_path) if f.endswith(".jpg")]
            train_x, train_y, test_x, test_y = [], [], [], []

            for item in train_files:
                img = img_to_array(load_img(item))
                img = img.mean(axis=2)
                assert img.shape == (shapex, shapey)
                img = np.array(img) / 255.0
                train_y.append(img)
                train_x.append(self.noisy(img, noiseVar))

            for item in test_files:
                img = img_to_array(load_img(item))
                img = img.mean(axis=2)
                assert img.shape == (shapex, shapey)
                img = np.array(img) / 255.0
                test_y.append(img)
                test_x.append(self.noisy(img, noiseVar))

            train_x = (1.0 - np.array(train_x))
            train_y = (1.0 - np.array(train_y))
            test_x = (1.0 - np.array(test_x))
            test_y = (1.0 - np.array(test_y))

            return train_x, train_y, test_x, test_y
        except:
            return None, None, None, None

    def noisy(self, image, var):
        row, col = image.shape
        mean = 0.0
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        noisy = image + gauss
        return noisy


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
