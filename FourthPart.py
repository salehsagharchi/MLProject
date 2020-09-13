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
from tensorflow.keras.preprocessing.image import load_img, img_to_array


class PlotValuesObject:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels


class CustomSignals(QObject):
    printStatus = pyqtSignal(int, str, bool)  # status_number, text, clear
    prgValue = pyqtSignal(int, int)
    plotSignal = pyqtSignal(object)


class FourthPart:
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

    def startLearning(self, address, batch, epoch, shape, layers):
        if self.myWorker is not None:
            self.printToStatus("\nAnother process is running, Try Again\n")
            return None

        try:
            shapex = int(shape.strip().split(",")[0].strip())
            assert shapex >= 1
            shapey = int(shape.strip().split(",")[1].strip())
            assert shapey >= 1
            assert (address != "" and self.is_digit(batch) and self.is_digit(epoch) and Checker.isTupleDigitValues([layers], True))
        except:
            self.printToStatus("Input types error, Try Again\n")
            return None

        layers = Checker.convertToNumbersFromTuple(layers, True)
        myWorker = AiWorker(self, address, batch, epoch, shapex, shapey, layers)
        self.threadpool.start(myWorker)
        self.myWorker = myWorker

    def printToStatus(self, str, clear=False):
        self.signals.printStatus.emit(4, str, clear)

    def changePrgVal(self, value):
        self.signals.prgValue.emit(4, value)

    def workerFinished(self, model, plotObject):
        self.myWorker = None
        self.model = model
        if plotObject is not None:
            self.signals.plotSignal.emit(plotObject)

    def stopLearning(self):
        if self.myWorker is not None:
            self.myWorker.stop()


class AiWorker(QRunnable):
    def __init__(self, fourthPart: FourthPart, address, batch, epoch, shapex, shapey, layers):
        super(AiWorker, self).__init__()
        self.fourthPart = fourthPart
        self.address = address
        self.shapex = shapex
        self.shapey = shapey
        self.epochs = max(1, int(epoch))
        self.batchSize = max(1, int(batch))
        self.callBack = CustomCallback(self)
        self.layers = layers

        self.tolearn = 0

    def stop(self):
        self.callBack.stop = True

    @pyqtSlot()
    def run(self):

        self.fourthPart.printToStatus("Start reading image files and prepare datas ...\n")

        train_images, train_labels, test_images, test_labels, label_counts = self.loadDatasFromImages(self.address, self.shapex, self.shapey)

        if train_images is None:
            self.fourthPart.printToStatus(f"\nCannot load images datas try again\n")
            self.fourthPart.workerFinished(None, None)
            return None

        activation = keras.activations.relu
        optimizer = 'adam'
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = ['accuracy']

        self.fourthPart.printToStatus(f"Building and compile model with following params :\n \t'activation'={activation.__name__}, ")
        self.fourthPart.printToStatus(
            f"\n\t'optimizer'={optimizer}, \n\t'loss'='{loss.__class__.__name__}', 'metrics'={metrics} ...\n")

        # 128

        model = keras.Sequential()
        model.add(keras.layers.Flatten(input_shape=(self.shapex, self.shapey)))
        for item in self.layers:
            model.add(keras.layers.Dense(item, activation=activation))
        model.add(keras.layers.Dense(label_counts, activation=keras.activations.softmax))

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        self.fourthPart.printToStatus(f"Model Compiling Finished\nModel Summary:\n\n{short_model_summary}\n\n")
        self.fourthPart.printToStatus(f"Start learning {len(train_images)} images with 'epochs'={self.epochs}, 'batch_size'={self.batchSize} ...\n")

        self.tolearn = self.epochs * math.ceil(len(train_images) / float(self.batchSize))
        model.fit(train_images, train_labels, epochs=self.epochs, batch_size=self.batchSize, callbacks=[self.callBack])

        if self.callBack.stop:
            self.fourthPart.printToStatus("Model training stopped by user request\n")

        self.updateProgress(self.tolearn)

        self.fourthPart.printToStatus(f"\nStart predicting for {len(test_images)} images ...\n")

        test_eval = model.evaluate(test_images, test_labels, return_dict=True)
        predicted_labels = model.predict(test_images)
        self.fourthPart.printToStatus(
            f"\nModel predicting finished with accuracy={round(test_eval['accuracy'], 2)} for TestData\n")

        self.fourthPart.printToStatus(f"\nStart plotting 25 random images of test images ...\n")

        indices = random.sample(range(0, test_images.shape[0]), 25)
        plot_labels = []
        for i in indices:
            predicted = np.argmax(predicted_labels[i])
            if predicted != test_labels[i]:
                plot_labels.append(f"{str(predicted)} {round(predicted_labels[i][predicted] * 100)}% (act: {test_labels[i]})")
            else:
                plot_labels.append(f"{str(predicted)} {round(predicted_labels[i][predicted] * 100)}%")
        self.fourthPart.workerFinished(model, PlotValuesObject(np.take(test_images, indices, axis=0), plot_labels))

    def updateProgress(self, value):
        self.fourthPart.changePrgVal(min(100, math.ceil(value / float(self.tolearn) * 100.0)))

    def epochsFinished(self, epoch, logs):
        self.fourthPart.printToStatus(f"\nEpoch {epoch} finished ==>\n{logs}\n")

    def trainFinished(self, logs):
        self.fourthPart.printToStatus(
            f"\nModel learning finished with accuracy={round(logs['accuracy'], 2)} for TrainData\n")

    def loadDatasFromImages(self, address, shapex, shapey):
        try:
            train_path = os.path.join(address, "train")
            test_path = os.path.join(address, "test")
            train_files = [os.path.join(train_path, f) for f in os.listdir(train_path) if f.endswith(".jpg")]
            test_files = [os.path.join(test_path, f) for f in os.listdir(test_path) if f.endswith(".jpg")]
            train_images, train_labels, test_images, test_labels = [], [], [], []

            for item in train_files:
                img = img_to_array(load_img(item))
                img = img.mean(axis=2)
                assert img.shape == (shapex, shapey)
                train_images.append(img)
                train_labels.append(int(os.path.basename(item).split("_")[0]))

            for item in test_files:
                img = img_to_array(load_img(item))
                img = img.mean(axis=2)
                assert img.shape == (shapex, shapey)
                test_images.append(img)
                test_labels.append(int(os.path.basename(item).split("_")[0]))

            train_images = (np.array(train_images) / 255.0)
            test_images = (np.array(test_images) / 255.0)

            train_labels = np.array(train_labels)
            test_labels = np.array(test_labels)

            label_count = np.unique(train_labels).size

            return train_images, train_labels, test_images, test_labels, label_count

        except:
            return None, None, None, None, None


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
