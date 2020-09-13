import random
import sys
import time
import traceback
from termcolor import colored, cprint

import matplotlib.pyplot as plt
from PyQt5.QtGui import QImage
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import pickle

from PyQt5.QtWidgets import QApplication
from PyQt5 import uic, QtGui, QtWidgets, QtCore

import FirstPart
import SecondPart
import ThirdPart
import FourthPart
import FifthPart
from UiFiles.MainWindow import Ui_MainWindow
from PyQt5.QtCore import *
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array

isLoadCache = True


class UiController(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        super(UiController, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.setFixedSize(self.size())

        self.statusArr = [self.status1, self.status2, self.status3, self.status4, self.status5]
        self.prgArr = [self.prgBar1, self.prgBar2, self.prgBar3, self.prgBar4, self.prgBar5]

        self.firstPart = FirstPart.FirstPart()
        self.secondPart = SecondPart.SecondPart()
        self.thirdPart = ThirdPart.ThirdPart()
        self.fourthPart = FourthPart.FourthPart()
        self.fifthPart = FifthPart.FifthPart()

        self.bindEvents()
        self.bindCustomSignals()

        self.canvas1 = Canvas(self)
        self.pltLayout1.addWidget(self.canvas1)

        self.canvas2 = Canvas3D(self)
        self.pltLayout2.addWidget(self.canvas2)

        self.canvas3 = Canvas(self)
        self.pltLayout3.addWidget(self.canvas3)

        self.canvas4 = CanvasImage(self)
        self.pltLayout4.addWidget(self.canvas4)

        self.loadCache()

    def bindCustomSignals(self):
        self.firstPart.signals.printStatus.connect(self.printToStatus)
        self.firstPart.signals.prgValue.connect(self.setPrgValue)
        self.firstPart.signals.plotSignal.connect(self.canvas1Plot)
        self.secondPart.signals.printStatus.connect(self.printToStatus)
        self.secondPart.signals.prgValue.connect(self.setPrgValue)
        self.secondPart.signals.plotSignal.connect(self.canvas2Plot)
        self.thirdPart.signals.printStatus.connect(self.printToStatus)
        self.thirdPart.signals.prgValue.connect(self.setPrgValue)
        self.thirdPart.signals.plotSignal.connect(self.canvas3Plot)
        self.fourthPart.signals.printStatus.connect(self.printToStatus)
        self.fourthPart.signals.prgValue.connect(self.setPrgValue)
        self.fourthPart.signals.plotSignal.connect(self.canvas4Plot)
        self.fifthPart.signals.printStatus.connect(self.printToStatus)
        self.fifthPart.signals.prgValue.connect(self.setPrgValue)
        self.fifthPart.signals.plotSignal.connect(self.drawSamples5)

    def bindEvents(self):
        self.closeBtn.clicked.connect(lambda: (
            self.makeCacheDict(),
            self.close()))
        self.learn1.clicked.connect(self.learn1Clicked)
        self.stop1.clicked.connect(lambda: self.firstPart.stopLearning())
        self.learn2.clicked.connect(self.learn2Clicked)
        self.stop2.clicked.connect(lambda: self.secondPart.stopLearning())
        self.learn3.clicked.connect(self.learn3Clicked)
        self.stop3.clicked.connect(lambda: self.thirdPart.stopLearning())
        self.learn4.clicked.connect(self.learn4Clicked)
        self.stop4.clicked.connect(lambda: self.fourthPart.stopLearning())
        self.learn5.clicked.connect(self.learn5Clicked)
        self.stop5.clicked.connect(lambda: self.fifthPart.stopLearning())

    def printToStatus(self, number, text, clear):
        if 1 <= number <= len(self.statusArr):
            number -= 1
            if clear:
                self.statusArr[number].clear()
            self.statusArr[number].setText(self.statusArr[number].toPlainText() + text)
            self.statusArr[number].verticalScrollBar().setValue(self.statusArr[number].verticalScrollBar().maximum())

    def setPrgValue(self, number, value):
        if 1 <= number <= len(self.prgArr):
            number -= 1
            self.prgArr[number].setValue(value)

    def canvas1Plot(self, plotObj):
        self.canvas1.axes.cla()
        self.canvas1.axes.plot(plotObj.x_list3, plotObj.y_list3, 'g')
        self.canvas1.axes.plot(plotObj.x_list1, plotObj.y_list1, 'b')
        self.canvas1.axes.plot(plotObj.x_list2, plotObj.y_list2, 'y')
        self.canvas1.draw()
        self.printToStatus(1, "Plotting Finished\n", False)

    def canvas2Plot(self, plotObj):
        self.canvas2.axes.cla()
        x = plotObj.x_list1
        y = plotObj.y_list1
        z = plotObj.z_list1
        x2 = plotObj.x_list2
        y2 = plotObj.y_list2
        z2 = plotObj.z_list2

        self.canvas2.axes.plot_surface(x, y, z, color='b', alpha=0.2)
        self.canvas2.axes.plot_surface(x2, y2, z2, color='y', alpha=0.6)
        self.canvas2.draw()

        self.printToStatus(2, "Plotting Finished\n", False)

    def canvas3Plot(self, plotObj):
        self.canvas3.axes.cla()
        self.canvas3.axes.plot(plotObj.x_list1, plotObj.y_list1, 'b')
        self.canvas3.axes.plot(plotObj.x_list2, plotObj.y_list2, 'y')
        self.canvas3.draw()
        self.printToStatus(3, "Plotting Finished\n", False)

    def canvas4Plot(self, plotObj):
        for i in range(0, min(25, len(plotObj.images))):
            self.canvas4.ax[i].cla()
            self.canvas4.ax[i].set_xticks([])
            self.canvas4.ax[i].set_yticks([])
            self.canvas4.ax[i].grid(False)
            self.canvas4.ax[i].imshow(plotObj.images[i], cmap=plt.cm.binary)
            self.canvas4.ax[i].set_xlabel(plotObj.labels[i], fontsize=5)
        self.canvas4.draw()
        self.printToStatus(4, "\nPlotting Finished\n", False)

    def drawSamples5(self, plotObj):
        baseImages = plotObj.baseImages
        noisyImages = plotObj.noisyImages
        cancelledImages = plotObj.cancelledImages

        self.table5.setRowCount(0)
        self.table5.setRowCount(len(baseImages))

        for i in range(len(baseImages)):
            lbl = QtWidgets.QLabel()
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setPixmap(baseImages[i])
            lbl2 = QtWidgets.QLabel()
            lbl2.setAlignment(Qt.AlignCenter)
            lbl2.setPixmap(noisyImages[i])
            lbl3 = QtWidgets.QLabel()
            lbl3.setAlignment(Qt.AlignCenter)
            lbl3.setPixmap(cancelledImages[i])
            self.table5.setRowHeight(i, 55)
            self.table5.setCellWidget(i, 0, lbl)
            self.table5.setCellWidget(i, 1, lbl2)
            self.table5.setCellWidget(i, 2, lbl3)
        self.printToStatus(5, "\nDrawing Finished\n", False)

    def learn1Clicked(self):
        self.makeCacheDict()
        funcExpr = self.funExp1.text().strip()
        trainDom = self.trainDom1.text().strip()
        trainStep = self.trainStep1.text().strip()
        testDom = self.testDom1.text().strip()
        testStep = self.testStep1.text().strip()
        epochs = self.epochs1.text().strip()
        batch = self.batch1.text().strip()
        layers = self.layers1.text().strip()
        noiseMean = self.noiseMean1.text().strip()
        noiseSd = self.noiseSd1.text().strip()
        self.firstPart.startLearning(funcExpr, trainDom, trainStep, testDom, testStep, epochs, batch, layers, noiseMean, noiseSd)

    def learn2Clicked(self):
        self.makeCacheDict()
        funcExpr = self.funExp2.text().strip()
        domainStart = self.domStart2.text().strip()
        domainEnd = self.domEnd2.text().strip()
        step = self.step2.text().strip()
        testStart = self.testStart2.text().strip()
        testEnd = self.testEnd2.text().strip()
        testStep = self.testStep2.text().strip()
        epochs = self.epochs2.text().strip()
        batch = self.batch2.text().strip()
        layers = self.layers2.text().strip()
        self.secondPart.startLearning(funcExpr, domainStart, domainEnd, step, testStart, testEnd, testStep, epochs, batch, layers)

    def learn3Clicked(self):
        self.makeCacheDict()
        points = self.trainPoints3.toPlainText().strip()
        testStart = self.testStart3.text().strip()
        testEnd = self.testEnd3.text().strip()
        testStep = self.testStep3.text().strip()
        epochs = self.epochs3.text().strip()
        batch = self.batch3.text().strip()
        layers = self.layers3.text().strip()
        normalFactor = self.normalize3.text().strip()
        spaceFactor = self.lineSpace3.text().strip()
        self.thirdPart.startLearning(points, testStart, testEnd, testStep, normalFactor, spaceFactor, epochs, batch, layers)

    def learn4Clicked(self):
        self.makeCacheDict()
        address = self.fileAddress4.text().strip()
        epochs = self.epochs4.text().strip()
        batch = self.batch4.text().strip()
        layers = self.layers4.text().strip()
        shape = self.shape4.text().strip()
        self.fourthPart.startLearning(address, batch, epochs, shape, layers)

    def learn5Clicked(self):
        self.makeCacheDict()
        address = self.fileAddress5.text().strip()
        epochs = self.epochs5.text().strip()
        batch = self.batch5.text().strip()
        layers = self.layers5.text().strip()
        shape = self.shape5.text().strip()
        noiseVar = self.noiseVar5.text().strip()
        sampleNum = self.samples5.text().strip()

        self.fifthPart.startLearning(address, shape, noiseVar, sampleNum, epochs, batch, layers)

    def makeCacheDict(self):
        cacheDict = {}
        lineEdits = self.findChildren(QtWidgets.QLineEdit)
        for item in lineEdits:
            cacheDict[item.objectName()] = (item.text().strip())
        cacheDict['trainPoints3'] = self.trainPoints3.toPlainText().strip()
        with open('program_cache.obj', 'wb') as cacheFile:
            pickle.dump(cacheDict, cacheFile)
        return cacheDict

    def loadCache(self):
        if not isLoadCache:
            return
        try:
            with open('program_cache.obj', 'rb') as cacheFile:
                cacheDict = pickle.load(cacheFile)
            lineEdits = self.findChildren(QtWidgets.QLineEdit)
            for item in lineEdits:
                try:
                    item.setText(cacheDict[item.objectName()])
                except:
                    pass
            self.trainPoints3.setText(cacheDict['trainPoints3'])
        except:
            pass


class Canvas(FigureCanvas):
    def __init__(self, parent=None, width=4.5, height=4.5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)


class CanvasImage(FigureCanvas):
    def __init__(self, parent=None, width=4.5, height=4.5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)

        self.ax = []
        for i in range(25):
            self.ax.append(fig.add_subplot(5, 5, i + 1))
            self.ax[-1].set_xticks([])
            self.ax[-1].set_yticks([])
            self.ax[-1].grid(False)
        fig.subplots_adjust(hspace=0.9)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)


class Canvas3D(FigureCanvas):
    def __init__(self, parent=None, width=4.5, height=4.5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111, projection='3d')
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)


class ExceptionHandler(QtCore.QObject):
    def __init__(self):
        super(ExceptionHandler, self).__init__()


def excepthook(exc_type, exc_value, exc_tb):
    tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print(colored("Error Catched!:", 'red'))
    print(colored("error message:\n", 'red'), colored(tb, 'red'))
    QtWidgets.QApplication.quit()


def main():
    sys.excepthook = excepthook

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = UiController()
    window.show()

    app.exec()


if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    main()
