# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UiMainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(940, 600)
        MainWindow.setMinimumSize(QtCore.QSize(940, 600))
        icon = QtGui.QIcon.fromTheme("base")
        MainWindow.setWindowIcon(icon)
        MainWindow.setLocale(QtCore.QLocale(QtCore.QLocale.Russian, QtCore.QLocale.Russia))
        MainWindow.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setMinimumSize(QtCore.QSize(720, 540))
        self.tabWidget.setMaximumSize(QtCore.QSize(720, 540))
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(9)
        self.tabWidget.setFont(font)
        self.tabWidget.setObjectName("tabWidget")
        self.tabNew = QtWidgets.QWidget()
        self.tabNew.setObjectName("tabNew")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.tabNew)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.imageNew = QtWidgets.QLabel(self.tabNew)
        self.imageNew.setMinimumSize(QtCore.QSize(720, 540))
        self.imageNew.setMaximumSize(QtCore.QSize(1920, 1440))
        self.imageNew.setText("")
        self.imageNew.setObjectName("imageNew")
        self.gridLayout_3.addWidget(self.imageNew, 0, 0, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_3, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tabNew, "")
        self.tabProcessed = QtWidgets.QWidget()
        self.tabProcessed.setObjectName("tabProcessed")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.tabProcessed)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.imageProcessed = QtWidgets.QLabel(self.tabProcessed)
        self.imageProcessed.setMinimumSize(QtCore.QSize(720, 540))
        self.imageProcessed.setMaximumSize(QtCore.QSize(1920, 1440))
        self.imageProcessed.setText("")
        self.imageProcessed.setObjectName("imageProcessed")
        self.gridLayout_5.addWidget(self.imageProcessed, 0, 0, 1, 1)
        self.gridLayout_6.addLayout(self.gridLayout_5, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tabProcessed, "")
        self.gridLayout_2.addWidget(self.tabWidget, 0, 0, 1, 1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout.setObjectName("gridLayout")
        self.buttonNetwork1 = QtWidgets.QPushButton(self.centralwidget)
        self.buttonNetwork1.setMinimumSize(QtCore.QSize(140, 50))
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(9)
        self.buttonNetwork1.setFont(font)
        self.buttonNetwork1.setLocale(QtCore.QLocale(QtCore.QLocale.Russian, QtCore.QLocale.Russia))
        self.buttonNetwork1.setObjectName("buttonNetwork1")
        self.gridLayout.addWidget(self.buttonNetwork1, 0, 0, 1, 1)
        self.buttonNetwork2 = QtWidgets.QPushButton(self.centralwidget)
        self.buttonNetwork2.setMinimumSize(QtCore.QSize(140, 50))
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(9)
        self.buttonNetwork2.setFont(font)
        self.buttonNetwork2.setLocale(QtCore.QLocale(QtCore.QLocale.Russian, QtCore.QLocale.Russia))
        self.buttonNetwork2.setObjectName("buttonNetwork2")
        self.gridLayout.addWidget(self.buttonNetwork2, 1, 0, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setMinimumSize(QtCore.QSize(140, 50))
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(9)
        self.label.setFont(font)
        self.label.setScaledContents(False)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.results = QtWidgets.QLabel(self.centralwidget)
        self.results.setMinimumSize(QtCore.QSize(140, 100))
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.results.setFont(font)
        self.results.setLocale(QtCore.QLocale(QtCore.QLocale.Russian, QtCore.QLocale.Russia))
        self.results.setAlignment(QtCore.Qt.AlignCenter)
        self.results.setWordWrap(True)
        self.results.setObjectName("results")
        self.verticalLayout.addWidget(self.results)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.gridLayout_2.addLayout(self.verticalLayout_2, 0, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 940, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuNetwork = QtWidgets.QMenu(self.menubar)
        self.menuNetwork.setObjectName("menuNetwork")
        self.menu = QtWidgets.QMenu(self.menuNetwork)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menuNetwork)
        self.menu_2.setObjectName("menu_2")
        self.menuAbout = QtWidgets.QMenu(self.menubar)
        self.menuAbout.setObjectName("menuAbout")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionAbout = QtWidgets.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.actionUse_network_12 = QtWidgets.QAction(MainWindow)
        self.actionUse_network_12.setObjectName("actionUse_network_12")
        self.actionUse_network_21 = QtWidgets.QAction(MainWindow)
        self.actionUse_network_21.setObjectName("actionUse_network_21")
        self.actionRetrainNW = QtWidgets.QAction(MainWindow)
        self.actionRetrainNW.setObjectName("actionRetrainNW")
        self.actionUseAnotherNW222 = QtWidgets.QAction(MainWindow)
        self.actionUseAnotherNW222.setObjectName("actionUseAnotherNW222")
        self.action = QtWidgets.QAction(MainWindow)
        self.action.setObjectName("action")
        self.actionUse_network_1 = QtWidgets.QAction(MainWindow)
        self.actionUse_network_1.setObjectName("actionUse_network_1")
        self.actionUse_network_2 = QtWidgets.QAction(MainWindow)
        self.actionUse_network_2.setObjectName("actionUse_network_2")
        self.actionUseAnotherNW = QtWidgets.QAction(MainWindow)
        self.actionUseAnotherNW.setObjectName("actionUseAnotherNW")
        self.actionUseAnotherCNN = QtWidgets.QAction(MainWindow)
        self.actionUseAnotherCNN.setObjectName("actionUseAnotherCNN")
        self.actionOpenBase = QtWidgets.QAction(MainWindow)
        self.actionOpenBase.setObjectName("actionOpenBase")
        self.actionSaveImage = QtWidgets.QAction(MainWindow)
        self.actionSaveImage.setObjectName("actionSaveImage")
        self.actionSaveNumbers = QtWidgets.QAction(MainWindow)
        self.actionSaveNumbers.setObjectName("actionSaveNumbers")
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionSaveImage)
        self.menu.addAction(self.actionUse_network_1)
        self.menu.addAction(self.actionUse_network_2)
        self.menu_2.addAction(self.actionUseAnotherNW)
        self.menu_2.addAction(self.actionUseAnotherCNN)
        self.menuNetwork.addAction(self.actionRetrainNW)
        self.menuNetwork.addAction(self.menu.menuAction())
        self.menuNetwork.addAction(self.menu_2.menuAction())
        self.menuAbout.addAction(self.actionAbout)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuNetwork.menuAction())
        self.menubar.addAction(self.menuAbout.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "DigitRecognition"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabNew), _translate("MainWindow", "Исходное изображение"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabProcessed), _translate("MainWindow", "Обработанное нейросетью"))
        self.buttonNetwork1.setText(_translate("MainWindow", "Распознать\n"
"(полносвязная нейросеть)"))
        self.buttonNetwork2.setText(_translate("MainWindow", "Распознать\n"
"(свёрточная нейросеть)"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt;\">Распознанные цифры</span></p></body></html>"))
        self.results.setText(_translate("MainWindow", "<html><head/><body><p/></body></html>"))
        self.menuFile.setTitle(_translate("MainWindow", "Файл"))
        self.menuNetwork.setTitle(_translate("MainWindow", "Нейросеть"))
        self.menu.setTitle(_translate("MainWindow", "Использовать нейросеть"))
        self.menu_2.setTitle(_translate("MainWindow", "Выбрать другую..."))
        self.menuAbout.setTitle(_translate("MainWindow", "Справка"))
        self.actionOpen.setText(_translate("MainWindow", "Открыть"))
        self.actionAbout.setText(_translate("MainWindow", "О разработчике"))
        self.actionUse_network_12.setText(_translate("MainWindow", "Использовать полносвязную нейросеть"))
        self.actionUse_network_21.setText(_translate("MainWindow", "Использовать свёрточную нейросеть"))
        self.actionRetrainNW.setText(_translate("MainWindow", "Переобучить полносвязную нейросеть"))
        self.actionUseAnotherNW222.setText(_translate("MainWindow", "Выбрать другую полносвязную нейросеть (.json)"))
        self.action.setText(_translate("MainWindow", "Выбрать другую свёрточную нейросеть (.h5)"))
        self.actionUse_network_1.setText(_translate("MainWindow", "Использовать полносвязную нейросеть"))
        self.actionUse_network_2.setText(_translate("MainWindow", "Использовать свёрточную нейросеть"))
        self.actionUseAnotherNW.setText(_translate("MainWindow", "... полносвязную нейросеть (.json)"))
        self.actionUseAnotherCNN.setText(_translate("MainWindow", "... свёрточную нейросеть"))
        self.actionOpenBase.setText(_translate("MainWindow", "Открыть изображение из базы"))
        self.actionSaveImage.setText(_translate("MainWindow", "Сохранить обработанное изображение"))
        self.actionSaveNumbers.setText(_translate("MainWindow", "Сохранить текстовый файл с цифрами"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
