from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import os
import cnn
import network
import realimage
import sys
from UiMainWindow import Ui_MainWindow
from UiNetworkTraining import Ui_NetworkTraining


class Window(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.buttonNetwork1.clicked.connect(self.handle_button_network1)
        self.ui.buttonNetwork2.clicked.connect(self.handle_button_network2)
        self.ui.actionOpen.triggered.connect(self.trigger_action_open)
        self.ui.actionAbout.triggered.connect(self.trigger_action_about)
        self.ui.actionUseAnotherNW.triggered.connect(self.trigger_action_use_another_nw)
        self.ui.actionRetrainNW.triggered.connect(self.trigger_action_retrain_nw1)
        self.ui.actionUse_network_1.triggered.connect(self.trigger_action_nw1)
        self.ui.actionUse_network_2.triggered.connect(self.trigger_action_nw2)
        self.ui.actionSaveImage.triggered.connect(self.trigger_action_save_image)
        self.ui.actionUseAnotherCNN.triggered.connect(self.trigger_action_use_another_cnn)
        self.network1 = network.load("../data/networks/network_30ep.json")
        self.network2 = cnn.ConvolutedNeuralNetwork(cnn.load("../data/networks/model_10epochs.h5"))
        self.image_filename = ""
        self.ui.tabWidget.setCurrentIndex(0)
        self.dialog = SecondaryWindow()

    def button_image_change(self, numbers):
        """Устанавливает картину после того, как она была обработана одной из нейросетей"""
        self.ui.results.clear()
        self.ui.imageProcessed.clear()
        self.ui.results.setText(" ".join(str(x) for x in numbers))
        pixmap = QPixmap("../data/resources/zbuf.jpg").scaled(720, 540, Qt.KeepAspectRatio)
        self.ui.imageProcessed.setPixmap(pixmap)
        self.ui.tabWidget.setCurrentIndex(1)

    def handle_button_network1(self):
        """Применяет полносвязную нейросеть"""
        if self.image_filename == "":
            return
        image_process = realimage.RealImageProcessing(self.image_filename)
        numbers = image_process.predict_mult_digits(self.network1)
        self.button_image_change(numbers)

    def handle_button_network2(self):
        """Применяет свёрточную нейросеть"""
        if self.image_filename == "":
            return
        image_process = realimage.RealImageProcessing(self.image_filename)
        numbers = image_process.predict_mult_digits(self.network2)
        self.button_image_change(numbers)

    def trigger_action_nw1(self):
        """Применяет полносвязную нейросеть"""
        self.handle_button_network1()

    def trigger_action_nw2(self):
        """Применяет свёрточную нейросеть"""
        self.handle_button_network2()

    def trigger_action_open(self):
        """Позволяет открыть изображение для обработки"""
        self.ui.imageNew.clear()
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Открыть файл изображения", "../data/images/ex1.jpg",
                                                            "Image Files (*.jpg *.png *.bmp *.jpeg)")
        if filename == "":
            return
        self.image_filename = filename
        self.ui.imageNew.setPixmap(QPixmap(filename).scaled(720, 540, Qt.KeepAspectRatio))
        self.ui.tabWidget.setCurrentIndex(0)

    def trigger_action_save_image(self):
        """Позволяет сохранить обработанное изображение"""
        if self.ui.imageProcessed.pixmap() is None:
            return
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Сохранить файл изображения",
                                                            "../data/output_files/ex1.jpg", "Image File (*.jpg)")
        if filename == "":
            return
        pix = self.ui.imageProcessed.pixmap().toImage()
        pix.save(filename, "jpg")

    def trigger_action_about(self):
        """Справка о разработчике"""
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("О разработчике")
        msg.setText("Программа разработана студентом группы ИВТ-3\n физико-технического факультета Храмогином А.А.")
        msg.setIcon(QtWidgets.QMessageBox.Information)
        x = msg.exec_()

    def trigger_action_use_another_nw(self):
        """Позволяет выбрать другую полносвязную нейросеть"""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Открыть файл .json", "../data/networks/nw1.json",
                                                            "Json Files (*.json)")
        if filename == "":
            return
        self.network1 = network.load(filename)

    def trigger_action_use_another_cnn(self):
        """Позволяет выбрать другую свёрточную нейросеть"""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Открыть файл .h5", "../data/networks/nw1.h5",
                                                            "h5 (*.h5)")
        if filename == "":
            return
        self.network2 = cnn.ConvolutedNeuralNetwork(cnn.load(filename))

    def trigger_action_retrain_nw1(self):
        """Позволяет обучить новую полносвязную нейросеть"""
        self.dialog.show()


class SecondaryWindow(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.ui = Ui_NetworkTraining()
        self.ui.setupUi(self)
        self.ui.buttonTrain.clicked.connect(self.handle_button_train)
        self.ui.buttonExit.clicked.connect(self.handle_button_exit)

    def handle_button_train(self):
        """Начать обучение нейросети"""
        layers = self.ui.neuronInput_1.text() + " " + self.ui.neuronInput_2.text() + " " + self.ui.neuronInput_3.text()
        numbers = [int(x) for x in layers.split()]
        print(numbers)
        net1 = network.Network(numbers, cost=network.CrossEntropyCost)
        epochs = int(self.ui.epochs.text())
        batch_size = int(self.ui.batch_size.text())
        eta = float(self.ui.eta.text())
        lmbd = float(self.ui.lmbda.text())
        filename = self.ui.neuronInput_4.text()
        if filename == "":
            return
        filename = "../data/networks/" + filename + ".json"
        training_data, validation_data, test_data = network.MnistLoadNN.load_data_wrap()
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("Внимание!")
        msg.setText("Обучение нейросети началось. Пожалуйста, дождитесь сообщения об успешном завершении.")
        msg.exec_()
        net1.model_train(training_data, epochs, batch_size, eta, lmbda=lmbd)
        msg.setText("Обучение завершено! Теперь программа сохранит нейросеть по указанному пути.")
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.exec_()
        net1.save(filename)

    def handle_button_exit(self):
        """Закрыть окно"""
        self.close()


def start():
    """Запустить программу"""
    m = Window()
    m.show()
    return m


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = start()
    path = "../data/resources/zbuf.jpg"
    if os.path.exists(path):
        os.remove(path)
    sys.exit(app.exec_())
