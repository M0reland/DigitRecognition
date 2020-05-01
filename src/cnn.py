import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from network import AbstractNetwork
from keras.datasets import mnist
from keras.models import load_model
from keras.layers import MaxPooling2D, Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.utils import to_categorical
from keras.preprocessing.image import load_img
from keras.layers.normalization import BatchNormalization


class MnistLoadCNN:
    """Класс для загрузки тренировочных и тестовых данных из базы MNIST."""

    def __init__(self):
        """Создание объекта с данными из базы MNIST."""
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.__load_data()
        self.__load_data_wrap()

    def __load_data(self):
        """Загрузка данных из базы MNIST с использованием стандартных средств библиотеки Keras."""
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

    def __load_data_wrap(self):
        """Придание данным правильного вида для обработки нейросетью."""
        self.x_train = self.x_train.reshape(60000, 28, 28, 1)
        self.x_test = self.x_test.reshape(10000, 28, 28, 1)
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)


class ConvolutedNeuralNetwork(AbstractNetwork):
    """Свёрточная нейросеть с использованием библиотеки Keras."""
    def __init__(self, mod=Sequential()):
        """Создание пустой модели (если не подать аргумент) или копирование существующей."""
        self.model = mod

    def model_compile(self):
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
        self.model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(BatchNormalization())
        self.model.add(Dense(512, activation="relu"))

        self.model.add(Dense(10, activation="softmax"))
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    def model_train(self, epoch=5):
        """Тренировка модели с заданным количеством эпох."""
        mnist_data = MnistLoadCNN()
        self.model.fit(mnist_data.x_train, mnist_data.y_train,
                       validation_data=(mnist_data.x_test, mnist_data.y_test), epochs=epoch)

    def save(self, filename="../data/networks/mlp_digits_28x28.h5"):
        """Сохраняет нейросеть в файл filename."""
        self.model.save(filename)

    def feedforward(self, a):
        prediction = self.model.predict(a)
        number = np.argmax(prediction)
        return number

    def feedforward_base(self, number_in_base=0):
        """Получение цифры на основе n-ной картинки из базы MNIST."""
        mnist_data = MnistLoadCNN()
        example = mnist_data.x_test[number_in_base]
        prediction = self.model.predict(example.reshape(1, 28, 28, 1))
        number = np.argmax(prediction)
        # plt.imshow(example.reshape(28, 28), cmap="gray")
        # plt.show()
        # print("\nOutput number: {}".format(number))
        return number


def load(filename="../data/networks/mlp_digits_28x28.h5"):
    """Загрузка нейросети из файла filename."""
    return load_model(filename)
