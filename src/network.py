import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gzip
import pickle
import json
import random
import sys
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path


class AbstractNetwork(ABC):
    """Абстрактный класс, обеспечивающий наличие трёх обязательных для нейросети методов: создания (тренировки),
    предсказания величины по поданным данным и сохранения в файл"""

    @abstractmethod
    def model_train(self):
        raise NotImplementedError()

    @abstractmethod
    def feedforward(self):
        raise NotImplementedError()

    @abstractmethod
    def save(self):
        raise NotImplementedError()


class MnistLoadNN:
    """Класс-оболочка для методов загрузки и обработки данных из локальной базы MNIST."""

    @staticmethod
    def __load_data():
        """Загрузка данных из базы."""
        f = gzip.open(Path("../data/resources/mnist.pkl.gz"), 'rb')
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
        f.close()
        return training_data, validation_data, test_data

    @staticmethod
    def load_data_wrap():
        """Обработка данных для подачи их нейросети."""
        tr_d, v_d, t_d = MnistLoadNN.__load_data()
        tr_images = [np.reshape(x, (784, 1)) for x in tr_d[0]]
        tr_numbers = [MnistLoadNN.numbers_to_vectors(y) for y in tr_d[1]]
        training_data = list(zip(tr_images, tr_numbers))
        v_images = [np.reshape(x, (784, 1)) for x in v_d[0]]
        validation_data = list(zip(v_images, v_d[1]))
        t_images = [np.reshape(x, (784, 1)) for x in t_d[0]]
        test_data = list(zip(t_images, t_d[1]))
        return training_data, validation_data, test_data

    @staticmethod
    def numbers_to_vectors(j):
        """Преобразует цифру в вектор."""
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e


class QuadraticCost:
    """Класс с квадратичной функцией потерь."""

    @staticmethod
    def fn(a, y):
        """Возвращает потери, связанные с выходом a и таргетным значением y."""
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        """Возвращает дельту ошибки выходного слоя."""
        return (a - y) * sigmoid_prime_vec(z)


class CrossEntropyCost:
    """Класс с функцией потерь кросс-энтропия."""

    @staticmethod
    def fn(a, y):
        """Возвращает потери, связанные с выходом a и таргетным значением y."""
        # Nan_to_num использована, чтобы не получить NaN в случае, если под знаком логаримфа будет ноль.
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        """Возвращает дельту ошибки выходного слоя. Первый параметр z здесь ради обеспечения совместимости с методом
        delta класса квадратичной функции потерь."""
        return a - y


class Network(AbstractNetwork):
    """Основной класс нейросети."""

    def __init__(self, sizes, cost=CrossEntropyCost):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).
        Список sizes хранит число нейронов на последовательных слоях
        нейросети. Bias'ы и веса для нейросети инициализируются
        рандомно, с помощью метода default_weight_initializer().
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.__large_weight_initializer()
        self.cost = cost

    def __default_weight_initializer(self):
        """Один из вариантов инициализации весов и bias'ов.
        Первый слой - входной, поэтому для него bias'ы не выставляются,
        поскольку они используются только при расчётах выходов из
        последующих слоёв.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in list(zip(self.sizes[:-1], self.sizes[1:]))]

    def __large_weight_initializer(self):
        """Один из вариантов инициализации весов и bias'ов.
        Первый слой - входной, поэтому для него bias'ы не выставляются,
        поскольку они используются только при расчётах выходов из
        последующих слоёв.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in list(zip(self.sizes[:-1], self.sizes[1:]))]

    def feedforward(self, a, vect=False):
        """Возвращает выходное значение нейросети при вводе a."""
        for b, w in list(zip(self.biases, self.weights)):
            a = sigmoid_vec(np.dot(w, a) + b)
        if not vect:
            a = np.argmax(a)
        return a

    def model_train(self, training_data, epochs, batch_size, eta,
                    lmbda=0.0,
                    evaluation_data=None,
                    monitor_evaluation_cost=False,
                    monitor_evaluation_accuracy=False,
                    monitor_training_cost=False,
                    monitor_training_accuracy=False):
        """Тренировка нейросети с использованием батчевого стохастического
        градиентного спуска. Training_data - список кортежей (x, y),
        которые представляют собой тренировочные входные данные
        и таргетные значения. Остальные обязательные параметры -
        количество эпох, размер батча, скорость обучения eta. lmbda -
        параметр регуляризации. evaluation_data - валидационные или
        тестовые данные. Есть возможность мониторить стоимость и точность
        на evaluation_data или training_data, проставив соответствующие
        флаги. Метод возвращает кортеж из четырёх списков: стоимости
        и точности на evaluation_data и на training_data, подсчитываемые
        в конце каждой эпохи. Списки пустые, если не выставлен
        сооответствующий флаг.
        """
        n_data = None
        if evaluation_data:
            n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            batches = [
                training_data[k:k + batch_size]
                for k in range(0, n, batch_size)]
            for batch in batches:
                self.__update_batch(
                    batch, eta, lmbda, len(training_data))
            print("Epoch {} training complete".format(j))
            if monitor_training_cost:
                cost = self.__total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.__accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(
                    accuracy, n))
            if monitor_evaluation_cost:
                cost = self.__total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.__accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(
                    accuracy, n_data))
            print("\n")
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def __update_batch(self, batch, eta, lmbda, n):
        """Обновление весов и bias'ов нейросети. К одиночному батчу
        применяется градиентный спуск с использованием метода
        обратного распространения ошибки (backpropagation).
        batch - список кортежей (x, y), eta - скорость
        обучения, lmbda - параметр регуляризации, n - общий
        размер тренировочного датасета
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.backpropagation(x, y)
            nabla_b = [nb + dnb for nb, dnb in list(zip(nabla_b, delta_nabla_b))]
            nabla_w = [nw + dnw for nw, dnw in list(zip(nabla_w, delta_nabla_w))]
        self.weights = [(1 - eta * (lmbda / n)) * w - (eta / len(batch)) * nw
                        for w, nw in list(zip(self.weights, nabla_w))]
        self.biases = [b - (eta / len(batch)) * nb
                       for b, nb in list(zip(self.biases, nabla_b))]

    def backpropagation(self, x, y):
        """Возвращает кортеж (nabla_b, nabla_w), представляющий собой
        градиент функции потерь C_x. nabla_b и nabla_w
        представляют собой послойные списки массивов numpy,
        подобно self.biases и self.weights
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Подача данных.
        activation = x
        activations = [x]  # Список со всеми значениями функции активации по слоям.
        zs = []  # Список, хранящие все векторы z послойно.
        for b, w in list(zip(self.biases, self.weights)):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid_vec(z)
            activations.append(activation)
        # Обратная проходка.
        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for k in range(2, self.num_layers):
            z = zs[-k]
            spv = sigmoid_prime_vec(z)
            delta = np.dot(self.weights[-k + 1].transpose(), delta) * spv
            nabla_b[-k] = delta
            nabla_w[-k] = np.dot(delta, activations[-k - 1].transpose())
        return nabla_b, nabla_w

    def __accuracy(self, data, convert=False):
        """Точность возвращает число входных данных из data, для которых
        нейросеть выдала правильный результат. Предполагается, то
        выход нейросети - индекс нейрона с самым высоким значением
        функции активации на последнем слое.
        Флаг convert должен быть False, если датасет - валидационные
        или тестовые данные, и True - если тренировочные данные.
        """
        if convert:
            results = [(np.argmax(self.feedforward(x, True)), np.argmax(y)) for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x, True)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def __total_cost(self, data, lmbda, convert=False):
        """Возвращает общую "стоимость" (общие потери) для датасета data.
        Флаг convert должен быть False, если датасет - тренировочные
        данные и True, если датасет - валидационные или тестовые данные.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x, True)
            if convert:
                y = vectorized_result(y)
            cost += self.cost.fn(a, y) / len(data)
        cost += 0.5 * (lmbda / len(data)) * sum(
            np.linalg.norm(w) ** 2 for w in self.weights)
        return cost

    def save(self, filename="../data/networks/mynetwork1.json"):
        """Сохранить готовую нейросеть в файл filename"""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


# Загрузка нейросети.
def load(filename="../data/networks/mynetwork1.json"):
    """Загрузить существующую нейросеть из файла filename"""
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


# Различные функции
def vectorized_result(j):
    """Возвращает десятимерный вектор с 1.0 на j-той позиции и с нулями на остальных. Используется для
    конвертации цифры в соответствующий желаемый выход из нейросети."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def sigmoid(z):
    """Сигмоидная функция"""
    np.seterr(over='ignore')
    return 1.0 / (1.0 + np.exp(-z))


sigmoid_vec = np.vectorize(sigmoid)


def sigmoid_prime(z):
    """Производная сигмоидной функции."""
    # numpy.expit вместо своего sigmoid используется из-за того, что такая функция выдаёт результат
    # даже при очень больших значениях аргумента.
    return sigmoid(z) * (1 - sigmoid(z))


sigmoid_prime_vec = np.vectorize(sigmoid_prime)
