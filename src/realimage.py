import cv2
import numpy as np
import network
from PIL import Image


class RealImageProcessing:
    """Класс, предназначенный для обработки нетестовых картинок: выделяет на картинке цифры и подаёт их нейросетям
    для распознания."""
    def __init__(self, filename):
        """Создание объекта, содержащего путь к файлу, саму картинку, набор вырезанных из неё цифр,
        распознанные числа."""
        self.imagepath = filename
        self.image = cv2.imread(filename)
        self.contoured_images = []
        self.digits = []
        self.__generate_preprocessed_digits()

    def __generate_preprocessed_digits(self):
        """С помощью средств opencv находит контуры цифр, выделяет их, заносит их изображения в объект класса."""
        grey = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # # Сортировка слева направо и сверху вниз
        # sort_contours = sorted([(c, cv2.boundingRect(c)[0] + cv2.boundingRect(c)[1]*3000) for c in contours],
        #                        key=lambda tup: tup[1])
        # contours = [x for (x, y) in sort_contours]
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(self.image, (x-5, y-5), (x+w+5, y+h+5), color=(0, 255, 0), thickness=2)
            digit = thresh[y:y+h, x:x+w]
            resized_digit = cv2.resize(digit, (18, 18))
            padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)
            self.contoured_images.append((padded_digit, x, y))

    def predict_mult_digits(self, nwork=None):
        """Распознаёт множество выделенных цифр, заносит их в объект класса, возвращает массив с ними."""
        for (digit_image, x, y) in self.contoured_images:
            if isinstance(nwork, network.Network):
                number = nwork.feedforward(digit_image.reshape(784, 1))
            else:
                number = nwork.feedforward(digit_image.reshape(1, 28, 28, 1))
            cv2.putText(self.image, str(int(number)), (x, y - 10), cv2.FONT_ITALIC, 2, (3, 3, 3), 3)
            self.digits.append(number)
        # print("\n\n\n----------------Contoured Image--------------------")
        # plt.imshow(self.image, cmap="gray")
        # plt.show()
        cv2.imwrite("../data/resources/zbuf.jpg", self.image)
        return self.digits
