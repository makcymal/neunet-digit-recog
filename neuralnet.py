import numpy as np


class NeuralNet:

    def __init__(self):

        # структура нейросети — количество узлов
        self.structure = [400, 600, 10]
        # коэффицент скорости обучения нейронной сети
        self.learn_rate = 0.25
        # количество эпох — повторений обучения на одинаковом наборе данных
        self.epochs = 6
        # показатель эффективности распознавания
        self.efficiency = 0
        # протяженность нейронной сети, количество слоев
        self.length = len(self.structure)
        # матрица выходных значений на каждом слое
        self.layers = [0 for i in range(self.length)]
        # матрица ошибок между слоями
        self.errors = [0 for i in range(self.length - 1)]
        # матрица весовых коэффицентов между слоями
        self.weights = [0 for i in range(self.length - 1)]
        # заполнение случайными числами от -0,5 до 0,5
        for i in range(self.length - 1):
            self.weights[i] = np.random.rand(self.structure[i + 1], self.structure[i])
            self.weights[i] -= 0.5

    def learn(self, record):

        # разбиение строки в список по пробелам
        record = record.split()
        # входные данные — список точек
        inputs = np.asfarray(record[1:])
        # нормализация входных данных к диапазону от 0.01 до 1.0
        inputs = inputs / 255.0 * 0.99 + 0.01

        # маркерное значение соответствует правильному ответу
        marker = int(record[0])
        # целевые данные представляются в виде списка нулей
        targets = np.zeros(self.structure[-1])
        # все целевые значения увеличиваются на 0.01
        targets += 0.01
        # целевое значение с правильным ответом равно 0.99
        targets[marker] = 0.99

        # преобразование входных и целевых значений в транспонированный двумерный массив
        inputs = np.array(inputs, ndmin=2).T
        targets = np.array(targets, ndmin=2).T

        # расчет работы каждого слоя
        self.layers[0] = inputs
        # перебор слоев нейросети, начиная со второго
        for i in range(1, self.length):
            # умножение матрицы весовых коэффицентов и матрицы предыдущего слоя
            layer = np.dot(self.weights[i - 1], self.layers[i - 1])
            # применение функции активации
            self.layers[i] = expit(layer)

        # расчет ошибок между слоями
        self.errors[-1] = targets - self.layers[-1]
        # перебор матриц ошибок от последнего слоя к первому
        for i in range(-2, -self.length, -1):
            # умножение матрицы весовых коэффицентов и матрицы ошибок предыдущего слоя
            self.errors[i] = np.dot(self.weights[i + 1].T, self.errors[i + 1])

        # перебор матриц весовых коэффицентов
        for i in range(-1, -self.length, -1):
            wei = np.dot(self.errors[i] * self.layers[i] * (1 - self.layers[i]), np.transpose(self.layers[i - 1]))
            # обновление весов между слоями
            self.weights[i] += self.learn_rate * wei

    def poll(self, record):

        # разбиение строки в список по пробелам
        record = record.split()
        # представление входных данных в виде списка вещественных чисел
        inputs = np.asfarray(record)
        # нормализация входных данных к диапазону от 0.01 до 1.0
        inputs = inputs / 255.0 * 0.99 + 0.01
        # преобразование массива входных значение в двумерный массив
        inputs = np.array(inputs, ndmin=2).T

        # вычисление результата работы
        # первый слой равен входным значениям
        self.layers[0] = inputs
        # перебор слоев нейросети, начиная с первого
        for i in range(1, self.length):
            # умножение матрицы весовых коэффицентов и матрицы предыдущего слоя
            layer = np.dot(self.weights[i - 1], self.layers[i - 1])
            # применение функции активации
            self.layers[i] = expit(layer)

        # ответом нейросети является индекс наибольшего элемента
        output = np.argmax(self.layers[-1])
        # выход из функции и возврат значения переменной output
        return output
