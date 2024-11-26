import math
import numpy as np


class SumError:

    def __init__(self, activation_function=None):
        self._activation_function = activation_function

    # Recibe:
    # w: np.array con los valores actuales de los w_i
    # results: todos los datos analizados para el entrenamiento
    # expected: los resultados esperados para cada dato
    def compute(self, data, expected, w):
        ans = 0
        for i in range(len(data)):
            ans += abs(expected[i] - self._activation_function.eval(w.dot(data[i])))
        return ans


class AccuracyError:

    def __init__(self, activation_function=None):
        self._activation_function = activation_function

    def compute(self, data, expected, w):
        correct = 0.0
        for i in range(len(data)):
            if self._activation_function.eval(w.dot(data[i])) == expected[i]:
                correct += 1
        #   Devuelve un valor entre 0 y 1, si se quiere % multiplicar por 100 y tener cuidado con la condición
        return correct / len(data)


class QuadraticError:

    def __init__(self, activation_function=None):
        self._activation_function = activation_function

    def compute(self, data, expected, w):
        ans = 0
        for i in range(len(data)):
            ans += math.pow((expected[i] - self._activation_function.eval(w.dot(data[i]))), 2)
        return ans / 2.0


class QuadraticErrorMultilayer:

    def difference(self, obtained, expected):
        ans = 0
        for j in range(len(expected)):
            ans += (expected[j] - obtained[j]) ** 2
        return ans / 2.0

    # expected: [[]] con los valores esperados para cada caso
    def compute(self, data, mlp, expected):
        ans = 0
        for i in range(len(expected)):
            obtained = mlp.forward(data[i])
            # index_max = np.argmax(obtained)
            # obtained_transformed = [0 if idx!=index_max else 1 for idx in range(len(obtained))]
            for j in range(len(expected[i])):
                ans += (expected[i][j] - obtained[j]) ** 2
        return ans / 2.0


class CrossEntropyError:

    def __init__(self, activation_function=None):
        self._activation_function = activation_function

    # TODO: Compute passes data, mlp and expected, in that order
    def compute(self, data, expected, w):
        ans = 0.0
        for i in range(len(data)):
            y_pred = self._activation_function.eval(w.dot(data[i]))
            y_true = expected[i]

            # Calculate the cross-entropy error for each data point and accumulate
            ans -= y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)

        # Average the total error for all data points
        return ans / len(data)


class Norm2Error:
    def __init__(self, activation_function=None):
        self._activation_function = activation_function

    def difference(self, obtained, expected):
        return np.linalg.norm(expected - obtained)

    def compute(self, data, mlp, expected):
        """
        :param data: e.g.   ([0 0 1 0 0
                            0 0 1 0 0
                            0 0 0 1 0
                            0 0 0 0 0
                            0 0 0 0 0
                            0 0 0 0 0
                            0 0 0 0 0], [..])
        :param mlp: MultilayerPerceptron instance
        :return: || X_i - X'_i ||
        """
        ans = 0.0
        for i in range(len(data)):
            obtained = mlp.forward(data[i])
            ans += np.linalg.norm(expected[i] - obtained)  # ||xn - xn'||
        return ans / len(data)  # || x1 - x1'|| + ||x2 - x2'|| + ... + ||xn - xn'||


class PixelDiff:

    def difference(self, obtained, expected):
        diff = 0
        for j in range(len(expected)):
            diff += 1 if (expected[j] != round(obtained[j])) else 0
        return diff

    def compute(self, data, mlp, expected):
        max_diff = 0
        for i in range(len(data)):
            obtained = mlp.forward(data[i])
            diff = 0
            for j in range(len(data[i])):
                diff += 1 if (expected[i][j] != round(obtained[j])) else 0
                # diff += (data[i][j] - np.round(np.abs(obtained[j])))
            max_diff = max(max_diff, diff)
        return max_diff


def from_str(string, activation_function=None):
    match string.upper():
        case "QUADRATIC_MULTILAYER":
            return QuadraticErrorMultilayer()
        case "NORM2":
            return Norm2Error()
        case "PIXEL_DIFF":
            return PixelDiff()
        case _:
            return QuadraticErrorMultilayer()