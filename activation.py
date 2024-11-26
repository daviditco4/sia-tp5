import math
from utils.Function import Function

class SignActivation(Function):
    def eval(self, x):
        return 1 if x >= 0 else -1

    def diff(self, x):
        return 1

    def scale(self, expected):
        return expected

class ReLUActivation(Function):
    def eval(self, x):
        return max(0, x)

    def diff(self, x):
        return 1 if x > 0 else 0

    def scale(self, expected):
        return expected



class LinearActivation(Function):
    def eval(self, x):
        return x

    def diff(self, x):
        # TODO: preguntar
        return 1

    def scale(self, expected):
        return expected


class SigmoidActivation(Function):

    def __init__(self, beta):
        self._beta = beta

    def eval(self, x):
        return math.tanh(self._beta * x)

    def diff(self, x):
        return self._beta * (1 - math.pow(self.eval(x), 2))

    def scale(self, expected):
#         Deben estar entre -1 y 1
#         En min debe dar -1, en max debe dar 1
        min_val = min(expected)
        max_val = max(expected)
        m = ((1-(-1))/(max_val-min_val))
        b = 1 - m*max_val
        for i in range(len(expected)):
            expected[i] = m * expected[i] + b


class LogisticActivation(Function):

    def __init__(self, beta):
        self._beta = beta

    def eval(self, x):
        return 1.0 / (1 + math.exp(-2 * self._beta * x))

    def diff(self, x):
        return 2 * self._beta * self.eval(x) * (1 - self.eval(x))

    def scale(self, expected):
    #   Deben estar entre 0 y 1
    #   en min debe dar 0, en max debe dar 1
        min_val = min(expected)
        max_val = max(expected)
        m = (1/(max_val-min_val))
        b = 1 - m*max_val
        for i in range(len(expected)):
            expected[i] = m*expected[i] + b

def from_str(string, beta=1):
    match string.upper():
        case "SIGN":
            return SignActivation()
        case "LINEAR":
            return LinearActivation()
        case "SIGMOID":
            return SigmoidActivation(beta)
        case "LOGISTIC":
            return LogisticActivation(beta)
        case "RELU":
            return ReLUActivation()
        case _:
            return SignActivation()