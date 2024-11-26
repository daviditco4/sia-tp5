import sys
import numpy as np


class GradientDescent:
    def __init__(self, eta):
        self.eta = eta

    #  El método que da los deltas sólo lo multiplica por el eta
    def get_deltas(self, gradients):
        deltas = []
        for layer in gradients:
            deltas.append(self.eta * layer)
        return deltas

    def on_epoch(self, new_error):
        pass


class AdaptiveEta:
    def __init__(self, config, initial_eta):
        self.last_error = sys.float_info.max
        self.error_tendency = 0
        self.iterations_increment = config['adaptive_eta_iterations_increment']
        self.increment = config['adaptive_eta_increment']
        self.iterations_decrement = config['adaptive_eta_iterations_decrement']
        self.decrement = config['adaptive_eta_decrement']
        self.eta = initial_eta

    # El método que da los deltas sólo lo multiplica por el eta
    def get_deltas(self, gradients):
        deltas = []
        for layer in gradients:
            deltas.append(self.eta * layer)
        return deltas

    def on_epoch(self, new_error):
        if self.last_error > new_error:
            if self.error_tendency < 0:
                self.error_tendency = 0
            self.error_tendency += 1
        if new_error >= self.last_error:
            if self.error_tendency > 0:
                self.error_tendency = 0
            self.error_tendency -= 1
        self.last_error = new_error
        if self.error_tendency >= self.iterations_increment:
            self.eta += self.increment
            self.error_tendency = 0
        if self.error_tendency <= self.iterations_decrement:
            self.eta -= self.decrement * self.eta
            self.error_tendency = 0


class Momemtum:

    def __init__(self, config, perceptrons_per_layer):
        self.beta = config["momentum_beta"]
        self.alpha = config["momentum_alpha"]
        self.m = [np.zeros((perceptrons_per_layer[indx], perceptrons_per_layer[indx - 1] + 1)) for indx in
                  range(len(perceptrons_per_layer) - 1, 0, -1)]

    def get_deltas(self, gradients):
        deltas = []
        for (j, gradient) in enumerate(gradients):
            self.m[j] = self.beta * self.m[j] + (1 - self.beta) * gradient
            deltas.append(self.alpha * self.m[j])

        # Update weights
        return deltas

    def on_epoch(self, new_error):
        pass


class ADAM:

    def __init__(self, config, perceptrons_per_layer):
        self.beta1 = config["adam_beta1"]
        self.beta2 = config["adam_beta2"]
        self.epsilon = config["adam_epsilon"]
        self.alpha = config["adam_alpha"]
        self.m = [np.zeros((perceptrons_per_layer[indx], perceptrons_per_layer[indx - 1] + 1)) for indx in
                  range(len(perceptrons_per_layer) - 1, 0, -1)]
        self.v = [np.zeros((perceptrons_per_layer[indx], perceptrons_per_layer[indx - 1] + 1)) for indx in
                  range(len(perceptrons_per_layer) - 1, 0, -1)]
        self.t = 1

    def get_deltas(self, gradients):
        deltas = []
        for (j, gradient) in enumerate(gradients):
            self.m[j] = self.beta1 * self.m[j] + (1 - self.beta1) * gradient
            self.v[j] = self.beta2 * self.v[j] + (1 - self.beta2) * (gradient ** 2)

            # Compute bias-corrected first and second moment estimates
            m_hat = self.m[j] / (1 - self.beta1 ** self.t)
            v_hat = self.v[j] / (1 - self.beta2 ** self.t)

            self.t += 1

            deltas.append(self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon))

        # Update weights
        return deltas

    def on_epoch(self, new_error):
        pass


def from_str(string, config, eta, perceptrons_per_layer):
    match string.upper():
        case "ADAM":
            return ADAM(config=config, perceptrons_per_layer=perceptrons_per_layer)
        case "ADAPTIVE_ETA":
            return AdaptiveEta(config=config, initial_eta=eta)
        case "GRADIENT_DESCENT":
            return GradientDescent(eta=eta)
        case "MOMENTUM":
            return Momemtum(config=config, perceptrons_per_layer=perceptrons_per_layer)
        case _:
            return GradientDescent(eta=eta)