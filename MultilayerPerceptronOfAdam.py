import numpy as np

from MultilayerPerceptron import MultilayerPerceptron


class MultilayerPerceptronOfAdam(MultilayerPerceptron):
    def __init__(self, layer_sizes, beta=1.0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 weight_updates_by_epoch=False):
        MultilayerPerceptron.__init__(self, layer_sizes, beta, learning_rate,
                                      weight_updates_by_epoch=weight_updates_by_epoch)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = None
        self.m_weights = None
        self.v_weights = None
        self.m_biases = None
        self.v_biases = None

    def initialize_weights(self):
        MultilayerPerceptron.initialize_weights(self)
        # Initialize Adam parameters (time step, moving averages of gradients and squared gradients)
        self.t = 0
        self.m_weights = [np.zeros_like(w) for w in self.weights]
        self.v_weights = [np.zeros_like(w) for w in self.weights]
        self.m_biases = [np.zeros_like(b) for b in self.biases]
        self.v_biases = [np.zeros_like(b) for b in self.biases]

    def calculate_weight_updates(self, weight_gradients, bias_gradients, learning_rate=None):
        if learning_rate is None:
            learning_rate = self.learning_rate
        self.t += 1
        weight_updates = [None] * len(self.weights)
        bias_updates = [None] * len(self.biases)
        for i in range(len(self.weights)):
            # Update biased first moment estimate (m) and second moment estimate (v) for weights
            self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * weight_gradients[i]
            self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * (weight_gradients[i] ** 2)
            # Update biased first moment estimate (m) and second moment estimate (v) for biases
            self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * bias_gradients[i]
            self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * (bias_gradients[i] ** 2)
            # Bias correction for weights
            m_weights_corr = self.m_weights[i] / (1 - self.beta1 ** self.t)
            v_weights_corr = self.v_weights[i] / (1 - self.beta2 ** self.t)
            # Bias correction for biases
            m_biases_corr = self.m_biases[i] / (1 - self.beta1 ** self.t)
            v_biases_corr = self.v_biases[i] / (1 - self.beta2 ** self.t)
            # Update weights using Adam update rule
            weight_updates[i] = -learning_rate * (
                        m_weights_corr / (np.sqrt(v_weights_corr) + self.epsilon) + 0.001 * self.weights[i])
            bias_updates[i] = -learning_rate * m_biases_corr / (np.sqrt(v_biases_corr) + self.epsilon)
        return weight_updates, bias_updates
