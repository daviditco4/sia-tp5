import numpy as np
from MLPLinearAutoencoder import MLPLinearAutoencoder


class MLPLinearAutoencoderOfAdam(MLPLinearAutoencoder):
    def __init__(self, encoder_layers, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize a deep autoencoder with Adam optimizer.
        Parameters:
        - encoder_layers: List of integers defining the encoder architecture, including input and latent sizes.
        - learning_rate: Learning rate for weight updates.
        - beta1: Exponential decay rate for the first moment estimates.
        - beta2: Exponential decay rate for the second moment estimates.
        - epsilon: Small value to prevent division by zero in Adam.
        """
        super().__init__(encoder_layers, learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = None
        self.m_weights = None
        self.v_weights = None

    def initialize_weights(self):
        super().initialize_weights()
        # Initialize Adam parameters (time step, moving averages of gradients and squared gradients)
        self.t = 0
        self.m_weights = [np.zeros_like(w) for w in self.weights]
        self.v_weights = [np.zeros_like(w) for w in self.weights]

    def calculate_weight_updates(self, weight_gradients):
        self.t += 1
        weight_updates = [None] * len(self.weights)
        for i in range(len(self.weights)):
            # Update biased first moment estimate (m) and second moment estimate (v) for weights
            self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * weight_gradients[i]
            self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * (weight_gradients[i] ** 2)
            # Bias correction for weights
            m_weights_corr = self.m_weights[i] / (1 - self.beta1 ** self.t)
            v_weights_corr = self.v_weights[i] / (1 - self.beta2 ** self.t)
            # Update weights using Adam update rule
            weight_updates[i] = -self.learning_rate * m_weights_corr / (np.sqrt(v_weights_corr) + self.epsilon)
        return weight_updates