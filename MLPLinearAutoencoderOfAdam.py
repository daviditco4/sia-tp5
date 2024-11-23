import numpy as np
import matplotlib.pyplot as plt

from MLPLinearAutoencoder import MLPLinearAutoencoder
from MultilayerPerceptronOfAdam import MultilayerPerceptronOfAdam


class MLPLinearAutoencoderOfAdam(MLPLinearAutoencoder, MultilayerPerceptronOfAdam):
    def __init__(self, encoder_layers, beta=1.0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize a linear autoencoder with Adam optimization.
        Parameters:
        - encoder_layers: List of integers defining the encoder architecture, e.g., [input_size, hidden1, latent_size].
        - learning_rate: Learning rate for Adam optimization.
        - beta1: Exponential decay rate for the first moment estimates (Adam parameter).
        - beta2: Exponential decay rate for the second moment estimates (Adam parameter).
        - epsilon: Small constant to prevent division by zero (Adam parameter).
        """
        # Call the parent constructors
        MLPLinearAutoencoder.__init__(self, encoder_layers, beta, learning_rate)
        MultilayerPerceptronOfAdam.__init__(
            self,
            encoder_layers + encoder_layers[-2::-1],  # Full autoencoder architecture
            beta,
            learning_rate,
            beta1,
            beta2,
            epsilon,
        )

    def calculate_weight_updates(self, weight_gradients, bias_gradients, learning_rate=None):
        """
        Override the weight and bias update calculation to use Adam optimization.
        """
        return MultilayerPerceptronOfAdam.calculate_weight_updates(self, weight_gradients, bias_gradients,
                                                                   learning_rate)

    def initialize_weights(self):
        """
        Override weight and bias initialization to ensure compatibility with Adam parameters and autoencoder structure.
        """
        MLPLinearAutoencoder.initialize_weights(self)  # Initialize weights
        MultilayerPerceptronOfAdam.initialize_weights(self)  # Initialize Adam-specific parameters
        for i in range(len(self.layer_sizes) - 1):
            if i == len(self.encoder_layers) - 2:
                # He initialization for the latent layer
                self.weights[i] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * np.sqrt(
                    2 / (self.layer_sizes[i]))
            elif i == len(self.layer_sizes) - 2:
                # Xavier initialization for the output layer
                self.weights[i] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * np.sqrt(
                    2 / (self.layer_sizes[i] + self.layer_sizes[i + 1]))
            else:
                # He initialization for the remaining layers
                self.weights[i] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * np.sqrt(
                    2 / self.layer_sizes[i])
            self.biases[i] = np.zeros((1, self.layer_sizes[i + 1]))

"""
    def train_autoencoder(self, x, epoch_limit, error_limit):
        y = x
        for epoch in range(int(epoch_limit)):
            error = MultilayerPerceptron.train(self, x, y, 1, error_limit)
            self.epoch_errors.append(error)
            if error <= error_limit:
                break
        return self.weights, min(self.epoch_errors), epoch + 1

    def plot_error_vs_epochs(self):
        plt.plot(self.epoch_errors)
        plt.xlabel('Epochs')
        plt.ylabel('Average Error')
        plt.title('Average Error vs. Number of Epochs')
        plt.grid(True)
        plt.show()

"""
# Example usage
if __name__ == "__main__":
    encode_layers = [5, 5, 2]
    learning_rate = 0.001
    autoencoder = MLPLinearAutoencoderOfAdam(encode_layers, learning_rate=learning_rate)

    x = np.rint(np.random.rand(5, 5))

    autoencoder.train_autoencoder(x, epoch_limit=1000, error_limit=0.01)
    #autoencoder.plot_error_vs_epochs()