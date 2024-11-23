import numpy as np
import matplotlib.pyplot as plt

from MLPLinearAutoencoder import MLPLinearAutoencoder
from MultilayerPerceptronOfAdam import MultilayerPerceptronOfAdam


class MLPLinearAutoencoderOfAdam(MLPLinearAutoencoder, MultilayerPerceptronOfAdam):
    def __init__(self, encoder_layers, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        MLPLinearAutoencoder.__init__(self, encoder_layers, learning_rate=learning_rate)
        MultilayerPerceptronOfAdam.__init__(
            self,
            layer_sizes=encoder_layers + encoder_layers[-2::-1],
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
        )

    def calculate_weight_updates(self, weight_gradients):
        return MultilayerPerceptronOfAdam.calculate_weight_updates(self, weight_gradients)

    def initialize_weights(self):
        MLPLinearAutoencoder.initialize_weights(self)
        MultilayerPerceptronOfAdam.initialize_weights(self)
        for i in range(len(self.layer_sizes) - 1):
            self.weights[i] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * np.sqrt(
                2 / (self.layer_sizes[i] + self.layer_sizes[i + 1]))

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