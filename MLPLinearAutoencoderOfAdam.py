import numpy as np

from MLPLinearAutoencoder import MLPLinearAutoencoder
from MultilayerPerceptronOfAdam import MultilayerPerceptronOfAdam


class MLPLinearAutoencoderOfAdam(MLPLinearAutoencoder, MultilayerPerceptronOfAdam):
    def __init__(self, encoder_layers, beta=1.0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, training_level=1):
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
        MLPLinearAutoencoder.__init__(self, encoder_layers, beta, learning_rate, training_level=training_level)
        MultilayerPerceptronOfAdam.__init__(
            self,
            encoder_layers + encoder_layers[-2::-1],  # Full autoencoder architecture
            beta,
            learning_rate,
            beta1,
            beta2,
            epsilon,
            training_level=training_level
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
            self.weights[i] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * np.sqrt(
                2 / (self.layer_sizes[i] + self.layer_sizes[i + 1]))
            # if i == len(self.encoder_layers) - 2:
            #     # He initialization for the latent layer
            #     self.weights[i] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * np.sqrt(
            #         2 / (self.layer_sizes[i]))
            # elif i == len(self.layer_sizes) - 2:
            #     # Xavier initialization for the output layer
            #     self.weights[i] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * np.sqrt(
            #         2 / (self.layer_sizes[i] + self.layer_sizes[i + 1]))
            # else:
            #     # He initialization for the remaining layers
            #     self.weights[i] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * np.sqrt(
            #         2 / self.layer_sizes[i])
            self.biases[i] = np.zeros((1, self.layer_sizes[i + 1]))

    # Other methods from MLPLinearAutoencoder, such as encode, decode, reconstruct, and train_autoencoder,
    # are inherited without changes, as they are compatible with Adam optimization.


# Example usage
if __name__ == "__main__":
    encode_layers = [5, 5, 2]  # Example architecture
    learnin_rate = 0.001
    autoencoder = MLPLinearAutoencoderOfAdam(encode_layers, learning_rate=learnin_rate)

    # Example input
    x = np.rint(np.random.rand(5, 5))  # 5 samples with 5 features each

    # Train the autoencoder
    autoencoder.train_autoencoder(x, epoch_limit=np.inf, error_limit=1)

    # Reconstruct data
    reconstructions = autoencoder.reconstruct(x)
    print("Reconstructed Data:", reconstructions)
