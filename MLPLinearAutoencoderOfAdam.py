import numpy as np

from MLPLinearAutoencoder import MLPLinearAutoencoder
from MultilayerPerceptronOfAdam import MultilayerPerceptronOfAdam


class MLPLinearAutoencoderOfAdam(MLPLinearAutoencoder, MultilayerPerceptronOfAdam):
    def __init__(self, encoder_layers, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
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
        MLPLinearAutoencoder.__init__(self, encoder_layers, learning_rate=learning_rate)
        MultilayerPerceptronOfAdam.__init__(
            self,
            layer_sizes=encoder_layers + encoder_layers[-2::-1],  # Full autoencoder architecture
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
        )

    def calculate_weight_updates(self, weight_gradients):
        """
        Override the weight update calculation to use Adam optimization.
        """
        return MultilayerPerceptronOfAdam.calculate_weight_updates(self, weight_gradients)

    def initialize_weights(self):
        """
        Override weight initialization to ensure compatibility with Adam parameters and autoencoder structure.
        """
        MLPLinearAutoencoder.initialize_weights(self)  # Initialize weights
        MultilayerPerceptronOfAdam.initialize_weights(self)  # Initialize Adam-specific parameters
        # for i in range(len(self.layer_sizes) - 1):
        #     # Initialize weights randomly from a uniform distribution
        #     self.weights[i] = np.random.rand(self.layer_sizes[i], self.layer_sizes[i + 1])

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
