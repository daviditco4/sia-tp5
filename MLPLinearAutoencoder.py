import numpy as np
from MultilayerPerceptron import MultilayerPerceptron


class MLPLinearAutoencoder(MultilayerPerceptron):
    def __init__(self, encoder_layers, learning_rate=0.001, momentum=0.0):
        """
        Initialize a deep autoencoder with a mirrored decoder architecture.
        Parameters:
        - encoder_layers: List of integers defining the encoder architecture, e.g., [input_size, hidden1, latent_size].
        - learning_rate: Learning rate for weight updates.
        - momentum: Momentum for weight updates.
        """
        if len(encoder_layers) < 2:
            raise ValueError("Autoencoder requires at least two layers in the encoder: input and latent layers.")
        # Mirror the encoder layers to form the full autoencoder architecture
        decoder_layers = encoder_layers[-2::-1]  # Reverse encoder layers, excluding the last one (latent)
        layer_sizes = encoder_layers + decoder_layers  # Full architecture: encoder + mirrored decoder
        MultilayerPerceptron.__init__(self, layer_sizes, learning_rate=learning_rate, momentum=momentum)

    # Override the activation function to be linear for reconstruction
    def sigmoid(self, x):
        return x  # Linear activation for the autoencoder

    def sigmoid_derivative(self, x):
        # For linear activation, the derivative is constant (1)
        return np.ones_like(x)

    def compute_error(self, x, _):
        reconstructions = self.reconstruct(x)
        return np.sum(np.abs(np.rint(reconstructions) - x))

    # Train the autoencoder
    def train_autoencoder(self, x, epoch_limit, error_limit):
        y = x  # In an autoencoder, the target is the input itself
        return MultilayerPerceptron.train(self, x, y, epoch_limit, error_limit)

    # Encode the input to its latent representation
    def encode(self, x):
        encoder_weights = self.weights[: len(self.layer_sizes) // 2]
        activations, _ = self.forward_propagation(x, encoder_weights)  # Use only the encoder weights
        return activations[-1]  # Latent representation

    # Decode the latent representation back to the original space
    def decode(self, latent):
        decoder_weights = self.weights[len(self.layer_sizes) // 2:]
        activations, _ = self.forward_propagation(latent, decoder_weights)  # Use only the decoder weights
        return activations[-1]

    # Reconstruct the input by encoding and decoding
    def reconstruct(self, x):
        latent = self.encode(x)
        return self.decode(latent)


# Example usage
if __name__ == '__main__':
    # Example dataset
    X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])

    # Create a deep autoencoder with a mirrored architecture
    # Encoder layers: 2 -> 4 -> 2
    # Decoder layers: 2 -> 4 -> 2 (automatically mirrored)
    autoencoder = MLPLinearAutoencoder(encoder_layers=[2, 4, 3], learning_rate=0.01, momentum=0.9)

    # Train the autoencoder
    trained_weights, min_error, epochs, _, _ = autoencoder.train_autoencoder(X, epoch_limit=np.inf, error_limit=1)
    print("Trained weights:", trained_weights)
    print("Minimum error:", min_error)
    print("Epochs used:", epochs)

    # Test encoding and reconstruction
    latent_representation = autoencoder.encode(X)
    reconstruction = autoencoder.reconstruct(X)
    print("Latent representation:", latent_representation)
    print("Reconstruction:", reconstruction)
