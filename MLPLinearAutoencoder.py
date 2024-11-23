import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean

from MultilayerPerceptron import MultilayerPerceptron


class MLPLinearAutoencoder(MultilayerPerceptron):
    def __init__(self, encoder_layers, beta=1.0, learning_rate=0.001, momentum=0.0):
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
        MultilayerPerceptron.__init__(self, layer_sizes, beta, learning_rate, momentum)
        self.encoder_layers = encoder_layers

    # Override the activation function to be linear for reconstruction
    def sigmoid(self, x, layer):
        if layer == len(self.encoder_layers) - 1:
            return x  # Tanh activation for the latent layer
        elif layer == len(self.layer_sizes) - 1:
            return MultilayerPerceptron.sigmoid(self, x, layer)  # Sigmoid activation for the output layer
        else:
            return np.tanh(self.beta * x)  # ReLU activation for the remaining layers

    def sigmoid_derivative(self, x, layer):
        if layer == len(self.encoder_layers) - 1:
            return np.ones_like(x)  # For linear activation, the derivative is constant (1)
        elif layer == len(self.layer_sizes) - 1:
            return MultilayerPerceptron.sigmoid_derivative(self, x, layer)
        else:
            return self.beta * (1 - np.tanh(self.beta * x) ** 2)  # Derivative for ReLU activation

    # def compute_error(self, x, _=None):
    #     reconstructions = self.reconstruct(x)
    #     # print(reconstructions)
    #     # print(np.abs(np.rint(reconstructions) - x))
    #     # exit()
    #     # Ensure numerical stability by clipping predictions to avoid log(0)
    #     y_pred = np.clip(reconstructions, a_min=1e-15, a_max=1 - 1e-15)
    #     # Compute BCE loss for each data point
    #     bce_loss = -(x * np.log(y_pred) + (1 - x) * np.log(1 - y_pred))
    #     return np.mean(bce_loss)  # Return the average loss
    #     # return np.sum(np.abs(np.rint(reconstructions) - x))

    # Train the autoencoder
    def train_autoencoder(self, x, epoch_limit, error_limit):
        y = x  # In an autoencoder, the target is the input itself
        return MultilayerPerceptron.train(self, x, y, epoch_limit, error_limit)

    # Encode the input to its latent representation
    def encode(self, x):
        encoder_weights = self.weights[: len(self.layer_sizes) // 2]
        encoder_biases = self.biases[: len(self.layer_sizes) // 2]
        activations, _ = self.forward_propagation(x, encoder_weights,
                                                  encoder_biases)  # Use only the encoder weights and biases
        return activations[-1]  # Latent representation

    # Decode the latent representation back to the original space
    def decode(self, latent):
        decoder_weights = self.weights[len(self.layer_sizes) // 2:]
        decoder_biases = self.biases[len(self.layer_sizes) // 2:]
        activations, _ = self.forward_propagation(latent, decoder_weights,
                                                  decoder_biases)  # Use only the decoder weights and biases
        return activations[-1]

    # Reconstruct the input by encoding and decoding
    def reconstruct(self, x):
        return self.forward_propagation(x)[0][-1]
        # latent = self.encode(x)
        # return self.decode(latent)

    def plot_latent_space(self, x, labels=None):
        if self.encoder_layers[-1] != 2:
            raise ValueError("The latent space must be 2D to visualize. Adjust the architecture.")
        # Encode input data into latent representations
        latent_representations = self.encode(x)
        # Compute distances from each point to the origin (0, 0)
        distances = [euclidean(point, [0, 0]) for point in latent_representations]
        # Normalize distances to [0, 1] for colormap scaling
        max_distance = max(distances)
        normalized_distances = [d / max_distance for d in distances]
        # Use a colormap (e.g., viridis) to assign colors based on distances
        colormap = plt.cm.viridis
        colors = [colormap(norm_dist) for norm_dist in normalized_distances]
        plt.figure(figsize=(8, 6))
        for i, point in enumerate(latent_representations):
            plt.scatter(point[0], point[1], color=colors[i], s=50, alpha=0.7)
            if labels is not None:
                plt.text(point[0], point[1], str(labels[i]), fontsize=9, ha='right', va='bottom')
        # Add a colorbar to represent the distances
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=max_distance))
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('Distance from Center')
        plt.title("2D Latent Space Representation")
        plt.xlabel("Latent Dimension 1")
        plt.ylabel("Latent Dimension 2")
        plt.grid(True)
        plt.show()


# Example usage
if __name__ == '__main__':
    # Example dataset
    X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])

    # Create a deep autoencoder with a mirrored architecture
    # Encoder layers: 2 -> 4 -> 2
    # Decoder layers: 2 -> 4 -> 2 (automatically mirrored)
    autoencoder = MLPLinearAutoencoder(encoder_layers=[2, 4, 3], learning_rate=0.01, momentum=0.9)

    # Train the autoencoder
    trained_weights, trained_biases, min_error, epochs, _, _, _ = autoencoder.train_autoencoder(X, epoch_limit=np.inf,
                                                                                                error_limit=1)
    print("Trained weights:", trained_weights)
    print("Trained biases:", trained_biases)
    print("Minimum error:", min_error)
    print("Epochs used:", epochs)

    # Test encoding and reconstruction
    latent_representation = autoencoder.encode(X)
    reconstruction = autoencoder.reconstruct(X)
    print("Latent representation:", latent_representation)
    print("Reconstruction:", reconstruction)
