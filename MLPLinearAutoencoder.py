import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean

from MultilayerPerceptron import MultilayerPerceptron


class MLPLinearAutoencoder(MultilayerPerceptron):
    def __init__(self, encoder_layers, beta=1.0, learning_rate=0.001, momentum=0.0, training_level=1):
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
        MultilayerPerceptron.__init__(self, layer_sizes, beta, learning_rate, momentum, training_level=training_level)
        self.encoder_layers = encoder_layers
        self.is_just_decoding = False

    # Override the activation function to be linear for reconstruction
    def sigmoid(self, x, layer_index=None):
        if self.is_just_decoding:
            layer_index += len(self.encoder_layers) - 1
        if layer_index == len(self.encoder_layers) - 1:
            return x  # Linear activation for the latent layer
        elif layer_index == len(self.layer_sizes) - 1:
            return MultilayerPerceptron.sigmoid(self, x)  # Sigmoid activation for the output layer
        else:
            return np.tanh(self.beta * x)  # Tanh activation for the remaining layers

    def sigmoid_derivative(self, x, layer_index=None):
        if self.is_just_decoding:
            layer_index += len(self.encoder_layers) - 1
        if layer_index == len(self.encoder_layers) - 1:
            return np.ones_like(x)  # For linear activation, the derivative is constant (1)
        elif layer_index == len(self.layer_sizes) - 1:
            return MultilayerPerceptron.sigmoid_derivative(self, x)
        else:
            return self.beta * (1 - np.tanh(self.beta * x) ** 2)  # Derivative for tanh activation

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
        self.is_just_decoding = True
        activations, _ = self.forward_propagation(latent, decoder_weights,
                                                  decoder_biases)  # Use only the decoder weights and biases
        self.is_just_decoding = False
        return activations[-1]

    # Reconstruct the input by encoding and decoding
    def reconstruct(self, x):
        # return self.forward_propagation(x)[0][-1]
        latent = self.encode(x)
        return self.decode(latent)

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
        # Plot the points
        plt.figure(figsize=(8, 6))
        for i, point in enumerate(latent_representations):
            plt.scatter(point[0], point[1], color=colors[i], s=50, alpha=0.7)
            # Adjust text labels with an offset for spacing
            if labels is not None:
                text_offset = 0.08  # Adjust the offset value to your preference
                plt.text(
                    point[0] + text_offset,
                    point[1] + text_offset,
                    str(labels[i]),
                    fontsize=9,
                    ha='center',  # Horizontal alignment
                    va='center'  # Vertical alignment
                )
        # Add a colorbar to represent the distances
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=max_distance))
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('Distance from Center')
        # Add titles, labels, and grid
        plt.title("2D Latent Space Representation")
        plt.xlabel("Latent Dimension 1")
        plt.ylabel("Latent Dimension 2")
        plt.grid(True)
        # Save and close the plot
        plt.savefig(
            f"task1_combined_activation_latent_space.png",
            # {varying_hyperparam.lower().replace(" ", "_")}_determined_latent_space.png",
            dpi=300, bbox_inches='tight')
        plt.close()

    def plot_reconstructions(self, original_bitmaps):
        """
        Plot the original and reconstructed bitmaps in a grid with rectangle borders for clarity.
        Parameters:
        - original_bitmaps: numpy array of shape (32, 35) where each row is a flattened 7x5 bitmap.
        """
        if original_bitmaps.shape[0] != 32 or original_bitmaps.shape[1] != 35:
            raise ValueError("Input must have shape (32, 35), representing 32 flattened 7x5 bitmaps.")
        # Reconstruct the bitmaps
        reconstructed_bitmaps = np.clip(self.reconstruct(original_bitmaps), a_min=0, a_max=1)
        # Prepare the figure
        fig, axes = plt.subplots(8, 8, figsize=(12, 12))  # 8x8 grid for 32 original + 32 reconstructed
        axes = axes.flatten()
        for i in range(32):
            # Plot original
            ax = axes[i * 2]
            ax.imshow(original_bitmaps[i].reshape(7, 5), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            ax.axis('off')
            ax.set_title("Original", fontsize=8)
            # Add a border rectangle
            rect = plt.Rectangle(
                (0, 0), 1, 1, transform=ax.transAxes,
                linewidth=2, edgecolor='blue', facecolor='none'
            )
            ax.add_patch(rect)
            # Plot reconstruction
            ax = axes[i * 2 + 1]
            ax.imshow(reconstructed_bitmaps[i].reshape(7, 5), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            ax.axis('off')
            ax.set_title("Reconstructed", fontsize=8)
            # Add a border rectangle
            rect = plt.Rectangle(
                (0, 0), 1, 1, transform=ax.transAxes,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
        # Adjust spacing and save the plot
        plt.tight_layout()
        plt.savefig(
            f"task1_combined_activation_reconstructions.png",
            # {varying_hyperparam.lower().replace(" ", "_")}_determined_reconstructions.png",
            dpi=300, bbox_inches='tight')
        plt.close()

    def plot_latent_grid(self, grid_size=21, pixel_dims=(7, 5), latent_range=(-1.0, 1.0)):
        """
        Plot a grid of 7x5 bitmaps decoded from a grid of latent space points, with labeled axes.
        The space between the subplots is filled with black.
        Parameters:
        - grid_size: Number of points along each dimension of the latent space grid.
        - pixel_dims: Dimensions of the output bitmaps (default is 7x5).
        - latent_range: Range of the latent space to sample from (default is [-1, 1]).
        """
        if self.encoder_layers[-1] != 2:
            raise ValueError("Latent space must be 2D to use this visualization.")
        # Create a 2D grid of points in the latent space
        x_coords = np.linspace(latent_range[0], latent_range[1], grid_size)
        y_coords = np.linspace(latent_range[0], latent_range[1], grid_size)
        latent_grid = np.array([[x, y] for y in y_coords for x in x_coords])
        # Decode each latent point
        decoded_images = self.decode(latent_grid)
        decoded_images = decoded_images.reshape((-1, *pixel_dims))  # Reshape to (grid_size^2, height, width)
        # Plot the grid of images
        fig, axes = plt.subplots(grid_size, grid_size,
                                 figsize=(12, 12))  # ,
                                 # gridspec_kw={'wspace': 0.0, 'hspace': 0.0})  # Remove all spacing
        # Set the figure background to black
        fig.patch.set_facecolor('black')
        for i in range(grid_size):
            for j in range(grid_size):
                index = i * grid_size + j
                axes[i, j].imshow(decoded_images[index], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                axes[i, j].set_aspect('equal')  # Ensure images don't stretch
                axes[i, j].set_facecolor('black')  # Set subplot background to black
                # Add X and Y labels for the first row and first column
                if j == 0:
                    axes[i, j].set_ylabel(f"{y_coords[i]:.1f}", fontsize=16, color='white')
                if i == grid_size - 1:
                    axes[i, j].set_xlabel(f"{x_coords[j]:.1f}", fontsize=16, color='white')
        # # Set common labels for axes
        # fig.text(0.5, 0.02, 'Latent X', ha='center', fontsize=10, color='white')
        # fig.text(0.02, 0.5, 'Latent Y', va='center', rotation='vertical', fontsize=10, color='white')
        # Save the plot with a black background
        plt.tight_layout(pad=0.5)  # No padding to make it as compact as possible
        plt.savefig(
            "task1_combined_activation_latent_grid.png",
            # {varying_hyperparam.lower().replace(" ", "_")}_determined_latent_grid.png",
            dpi=300, bbox_inches='tight',  # facecolor=fig.get_facecolor()
        )
        plt.close()


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
