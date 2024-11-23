import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the path to the folder containing MLPLinearAutoencoderOfAdam.py
sys.path.append(os.path.abspath('.'))

# Import necessary modules
from MLPLinearAutoencoderOfAdam import MLPLinearAutoencoderOfAdam
from utils.character_font import binary_arrays_from_font3
from assets.fonts import font3_labels


def plot_latent_space(autoencoder, grid_size=15, latent_range=(-2, 2)):
    """
    Plots the latent space reconstruction grid.

    Parameters:
    - autoencoder: The trained autoencoder object (with `decode` method).
    - grid_size: Number of points along each dimension in the latent space.
    - latent_range: Range of the latent space to sample.
    """
    # Create a grid of latent points
    latent_points = np.linspace(latent_range[0], latent_range[1], grid_size)
    grid = np.array([[np.array([x, y]) for x in latent_points] for y in latent_points])

    # Prepare the plot
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    for i in range(grid_size):
        for j in range(grid_size):
            latent_vector = grid[i, j]
            reconstruction = autoencoder.decode(latent_vector.reshape(1, -1))  # Decode the latent vector
            reconstruction_image = reconstruction.reshape((7, 5))  # Shape of character images

            # Plot the reconstruction
            axes[i, j].imshow(reconstruction_image, cmap="gray", interpolation="nearest")
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Get the characters from the font dataset
    characters = binary_arrays_from_font3()

    # Create a deep autoencoder with a mirrored architecture
    autoencoder = MLPLinearAutoencoderOfAdam(
        encoder_layers=[35, 25, 15, 5, 2, 2], 
        learning_rate=0.0001
    )

    # Train the autoencoder
    trained_weights, min_error, epochs, _ = autoencoder.train_autoencoder(
        characters, 
        epoch_limit=20000,   # High limit for training epochs
        error_limit=0.01     # Stop when error reaches 0.01
    )

    # Print training statistics
    print("Training complete!")
    print("Minimum error (binary difference):", np.sum(np.abs(np.rint(autoencoder.reconstruct(characters)) - characters)))
    print("Epochs used:", epochs)

    # Test encoding and reconstruction
    latent_representation = autoencoder.encode(characters)
    reconstruction = autoencoder.reconstruct(characters)
    print("Latent representation shape:", latent_representation.shape)
    print("Example reconstruction (reshaped):")
    print(np.resize(np.rint(reconstruction[1]), new_shape=(7, 5)))

    # Plot the latent space
    plot_latent_space(autoencoder, grid_size=15, latent_range=(-2, 2))