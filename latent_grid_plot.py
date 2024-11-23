import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the path to the folder containing MLPLinearAutoencoderOfAdam.py
sys.path.append(os.path.abspath('.'))

# Import necessary modules
from MLPLinearAutoencoderOfAdam import MLPLinearAutoencoderOfAdam
from utils.character_font import binary_arrays_from_font3


def plot_reconstructed_grid(autoencoder, grid_size=20, latent_range=(-1, 1)):
    """
    Plots a grid of reconstructed characters from the latent space.

    Parameters:
    - autoencoder: Trained autoencoder instance.
    - grid_size: Number of points along each latent dimension.
    - latent_range: Range of the latent space (-1 to 1 by default).
    """
    # Create a grid of points in the latent space
    latent_points = np.linspace(latent_range[0], latent_range[1], grid_size)
    grid = np.array([[np.array([x, y]) for x in latent_points] for y in latent_points])

    # Prepare a canvas for the plot
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    for i in range(grid_size):
        for j in range(grid_size):
            latent_vector = grid[i, j]  # Latent point
            reconstruction = autoencoder.decode(latent_vector.reshape(1, -1))  # Decode to reconstruct
            reconstruction_image = reconstruction.reshape((7, 5))  # Adjust to character dimensions

            # Plot each reconstructed character
            axes[i, j].imshow(reconstruction_image, cmap="gray", interpolation="nearest")
            axes[i, j].axis('off')  # Remove axes for cleaner visualization

    plt.subplots_adjust(wspace=0, hspace=0)  # Remove space between plots
    plt.show()


if __name__ == '__main__':
    # Load the dataset
    characters = binary_arrays_from_font3()

    # Create and train the autoencoder
    autoencoder = MLPLinearAutoencoderOfAdam(
        encoder_layers=[35, 25, 15, 5, 2, 2],  # 2D latent space
        learning_rate=0.0001
    )
    trained_weights, min_error, epochs, _, _ = autoencoder.train_autoencoder(
        characters, epoch_limit=5000, error_limit=0.01
    )
    print("Training complete!")
    print(f"Minimum error: {min_error}")
    print(f"Epochs used: {epochs}")

    # Generate and display the reconstructed grid
    plot_reconstructed_grid(autoencoder, grid_size=20, latent_range=(-1, 1))