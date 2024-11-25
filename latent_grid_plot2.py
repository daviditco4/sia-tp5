import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the path to the folder containing MLPLinearAutoencoderOfAdam.py
sys.path.append(os.path.abspath('.'))

from MLPLinearAutoencoderOfAdam import MLPLinearAutoencoderOfAdam
from utils.character_font import binary_arrays_from_font3


def plot_reconstructed_grid(autoencoder, grid_size=20, latent_range=(-1, 1)):
    """
    Plots a clear and structured grid of reconstructed characters from the latent space
    with a black grid background and white symbols, while keeping the axes visible.

    Parameters:
    - autoencoder: Trained autoencoder instance.
    - grid_size: Number of points along each latent dimension.
    - latent_range: Range of the latent space (-1, 1 by default).
    """
    from matplotlib.colors import ListedColormap

    # Define a custom black-and-white colormap
    custom_cmap = ListedColormap(["black", "white"])

    # Generate latent points based on grid size
    latent_points = np.linspace(latent_range[0], latent_range[1], grid_size)

    # Create a figure with axes
    fig, ax = plt.subplots(figsize=(10, 10))

    # Loop through latent space and plot each reconstructed character
    for i, y in enumerate(latent_points):
        for j, x in enumerate(latent_points):
            latent_vector = np.array([x, y])  # Create a latent vector for each grid position
            reconstruction = autoencoder.decode(latent_vector.reshape(1, -1))  # Decode the latent vector
            reconstruction_image = reconstruction.reshape((7, 5))  # Reshape to match input dimensions

            # Offset for plotting each reconstructed image
            #x_offset = j
            #y_offset = i

            # Plot the reconstructed image
            ax.imshow(
                reconstruction_image,
                cmap=custom_cmap,
                extent=(j, j + 1, grid_size - i - 1, grid_size - i),
                interpolation="nearest",
            )

    # Set ticks and labels
    ax.set_xticks(np.arange(0.5, grid_size + 0.5))
    ax.set_yticks(np.arange(0.5, grid_size + 0.5))
    ax.set_xticklabels([f"{x:.1f}" for x in latent_points], fontsize=8)
    ax.set_yticklabels([f"{y:.1f}" for y in latent_points], fontsize=8)
    ax.set_xlabel("Latent Dimension 1", fontsize=12)
    ax.set_ylabel("Latent Dimension 2", fontsize=12)
    ax.set_title("Reconstructed Grid from Latent Space", fontsize=16)

    # Show the grid for clarity
    #ax.grid(color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)

    # Adjust layout for better clarity
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Load the dataset
    characters = binary_arrays_from_font3()

    # Create and train the autoencoder
    autoencoder = MLPLinearAutoencoderOfAdam(
        encoder_layers=[35, 25, 15, 5, 2, 2],  # Encoder layers, with 2D latent space
        learning_rate=0.001
    )
    result = autoencoder.train_autoencoder(
        characters, epoch_limit=10000, error_limit=0.01
    )
    trained_weights, trained_biases, min_error, epochs = result[:4]

    # Print training results
    print("Training complete!")
    print(f"Minimum error: {min_error}")
    print(f"Epochs used: {epochs}")

    # Plot the reconstructed grid with a grid size of 20
    plot_reconstructed_grid(autoencoder, grid_size=20, latent_range=(-1, 1))