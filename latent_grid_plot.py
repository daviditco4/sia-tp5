import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the path to the folder containing MLPLinearAutoencoderOfAdam.py
sys.path.append(os.path.abspath('.'))

from MLPLinearAutoencoderOfAdam import MLPLinearAutoencoderOfAdam
from utils.character_font import binary_arrays_from_font3


def plot_reconstructed_grid(autoencoder, latent_range=(-1, 1), step=0.1):
    """
    Plots a clear and structured grid of reconstructed characters from the latent space with visible axis values.

    Parameters:
    - autoencoder: Trained autoencoder instance.
    - latent_range: Range of the latent space (-1 to 1 by default).
    - step: Step size for latent space sampling (default: 0.1).
    """
    # Generate latent points with the specified step
    latent_points = np.arange(latent_range[0], latent_range[1] + step, step)
    grid_size = len(latent_points)

    # Prepare the figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # Generate grid of reconstructions
    for i, y in enumerate(latent_points):
        for j, x in enumerate(latent_points):
            latent_vector = np.array([x, y])  # Create a latent vector for each grid position
            reconstruction = autoencoder.decode(latent_vector.reshape(1, -1))  # Decode the latent vector
            reconstruction_image = reconstruction.reshape((7, 5))  # Reshape to match input dimensions

            # Calculate grid cell placement
            x_offset = j * 8  # 5 pixels for width + 3 pixels padding
            y_offset = i * 8  # 7 pixels for height + 1 pixel padding
            ax.imshow(reconstruction_image, cmap="gray_r", extent=(x_offset, x_offset + 5, y_offset, y_offset + 7))

    # Add ticks and labels
    ax.set_xticks([i * 8 for i in range(grid_size)])
    ax.set_xticklabels([f"{x:.1f}" for x in latent_points], fontsize=8)
    ax.set_yticks([i * 8 for i in range(grid_size)])
    ax.set_yticklabels([f"{y:.1f}" for y in latent_points], fontsize=8)

    ax.set_xlabel("Latent Dimension 1", fontsize=12)
    ax.set_ylabel("Latent Dimension 2", fontsize=12)
    ax.set_title("Reconstructed Grid from Latent Space", fontsize=16)

    # Adjust layout for clarity
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

    # Plot the reconstructed grid
    plot_reconstructed_grid(autoencoder, latent_range=(-1, 1), step=0.1)