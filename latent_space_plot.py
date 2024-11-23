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


def plot_latent_space(autoencoder, data, labels=None):
    """
    Plots the 2D latent space of the input data.
    
    Parameters:
    - autoencoder: The trained autoencoder object.
    - data: Input data to encode.
    - labels: Optional labels for the data points.
    """
    # Ensure the latent space is 2D
    if autoencoder.encoder_layers[-1] != 2:
        raise ValueError("The latent space must be 2D for visualization.")
    
    # Encode input data into latent space
    latent_representations = autoencoder.encode(data)

    # Plot latent space
    plt.figure(figsize=(8, 6))
    for i, point in enumerate(latent_representations):
        plt.scatter(point[0], point[1], color='blue', s=50, alpha=0.7)
        if labels:
            plt.text(point[0], point[1], str(labels[i]), fontsize=9, ha='right', va='bottom')
    plt.title("2D Latent Space Representation")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.grid(True)
    plt.show()


def plot_reconstructed_latent_space(autoencoder, grid_size=10, latent_range=(-2, 2)):
    """
    Plots a grid of reconstructed characters from the latent space.

    Parameters:
    - autoencoder: Trained autoencoder instance.
    - grid_size: Number of points along each latent dimension.
    - latent_range: Range of the latent space (-2 to 2 by default).
    """
    latent_points = np.linspace(latent_range[0], latent_range[1], grid_size)
    grid = np.array([[np.array([x, y]) for x in latent_points] for y in latent_points])

    # Prepare a grid for plotting
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    for i in range(grid_size):
        for j in range(grid_size):
            latent_vector = grid[i, j]
            reconstruction = autoencoder.decode(latent_vector.reshape(1, -1))
            reconstruction_image = reconstruction.reshape((7, 5))  # Adjust shape for characters

            # Plot the reconstructed character
            axes[i, j].imshow(reconstruction_image, cmap="gray", interpolation="nearest")
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Load the dataset of characters
    characters = binary_arrays_from_font3()

    # Create and train the autoencoder
    autoencoder = MLPLinearAutoencoderOfAdam(
        encoder_layers=[35, 30, 20, 10, 4, 2], 
        learning_rate=0.001
    )
    trained_weights, min_error, epochs, _, error_history = autoencoder.train_autoencoder(
        characters, epoch_limit=10000, error_limit=0.01
    )
    print("Training complete!")
    print(f"Minimum error: {min_error}")
    print(f"Epochs used: {epochs}")

    # Test reconstruction
    reconstruction = autoencoder.reconstruct(characters)
    print("Example reconstruction (reshaped):")
    print(reconstruction[0].reshape((7, 5)))

    # Plot the latent space of the input data
    plot_latent_space(autoencoder, characters, labels=font3_labels)

    # Plot reconstructed grid of the latent space
    plot_reconstructed_latent_space(autoencoder, grid_size=10, latent_range=(-2, 2))