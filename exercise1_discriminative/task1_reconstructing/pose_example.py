import os
import sys

import numpy as np

# Add the path to the folder containing MLPLinearAutoencoderOfAdam.py
sys.path.append(os.path.abspath('.'))

# Now you can import MLPLinearAutoencoderOfAdam
from MLPLinearAutoencoderOfAdam import MLPLinearAutoencoderOfAdam
from assets.fonts import font3_labels
from utils.character_font import binary_arrays_from_font3

# Example solution
if __name__ == '__main__':
    # Get the characters from our font
    characters = binary_arrays_from_font3()
    # characters[characters == 0] = -1

    # Create a deep autoencoder with a mirrored architecture
    autoencoder = MLPLinearAutoencoderOfAdam(encoder_layers=[35, 50, 70, 30, 12, 5, 2], learning_rate=0.0001)

    # Train the autoencoder
    trained_weights, trained_biases, min_error, epochs, _, _, _ = autoencoder.train_autoencoder(characters,
                                                                                                epoch_limit=2000,
                                                                                                error_limit=0.002)
    print("Trained weights:", trained_weights)
    print("Trained biases:", trained_biases)
    print("Minimum error:", np.sum(np.abs(np.rint(autoencoder.reconstruct(characters)) - characters)))
    print("Epochs used:", epochs)

    # Test encoding and reconstruction
    latent_representation = autoencoder.encode(characters)
    reconstruction = autoencoder.reconstruct(characters)
    print("Latent representation:", latent_representation)
    print("Reconstruction:", np.resize(np.rint(reconstruction[31]), new_shape=(7, 5)))

    # Plot the latent space
    autoencoder.plot_latent_space(characters, font3_labels)
    # labels = [None] * len(characters)
    # for i in range(len(characters)):
    #     labels[i] = font3_labels[i] + str(autoencoder.compute_error(characters[i]))
    # autoencoder.plot_latent_space(characters, labels)

    # Plot the reconstructions
    autoencoder.plot_reconstructions(characters)

    # Plot the latent grid
    autoencoder.plot_latent_grid()
