import os
import sys
import matplotlib.pyplot as plt
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

    # Run the training process 10 times and collect the error histories
    num_runs = 10
    all_error_histories = []
    max_epochs = 0
    for _ in range(num_runs):
        trained_weights, trained_biases, min_error, epochs, _, _, error_history = autoencoder.train_autoencoder(
            characters,
            epoch_limit=np.inf,
            error_limit=0.05)
        all_error_histories.append(error_history)
        max_epochs = max(max_epochs, epochs)

    # Calculate the average error and standard deviation for each epoch
    avg_errors = np.zeros(max_epochs)
    std_devs = np.zeros(max_epochs)
    for epoch in range(max_epochs):
        errors_at_epoch = [history[epoch] for history in all_error_histories if epoch < len(history)]
        avg_errors[epoch] = np.mean(errors_at_epoch)
        std_devs[epoch] = np.std(errors_at_epoch)

    print("Trained weights:", trained_weights)
    print("Trained biases:", trained_biases)
    print("Minimum error:", np.sum(np.abs(np.rint(autoencoder.reconstruct(characters)) - characters)))
    print("Epochs used:", epochs)

    # Plot the average error with error bars representing the standard deviation

    # Plot the averages and standard deviation
    #plt.plot(columns, avg_0_1, marker='o', color='skyblue', label='Learning Rate 0.1', linestyle='-')
    #plt.errorbar(columns, avg_0_1, yerr=std_0_1, fmt='o', color='skyblue', capsize=5)

    plt.plot(range(max_epochs), avg_errors, marker='o', color='orange', label='Learning Rate 0.0001', linestyle='-')
    plt.errorbar(range(max_epochs), avg_errors, yerr=std_devs, fmt='o', color='skyblue', capsize=5)
    plt.xlabel('Epochs')
    plt.ylabel('Average Error')
    plt.title('Average Error over Multiple Runs')
    plt.grid(True)
    plt.show()

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
