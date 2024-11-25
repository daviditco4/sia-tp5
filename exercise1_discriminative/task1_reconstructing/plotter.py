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

# Example solution with multiple learning rates
if __name__ == '__main__':
    # Get the characters from our font
    characters = binary_arrays_from_font3()

    # Define different learning rates
    learning_rates = [0.0005, 0.001, 0.003, 0.005]
    num_runs = 10
    max_epochs = 0

    # Create placeholders for data
    all_avg_errors = {}
    all_std_devs = {}

    # Loop through each learning rate
    for lr in learning_rates:
        avg_errors = np.zeros(max_epochs)
        std_devs = np.zeros(max_epochs)
        autoencoder = MLPLinearAutoencoderOfAdam(encoder_layers=[35, 50, 70, 30, 12, 5, 2], learning_rate=lr)
        all_error_histories = []

        for _ in range(num_runs):
            trained_weights, trained_biases, min_error, epochs, _, _, error_history = autoencoder.train_autoencoder(
                characters,
                epoch_limit=1000,
                error_limit=0.0)
            all_error_histories.append(error_history)
            max_epochs = max(max_epochs, epochs)

        # Calculate the average error and standard deviation for each epoch
        avg_errors = np.zeros(max_epochs)
        std_devs = np.zeros(max_epochs)
        for epoch in range(max_epochs):
            errors_at_epoch = [history[epoch] for history in all_error_histories if epoch < len(history)]
            avg_errors[epoch] = np.mean(errors_at_epoch)
            std_devs[epoch] = np.std(errors_at_epoch)

        all_avg_errors[lr] = avg_errors
        all_std_devs[lr] = std_devs

    # Plot the results
    plt.figure(figsize=(12, 6))
    for lr, avg_errors in all_avg_errors.items():
        std_devs = all_std_devs[lr]
        plt.errorbar(range(max_epochs), avg_errors, yerr=std_devs, label=f"Learning Rate {lr}",
                     alpha=0.7, fmt='o', capsize=3, linestyle='-', markersize=3)

    # Customize the plot
    plt.title("Average Error per Iteration of Linear Perceptron with Different Learning Rates")
    plt.xlabel("Iteration of Perceptron")
    plt.ylabel("Average Error")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Show the plot
    plt.show()