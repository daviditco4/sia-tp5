import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python training_error_vs_epoch.py <data_csv_file> <varying_hyperparameter_name>")
        sys.exit(1)

    # Read the CSV file
    data = pd.read_csv(sys.argv[1])
    varying_hyperparam = sys.argv[2]

    # Convert the MSE column from string to list of floats
    data['Testing Error'] = data['Testing Error'].apply(eval)

    # Group data by the chosen hyperparameter's column
    unique_values = data[varying_hyperparam].unique()

    # Prepare the plot
    plt.figure(figsize=(10, 6))

    # For each unique value of the chosen hyperparameter, compute and plot MSE mean and std
    for val in unique_values:
        subset = data[data[varying_hyperparam] == val]

        # Determine the number of epochs based on the longest MSE list
        max_epochs = max(subset['Testing Error'].apply(len))
        all_mse = []

        # Collect all MSE values for each epoch
        for mse_list in subset['Testing Error']:
            padded_mse = mse_list + [np.nan] * (max_epochs - len(mse_list))  # Pad with NaN to handle shorter lists
            all_mse.append(padded_mse)

        all_mse = np.array(all_mse)

        # Compute mean and std, ignoring NaNs
        mse_means = np.nanmean(all_mse, axis=0)
        mse_stds = np.nanstd(all_mse, axis=0)

        # Generate epoch indices (1-based)
        epochs = np.arange(1, max_epochs + 1)

        # Plot MSE mean and std shadow for this hyperparameter value
        plt.plot(epochs, mse_means, label=f'{varying_hyperparam}: {val}')
        plt.fill_between(epochs, mse_means - mse_stds, mse_means + mse_stds, alpha=0.2)

    # # Limit X-axis to 2500 epochs
    # plt.xlim([1, 2500])

    # # Set y-axis to logarithmic scale
    # plt.yscale('log')

    # Add labels, title, and legend
    plt.xlabel('Epochs')
    plt.ylabel('Testing Error')
    plt.title(f'MSE per Epoch for varying {varying_hyperparam}')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(
        f"task1_{varying_hyperparam.lower().replace('/', '_')}_determined_testing_error_vs_epoch_plot.png",
        dpi=300, bbox_inches='tight')
    plt.close()