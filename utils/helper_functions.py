import csv
import json
import os
import sys

# Add the path to the folder containing MLPLinearAutoencoderOfAdam.py
sys.path.append(os.path.abspath("."))

# Now you can import MLPLinearAutoencoderOfAdam
from MLPLinearAutoencoderOfAdam import MLPLinearAutoencoderOfAdam
from MLPLinearAutoencoder import MLPLinearAutoencoder


def read_hyperparameters_from_json(file_path):
    with open(file_path, 'r') as file:
        hyperparameters = json.load(file)
    return hyperparameters


def train_autoencoder(inputs, labels, hyperparameters):
    if 'adam' in hyperparameters["optimizer"]:
        # Initialize autoencoder using the hyperparameters
        ae = MLPLinearAutoencoderOfAdam(hyperparameters["encoder_layers"], beta=hyperparameters["beta"],
                                        learning_rate=hyperparameters["learning_rate"],
                                        beta1=hyperparameters["momentum"] if 'momentum' in hyperparameters else 0.9)
    else:
        # Initialize autoencoder using the hyperparameters
        ae = MLPLinearAutoencoder(hyperparameters["encoder_layers"], beta=hyperparameters["beta"],
                                  learning_rate=hyperparameters["learning_rate"],
                                  momentum=hyperparameters["momentum"] if 'momentum' in hyperparameters else 0.0)

    # Train the autoencoder
    _, _, _, epochs, weight_history, bias_history, error_history = ae.train(inputs, labels, epoch_limit=2000,
                                                                            error_limit=hyperparameters["error_limit"])

    return ae, epochs, weight_history, bias_history, error_history


def append_results_to_csv(file_path, elapsed_time, hyperparameters, noise_level, epochs, training_errors,
                          testing_errors):
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        # If the file is new, write the header first
        if not file_exists:
            header = ["Elapsed Seconds", "Encoder Layers", "Beta", "Learning Rate", "Momentum", "Error Epsilon",
                      "Optimizer", "Noise Level", "Epochs", "Training Error", "Testing Error"]
            if noise_level == 0.0:
                header.remove("Testing Error")
            csvwriter.writerow(header)

        row = [elapsed_time, hyperparameters["encoder_layers"], hyperparameters["beta"],
               hyperparameters["learning_rate"],
               hyperparameters["momentum"] if 'momentum' in hyperparameters else 0.0, hyperparameters["error_limit"],
               hyperparameters["optimizer"], noise_level, epochs, training_errors, testing_errors]
        if noise_level == 0.0:
            del row[10]
        csvwriter.writerow(row)
