import os
# import random
import sys
import time

import numpy as np

# Add the path to the folder containing utils/character_font.py
sys.path.append(os.path.abspath('.'))

# Now you can import binary_arrays_from_font3
from utils.character_font import binary_arrays_from_font3
from utils.helper_functions import read_hyperparameters_from_json, train_autoencoder, append_results_to_csv


def apply_noise_to_bitmaps(bitmaps, noise_lvl):
    # Add Gaussian noise
    mean = 0
    std_dev = noise_lvl
    noise = np.random.normal(mean, std_dev, bitmaps.shape)
    noisy_bitmaps = np.clip(bitmaps + noise, a_min=0, a_max=1)
    # """
    # Applies noise to the bitmaps by flipping a fraction of the bits.
    # noise_level: The fraction of bits to flip (between 0 and 1).
    # """
    # noisy_bitmaps = bitmaps.copy()
    # num_bits = noisy_bitmaps.shape[1]  # Number of bits in each bitmap (should be 35 for 5x7)
    # num_characters = noisy_bitmaps.shape[0]  # Number of characters
    # num_bits_to_flip = int(num_bits * noise_lvl)  # How many bits to flip based on noise level
    #
    # for digit_index in range(num_characters):
    #     # Randomly select which bits to flip
    #     flip_indices = random.sample(range(num_bits), num_bits_to_flip)
    #     for bit_index in flip_indices:
    #         # Flip the bit (0 -> 1, 1 -> 0)
    #         noisy_bitmaps[digit_index, bit_index] = 1 - noisy_bitmaps[digit_index, bit_index]

    return noisy_bitmaps


def _test_autoencoder(a, chars, noise_lvl=0.0, weights=None, biases=None):
    noisy_characters = apply_noise_to_bitmaps(chars, noise_lvl)
    predictions = a.predict(noisy_characters, weights, biases)
    predicted_labels = predictions
    lbls = chars
    return np.mean((predicted_labels - lbls) ** 2)


def test_autoencoder(a, chars, w8_hist, bias_hist, noise_lvl=0.0):
    test_errs = [0.0] * len(w8_hist)

    for i in range(len(w8_hist)):
        test_error = 0.0
        for _ in range(10):
            test_error += _test_autoencoder(a, chars, noise_lvl, w8_hist[i], bias_hist[i])
        test_errs[i] = test_error / 10

    return test_errs


if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print("Usage: python runner_main.py <hyperparameters_json_file> <output_csv_file> [<noise_level> [<training_level>]]")
        sys.exit(1)

    hyperparameters_json_file = sys.argv[1]
    output_csv_file = sys.argv[2]
    noise_level = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    training_level = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    # Read the characters from our font
    characters = binary_arrays_from_font3()

    # Read hyperparameters from JSON
    hyperparameters = read_hyperparameters_from_json(hyperparameters_json_file)

    # Use same characters as `true` labels
    labels = characters if noise_level == 0.0 else np.tile(characters, reps=(training_level, 1))

    # Use noisy characters if it is denoising
    inputs = characters if noise_level == 0.0 else np.concatenate(
        [apply_noise_to_bitmaps(characters, noise_level) for _ in range(training_level)])

    for _ in range(4):
        # Start timing
        start_time = time.time()
        # Train the autoencoder
        ae, epochs, weight_history, bias_history, error_history = train_autoencoder(inputs, labels, hyperparameters, None if noise_level == 0.0 else training_level)
        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        testing_errors = test_autoencoder(ae, characters, weight_history, bias_history,
                                          noise_level) if noise_level != 0.0 else None

        # Append results to CSV
        append_results_to_csv(output_csv_file, elapsed_time, hyperparameters, noise_level, epochs, error_history,
                              testing_errors)
        # print(weight_history[0][:2][:2])
        # print(weight_history[1][:2][:2])
        # print(bias_history[0][:2][:2])
        # print(bias_history[1][:2][:2])
        print(f"Training completed in {epochs} epochs with {error_history[-1]:.4f} training error")
        print(np.mean((ae.predict(apply_noise_to_bitmaps(characters, noise_level), weight_history[0], bias_history[0]) - characters) ** 2))
