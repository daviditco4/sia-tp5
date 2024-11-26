# SIA TP3

Implementation of autoencoder to represent bitmap characters in 2D built with Python

## System requirements

* Python 3.7+

## How to use

* Clone or download this repository in the folder you desire
* When you are ready, enter a command as follows:
```sh
python3 exercise1_discriminative/task2_reconstructing/runner_main.py <config.json> <output_file.csv> [<noise_level> [<training_level>]]
```

### Hyperparameters

The configuration for the algorithm's options is a JSON file with the following structure:

* `"layer_sizes"`: The layer architecture
* `"beta"`: The multiplication factor of the sigmoid activation
* `"learning_rate"`: The multilayer perceptron's learning factor
* `"momentum"`: The momentum factor (0 by default)
* `"error_limit"`: The upper bound of the error
* `"optimizer"`: The optimizer to use ("none", "adam")

## License

This project is licensed under the MIT License - see the LICENSE file for details.
