import random
import numpy as np


class MultilayerPerceptron:
    def __init__(self, layer_sizes, beta=1.0, learning_rate=0.001, momentum=0.0, weight_updates_by_epoch=False, training_level=1):
        self.layer_sizes = layer_sizes  # List defining the number of neurons per layer
        self.beta = beta
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_updates_by_epoch = weight_updates_by_epoch
        self.weights = None
        self.biases = None
        self.prev_weight_updates = None
        self.prev_bias_updates = None
        self.training_level = training_level

    # Sigmoid activation function
    def sigmoid(self, x, _=None):
        return 1 / (1 + np.exp(-self.beta * x))

    # Derivative of the sigmoid (calculated on weighted sums, not activations)
    def sigmoid_derivative(self, x, _=None):
        sigmoid_val = MultilayerPerceptron.sigmoid(self, x)
        return self.beta * sigmoid_val * (1 - sigmoid_val)

    # Initialize the weights for all layers
    def initialize_weights(self):
        self.weights = [np.zeros(0)] * (len(self.layer_sizes) - 1)
        self.biases = [np.zeros(0)] * (len(self.layer_sizes) - 1)
        for i in range(len(self.layer_sizes) - 1):
            # Initialize weights and biases randomly from a normal distribution
            self.weights[i] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) / 3
            self.biases[i] = np.random.randn(1, self.layer_sizes[i + 1]) / 3
        self.prev_weight_updates = [np.zeros_like(w) for w in self.weights]
        self.prev_bias_updates = [np.zeros_like(w) for w in self.biases]

    # Forward propagation with returning activations and excitations (weighted sums)
    def forward_propagation(self, x, weights=None, biases=None):
        if weights is None:
            weights = self.weights
        if biases is None:
            biases = self.biases
        activations = [x] * (len(weights) + 1)  # Store activations for each layer
        excitations = [0.0] * len(weights)  # Store weighted sums for each layer (before activation)
        for i in range(len(weights)):
            net_input = np.dot(activations[i], weights[i]) + biases[i]  # Weighted sum + bias (excitation)
            excitations[i] = net_input
            activation = self.sigmoid(net_input, i + 1)  # Apply activation function (sigmoid)
            activations[i + 1] = activation
        return activations, excitations

    def calculate_weight_updates(self, weight_gradients, bias_gradients, learning_rate=None):
        if learning_rate is None:
            learning_rate = self.learning_rate
        weight_updates = [0.0] * len(self.weights)
        bias_updates = [0.0] * len(self.biases)
        for i in range(len(self.weights)):
            weight_updates[i] = -learning_rate * (weight_gradients[i] + 0.001 * self.weights[i]) + self.momentum * \
                                self.prev_weight_updates[i]
            bias_updates[i] = -learning_rate * bias_gradients[i] + self.momentum * self.prev_bias_updates[i]
            self.prev_weight_updates[i] = weight_updates[i]
            self.prev_bias_updates[i] = bias_updates[i]
        return weight_updates, bias_updates

    # Backpropagation for multiple layers
    def back_propagation(self, y_true, activations, excitations):
        errors = [0.0] * len(self.weights)  # Initialize error list
        weight_gradients = [0.0] * len(self.weights)
        bias_gradients = [0.0] * len(self.biases)
        # Error at the output layer (last layer)
        output_error = (y_true - activations[-1]) * self.sigmoid_derivative(excitations[-1], len(self.layer_sizes) - 1)
        errors[-1] = output_error
        # Back-propagate error through hidden layers
        for i in reversed(range(len(self.weights) - 1)):
            errors[i] = np.dot(errors[i + 1], self.weights[i + 1].T) * self.sigmoid_derivative(excitations[i], i + 1)
        for i in range(len(self.weights)):
            weight_gradients[i] = np.clip(-np.dot(activations[i].T, errors[i]), a_min=-5, a_max=5)
            bias_gradients[i] = np.clip(-np.sum(errors[i], axis=0, keepdims=True), a_min=-5, a_max=5)
        return weight_gradients, bias_gradients

    # Compute mean squared error
    def compute_error(self, x, y):
        predictions, _ = self.forward_propagation(x)
        return np.mean((predictions[-1] - y) ** 2)

    # Train the perceptron using gradient descent
    def train(self, x, y, epoch_limit, error_limit):
        self.initialize_weights()
        weight_updates = [np.zeros_like(w) for w in self.weights]
        bias_updates = [np.zeros_like(w) for w in self.biases]
        min_error = np.inf  # Initialize minimum error
        best_weights = None  # To store the best weights
        best_biases = None  # To store the best biases
        training_done = False
        epoch = 0
        weight_history = []
        bias_history = []
        error_history = []
        while not training_done and epoch < epoch_limit:
            indexes_shuffled = list(range(len(x)))
            random.shuffle(indexes_shuffled)
            for sample_idx in indexes_shuffled:
                x_sample = x[sample_idx:sample_idx + 1]
                y_sample = y[sample_idx:sample_idx + 1]
                # Forward pass
                activations, excitations = self.forward_propagation(x_sample)
                # Backpropagation, and weight and bias updates
                weight_gradients, bias_gradients = self.back_propagation(y_sample, activations, excitations)
                weight_updates_aux, bias_updates_aux = self.calculate_weight_updates(weight_gradients, bias_gradients)
                if not self.weight_updates_by_epoch:
                    for i in range(len(self.weights)):
                        self.weights[i] += weight_updates_aux[i]
                        self.biases[i] += bias_updates_aux[i]
                else:
                    for i in range(len(self.weights)):
                        weight_updates[i] += weight_updates_aux[i]
                        bias_updates[i] += bias_updates_aux[i]
                if not self.weight_updates_by_epoch:
                    error = self.compute_error(x, y)
                    # Update the minimum error and best weights if the current error is lower or equal
                    if error <= min_error:
                        # print(np.sum(np.abs(np.rint(self.forward_propagation(x)[0][-1]) - x)), error)
                        min_error = error
                        best_weights = [w.copy() for w in self.weights]  # Store the best weights
                        best_biases = [w.copy() for w in self.biases]  # Store the best biases
                        if min_error < error_limit:
                            training_done = True
                            break
            # Weight and bias updates
            if self.weight_updates_by_epoch:
                for i in range(len(self.weights)):
                    self.weights[i] += weight_updates[i]
                    self.biases[i] += bias_updates[i]
                weight_updates = [np.zeros_like(w) for w in self.weights]
                bias_updates = [np.zeros_like(w) for w in self.biases]
            epoch += 1
            error = self.compute_error(x, y)
            if epoch % self.training_level == 0:
                weight_history.append([np.copy(w) for w in self.weights])
                bias_history.append([np.copy(b) for b in self.biases])
                error_history.append(error)
            if self.weight_updates_by_epoch:
                # Update the minimum error, and best weights and biases if the current error is lower or equal
                if error <= min_error:
                    print(error)
                    min_error = error
                    best_weights = [w.copy() for w in self.weights]  # Store the best weights
                    best_biases = [w.copy() for w in self.biases]  # Store the best biases
                    if min_error < error_limit:
                        break
        return best_weights, best_biases, min_error, epoch, weight_history, bias_history, error_history

    # Predict output for given input X
    def predict(self, x, weights=None, biases=None):
        activations, _ = self.forward_propagation(x, weights, biases)
        return activations[-1]  # Return the output from the last layer


# Example usage
if __name__ == '__main__':
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
    Y = np.array([[0], [1], [1], [0]])  # Expected outputs
    # Instantiate the MultilayerPerceptron class
    mlp = MultilayerPerceptron([2, 8, 1], beta=1.25, learning_rate=0.05, momentum=0.8, weight_updates_by_epoch=False)
    # Train the MLP
    trained_weights, err, epochs, _, _ = mlp.train(X, Y, np.inf, 0.005)
    print("Trained weights:", trained_weights)
    print("Minimum error:", err)
    print("Epoch reached:", epochs)
    # Testing the trained network on the XOR problem
    prediction = mlp.predict(X)
    print("Predictions:", prediction)
