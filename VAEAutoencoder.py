import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean

from MultilayerPerceptron import MultilayerPerceptron

class VAEAutoencoder:
    
    def __init__(self, encoder_layers, learning_rate=0.001, momentum=0.0, weight_updates_by_epoch = False):
        """
        Initialize a VAE with a mirrored decoder architecture.
        Parameters:
        - encoder_layers: List of integers defining the encoder architecture, e.g., [input_size, hidden1, latent_size].
        - learning_rate: Learning rate for weight updates.
        - momentum: Momentum for weight updates.
        """
        if len(encoder_layers) < 2:
            raise ValueError("Autoencoder requires at least two layers in the encoder: input and latent layers.")
        # Mirror the encoder layers to form the full autoencoder architecture
        decoder_layers = encoder_layers[-2::-1]  # Reverse encoder layers, excluding the last one (latent)
        encoder_layers[-1] *= 2 #Beacuse each neuron of the latent layer has both a median and deviation value 
        self.total_layers = encoder_layers + decoder_layers
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_updates_by_epoch = weight_updates_by_epoch
        
    def relu(self, x):
        return np.maximum(0, x)
        
    def relu_der(self, x):
        return np.where(x > 0, 1, 0)
    
    def linear(self, x):
        return x
    
    def linear_der(self, x):
        return 1
    
    # Initialize the weights for all layers
    def initialize_weights(self):
        self.weights = [np.zeros(0)] * (len(self.total_layers) - 1)
        for i in range(len(self.total_layers) - 1):
            if(i == (len(self.total_layers)//2)):
                self.weights[i] = np.random.randn(self.total_layers[i] // 2, self.total_layers[i + 1]) / 3
            else:
                # Initialize weights randomly from a normal distribution
                self.weights[i] = np.random.randn(self.total_layers[i], self.total_layers[i + 1]) / 3
        self.prev_weight_updates = [np.zeros_like(w) for w in self.weights]
        return self.weights
    
    def forward_propagation(self, x, weights=None):
        if weights is None:
            weights = self.weights
        activations = [x] * (len(weights) + 1)  # Store activations for each layer
        excitations = [None] * len(weights)  # Store weighted sums for each layer (before activation)
        for i in range(len(weights)):
            net_input = np.dot(activations[i], weights[i])  # Weighted sum (excitation)
            excitations[i] = net_input
            activation = self.linear(net_input)  # Apply activation function (sigmoid)
            activations[i + 1] = activation
        return activations, excitations
    
    def encode(self, x_input):
        encoder_weights = self.weights[: len(self.total_layers) // 2]
        activations, excitations = self.forward_propagation(x_input, encoder_weights)
        print(excitations)
        print(activations)
        return activations, excitations
    
    def reparametrization_trick(self, array):
        aux = [(array[i], array[i + 1]) for i in range(0, len(array), 2)]
        print(aux)
        epsilon = np.random.normal(0,1)
        Z = []
        for x,y in aux:
            Z.append(np.exp(0.5 * x) * epsilon + y)
        print(Z)
        return Z, epsilon
    
    def decode(self, Z):
        decoder_weights = self.weights[len(self.total_layers) // 2 :]
        print(decoder_weights)
        activations, excitations = self.forward_propagation(Z, decoder_weights)
        print(excitations)
        print(activations)
        return activations, excitations
    
    def train(self, x, epoch_limit, error_limit):
        y = x
        self.initialize_weights()
        weight_updates = [np.zeros_like(w) for w in self.weights]
        min_error = np.inf  # Initialize minimum error
        best_weights = None  # To store the best weights
        training_done = False
        epoch = 0
        weight_history = []
        error_history = []
        while not training_done and epoch < epoch_limit:
            indexes_shuffled = list(range(len(x)))
            random.shuffle(indexes_shuffled)
            for sample_idx in indexes_shuffled:
                x_sample = x[sample_idx:sample_idx + 1]
                y_sample = y[sample_idx:sample_idx + 1]
                # Forward pass
                activations, excitations = self.encode(x_sample)
                Z = self.reparametrization_trick(activations[-1])
                activations_dec, excitations_dec = self.decode(Z)
                # Backpropagation and weight updates
                weight_gradients = self.back_propagation(x_sample, y_sample, activations, excitations)
                weight_updates_aux = self.calculate_weight_updates(weight_gradients)
                if not self.weight_updates_by_epoch:
                    for i in range(len(self.weights)):
                        self.weights[i] += weight_updates_aux[i]
                else:
                    for i in range(len(self.weights)):
                        weight_updates[i] += weight_updates_aux[i]
                if not self.weight_updates_by_epoch:
                    error = self.compute_error(x, y)
                    # Update the minimum error and best weights if the current error is lower or equal
                    if error <= min_error:
                        print(error)
                        min_error = error
                        best_weights = [w.copy() for w in self.weights]  # Store the best weights
                        if min_error < error_limit:
                            training_done = True
                            break
            # Weight updates
            if self.weight_updates_by_epoch:
                for i in range(len(self.weights)):
                    self.weights[i] += weight_updates[i]
                weight_updates = [np.zeros_like(w) for w in self.weights]
            epoch += 1
            error = self.compute_error(x, y)
            weight_history.append(self.weights)
            error_history.append(error)
            if self.weight_updates_by_epoch:
                # Update the minimum error and best weights if the current error is lower or equal
                if error <= min_error:
                    print(error)
                    min_error = error
                    best_weights = [w.copy() for w in self.weights]  # Store the best weights
                    if min_error < error_limit:
                        break
        return best_weights, min_error, epoch, weight_history, error_history
        
        
    def decoder_backprop(self, output, activations, excitations):
        decoder_weights = self.weights[len(self.total_layers) // 2 :]
        errors = [None] * len(decoder_weights)  # Initialize error list
        weight_gradients = [None] * len(decoder_weights)
        # Error at the output layer (last layer)
        output_error = (output - activations[-1]) * self.linear_der(excitations[-1])
        errors[-1] = output_error
        # Back-propagate error through hidden layers
        for i in reversed(range(len(decoder_weights) - 1)):
            errors[i] = np.dot(errors[i + 1], decoder_weights[i + 1].T) * self.linear_der(excitations[i])
        for i in range(len(decoder_weights)):
            weight_gradients[i] = -np.dot(activations[i].T, errors[i])
        return weight_gradients
    
    def calculate_weight_updates(self, weight_gradients):
        weight_updates = [None] * len(self.weights)
        for i in range(len(self.weights)):
            weight_updates[i] = -self.learning_rate * weight_gradients[i] + self.momentum * self.prev_weight_updates[i]
            self.prev_weight_updates[i] = weight_updates[i]
        return weight_updates
    
    def encoder_backprop(self, Z_grad, epsilon, activations, excitations):
        array = activations[-1]
        aux = [(array[i], array[i + 1]) for i in range(0, len(array), 2)]
        latent_gradients = []
        for i in range(len(aux)):
            sigma = Z_grad[i] * epsilon
            log_var = sigma * np.exp(0.5 * aux[i][0]) * 0.5 - 0.5 * (1 - np.exp(aux[i][0]))
            mu = Z_grad[i] + aux[i][1]
            latent_gradients.append(np.array(mu, log_var))

        
        
        
# Example usage
if __name__ == '__main__':
    # Example dataset
    X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
    
    # Create a deep autoencoder with a mirrored architecture
    # Encoder layers: 2 -> 1 
    # Decoder layers: 1 -> 2 (automatically mirrored)
    autoencoder = VAEAutoencoder(encoder_layers=[2, 2], learning_rate=0.1, momentum=0.0)
    print(autoencoder.initialize_weights())
    latent = autoencoder.encode(X[0])
    Z = autoencoder.reparametrization_trick(latent)
    out = autoencoder.decode(Z)
    
