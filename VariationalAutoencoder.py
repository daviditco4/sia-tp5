import numpy as np
from MLPLinearAutoencoder import MLPLinearAutoencoder


class MLPVariationalAutoencoder(MLPLinearAutoencoder):
    def __init__(self, encoder_layers, beta=1.0, learning_rate=0.001, momentum=0.0):
        """
        Initialize a Variational Autoencoder with a mirrored decoder architecture.
        Parameters:
        - encoder_layers: List of integers defining the encoder architecture, e.g., [input_size, hidden1, latent_size].
        - learning_rate: Learning rate for weight updates.
        - momentum: Momentum for weight updates.
        """
        super().__init__(encoder_layers, beta, learning_rate, momentum)
        self.latent_size = encoder_layers[-1]

    def initialize_weights(self):
        super().initialize_weights()
        input_dim = self.encoder_layers[-2]
        self.weights[len(self.encoder_layers) - 1] = np.random.randn(self.latent_size // 2, input_dim) / 3
        self.biases[len(self.encoder_layers) - 1] = np.random.randn(1, input_dim) / 3

    def encode(self, x):
        """
        Override the encode method to compute μ and log(σ²) for the latent space.
        """
        encoder_weights = self.weights[: len(self.layer_sizes) // 2]
        encoder_biases = self.biases[: len(self.layer_sizes) // 2]
        activations, excitations = self.forward_propagation(x, encoder_weights, encoder_biases)
        latent_mean = np.array(*(activations[-1][:, : self.latent_size // 2]))
        latent_log_var = np.array(*(activations[-1][:, self.latent_size // 2:]))
        return latent_mean, latent_log_var, activations, excitations

    def reparameterize(self, latent_mean, latent_log_var):
        """
        Reparameterization trick to sample from N(μ, σ²) using N(0, 1).
        """
        epsilon = np.random.randn(self.latent_size // 2)
        latent_sample = latent_mean + epsilon * np.exp(0.5 * latent_log_var)  # σ = exp(0.5 * log(σ²))
        return latent_sample

    def compute_loss(self, x, reconstruction, latent_mean, latent_log_var):
        """
        Compute the VAE loss: reconstruction loss + KL divergence.
        """
        # Reconstruction loss (Mean Squared Error)
        reconstruction_loss = np.mean(np.sum((x - reconstruction) ** 2, axis=1))
        # KL divergence
        kl_divergence = -0.5 * np.mean(np.sum(1 + latent_log_var - latent_mean ** 2 - np.exp(latent_log_var), axis=1))
        return reconstruction_loss + kl_divergence, reconstruction_loss, kl_divergence

    def back_propagation_vae(self, x_sample, reconstruction, excitations, latent_mean, latent_log_var):
        """
        Backpropagation for VAE to compute weight and bias gradients, accounting for both reconstruction and KL divergence.
        """
        errors = [0.0] * len(self.weights)
        weight_gradients = [0.0] * len(self.weights)
        bias_gradients = [0.0] * len(self.biases)

        # Compute reconstruction error
        reconstruction_error = (x_sample - reconstruction[-1])  # Reconstruction loss gradient
        output_error = reconstruction_error * self.sigmoid_derivative(excitations[-1], len(self.layer_sizes) - 1)
        errors[-1] = output_error

        # KL divergence error (applied directly to the encoder)
        kl_divergence_grad_mean = latent_mean  # d(μ²)/dμ = 2μ
        kl_divergence_grad_log_var = 0.5 * (np.exp(latent_log_var) - 1)  # d(-log(σ²))/dσ²

        latent_e = kl_divergence_grad_mean + kl_divergence_grad_log_var

        # Backpropagate through the decoder
        self.is_just_decoding = True
        for i in reversed(range(len(self.weights) // 2, len(self.weights) - 1)):
            errors[i] = np.dot(errors[i + 1], self.weights[i + 1].T) * self.sigmoid_derivative(excitations[i], i + 1)
        for i in range(len(self.weights) // 2, len(self.weights)):
            weight_gradients[i] = -np.dot(reconstruction[i + 1].T, errors[i])
            bias_gradients[i] = -np.sum(errors[i], axis=0)
        self.is_just_decoding = False

        # Total error at the latent layer: reconstruction error + KL divergence
        latent_error = np.resize(kl_divergence_grad_mean + kl_divergence_grad_log_var, new_shape=(1, latent_e.shape[0]))
        errors[len(self.weights) // 2 - 1] = latent_error

        # Backpropagate through the encoder
        for i in reversed(range(len(self.weights) // 2 - 1)):
            errors[i] = np.dot(errors[i + 1], self.weights[i + 1].T) * self.sigmoid_derivative(excitations[i], i + 1)
        for i in range(len(self.weights) // 2):
            weight_gradients[i] = -np.dot(reconstruction[i].T, errors[i])
            bias_gradients[i] = -np.sum(errors[i], axis=0)

        return weight_gradients, bias_gradients

    def train_vae(self, x, epoch_limit, error_limit):
        """
        Train the VAE using the reparameterization trick and the VAE loss.
        """
        self.initialize_weights()
        epoch = 0
        min_error = np.inf
        best_weights = None
        best_biases = None

        while epoch < epoch_limit:
            for sample_idx in range(len(x)):
                x_sample = x[sample_idx:sample_idx + 1]

                # Forward pass: Encode to get latent mean and log variance
                latent_mean, latent_log_var, encoding_activations, encoding_excitations = self.encode(x_sample)

                # Reparameterization trick
                latent_sample = self.reparameterize(latent_mean, latent_log_var)

                self.is_just_decoding = True
                # Decode the latent sample to reconstruct the input
                reconstruction, decoding_excitations = self.forward_propagation(latent_sample, self.weights[len(self.layer_sizes) // 2:], self.biases[len(self.layer_sizes) // 2:])
                self.is_just_decoding = False
                reconstruction[0] = np.resize(reconstruction[0], new_shape=(1, latent_sample.shape[0]))

                # Backpropagate and update weights
                weight_gradients, bias_gradients = self.back_propagation_vae(
                    x_sample, encoding_activations + reconstruction, encoding_excitations + decoding_excitations, latent_mean, latent_log_var
                )
                weight_updates, bias_updates = self.calculate_weight_updates(weight_gradients, bias_gradients)

                for i in range(len(self.weights)):
                    self.weights[i] += weight_updates[i]
                    self.biases[i] += bias_updates[i]

            # Compute loss for the entire dataset
            loss, reconstruction_loss_total, kl_divergence_total = 0, 0, 0
            for sample_idx in range(len(x)):
                x_sample = x[sample_idx:sample_idx + 1]
                latent_mean, latent_log_var, _ = self.encode(x_sample)
                latent_sample = self.reparameterize(latent_mean, latent_log_var)

                self.is_just_decoding = True
                # Decode the latent sample to reconstruct the input
                recons, decoding_excitations = self.forward_propagation(latent_sample, self.weights[len(self.layer_sizes) // 2:], self.biases[len(self.layer_sizes) // 2:])
                self.is_just_decoding = False

                # Compute gradients for VAE loss
                loss, reconstruction_loss, kl_divergence = self.compute_loss(
                    x_sample, recons[-1], latent_mean, latent_log_var
                )
                reconstruction_loss_total += reconstruction_loss
                kl_divergence_total += kl_divergence

            if loss < min_error:
                min_error = loss
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]

            epoch += 1
            if min_error < error_limit:
                break

        return best_weights, best_biases, min_error, epoch


if __name__ == '__main__':
    emoji_1 = np.array([
        1, 0, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 0, 1,
        1, 0, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 0, 1,
        1, 0, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 0, 1,
        1, 0, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 0, 1
    ])

    emoji_2 = np.array([
        0, 1, 0, 1, 0, 1, 0, 1,
        1, 0, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 0, 1,
        1, 0, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 0, 1,
        1, 0, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 0, 1,
        1, 0, 1, 0, 1, 0, 1, 0
    ])

    emoji_3 = np.array([
        1, 1, 0, 0, 1, 1, 0, 0,
        1, 1, 0, 0, 1, 1, 0, 0,
        0, 0, 1, 1, 0, 0, 1, 1,
        0, 0, 1, 1, 0, 0, 1, 1,
        1, 1, 0, 0, 1, 1, 0, 0,
        1, 1, 0, 0, 1, 1, 0, 0,
        0, 0, 1, 1, 0, 0, 1, 1,
        0, 0, 1, 1, 0, 0, 1, 1
    ])

    # Stack the emojis into a dataset (3 examples of 8x8 bitmaps)
    X = np.array([emoji_1, emoji_2, emoji_3])

    # Define the encoder architecture [64, 32, 16]
    encoder_layers = [64, 16, 4]

    # Instantiate the VAE
    vae = MLPVariationalAutoencoder(encoder_layers, learning_rate=0.0001, momentum=0.9)

    # Train the VAE on the emoji dataset (3 emojis)
    trained_weights, trained_biases, min_error, epochs = vae.train_vae(X, epoch_limit=np.inf, error_limit=0.01)

    print("Trained weights:", trained_weights)
    print("Trained biases:", trained_biases)
    print("Minimum error:", min_error)
    print("Epochs used:", epochs)

    # Test encoding and reconstruction
    latent_representation = vae.encode(X)
    reconstructions = vae.decode(vae.encode(X)[:2])
    print("Latent representation:", np.resize(np.rint(latent_representation[0]), new_shape=(7, 5)))
    print("Reconstruction:", reconstructions)