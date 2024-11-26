import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

def lineal(k):
    return k

def lineal_der(k):
    return np.ones_like(k)

def tanh(k, beta):
    return np.tanh(beta * k)

def tanh_der(k, beta):
    tanh_k = tanh(beta * k)
    return beta * (1 - (tanh_k ** 2))

# Función de activación Sigmoid
def sigmoid(x, beta):
    return 1 / (1 + np.exp(-2 * beta * x))

def sigmoid_derivative(x, beta):
    s = sigmoid(x, beta)
    return 2 * beta * s * (1 - s)

# Función de la pérdida MSE
def mse_loss(y_pred, y_true):
    return np.mean(np.square(y_pred - y_true))

# Derivada de la función de pérdida MSE
def mse_loss_derivative(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.size

# Función para calcular la divergencia KL
def kl_divergence_loss(mu, logvar):
    return -0.5 * np.sum(1 + logvar - np.square(mu) - np.exp(logvar))

# Derivadas de la divergencia KL
def kl_divergence_grad(mu, logvar):
    d_mu = -mu
    d_logvar = -0.5 + 0.5 * np.exp(logvar)
    return d_mu, d_logvar

class VAEAutoencoder:
    def __init__(self, structure, beta=1.0, learning_rate=0.001, epochs = 1000):
        self.structure = structure  # List defining the number of neurons per layer
        self.beta = beta
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.decoder_weights = None
        self.decoder_biases = None
        self.encoder_weights = None
        self.encoder_biases = None
        self.W_mu = None
        self.W_logvar = None
        self.b_mu = None
        self.b_logvar = None


    # Función para inicializar los pesos y los sesgos
    def initialize_weights(self, structure):
        weights = []
        biases = []
        
        for i in range(len(structure) - 1):
            weight_matrix = np.random.randn(structure[i], structure[i + 1]) * 0.01
            bias_vector = np.zeros((1, structure[i + 1]))
            weights.append(weight_matrix)
            biases.append(bias_vector)
        
        return weights, biases

    # Función de forward propagation
    def forward_propagation(self, x, weights, biases, beta):
        activations = [x]
        excitations= []
        for W, b in zip(weights, biases):
            excitations.append((np.dot(x,W) + b))
            x = sigmoid(np.dot(x, W) + b, beta)
            activations.append(x)
        return activations, excitations

    # Backpropagation general
    def backpropagation(self, activations, weights, biases, gradients, lr):
        for i in range(len(weights) - 1, -1, -1):
            # Compute gradients for weights and biases
            grad_activation = gradients.pop()
            grad_weight = np.dot(activations[i].T, grad_activation)
            grad_bias = np.sum(grad_activation, axis=0, keepdims=True)

            # Update weights and biases
            weights[i] -= lr * grad_weight
            biases[i] -= lr * grad_bias

            # Compute the gradient for the previous layer
            if i > 0:
                grad_activation = np.dot(grad_activation, weights[i].T) * activations[i] * (1 - activations[i])
                gradients.append(grad_activation)
                
    def decoder_backprop(self, output, activations, excitations, weights, biases, beta):
        errors = [0.0] * len(weights)  # Initialize error list
        weight_gradients = [0.0] * len(weights)
        bias_gradients = [0.0] * len(biases)
        # Error at the output layer (last layer)
        aux = output - activations[-1]
        output_error = aux * sigmoid_derivative(excitations[-1], beta)
        errors[-1] = output_error
        # Back-propagate error through hidden layers
        for i in reversed(range(len(weights) - 1)):
            errors[i] = np.dot(errors[i + 1], weights[i + 1].T) * sigmoid_derivative(excitations[i], beta)
        for i in range(len(weights)):
            weight_gradients[i] = np.clip(-np.dot(activations[i].T, errors[i]), a_min=-5, a_max=5)
            bias_gradients[i] = np.clip(-np.sum(errors[i], axis=0, keepdims=True), a_min=-5, a_max=5)
        return weight_gradients, bias_gradients, errors

    def encoder_backprop(self, grad_mu, grad_logvar, W_mu, W_logvar,  activations, excitations, weights, biases, beta):
        errors = [0.0] * (len(weights))  # Initialize error list
        weight_gradients = [0.0] * (len(weights))
        bias_gradients = [0.0] * (len(biases))
        grad_anterior_capa = np.dot(grad_mu, W_mu) + np.dot(grad_logvar, W_logvar)
        errors[-1] = grad_anterior_capa * sigmoid_derivative(excitations[-1], beta)
        for i in reversed(range(len(weights) - 1)):
            errors[i] = np.dot(errors[i+1], weights[i+1].T) * sigmoid_derivative(excitations[i], beta)
        for i in range(len(weights)):
            weight_gradients[i] = np.clip(-np.dot(activations[i].T, errors[i]), a_min=-5, a_max=5)
            bias_gradients[i] = np.clip(-np.sum(errors[i], axis=0, keepdims=True), a_min=-5, a_max=5)   
        return weight_gradients, bias_gradients, errors

    def wab_actualizations(self, W_grad, b_grad, weigths, biases, lr):
        for i in range(len(weigths)):
            delta_W = lr * W_grad[i]
            delta_b = lr * b_grad[i]
            weigths[i] -= delta_W
            biases[i] -= delta_b
        
    # Función de entrenamiento
    def train_vae(self, X):
        input_dim = self.structure[0]
        hidden_layers = self.structure[1:-1]
        latent_dim = self.structure[-1]
        
        # Inicializar pesos y sesgos para encoder y decoder
        self.encoder_weights, self.encoder_biases = self.initialize_weights(self.structure)
        
        # Asegúrate de que el encoder tenga un conjunto de pesos y sesgos separados para 'mu' y 'logvar'
        self.W_mu = np.random.randn(self.structure[-1], self.structure[-2]) * 0.01  # Pesos para 'mu'
        self.W_logvar = np.random.randn(self.structure[-1], self.structure[-2]) * 0.01  # Pesos para 'logvar'
        self.b_mu = np.zeros((1, self.structure[-1]))  # Sesgos para 'mu'
        self.b_logvar = np.zeros((1, self.structure[-1]))  # Sesgos para 'logvar'

        # Para el decoder, invertimos la estructura
        decoder_structure = [latent_dim] + hidden_layers[::-1] + [input_dim]
        self.decoder_weights, self.decoder_biases = self.initialize_weights(decoder_structure)

        total_losses = []

        for epoch in range(self.epochs):
            total_loss = 0
            for x in X:
                x = x.reshape(1, -1)  # Reshape the input to match the expected dimensions

                # Encoder forward pass
                encoder_activations, encoder_excitations = self.forward_propagation(x, self.encoder_weights[:-1], self.encoder_biases[:-1], self.beta)
                h_enc = encoder_activations[-1]  # Última capa oculta del encoder

                # Cálculo de 'mu' y 'logvar' con pesos separados
                mu = np.dot(h_enc, self.W_mu.T) + self.b_mu  # Capa de salida para 'mu'
                logvar = np.dot(h_enc, self.W_logvar.T) + self.b_logvar  # Capa de salida para 'logvar'

                # Reparameterization trick
                std = np.exp(0.5 * logvar)
                z = mu + std * np.random.randn(*std.shape)

                # Decoder forward pass
                decoder_activations, decoder_excitations = self.forward_propagation(z, self.decoder_weights, self.decoder_biases, self.beta)
                reconstructed = decoder_activations[-1]

                # Compute losses
                mse = mse_loss(reconstructed, x)
                kl = kl_divergence_loss(mu, logvar)
                loss = mse + kl
                total_loss += loss

                # Decoder backpropagation
                #d_reconstructed = mse_loss_derivative(reconstructed, x)
                #decoder_gradients = [d_reconstructed]
                #backpropagation(decoder_activations, decoder_weights, decoder_biases, decoder_gradients, lr)
                decoder_w_gradients, decoder_b_gradients, decoder_errors = self.decoder_backprop(x, decoder_activations, decoder_excitations, self.decoder_weights, self.decoder_biases, self.beta)
                
                #Compute latent gradients
                z_gradient = np.dot(decoder_errors[0], self.decoder_weights[0].T)
                grad_mu_kl = mu  # Gradiente de KL respecto a mu
                grad_logvar_kl = 0.5 * (np.exp(logvar) - 1)  # Gradiente de KL respecto a logvar
                grad_mu = z_gradient + grad_mu_kl
                grad_logvar = z_gradient * (z-mu) * 0.5 + grad_logvar_kl
                
                #Calculando el gradiente para W_mu y W_logvar
                grad_W_mu = np.dot((grad_mu * sigmoid_derivative(mu, self.beta)).T, h_enc)
                grad_W_logvar = np.dot((grad_logvar * sigmoid_derivative(logvar, self.beta)).T, h_enc)
                grad_b_mu = grad_mu
                grad_b_logvar = grad_logvar

                encoder_w_gradients, encoder_b_gradients, encoder_errors = self.encoder_backprop(grad_mu, grad_logvar, self.W_mu, self.W_logvar, encoder_activations, encoder_excitations, self.encoder_weights[:-1], self.encoder_biases[:-1], self.beta)

                #Actualizacion de pesos y sesgos del decoder
                self.wab_actualizations(decoder_w_gradients, decoder_b_gradients, self.decoder_weights, self.decoder_biases,self.learning_rate)
                
                # Actualización de los pesos y sesgos de mu y logvar
                self.W_mu -= self.learning_rate * grad_W_mu
                self.b_mu -= self.learning_rate * grad_b_mu
                self.W_logvar -= self.learning_rate * grad_W_logvar
                self.b_logvar -= self.learning_rate * grad_b_logvar
                
                #Actualizacion de pesos y sesgos en el encoder
                self.wab_actualizations(encoder_w_gradients, encoder_b_gradients, self.encoder_weights[:-1], self.encoder_biases[:-1], self.learning_rate)

            total_losses.append(total_loss / len(X))

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss / len(X):.4f}")

        # Plot the total loss
        plt.plot(total_losses)
        plt.xlabel("Epochs")
        plt.ylabel("Loss (MSE + KL)")
        plt.title("Training Loss")
        plt.show()
        
    def decode(self, Z):
        reconstructed_array = []
        for x in Z:
            decoder_activations, decoder_excitations = self.forward_propagation(x, self.decoder_weights, self.decoder_biases, self.beta)
            reconstructed = decoder_activations[-1]
            reconstructed_array.append(reconstructed)
        return reconstructed_array
        
    def example(self, x):
        x = x.reshape(1, -1)  # Reshape the input to match the expected dimensions

        # Encoder forward pass
        encoder_activations, encoder_excitations = self.forward_propagation(x, self.encoder_weights[:-1], self.encoder_biases[:-1], self.beta)
        h_enc = encoder_activations[-1]  # Última capa oculta del encoder

        # Cálculo de 'mu' y 'logvar' con pesos separados
        mu = np.dot(h_enc, self.W_mu.T) + self.b_mu  # Capa de salida para 'mu'
        logvar = np.dot(h_enc, self.W_logvar.T) + self.b_logvar  # Capa de salida para 'logvar'

        # Reparameterization trick
        std = np.exp(0.5 * logvar)
        z = mu + std * np.random.randn(*std.shape)

        # Decoder forward pass
        decoder_activations, decoder_excitations = self.forward_propagation(z, self.decoder_weights, self.decoder_biases, self.beta)
        reconstructed = decoder_activations[-1]

        return reconstructed
    
    def plot_latent_grid(self, grid_size=25, pixel_dims=(7, 7), latent_range=(-2.5, 2.5)):
        """
        Plot a grid of 7x7 bitmaps decoded from a grid of latent space points, with labeled axes.
        The space between the subplots is filled with black.
        Parameters:
        - grid_size: Number of points along each dimension of the latent space grid.
        - pixel_dims: Dimensions of the output bitmaps (default is 7x7).
        - latent_range: Range of the latent space to sample from (default is [-1, 1]).
        """
        if self.structure[-1] != 2:
            raise ValueError("Latent space must be 2D to use this visualization.")
        # Create a 2D grid of points in the latent space
        x_coords = np.linspace(latent_range[0], latent_range[1], grid_size)
        y_coords = np.linspace(latent_range[0], latent_range[1], grid_size)
        latent_grid = np.array([[x, y] for y in y_coords for x in x_coords])
        # Decode each latent point
        decoded_images = np.array(self.decode(latent_grid))
        decoded_images = decoded_images.reshape((-1, *pixel_dims))  # Reshape to (grid_size^2, height, width)
        # Plot the grid of images
        fig, axes = plt.subplots(grid_size, grid_size,
                                 figsize=(12, 12))  # ,
                                 # gridspec_kw={'wspace': 0.0, 'hspace': 0.0})  # Remove all spacing
        # Set the figure background to black
        fig.patch.set_facecolor('black')
        for i in range(grid_size):
            for j in range(grid_size):
                index = i * grid_size + j
                axes[i, j].imshow(decoded_images[index], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                axes[i, j].set_aspect('equal')  # Ensure images don't stretch
                axes[i, j].set_facecolor('black')  # Set subplot background to black
                # Add X and Y labels for the first row and first column
                if j == 0:
                    axes[i, j].set_ylabel(f"{y_coords[i]:.1f}", fontsize=16, color='white')
                if i == grid_size - 1:
                    axes[i, j].set_xlabel(f"{x_coords[j]:.1f}", fontsize=16, color='white')
        # # Set common labels for axes
        # fig.text(0.5, 0.02, 'Latent X', ha='center', fontsize=10, color='white')
        # fig.text(0.02, 0.5, 'Latent Y', va='center', rotation='vertical', fontsize=10, color='white')
        # Save the plot with a black background
        plt.tight_layout(pad=0.5)  # No padding to make it as compact as possible
        plt.savefig(
            "task1_combined_activation_latent_grid.png",
            # {varying_hyperparam.lower().replace(" ", "_")}_determined_latent_grid.png",
            dpi=300, bbox_inches='tight',  # facecolor=fig.get_facecolor()
        )
        plt.close()
        
    def encode(self, input):
        z_values = []
        
        for k in input:
        
            k = k.reshape(1, -1)  # Reshape the input to match the expected dimensions

            # Encoder forward pass
            encoder_activations, encoder_excitations = self.forward_propagation(k, self.encoder_weights[:-1], self.encoder_biases[:-1], self.beta)
            h_enc = encoder_activations[-1]  # Última capa oculta del encoder

            # Cálculo de 'mu' y 'logvar' con pesos separados
            mu = np.dot(h_enc, self.W_mu.T) + self.b_mu  # Capa de salida para 'mu'
            logvar = np.dot(h_enc, self.W_logvar.T) + self.b_logvar  # Capa de salida para 'logvar'

            # Reparameterization trick
            std = np.exp(0.5 * logvar)
            new_z = mu + std * np.random.randn(*std.shape)
            
            z_values.append(*new_z)
        
        return z_values
        
    def plot_latent_space(self, x, labels=None):
        if self.structure[-1] != 2:
            raise ValueError("The latent space must be 2D to visualize. Adjust the architecture.")
        # Encode input data into latent representations
        latent_representations = self.encode(x)
        # Compute distances from each point to the origin (0, 0)
        #distances = [euclidean(point, [0, 0]) for point in latent_representations]
        # Normalize distances to [0, 1] for colormap scaling
        #max_distance = max(distances)
        #normalized_distances = [d / max_distance for d in distances]
        # Use a colormap (e.g., viridis) to assign colors based on distances
        colormap = plt.cm.viridis
        #colors = [colormap(norm_dist) for norm_dist in normalized_distances]
        # Plot the points
        plt.figure(figsize=(8, 6))
        for i, point in enumerate(latent_representations):
            plt.scatter(point[0], point[1], color="blue", s=50, alpha=0.7)
            # Adjust text labels with an offset for spacing
            if labels is not None:
                text_offset = 0.08  # Adjust the offset value to your preference
                plt.text(
                    point[0] + text_offset,
                    point[1] + text_offset,
                    str(labels[i]),
                    fontsize=9,
                    ha='center',  # Horizontal alignment
                    va='center'  # Vertical alignment
                )
        # Add a colorbar to represent the distances
        #sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=max_distance))
        #sm.set_array([])
        #cbar = plt.colorbar(sm)
        #cbar.set_label('Distance from Center')
        # Add titles, labels, and grid
        plt.title("2D Latent Space Representation")
        plt.xlabel("Latent Dimension 1")
        plt.ylabel("Latent Dimension 2")
        plt.grid(True)
        # Save and close the plot
        plt.savefig(
            f"task1_combined_activation_latent_space.png",
            # {varying_hyperparam.lower().replace(" ", "_")}_determined_latent_space.png",
            dpi=300, bbox_inches='tight')
        plt.close()

inputs = np.array([
    # Cara feliz 🙂
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
    ],
    # Cara triste ☹️
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
    ],
    # Cara sorprendida 😮
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
    ],
    # Corazón ❤️
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ],
    # Estrella ⭐
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ],
    # Pulgar arriba 👍
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
    ],
    # Carita guiñando 😉
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
    ],
    # Cara con gafas 😎
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 1, 0],
    ],
    # Carita llorando 😢
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
    ],
    # Carita enfadada 😡
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
    ],
])

# Graficar los emojis
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.flatten()

for i, emoji in enumerate(inputs):
    axes[i].imshow(emoji, cmap="Greys", interpolation="nearest")
    axes[i].axis("off")
    axes[i].set_title(f"Emoji {i+1}")

plt.tight_layout()
plt.show()

inputs = inputs / 1.0

new_input = inputs[:6]

vae = VAEAutoencoder([49, 21, 7, 2], beta=1, learning_rate=0.001, epochs=2500)
vae.train_vae(new_input)

'''
index = 0
emoji_example = inputs[index]
reconstructed_emoji = vae.example(emoji_example)
reconstructed_matrix = np.array(reconstructed_emoji).reshape(7,7)

plt.figure(figsize=(3, 3))  # Ajusta el tamaño para que se vea bien como cuadrícula
plt.imshow(reconstructed_matrix, cmap="Greys", interpolation="nearest")
plt.axis("off")
plt.title(f"Emoji {index + 1}")
plt.gca().set_aspect("equal")  # Asegurar que las celdas sean cuadradas
plt.show()
'''

#zed = vae.encode(new_input)

#print(zed)

#vae.plot_latent_space(new_input, ["feliz", "triste", "sorprendida", "corazon", "estrella", "pulgar arriba"])


index = 3
emoji_example = new_input[index]
reconstructed_emoji = vae.example(emoji_example)
reconstructed_matrix = np.array(reconstructed_emoji).reshape(7,7)

plt.figure(figsize=(3, 3))  # Ajusta el tamaño para que se vea bien como cuadrícula
plt.imshow(reconstructed_matrix, cmap="Greys", interpolation="nearest")
plt.axis("off")
plt.title(f"Emoji {index + 1}")
plt.gca().set_aspect("equal")  # Asegurar que las celdas sean cuadradas
plt.show()

#vae.plot_latent_grid()


