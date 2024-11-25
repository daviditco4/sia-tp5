import numpy as np
import matplotlib.pyplot as plt

# Función para inicializar los pesos y los sesgos
def initialize_weights(structure):
    weights = []
    biases = []
    
    for i in range(len(structure) - 1):
        weight_matrix = np.random.randn(structure[i], structure[i + 1]) * 0.01
        bias_vector = np.zeros((1, structure[i + 1]))
        weights.append(weight_matrix)
        biases.append(bias_vector)
    
    return weights, biases

# Función de forward propagation
def forward_propagation(x, weights, biases):
    activations = [x]
    excitations= []
    for W, b in zip(weights, biases):
        excitations.append((np.dot(x,W) + b))
        x = sigmoid(np.dot(x, W) + b)
        activations.append(x)
    return activations, excitations

# Función de activación Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

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

# Backpropagation general
def backpropagation(activations, weights, biases, gradients, lr):
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
            
def decoder_backprop(output, activations, excitations, weights, biases):
    errors = [0.0] * len(weights)  # Initialize error list
    weight_gradients = [0.0] * len(weights)
    bias_gradients = [0.0] * len(biases)
    # Error at the output layer (last layer)
    aux = output - activations[-1]
    output_error = aux * sigmoid_derivative(excitations[-1])
    errors[-1] = output_error
    # Back-propagate error through hidden layers
    for i in reversed(range(len(weights) - 1)):
        errors[i] = np.dot(errors[i + 1], weights[i + 1].T) * sigmoid_derivative(excitations[i])
    for i in range(len(weights)):
        weight_gradients[i] = np.clip(-np.dot(activations[i].T, errors[i]), a_min=-5, a_max=5)
        bias_gradients[i] = np.clip(-np.sum(errors[i], axis=0, keepdims=True), a_min=-5, a_max=5)
    return weight_gradients, bias_gradients, errors

def encoder_backprop(grad_mu, grad_logvar, W_mu, W_logvar,  activations, excitations, weights, biases):
    errors = [0.0] * (len(weights))  # Initialize error list
    weight_gradients = [0.0] * (len(weights))
    bias_gradients = [0.0] * (len(biases))
    grad_anterior_capa = np.dot(grad_mu, W_mu) + np.dot(grad_logvar, W_logvar)
    errors[-1] = grad_anterior_capa * sigmoid_derivative(excitations[-1])
    for i in reversed(range(len(weights) - 1)):
        errors[i] = np.dot(errors[i+1], weights[i+1].T) * sigmoid_derivative(excitations[i])
    for i in range(len(weights)):
        weight_gradients[i] = np.clip(-np.dot(activations[i].T, errors[i]), a_min=-5, a_max=5)
        bias_gradients[i] = np.clip(-np.sum(errors[i], axis=0, keepdims=True), a_min=-5, a_max=5)   
    return weight_gradients, bias_gradients, errors

def wab_actualizations(W_grad, b_grad, weigths, biases, lr):
    for i in range(len(weigths)):
        delta_W = lr * W_grad[i]
        delta_b = lr * b_grad[i]
        weigths[i] -= delta_W
        biases[i] -= delta_b
    
# Función de entrenamiento
def train_vae(X, structure, epochs, lr):
    input_dim = structure[0]
    hidden_layers = structure[1:-1]
    latent_dim = structure[-1]
    
    # Inicializar pesos y sesgos para encoder y decoder
    encoder_weights, encoder_biases = initialize_weights(structure)
    
    # Asegúrate de que el encoder tenga un conjunto de pesos y sesgos separados para 'mu' y 'logvar'
    W_mu = np.random.randn(structure[-1], structure[-2]) * 0.01  # Pesos para 'mu'
    W_logvar = np.random.randn(structure[-1], structure[-2]) * 0.01  # Pesos para 'logvar'
    b_mu = np.zeros((1, structure[-1]))  # Sesgos para 'mu'
    b_logvar = np.zeros((1, structure[-1]))  # Sesgos para 'logvar'

    # Para el decoder, invertimos la estructura
    decoder_structure = [latent_dim] + hidden_layers[::-1] + [input_dim]
    decoder_weights, decoder_biases = initialize_weights(decoder_structure)

    total_losses = []

    for epoch in range(epochs):
        total_loss = 0
        for x in X:
            x = x.reshape(1, -1)  # Reshape the input to match the expected dimensions

            # Encoder forward pass
            encoder_activations, encoder_excitations = forward_propagation(x, encoder_weights[:-1], encoder_biases[:-1])
            h_enc = encoder_activations[-1]  # Última capa oculta del encoder

            # Cálculo de 'mu' y 'logvar' con pesos separados
            mu = np.dot(h_enc, W_mu.T) + b_mu  # Capa de salida para 'mu'
            logvar = np.dot(h_enc, W_logvar.T) + b_logvar  # Capa de salida para 'logvar'

            # Reparameterization trick
            std = np.exp(0.5 * logvar)
            z = mu + std * np.random.randn(*std.shape)

            # Decoder forward pass
            decoder_activations, decoder_excitations = forward_propagation(z, decoder_weights, decoder_biases)
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
            decoder_w_gradients, decoder_b_gradients, decoder_errors = decoder_backprop(x, decoder_activations, decoder_excitations, decoder_weights, decoder_biases)
            
            #Compute latent gradients
            z_gradient = np.dot(decoder_errors[0], decoder_weights[0].T)
            grad_mu_kl = mu  # Gradiente de KL respecto a mu
            grad_logvar_kl = 0.5 * (np.exp(logvar) - 1)  # Gradiente de KL respecto a logvar
            grad_mu = z_gradient + grad_mu_kl
            grad_logvar = z_gradient * (z-mu) * 0.5 + grad_logvar_kl
            
            #Calculando el gradiente para W_mu y W_logvar
            grad_W_mu = np.dot((grad_mu * sigmoid_derivative(mu)).T, h_enc)
            grad_W_logvar = np.dot((grad_logvar * sigmoid_derivative(logvar)).T, h_enc)
            grad_b_mu = grad_mu
            grad_b_logvar = grad_logvar

            encoder_w_gradients, encoder_b_gradients, encoder_errors = encoder_backprop(grad_mu, grad_logvar, W_mu, W_logvar, encoder_activations, encoder_excitations, encoder_weights[:-1], encoder_biases[:-1])

            #Actualizacion de pesos y sesgos del decoder
            wab_actualizations(decoder_w_gradients, decoder_b_gradients, decoder_weights, decoder_biases,lr)
            
            # Actualización de los pesos y sesgos de mu y logvar
            W_mu -= lr * grad_W_mu
            b_mu -= lr * grad_b_mu
            W_logvar -= lr * grad_W_logvar
            b_logvar -= lr * grad_b_logvar
            
            #Actualizacion de pesos y sesgos en el encoder
            wab_actualizations(encoder_w_gradients, encoder_b_gradients, encoder_weights[:-1], encoder_biases[:-1], lr)

        total_losses.append(total_loss / len(X))

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(X):.4f}")

    # Plot the total loss
    plt.plot(total_losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE + KL)")
    plt.title("Training Loss")
    plt.show()


# Example usage
X = np.random.binomial(1, 0.5, (100, 25))  # Example binary data for letters
train_vae(X, structure=[25, 10, 5, 2], epochs=10000, lr=0.01)
