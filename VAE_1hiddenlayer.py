import numpy as np
import matplotlib.pyplot as plt

# Funciones auxiliares
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def mse_loss(reconstructed, original):
    return np.mean((reconstructed - original) ** 2)

def mse_loss_derivative(reconstructed, original):
    return 2 * (reconstructed - original) / original.size

def kl_divergence_loss(mu, logvar):
    return -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar))

def kl_divergence_grad(mu, logvar):
    d_mu = mu
    d_logvar = -0.5 * (1 - np.exp(logvar))
    return d_mu, d_logvar

# Inicialización de pesos
def initialize_weights(input_dim, hidden_dim, latent_dim):
    np.random.seed(42)
    W1 = np.random.randn(input_dim, hidden_dim) * 0.01
    b1 = np.zeros((1, hidden_dim))
    W2_mu = np.random.randn(hidden_dim, latent_dim) * 0.01
    b2_mu = np.zeros((1, latent_dim))
    W2_logvar = np.random.randn(hidden_dim, latent_dim) * 0.01
    b2_logvar = np.zeros((1, latent_dim))
    W3 = np.random.randn(latent_dim, hidden_dim) * 0.01
    b3 = np.zeros((1, hidden_dim))
    W4 = np.random.randn(hidden_dim, input_dim) * 0.01
    b4 = np.zeros((1, input_dim))
    return W1, b1, W2_mu, b2_mu, W2_logvar, b2_logvar, W3, b3, W4, b4

# Implementación del entrenamiento de un VAE
def train_vae(X, input_dim, hidden_dim, latent_dim, epochs=1000, lr=0.001):
    W1, b1, W2_mu, b2_mu, W2_logvar, b2_logvar, W3, b3, W4, b4 = initialize_weights(input_dim, hidden_dim, latent_dim)
    total_losses = []  # Para almacenar las pérdidas totales

    for epoch in range(epochs):
        total_loss = 0

        for x in X:
            x = x.reshape(1, -1)

            # Encoder
            h1 = sigmoid(np.dot(x, W1) + b1)
            mu = np.dot(h1, W2_mu) + b2_mu
            logvar = np.dot(h1, W2_logvar) + b2_logvar
            
            # Reparametrización
            std = np.exp(0.5 * logvar)
            z = mu + std * np.random.randn(*std.shape)

            # Decoder
            h2 = sigmoid(np.dot(z, W3) + b3)
            reconstructed = sigmoid(np.dot(h2, W4) + b4)

            # Pérdida
            recon_loss = mse_loss(reconstructed, x)
            kl_loss = kl_divergence_loss(mu, logvar)
            loss = recon_loss + kl_loss
            total_loss += loss

            # Backpropagation
            d_recon = mse_loss_derivative(reconstructed, x) * sigmoid_derivative(reconstructed)
            d_W4 = np.dot(h2.T, d_recon)
            d_b4 = np.sum(d_recon, axis=0, keepdims=True)

            d_h2 = np.dot(d_recon, W4.T) * sigmoid_derivative(h2)
            d_W3 = np.dot(z.T, d_h2)
            d_b3 = np.sum(d_h2, axis=0, keepdims=True)

            d_z = np.dot(d_h2, W3.T)
            d_mu, d_logvar = kl_divergence_grad(mu, logvar)
            d_z += d_mu + d_logvar * std

            d_h1 = np.dot(d_z, W2_mu.T + W2_logvar.T) * sigmoid_derivative(h1)
            d_W2_mu = np.dot(h1.T, d_z)
            d_b2_mu = np.sum(d_z, axis=0, keepdims=True)
            d_W2_logvar = np.dot(h1.T, d_logvar)
            d_b2_logvar = np.sum(d_logvar, axis=0, keepdims=True)

            d_W1 = np.dot(x.T, d_h1)
            d_b1 = np.sum(d_h1, axis=0, keepdims=True)

            # Actualización de pesos
            W1 -= lr * d_W1
            b1 -= lr * d_b1
            W2_mu -= lr * d_W2_mu
            b2_mu -= lr * d_b2_mu
            W2_logvar -= lr * d_W2_logvar
            b2_logvar -= lr * d_b2_logvar
            W3 -= lr * d_W3
            b3 -= lr * d_b3
            W4 -= lr * d_W4
            b4 -= lr * d_b4

        total_losses.append(total_loss / len(X))  # Pérdida promedio

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(X)}")

    # Gráfico del error
    plt.figure(figsize=(8, 6))
    plt.plot(total_losses, label="Loss (MSE + KL)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("VAE Training Loss")
    plt.legend()
    plt.grid()
    plt.show()

# Letras pixeladas como entrada (ejemplo simplificado)
X = np.array([
    # Letra 'A'
    [0, 1, 1, 1, 0,
     1, 0, 0, 0, 1,
     1, 1, 1, 1, 1,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1],
    # Letra 'B'
    [1, 1, 1, 1, 0,
     1, 0, 0, 0, 1,
     1, 1, 1, 1, 0,
     1, 0, 0, 0, 1,
     1, 1, 1, 1, 0]
])  # Cada fila es una letra, aplanada a un vector

# Normalización de los datos
X = X / 1.0  # Mantener como valores binarios (0 y 1)

# Entrenamiento
train_vae(X, input_dim=25, hidden_dim=15, latent_dim=3, epochs=10000, lr=0.001)
