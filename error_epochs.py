import matplotlib.pyplot as plt
import numpy as np
from MLPLinearAutoencoder import MLPLinearAutoencoder

if __name__ == '__main__':
    # Example dataset
    X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])

    # Create a deep autoencoder with a mirrored architecture
    autoencoder = MLPLinearAutoencoder(encoder_layers=[2, 4, 2], learning_rate=0.01, momentum=0.9)

    # Train the autoencoder and collect error history
    trained_weights, min_error, epochs, errors = autoencoder.train_autoencoder(X, epoch_limit=1000, error_limit=0.01)

    # Print some training stats
    print("Trained weights:", trained_weights)
    print("Minimum error:", min_error)
    print("Epochs used:", epochs)

    # Plot the training error vs. epochs
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(errors) + 1), errors, label='Training Error', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Training Error vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
    