import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
from MLPLinearAutoencoderOfAdam import MLPLinearAutoencoderOfAdam
from assets.fonts import font3, font3_labels

# Step 1: Convert font3 binary data into usable format
def binary_arrays_from_font3():
    """
    Converts hexadecimal character data into binary arrays.
    """
    binary_data = []
    for character in font3:
        # Convert each hexadecimal row into a binary array of size 5
        binary_char = [list(format(row, '05b')) for row in character]
        # Convert binary strings ('0'/'1') to integers
        binary_char = np.array(binary_char, dtype=int)
        binary_data.append(binary_char)
    return np.array(binary_data)

# Step 2: Prepare the dataset
characters = binary_arrays_from_font3()
characters = characters.reshape(len(font3), -1)  # Flatten each 7x5 matrix into a 35-element vector

# Step 3: Define and train the autoencoder
autoencoder = MLPLinearAutoencoderOfAdam(encoder_layers=[35, 25, 15, 5, 2], learning_rate=0.0001)

# Train the autoencoder (reduce error below 0.01 or for a set number of epochs)
trained_weights, min_error, epochs, _, _ = autoencoder.train_autoencoder(
    characters, epoch_limit=10000, error_limit=0.01
)
print(f"Training completed in {epochs} epochs with minimum error: {min_error}")

# Step 4: Encode the characters into latent space
latent_space = autoencoder.encode(characters)

# Ensure latent space is 2D (as defined in the autoencoder architecture)
assert latent_space.shape[1] == 2, "Latent space must have exactly 2 dimensions for plotting."

# Step 5: Plot the latent space
plt.figure(figsize=(10, 8))
texts = []
for i, (x, y) in enumerate(latent_space):
    plt.scatter(x, y, label=font3_labels[i])  # Scatter plot for each character
    texts.append(plt.text(x, y, font3_labels[i], fontsize=9))

# Automatically adjust labels to avoid overlaps
adjust_text(texts, arrowprops=dict(arrowstyle="->", color='gray'))

plt.title("Latent Space Representation of Binary Characters")
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.grid(True)
plt.show()

# Step 6: Evaluate Reconstruction Error
reconstruction = autoencoder.reconstruct(characters)
total_error = np.sum(np.abs(reconstruction - characters))
print(f"Total reconstruction error: {total_error}")

# Step 7: Visualize Original vs Reconstructed Characters
for i, char in enumerate(characters[:5]):  # Visualize first 5 characters
    original = char.reshape(7, 5)
    reconstructed = reconstruction[i].reshape(7, 5)
    
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='binary')
    plt.title(f"Original: {font3_labels[i]}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed, cmap='binary')
    plt.title(f"Reconstructed")
    plt.axis('off')
    plt.show()