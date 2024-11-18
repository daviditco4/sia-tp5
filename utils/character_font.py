import os
import sys

import numpy as np

# Add the path to the folder containing font3.py (with nesting)
sys.path.append(os.path.abspath('.'))

# Now you can import font3
from assets.fonts import font3


def binary_arrays_from_font3():
    # Assuming `font3` is defined elsewhere and is a list of 7x5 matrices
    num_chars = len(font3)
    binary_arrays = np.zeros((num_chars, 35), dtype=int)  # Preallocate with zeros
    for i, character in enumerate(font3):
        for row in range(7):
            processing_row = character[row]
            for col in range(5):
                binary_arrays[i, row * 5 + (4 - col)] = processing_row & 1
                processing_row >>= 1
    return binary_arrays
