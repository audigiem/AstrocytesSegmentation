"""
@file spaceMorphology.py
@brief This module provides functionality to fill/connect the structure in space, with a ball-like morphology of radius 1.
"""

import numpy as np
from scipy.ndimage import median_filter, binary_dilation
from skimage.morphology import ball




def fill_space_morphology(data: np.ndarray, radius: int = 1) -> np.ndarray:
    """
    @brief Fill/connect the structure in space with a ball-like morphology of radius 1.

    @param data: 4D numpy array of shape (T, Z, Y, X) representing the image sequence.
    @param radius: Radius of the ball-like morphology to use for filling.
    @return: 4D numpy array with the filled structure.
    @raises ValueError: If the radius is less than 1 or if the input data is not a 4D numpy array.
    """
    if radius < 1:
        raise ValueError("Radius must be at least 1.")
    if data.ndim != 4:
        raise ValueError("Input must be 4D (T, Z, Y, X).")

    # for each voxel, if it is non-zero, apply binary dilation with a ball-like structure
    data_binary = (data == 255)  # Assuming active voxels are marked with 255
    struct = ball(radius)
    filled_data = np.array([binary_dilation(frame, structure=struct) for frame in data_binary])
    filled_data = filled_data.astype(data.dtype) * 255  # Return in original data type
    print(f"Filled space morphology with radius: {radius}, resulting shape: {filled_data.shape}")
    return filled_data

def apply_median_filter(data: np.ndarray, size: int = 2) -> np.ndarray:
    """
    @brief Apply a median filter to the 3D image sequence to smooth the data.

    @param data: 4D numpy array of shape (T, Z, Y, X) representing the image sequence.
    @param size: Size of the median filter to apply.
    @return: 4D numpy array with the median filter applied.
    """
    if size < 1:
        raise ValueError("Size must be at least 1.")

    # Apply median filter to each time frame
    filtered_data = np.array([median_filter(frame, size=size) for frame in data])

    print(f"Applied median filter with size: {size}, resulting shape: {filtered_data.shape}")

    return filtered_data.astype(data.dtype)  # Return in original data type

def main():
    # Example usage
    data = np.zeros((5, 5,5,5))  # Create a dummy 4D array (T, Z, Y, X)

    # set some random voxels to 255
    data[0, 2, 2, 2] = 255
    data[1, 2, 2, 3] = 255
    data[2, 3, 3, 3] = 255
    data[3, 1, 1, 1] = 255
    data[4, 0, 0, 0] = 255

    # show original data
    print("Original Data Shape:", data.shape)
    for i in range(data.shape[0]):
        print(f"Frame {i}:\n{data[i]}")
    filled_data = fill_space_morphology(data, radius=1)

    print("Filled Data Shape:", filled_data.shape)
    # show each frame
    for i in range(filled_data.shape[0]):
        print(f"Frame {i}:\n{filled_data[i]}")


if __name__ == "__main__":
    main()
