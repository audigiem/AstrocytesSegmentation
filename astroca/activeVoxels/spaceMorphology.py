"""
@file spaceMorphology.py
@brief This module provides functionality to fill/connect the structure in space, with a ball-like morphology of radius 1.
"""

import numpy as np
from scipy.ndimage import median_filter, binary_closing, binary_dilation, binary_erosion
from skimage.morphology import ball


def create_ball_structuring_element(radius_z, radius_y, radius_x):
    """
    Simulate Strel3D.Shape.BALL.fromRadiusList in Java using a Manhattan ellipsoid.
    """
    zz, yy, xx = np.ogrid[
        -radius_z:radius_z + 1,
        -radius_y:radius_y + 1,
        -radius_x:radius_x + 1
    ]
    mask = (np.abs(zz) / radius_z + np.abs(yy) / radius_y + np.abs(xx) / radius_x) <= 1
    return mask

def fill_space_morphology(data: np.ndarray, radius: tuple) -> np.ndarray:
    """
    Apply 3D closing morphology with a discrete ball (octahedron-like) shape on each time frame,
    mimicking ImageJ behavior more closely at Z-boundaries using edge-padding.
    @param data: 4D numpy array of shape (T, Z, Y, X) representing the image sequence (binary mask).
    @param radius: Tuple specifying the radius for the ball structuring element (radius_z, radius_y, radius_x).
    @return : 4D numpy array with the same shape as input data, where the structure is filled.
    """
    if data.ndim != 4:
        raise ValueError("Expected 4D input array (T, Z, Y, X)")

    radius_x, radius_y, radius_z = radius
    struct_elem = create_ball_structuring_element(radius_z, radius_y, radius_x)

    pad_width = ((radius_z, radius_z), (radius_y, radius_y), (radius_x, radius_x))

    result = np.zeros_like(data, dtype=np.uint8)
    binary_input = (data == 255)

    for t in range(data.shape[0]):
        frame = binary_input[t]

        # Pad Z, Y, X with 'edge' to replicate border values
        padded = np.pad(frame, pad_width, mode='edge')

        # Morphological closing (dilation followed by erosion)
        closed = binary_erosion(binary_dilation(padded, structure=struct_elem), structure=struct_elem)

        # Crop to original shape
        closed = closed[radius_z:-radius_z, radius_y:-radius_y, radius_x:-radius_x]

        result[t] = closed.astype(np.uint8) * 255

    print(f"Applied manual closing with (X,Y,Z)=({radius_x},{radius_y},{radius_z}) + 'edge' padding on all axes")
    return result

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
    filled_data = fill_space_morphology(data, radius=(1,1,1))

    print("Filled Data Shape:", filled_data.shape)
    # show each frame
    for i in range(filled_data.shape[0]):
        print(f"Frame {i}:\n{filled_data[i]}")


if __name__ == "__main__":
    main()
