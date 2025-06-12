"""
@file spaceMorphology.py
@brief This module provides functionality to fill/connect the structure in space, with a ball-like morphology of radius 1.
"""

from scipy.ndimage import median_filter, binary_closing, binary_dilation, binary_erosion
import numpy as np
from scipy.ndimage import generic_filter
from math import ceil
from skimage.morphology import ball
from skimage.util import view_as_windows
from numba import njit, prange, config


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
    print(f"Apply manual closing with (X,Y,Z)={radius} + 'edge' padding on all axes")

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

    return result



def apply_median_filter_3d_per_time(data: np.ndarray, size: float = 1.5) -> np.ndarray:
    """
    Apply a 3D median filter to each time frame (Z, Y, X).
    @param data: 4D numpy array of shape (T, Z, Y, X)
    @param size: isotropic kernel size. Will be rounded up to nearest odd int.
    @return: filtered 4D data.
    """

    # Convert float size to nearest odd integer ≥ size
    def to_odd_int(s):
        return int(ceil(s)) | 1  # make sure it's odd

    size_3d = (to_odd_int(size),) * 3  # (z, y, x)

    filtered = np.empty_like(data)
    for t in range(data.shape[0]):
        filtered[t] = median_filter(data[t], size=size_3d)

    print(f"Applied 3D median filter per frame with size={size_3d}")
    return filtered

def spherical_mask(radius: float) -> np.ndarray:
    """
    Create a 3D spherical structuring element with a given radius (can be float).
    """
    r = ceil(radius)
    L = np.arange(-r, r + 1)
    Z, Y, X = np.meshgrid(L, L, L, indexing='ij')
    sphere = (X**2 + Y**2 + Z**2) <= radius**2
    return sphere

def apply_median_filter_spherical(data: np.ndarray, radius: float = 1.5) -> np.ndarray:
    """
    Apply a median filter with a spherical structuring element to each time frame.
    @param data: 4D array (T, Z, Y, X)
    @param radius: Spherical radius (float), like in Filters3D.MEDIAN
    """
    struct = spherical_mask(radius)
    filtered = np.empty_like(data)
    for t in range(data.shape[0]):
        filtered[t] = generic_filter(data[t], np.median, footprint=struct, mode='mirror')
    print(f"Applied spherical median filter with radius={radius} (shape {struct.shape}, {struct.sum()} voxels)")
    return filtered


def fast_spherical_median_filter(frame: np.ndarray, radius: float) -> np.ndarray:
    """
    Apply a fast spherical median filter to a single 3D frame (Z, Y, X)
    @param frame: 3D array (Z, Y, X)
    @param radius: Spherical radius (float)
    @return: 2D array (Z, Y, X) with the median values within the spherical mask.
    """
    mask = spherical_mask(radius)
    patch_size = mask.shape
    pad_width = tuple((s // 2, s // 2) for s in patch_size)
    padded = np.pad(frame, pad_width, mode='reflect')

    # Extract sliding windows (patches)
    patches = view_as_windows(padded, patch_size)
    Z, Y, X, *_ = patches.shape

    # Reshape and mask
    patches = patches.reshape(Z, Y, X, -1)  # (Z, Y, X, N)
    masked = patches[..., mask.flatten()]  # keep only values in sphere

    # Compute median along last axis
    return np.median(masked, axis=-1).astype(frame.dtype)


def apply_median_filter_spherical_fast(data: np.ndarray, radius: float = 1.5) -> np.ndarray:
    """
    Vectorized fast spherical median filter over 4D (T, Z, Y, X)
    @param data: 4D array (T, Z, Y, X)
    @param radius: Spherical radius (float)
    @return: 4D array with the same shape as input data, where each frame is filtered.
    """
    filtered = np.empty_like(data)
    for t in range(data.shape[0]):
        filtered[t] = fast_spherical_median_filter(data[t], radius)
    print(f"Applied fast spherical median filter with radius={radius}")
    return filtered


def spherical_offsets(radius: float):
    """Get relative offsets of a spherical mask."""
    r = ceil(radius)
    offsets = []
    for dz in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx**2 + dy**2 + dz**2 <= radius**2:
                    offsets.append((dz, dy, dx))
    return np.array(offsets, dtype=np.int32)

@njit(parallel=True)
def median_filter_sphere_3d(frame, offsets):
    Z, Y, X = frame.shape
    output = np.empty_like(frame)
    for z in prange(Z):
        for y in range(Y):
            for x in range(X):
                values = []
                for k in range(offsets.shape[0]):
                    dz, dy, dx = offsets[k]
                    zz = min(max(z + dz, 0), Z - 1)
                    yy = min(max(y + dy, 0), Y - 1)
                    xx = min(max(x + dx, 0), X - 1)
                    values.append(frame[zz, yy, xx])
                output[z, y, x] = np.median(np.array(values))
    return output

def apply_median_filter_spherical_numba(data: np.ndarray, radius: float = 1.5) -> np.ndarray:
    """
    Very fast median filter with spherical mask using Numba over (T, Z, Y, X)
    """
    print(f"Apply Numba-accelerated spherical median filter with radius={radius}")
    offsets = spherical_offsets(radius)
    filtered = np.empty_like(data)
    for t in range(data.shape[0]):
        filtered[t] = median_filter_sphere_3d(data[t], offsets)
    return filtered

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
