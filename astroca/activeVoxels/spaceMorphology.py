"""
@file spaceMorphology.py
@brief This module provides functionality to fill/connect the structure in space, with a ball-like morphology of radius 1.
"""
from scipy.ndimage import median_filter, binary_closing, generate_binary_structure
import numpy as np
from scipy.ndimage import generic_filter
from math import ceil
from skimage.morphology import ball
from skimage.util import view_as_windows
from numba import njit, prange, config
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def closing_morphology_in_space(data: np.ndarray, radius: int, border_mode: str='reflect') -> np.ndarray:
    """
    Apply 3D morphological closing (dilation followed by erosion) with a spherical structuring element
    to each time frame of a 4D sequence (T, Z, Y, X).
    @param data: 4D numpy array (T, Z, Y, X), binary (0/255 or 0/1)
    @param radius: Radius of the spherical structuring element
    @param border_mode: Padding mode for borders, e.g., 'reflect', 'edge', 'constant', etc.
    @return: 4D numpy array (T, Z, Y, X) after closing
    """
    print(f" - Apply morphological closing with radius={radius} and border mode='{border_mode}'")
    if border_mode not in ['reflect', 'edge']:
        raise ValueError("Unsupported border mode. Use 'reflect' or 'edge'.")
    # Spherical structuring element
    struct_elem = ball(radius)
    pad_width = ((radius, radius), (radius, radius), (radius, radius))  # (Z, Y, X)

    # Ensure binary input
    binary = (data > 0)
    result = np.empty_like(binary, dtype=np.uint8)
    if border_mode == 'edge':
        for t in tqdm(range(binary.shape[0]), desc="Morphological closing over time", unit="frame"):
            padded = np.pad(binary[t], pad_width=pad_width, mode='edge')
            # Apply 3D closing to each time frame
            closed = binary_closing(padded, structure=struct_elem)
            result[t] = closed.astype(np.uint8) * 255

        return result
    else:
        for t in tqdm(range(binary.shape[0]), desc="Morphological closing over time", unit="frame"):
            padded = np.pad(binary[t], pad_width=pad_width, mode='reflect')
            # Apply 3D closing to each time frame
            closed = binary_closing(padded, structure=struct_elem)
            # Crop to original shape
            closed = closed[radius:-radius, radius:-radius, radius:-radius]
            result[t] = closed.astype(np.uint8) * 255

        return result


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
    # mask = (zz / radius_z) ** 2 + (yy / radius_y) ** 2 + (xx / radius_x) ** 2 <= 1.0

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

    T, Z, Y, X = data.shape
    result = np.zeros_like(data, dtype=np.uint8)

    for t in range(T):
        binary_frame = (data[t] > 0)

        # Pad Z, Y, X with 'edge' to replicate border values
        padded = np.pad(binary_frame, pad_width, mode='reflect')

        # Morphological closing (dilation followed by erosion)
        # closed = binary_erosion(binary_dilation(padded, structure=struct_elem), structure=struct_elem)
        closed = binary_closing(padded, structure=struct_elem)
        # Crop to original shape
        closed = closed[radius_z:-radius_z, radius_y:-radius_y, radius_x:-radius_x]

        result[t] = closed.astype(np.uint8) * 255

    return result



def median_3d_for_4d_stack(fourd_stack, radius=2, n_workers=None):
    """
    Applique un filtre médian 3D à chaque frame 3D d'un stack 4D.

    Args:
        fourd_stack (numpy.ndarray): Stack 4D (T, Z, Y, X)
        radius (float): Rayon du voisinage (1.5 donne un voisinage 3×3×3)
        n_workers (int): Nombre de workers pour le traitement parallèle

    Returns:
        numpy.ndarray: Stack 4D filtré
    """
    # round up radius to nearest bigger odd integer
    footprint = ball(ceil(radius))
    n_frames = fourd_stack.shape[0]
    filtered_stack = np.empty_like(fourd_stack)

    def process_frame(t):
        filtered_stack[t] = median_filter(
            fourd_stack[t],
            footprint=footprint,
            mode='reflect'
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        list(tqdm(executor.map(process_frame, range(n_frames)), total=n_frames, desc="Processing frames", unit="frame"))

    return filtered_stack


def median_filter_3d(data: np.ndarray, size_radius: float = 1.5, border_mode: str='reflect') -> np.ndarray:
    """
    Apply a 3D median filter to each time frame (Z, Y, X).
    @param data: 4D numpy array of shape (T, Z, Y, X)
    @param size: Size of the median filter kernel (should be odd)
    @param border_mode: Padding mode for borders, e.g., 'reflect', 'edge', etc.
    @return: Filtered 4D data.
    """

    if border_mode not in ['reflect', 'edge']:
        raise ValueError("Unsupported border mode. Use 'reflect' or 'edge'.")

    struct = ball(size_radius)
    int_size_radius = ceil(size_radius)
    if int_size_radius % 2 == 0:
        int_size_radius += 1
    pad_width = ((int_size_radius, int_size_radius), (int_size_radius, int_size_radius), (int_size_radius, int_size_radius))

    filtered = np.empty_like(data)
    if border_mode == 'edge':
        for t in range(data.shape[0]):
            padded = np.pad(data[t], pad_width=pad_width, mode='edge')
            filtered_temps = median_filter(padded, footprint=struct, mode='edge')
            filtered_temps = filtered_temps[int_size_radius:-int_size_radius, int_size_radius:-int_size_radius, int_size_radius:-int_size_radius]
            filtered[t] = filtered_temps

        return filtered
    else:
        for t in range(data.shape[0]):
            padded = np.pad(data[t], pad_width=pad_width, mode='reflect')
            filtered_temps = median_filter(padded, footprint=struct, mode='reflect')
            filtered_temps = filtered_temps[int_size_radius:-int_size_radius, int_size_radius:-int_size_radius, int_size_radius:-int_size_radius]
            filtered[t] = filtered_temps

        return filtered




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
        filtered[t] = median_filter(data[t], size=size_3d, mode='mirror')

    print(f"Applied 3D median filter per frame with size={size_3d}")
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


@njit
def compute_median(values):
    n = len(values)
    sorted_vals = np.sort(values)
    if n % 2 == 0:
        return 0.5 * (sorted_vals[n // 2 - 1] + sorted_vals[n // 2])
    else:
        return sorted_vals[n // 2]

@njit(parallel=True)
def median_filter_sphere_3d(frame, offsets, border_condition='reflect'):
    Z, Y, X = frame.shape
    output = np.empty_like(frame)
    if border_condition not in ['nearest', 'reflect']:
        raise ValueError("Unsupported border condition. Use 'nearest', 'reflect'.")
    if border_condition == 'nearest':
        # Nearest neighbor padding
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
    else:
        # Reflective padding
        for z in prange(Z):
            for y in range(Y):
                for x in range(X):
                    values = []
                    for k in range(offsets.shape[0]):
                        dz, dy, dx = offsets[k]
                        zz = z + dz
                        yy = y + dy
                        xx = x + dx
                        if zz < 0:
                            zz = -zz
                        elif zz >= Z:
                            zz = 2 * Z - zz - 2
                        if yy < 0:
                            yy = -yy
                        elif yy >= Y:
                            yy = 2 * Y - yy - 2
                        if xx < 0:
                            xx = -xx
                        elif xx >= X:
                            xx = 2 * X - xx - 2
                        values.append(frame[zz, yy, xx])
                    output[z, y, x] = compute_median(np.array(values))
        return output

def apply_median_filter_spherical_numba(data: np.ndarray, radius: float = 1.5, border_condition: str = 'reflect') -> np.ndarray:
    """
    Very fast median filter with spherical mask using Numba over (T, Z, Y, X)
    @param data: 4D array (T, Z, Y, X)
    @param radius: Spherical radius (float)
    @param border_condition: 'nearest' or 'reflect' for handling borders
    """
    print(f"Apply Numba-accelerated spherical median filter with radius={radius}, and border condition='{border_condition}'")
    offsets = spherical_offsets(radius)
    filtered = np.empty_like(data)
    for t in range(data.shape[0]):
        try:
            filtered[t] = median_filter_sphere_3d(data[t], offsets, border_condition)
        except Exception as e:
            print(f"Error processing frame {t}: {e}")
            filtered[t] = data[t]

    return filtered




def unified_median_filter_3d(
        data: np.ndarray,
        radius: float = 1.5,
        border_mode: str = 'reflect',
        n_workers: int = None
) -> np.ndarray:
    """
    Median filter 3D unifié pour stacks 4D (T,Z,Y,X)

    Args:
        data: Input stack (T,Z,Y,X)
        radius: Rayon de la sphère (1.5 → voisinage 3×3×7)
        border_mode: 'reflect', 'nearest', 'constant', etc.
        n_workers: Nombre de threads
    """
    print(f" - Apply 3D median filter with radius={radius}, border mode='{border_mode}'")
    r = int(np.ceil(radius))

    # Créer le masque sphérique
    shape = (2 * r + 1, 2 * r + 1, 2 * r + 1)
    mask = np.zeros(shape, dtype=bool)
    center = np.array([r, r, r])
    for idx in np.ndindex(shape):
        if np.linalg.norm(np.array(idx) - center) <= radius:
            mask[idx] = True

    # Padding manuel : seulement sur les axes Z, Y, X
    pad_width = [(0, 0), (r, r), (r, r), (r, r)]
    padded = np.pad(data, pad_width=pad_width, mode=border_mode)

    filtered = np.empty_like(data)

    def process_frame(t):
        result = median_filter(
            padded[t], footprint=mask, mode='constant', cval=0.0  # on ignore mode ici
        )
        # Enlever le padding
        filtered[t] = result[r:-r, r:-r, r:-r]

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        list(tqdm(
            executor.map(process_frame, range(data.shape[0])),
            total=data.shape[0], desc="Processing frames with median filter", unit="frame"
        ))

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
