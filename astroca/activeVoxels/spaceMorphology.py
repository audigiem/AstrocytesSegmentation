"""
@file spaceMorphology.py
@brief This module provides functionality to fill/connect the structure in space, with a ball-like morphology of radius 1.
"""
from scipy.ndimage import binary_closing
import numpy as np
from skimage.morphology import ball
from numba import njit, prange, config
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from astroca.activeVoxels.medianFilter import generate_spherical_offsets

def closing_morphology_in_space(data: np.ndarray, radius: int, border_mode: str = 'constant') -> np.ndarray:
    """
    Apply 3D morphological closing with a spherical structuring element
    to each time frame of a 4D sequence (T, Z, Y, X).

    @param data: 4D numpy array (T, Z, Y, X), binary (0/255 or 0/1)
    @param radius: Radius of the spherical structuring element
    @param border_mode: 'constant', 'reflect', etc. OR 'ignore' to skip voxels near borders
    @return: 4D numpy array (T, Z, Y, X) after closing
    """

    if border_mode == 'ignore':
        return closing_morphology_in_space_ignore_border(data, radius)

    print(f" - Apply morphological closing with radius={radius} and border mode='{border_mode}'")
    struct_elem = ball(radius)
    print(f"Structuring element shape: {struct_elem.shape}, active pixels: {struct_elem.sum()}")
    result = np.zeros_like(data, dtype=np.uint8)

    # Ensure binary input
    binary = (data > 0)

    for t in tqdm(range(binary.shape[0]), desc=f"Morphological closing", unit="frame"):
        pad_width = ((radius, radius), (radius, radius), (radius, radius))
        padded = np.pad(binary[t], pad_width=pad_width, mode=border_mode)
        closed = binary_closing(padded, structure=struct_elem)
        closed = closed[radius:-radius, radius:-radius, radius:-radius]
        result[t] = closed.astype(np.uint8) * 255

    return result

@njit(parallel=True)
def morphological_erosion_ignore_border(frame: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """
    Erosion morphologique : prendre le minimum dans le voisinage sphérique
    Ne considère que les voxels valides (dans le volume)
    """
    Z, Y, X = frame.shape
    eroded = np.zeros((Z, Y, X), dtype=np.uint8)

    for z in prange(Z):
        for y in range(Y):
            for x in range(X):
                min_val = 1  # Valeur max possible
                
                for i in range(offsets.shape[0]):
                    dz, dy, dx = offsets[i]
                    zz, yy, xx = z + dz, y + dy, x + dx
                    
                    if 0 <= zz < Z and 0 <= yy < Y and 0 <= xx < X:
                        if frame[zz, yy, xx] < min_val:
                            min_val = frame[zz, yy, xx]
                    # Les voxels hors du volume sont ignorés
                
                eroded[z, y, x] = min_val
    return eroded

@njit(parallel=True)
def morphological_dilation_ignore_border(frame: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """
    Dilatation morphologique : prendre le maximum dans le voisinage sphérique
    Ne considère que les voxels valides (dans le volume)
    """
    Z, Y, X = frame.shape
    dilated = np.zeros((Z, Y, X), dtype=np.uint8)

    for z in prange(Z):
        for y in range(Y):
            for x in range(X):
                max_val = 0  # Valeur min possible
                
                for i in range(offsets.shape[0]):
                    dz, dy, dx = offsets[i]
                    zz, yy, xx = z + dz, y + dy, x + dx
                    
                    if 0 <= zz < Z and 0 <= yy < Y and 0 <= xx < X:
                        if frame[zz, yy, xx] > max_val:
                            max_val = frame[zz, yy, xx]
                    # Les voxels hors du volume sont ignorés
                
                dilated[z, y, x] = max_val
    return dilated


def closing_morphology_in_space_ignore_border(
    data: np.ndarray,
    radius: int,
    n_workers: int = None
) -> np.ndarray:
    """
    Fast 3D morphological closing on a 4D binary array (T,Z,Y,X), with border_mode='ignore'.
    """
    print(f" - Fast morphological closing with radius={radius} and border mode='ignore'")

    binary = (data > 0).astype(np.uint8)
    T, Z, Y, X = binary.shape
    
    result = np.empty((T, Z, Y, X), dtype=np.uint8)
    offsets = generate_spherical_offsets(radius)

    def process(t):
        frame = binary[t]

        closed = morphological_dilation_ignore_border(frame, offsets)

        eroded = morphological_erosion_ignore_border(closed, offsets)

        result[t] = eroded.astype(np.uint8) * 255  # Convert to binary (0/255)

    # Traiter les autres frames
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        list(tqdm(
            executor.map(process, range(T)),
            total=T-1,
            desc="Morphological closing (ignore border)",
            unit="frame"
        ))

    return result

