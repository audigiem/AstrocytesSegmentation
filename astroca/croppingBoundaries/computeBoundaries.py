"""
@file computeBoundaries.py
@brief This module provides functionality to compute cropping boundaries for 3D image sequences with time dimension.
@detail Due to the acquisition process, the data is supposed to have empty band(s) for each z-slice.
To speed up the calculations over the images, we spot those empty bands whose value is equal to default_value
(0.0 or 50.0) and compute the cropping boundaries for each z-slice. The cropping boundaries are computed as
the first and last non-empty band in each z-slice.
"""

import numpy as np
from astroca.tools.exportData import export_data, save_numpy_tab 
import os
from tqdm import tqdm
from typing import List, Dict, Tuple, Any
import cupy as cp
from numba import cuda



def compute_boundaries_CPU(data: np.ndarray, params: dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Compute cropping boundaries in X for each Z slice based on background values.
    Sets boundary pixels to default_value and saves results optionally.

    @param data: 4D data (T, Z, Y, X)
    @param params: Dictionary containing the parameters:
        - pixel_cropped: Number of pixels to crop from the height dimension.
        - save_results: Boolean indicating whether to save the results.
        - output_directory: Directory to save the results if save_results is True.
    @return: (index_xmin, index_xmax, default_value)
    """
    print(" - Computing cropping boundaries in X for each Z slice (CPU version)...")
    
    # extract necessary parameters
    required_keys = {'preprocessing', 'files', 'paths'}
    if not required_keys.issubset(params.keys()):
        raise ValueError(f"Missing required parameters: {required_keys - params.keys()}")
    pixel_cropped = int(params['preprocessing']['pixel_cropped'])
    save_results = int(params['files']['save_results']) == 1
    output_directory = params['paths']['output_dir']
    
    T, Z, Y, X = data.shape
    t = 0  # analyse du premier temps uniquement

    nb_y_tested = max(1, int(0.1 * Y))
    y_array = np.random.choice(Y, size=nb_y_tested, replace=False)

    default_value = float(data[t, 0, 0, X - 1])

    index_xmin = np.full(Z, -1, dtype=int)
    index_xmax = np.full(Z, X - 1, dtype=int)

    for z in tqdm(range(Z), desc="Computing X bounds per Z-slice", unit="slice"):
        found = False
        for x in range(X):
            values = data[t, z, y_array, x]
            if not np.all(values == default_value):
                if not found:
                    index_xmin[z] = x
                    found = True
            elif found:
                index_xmax[z] = x - 1
                break

    for t in tqdm(range(T), desc="Cropping borders", unit="frame"):
        for z in range(Z):
            x_start = index_xmin[z]
            x_end = index_xmax[z]
            if x_start < 0 or x_end <= x_start:
                continue  # skip invalid bounds

            crop_start = x_start
            crop_end = x_end + 1

            data[t, z, :, crop_start:crop_start + pixel_cropped] = default_value
            data[t, z, :, crop_end - pixel_cropped:crop_end] = default_value


    index_xmin += pixel_cropped
    index_xmax -= pixel_cropped

    print(f"    index_xmin = {index_xmin}\n     index_xmax = {index_xmax}\n     default_value = {default_value}")

    if save_results:
        if output_directory is None:
            raise ValueError("output_directory must be specified if save_results is True.")
        os.makedirs(output_directory, exist_ok=True)
        export_data(data, output_directory, export_as_single_tif=True, file_name="bounded_image_sequence")
        save_numpy_tab(index_xmin, output_directory, file_name="index_Xmin.npy")
        save_numpy_tab(index_xmax, output_directory, file_name="index_Xmax.npy")
        
    print(60*"=")
    print()
    return index_xmin, index_xmax, default_value, data


@cuda.jit
def _find_bounds(frame, default_val, xmin, xmax, pixel_cropped):
    """Kernel par Z pour détecter xmin / xmax."""
    z = cuda.grid(1)
    Z, Y, X = frame.shape
    if z >= Z:
        return
    found = False
    for x in range(X):
        # échantillonner 10% des Y (heuristique simplifiée)
        step = max(1, Y // 10)
        all_default = True
        y = 0
        while y < Y:
            if frame[z, y, x] != default_val:
                all_default = False
                break
            y += step
        if not all_default and not found:
            xmin[z] = x
            found = True
        elif all_default and found:
            xmax[z] = x - 1
            break
    # post‑traitement (padding)
    xmin[z] += pixel_cropped
    xmax[z] -= pixel_cropped

def compute_boundaries_GPU(data: cp.ndarray, params: dict) -> Tuple[cp.ndarray, cp.ndarray, float, cp.ndarray]:
    """
    Compute cropping boundaries in X for each Z slice based on background values using GPU.

    @param data: 4D data (T, Z, Y, X) as a cupy array
    @param params: Dictionary containing the parameters:
        - pixel_cropped: Number of pixels to crop from the height dimension.
        - save_results: Boolean indicating whether to save the results.
        - output_directory: Directory to save the results if save_results is True.
    @return: (index_xmin, index_xmax, default_value, data)
    """
    print(" - Computing cropping boundaries in X for each Z slice (GPU)...")
    
    # extract necessary parameters
    required_keys = {'preprocessing', 'files', 'paths'}
    if not required_keys.issubset(params.keys()):
        raise ValueError(f"Missing required parameters: {required_keys - params.keys()}")
    pixel_cropped = int(params['preprocessing']['pixel_cropped'])
    save_results  = int(params['files']['save_results']) == 1
    out_dir       = params['paths']['output_dir']

    T, Z, Y, X = data.shape
    default_val = float(data[0, 0, 0, X-1])

    xmin = cp.full(Z, -1, dtype=cp.int32)
    xmax = cp.full(Z,  X-1, dtype=cp.int32)
    threads, blocks = 128, (Z + 127)//128
    _find_bounds[blocks, threads](data[0], default_val, xmin, xmax, pixel_cropped)
    cuda.synchronize()

    arange_x = cp.arange(X)                          # shape (X,)
    # Broadcasting : (Z,1)  vs  (1,X)  → (Z,X)
    mask_left  = arange_x[None, :] <  (xmin + pixel_cropped)[:, None]
    mask_right = arange_x[None, :] >= (xmax - pixel_cropped + 1)[:, None]
    mask_zx    = cp.logical_or(mask_left, mask_right)           # shape (Z,X)

    # Diffusion paresseuse : (1,Z,1,X) se propage sur T et Y
    mask_4d = mask_zx[None, :, None, :]              # shape (1,Z,1,X)
    data[mask_4d] = default_val                      # remplissage GPU

    if save_results:
        os.makedirs(out_dir, exist_ok=True)
        export_data(data, out_dir, export_as_single_tif=True,
                    file_name="bounded_image_sequence")
        save_numpy_tab(xmin, out_dir, file_name="index_Xmin.npy")
        save_numpy_tab(xmax, out_dir, file_name="index_Xmax.npy")

    return xmin, xmax, default_val, data


def compute_boundaries(data: np.ndarray, params: dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Compute cropping boundaries in X for each Z slice based on background values.
    Sets boundary pixels to default_value and saves results optionally.

    @param data: 4D data (T, Z, Y, X)
    @param params: Dictionary containing the parameters:
        - pixel_cropped: Number of pixels to crop from the height dimension.
        - save_results: Boolean indicating whether to save the results.
        - output_directory: Directory to save the results if save_results is True.
    @return: (index_xmin, index_xmax, default_value, data)
    """
    required_keys = {'GPU_AVAILABLE'}
    if not required_keys.issubset(params.keys()):
        raise ValueError(f"Missing required parameters: {required_keys - params.keys()}")
    _GPU_AVAILABLE = int(params['GPU_AVAILABLE']) == 1  # Convert to boolean
    if _GPU_AVAILABLE:
        return compute_boundaries_GPU(data, params)
    else:
        return compute_boundaries_CPU(data, params)