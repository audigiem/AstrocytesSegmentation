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


def compute_boundaries(data: np.ndarray, params: dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
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
    print(" - Computing cropping boundaries in X for each Z slice...")
    
    # extract necessary parameters
    required_keys = {'preprocessing', 'save', 'paths'}
    if not required_keys.issubset(params.keys()):
        raise ValueError(f"Missing required parameters: {required_keys - params.keys()}")
    pixel_cropped = int(params['preprocessing']['pixel_cropped'])
    save_results = int(params['save']['save_boundaries']) == 1
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
        export_data(data, output_directory, export_as_single_tif=True, file_name="data")
        save_numpy_tab(index_xmin, output_directory, file_name="index_Xmin.npy")
        save_numpy_tab(index_xmax, output_directory, file_name="index_Xmax.npy")
        
    print(60*"=")
    print()
    return index_xmin, index_xmax, default_value, data