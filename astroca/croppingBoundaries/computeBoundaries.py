"""
@file computeBoundaries.py
@brief This module provides functionality to compute cropping boundaries for 3D image sequences with time dimension.
@detail Due to the acquisition process, the data is supposed to have empty band(s) for each z-slice.
To speed up the calculations over the images, we spot those empty bands whose value is equal to default_value
(0.0 or 50.0) and compute the cropping boundaries for each z-slice. The cropping boundaries are computed as
the first and last non-empty band in each z-slice.
"""

import numpy as np
from astroca.tools.scene import ImageSequence3DPlusTime
from astroca.tools.exportData import export_data
import os
from tqdm import tqdm

def compute_boundaries(image_sequence: 'ImageSequence3DPlusTime',
                       pixel_cropped: int = 2,
                       save_results: bool = False,
                       output_directory: str = None) -> tuple:
    """
    Compute cropping boundaries in X for each Z slice based on background values.
    Sets boundary pixels to default_value and saves results optionally.

    @param image_sequence: ImageSequence3DPlusTime instance with 4D data (T, Z, Y, X)
    @param pixel_cropped: Number of pixels to trim inside each boundary
    @param save_results: Whether to save the updated sequence
    @param output_directory: Directory to save the bounded sequence if save_results is True
    @return: (index_xmin, index_xmax, default_value)
    """
    print(" - Computing cropping boundaries in X for each Z slice...")
    data = image_sequence.get_data()
    T, Z, Y, X = data.shape
    t = 0  # analyse du premier temps uniquement

    nb_y_tested = max(1, int(0.1 * Y))
    y_array = np.random.choice(Y, size=nb_y_tested, replace=False)

    default_value = float(data[t, 0, 0, X - 1])
    image_sequence.set_default_value(default_value)

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

    image_sequence.set_data(data)

    print(f"    index_xmin = {index_xmin}\n     index_xmax = {index_xmax}\n     default_value = {default_value}")

    if save_results:
        if output_directory is None:
            raise ValueError("output_directory must be specified if save_results is True.")
        os.makedirs(output_directory, exist_ok=True)
        export_data(data, output_directory, export_as_single_tif=True, file_name="bounded_image_sequence")

    print(60*"=")
    print()
    return index_xmin, index_xmax, default_value