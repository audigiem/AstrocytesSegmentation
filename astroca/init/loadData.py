"""
@file loadData.py
@brief This module provides functionality to load and manage 3D image sequences with time dimension.
"""

import numpy as np
import os
import tifffile as tif
import glob

def load_data_from_file(dir_path: str) -> np.ndarray:
    """
    @brief Load 3D image sequence data from a file, we assume the directory contains .tif files

    @param dir_path: Path to the directory containing the image sequence data.
    @return: 4D numpy array with shape (timeLength, depth, height, width).
    @raises ValueError: If no .tif files are found in the specified directory.
    """
    list_of_files = sorted(glob.glob(os.path.join(dir_path, "*.tif")))

    if not list_of_files:
        raise ValueError("No .tif files found in the specified directory.")

    time_length = len(list_of_files)
    data = np.zeros((time_length, 1, 1, 1), dtype=np.float32)  # Placeholder for the actual shape
    for i, file_path in enumerate(list_of_files):
        img = tif.imread(file_path)
        if i == 0:
            data = np.zeros((time_length, img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)
        data[i] = img

    return data
