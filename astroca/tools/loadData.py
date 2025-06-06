"""
@file loadData.py
@brief This module provides functionality to load and manage 3D image sequences with time dimension.
"""

import numpy as np
import os
import tifffile as tif
import glob

def load_data(file_path: str) -> np.ndarray:
    """
    Load 3D image sequence data from a .tif file or a directory containing .tif files.

    @param file_path: Path to the .tif file or directory containing .tif files.
    @return: 4D numpy array of shape (T, Z, Y, X) representing the image sequence.
    """
    if os.path.isdir(file_path):
        # Load all .tif files in the directory
        file_list = sorted(glob.glob(os.path.join(file_path, '*.tif')))
        if not file_list:
            raise ValueError(f"No .tif files found in directory: {file_path}")
        data = [tif.imread(f) for f in file_list]
        return np.array(data)
    elif os.path.isfile(file_path) and file_path.endswith('.tif'):
        # Load single .tif file
        return tif.imread(file_path)
    else:
        raise ValueError(f"Invalid file path: {file_path}. Must be a .tif file or a directory containing .tif files.")
