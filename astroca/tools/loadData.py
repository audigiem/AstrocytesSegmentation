"""
@file loadData.py
@brief This module provides functionality to load and manage 3D image sequences with time dimension.
"""

import numpy as np
import os
import tifffile as tif
import glob
import configparser
from typing import List, Dict, Tuple, Any
import cupy as cp




def load_data(file_path: str, GPU_AVAILABLE: bool = False) -> np.ndarray:
    """
    Load 3D image sequence data from a .tif file or a directory containing .tif files.

    @param file_path: Path to the .tif file or directory containing .tif files.
    @param GPU_AVAILABLE: Boolean indicating whether to use GPU for processing. If True, data will be loaded as a cupy array.
    @raises ValueError: If the input path is invalid or if no .tif files are found in the directory.
    @return: 4D numpy array of shape (T, Z, Y, X) representing the image sequence.
    """

    if os.path.isdir(file_path):
        # Load all .tif files in the directory
        file_list = sorted(glob.glob(os.path.join(file_path, '*.tif')))
        if not file_list:
            raise ValueError(f"No .tif files found in directory: {file_path}")
        data = [tif.imread(f) for f in file_list]
    elif os.path.isfile(file_path) and file_path.endswith('.tif'):
        # Load single .tif file
        data = tif.imread(file_path)
        # prevent (1, T, Z, Y, X) shape
        if data.ndim == 5 and data.shape[0] == 1:
            data = np.squeeze(data, axis=0)
        if len(data.shape) != 4 and len(data.shape) != 3:
            raise ValueError(f"Loaded data must be a 4D array (T, Z, Y, X) or 3D array (Z, Y, X), but got shape {data.shape}.")

    else:
        raise ValueError(f"Invalid file path: {file_path}. Must be a .tif file or a directory containing .tif files.")
    if GPU_AVAILABLE:
        # Convert to cupy array for GPU processing
        print(f"Reading data on GPU...")
        data = cp.asarray(data)
    else:
        # Ensure data is a numpy array for CPU processing
        print(f"Reading data on CPU...")
        data = np.asarray(data)
    
    
def read_config(config_file: str | None = None) -> dict:
    """
    Read configuration parameters from a .ini file.

    @param config_file: Path to the configuration file. If None, default is relative to this file.
    @return: Dictionary containing configuration parameters.
    """
    if config_file is None:
        # Default path to the configuration file
        base_dir = os.path.dirname(os.path.abspath(__file__)) 
        config_file = os.path.join(base_dir, "..", "..", "config.ini")
        config_file = os.path.normpath(config_file)

    config = configparser.ConfigParser()
    config.read(config_file)

    if not config.sections():
        raise ValueError(f"No sections found in configuration file: {config_file}")

    return {section: dict(config.items(section)) for section in config.sections()}

