"""
@file loadData_torch.py
@brief This module provides functionality to load and manage 3D image sequences with time dimension using PyTorch.
"""

import os
import glob
import tifffile as tif
import configparser
from typing import Dict
import numpy as np
import torch


def load_data(file_path: str, GPU_AVAILABLE: bool = False) -> torch.Tensor | np.ndarray:
    """
    Load 3D image sequence data from a .tif file or a directory containing .tif files.

    @param file_path: Path to the .tif file or directory containing .tif files.
    @param GPU_AVAILABLE: If True, the data is moved to GPU as a PyTorch tensor.
    @raises ValueError: If the input path is invalid or if no .tif files are found in the directory.
    @return: 4D array (T, Z, Y, X) either as torch.Tensor or numpy.ndarray depending on GPU_AVAILABLE.
    """

    if os.path.isdir(file_path):
        # Load all .tif files in the directory
        file_list = sorted(glob.glob(os.path.join(file_path, '*.tif')))
        if not file_list:
            raise ValueError(f"No .tif files found in directory: {file_path}")
        data = [tif.imread(f) for f in file_list]
        data = np.stack(data)
    elif os.path.isfile(file_path) and file_path.endswith('.tif'):
        data = tif.imread(file_path)
        # prevent (1, T, Z, Y, X) shape
        if data.ndim == 5 and data.shape[0] == 1:
            data = np.squeeze(data, axis=0)
        if len(data.shape) not in (3, 4):
            raise ValueError(f"Loaded data must be 3D (Z,Y,X) or 4D (T,Z,Y,X), got shape {data.shape}")
    else:
        raise ValueError(f"Invalid file path: {file_path}. Must be a .tif file or a directory containing .tif files.")

    if GPU_AVAILABLE:
        # print("Reading data on GPU...")
        tensor = torch.tensor(data, dtype=torch.float32, device='cuda')
        return tensor
    else:
        # print("Reading data on CPU...")
        return np.asarray(data)


def read_config(config_file: str | None = None) -> Dict[str, Dict[str, str]]:
    """
    Read configuration parameters from a .ini file.

    @param config_file: Path to the configuration file. If None, default path is used.
    @return: Dictionary containing configuration parameters.
    """
    if config_file is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.normpath(os.path.join(base_dir, "..", "..", "config.ini"))

    config = configparser.ConfigParser()
    config.read(config_file)

    if not config.sections():
        raise ValueError(f"No sections found in configuration file: {config_file}")

    return {section: dict(config.items(section)) for section in config.sections()}
