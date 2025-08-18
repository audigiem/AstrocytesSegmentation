"""
@file loadData.py
@brief This module provides functionality to load and manage 3D image sequences with time dimension.
"""

import os
import tifffile as tif
import glob
import configparser
from typing import Dict
import numpy as np
import torch

@profile
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
        file_list = sorted(glob.glob(os.path.join(file_path, "*.tif")))
        if not file_list:
            raise ValueError(f"No .tif files found in directory: {file_path}")
        data = [tif.imread(f) for f in file_list]
        if GPU_AVAILABLE:
            data = [torch.tensor(d, dtype=torch.float32) for d in data]
            data = torch.stack(data, dim=0)
        else:
            data = np.array(data, dtype=np.float32)
        return data

    elif os.path.isfile(file_path) and file_path.endswith(".tif"):
        # Load single .tif file
        data = tif.imread(file_path)
        # prevent (1, T, Z, Y, X) shape
        if data.ndim == 5 and data.shape[0] == 1:
            data = np.squeeze(data, axis=0)
        if len(data.shape) != 4 and len(data.shape) != 3:
            raise ValueError(
                f"Loaded data must be a 4D array (T, Z, Y, X) or 3D array (Z, Y, X), but got shape {data.shape}."
            )
        if GPU_AVAILABLE:
            data = torch.tensor(data, dtype=torch.float32)
            if data.ndim == 3:
                # Add time dimension if missing
                data = data.unsqueeze(0)

        else:
            data = np.array(data, dtype=np.float32)
            if data.ndim == 3:
                # Add time dimension if missing
                data = np.expand_dims(data, axis=0)
        return data

    else:
        raise ValueError(
            f"Invalid file path: {file_path}. Must be a .tif file or a directory containing .tif files."
        )


def read_config(config_file: str | None = None) -> Dict[str, Dict[str, str]]:
    """
    Read configuration parameters from a .ini file.

    @param config_file: Path to the configuration file. If None, default path is used.
    @return: Dictionary containing configuration parameters.
    """
    if config_file is None:
        # Default path to the configuration file
        base_dir = os.path.dirname(os.path.abspath(__file__))  # /.../astroca/tools
        config_file = os.path.join(base_dir, "..", "..", "config.ini")
        config_file = os.path.normpath(config_file)

    config = configparser.ConfigParser()
    config.read(config_file)

    if not config.sections():
        raise ValueError(f"No sections found in configuration file: {config_file}")

    return {section: dict(config.items(section)) for section in config.sections()}
