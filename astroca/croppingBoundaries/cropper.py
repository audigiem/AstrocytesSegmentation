"""
@file cropper.py
@brief This module provides functionality to crop boundaries of 3D image sequences with time dimension (if needed).
"""

from astroca.tools.exportData import export_data
import os
import numpy as np

import numpy as np
import os
import torch
from typing import Union



def crop_boundaries_CPU(data: np.ndarray, params: dict) -> np.ndarray:
    print("=== Cropping boundaries and compute boundaries (CPU) ===")
    print(" - Cropping the boundaries of the image sequence...")

    required_keys = {'preprocessing', 'save', 'paths'}
    if not required_keys.issubset(params.keys()):
        raise ValueError(f"Missing required parameters: {required_keys - params.keys()}")

    x_min = int(params['preprocessing']['x_min'])
    x_max = int(params['preprocessing']['x_max'])
    pixel_cropped = int(params['preprocessing']['pixel_cropped'])
    save_results = int(params['save']['save_cropp_boundaries']) == 1
    output_directory = params['paths']['output_dir']

    if len(data.shape) not in [3, 4]:
        raise ValueError(f"Input must be a 3D or 4D numpy array, got {data.shape}")

    if data.ndim == 3:
        data = np.expand_dims(data, axis=0)

    T, Z, Y, X = data.shape
    cropped_data = data[:, :, pixel_cropped:Y, x_min:x_max + 1]

    print(f"    Cropped data shape: {cropped_data.shape}")

    if save_results:
        if output_directory is None:
            raise ValueError("output_directory must be specified if save_results is True.")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        export_data(cropped_data, output_directory, export_as_single_tif=True, file_name="cropped_image_sequence")

    print()
    return cropped_data


def crop_boundaries_GPU(data: torch.Tensor, params: dict) -> torch.Tensor:
    print("=== Cropping boundaries and compute boundaries (GPU) ===")
    print(" - Cropping the boundaries of the image sequence...")

    required_keys = {'preprocessing', 'save', 'paths'}
    if not required_keys.issubset(params.keys()):
        raise ValueError(f"Missing required parameters: {required_keys - params.keys()}")

    x_min = int(params['preprocessing']['x_min'])
    x_max = int(params['preprocessing']['x_max'])
    pixel_cropped = int(params['preprocessing']['pixel_cropped'])
    save_results = int(params['save']['save_cropp_boundaries']) == 1
    output_directory = params['paths']['output_dir']

    if data.ndim != 4:
        raise ValueError(f"Input data must be a 4D torch tensor (T, Z, Y, X), got shape {data.shape}.")

    T, Z, Y, X = data.shape
    cropped_data = data[:, :, pixel_cropped:Y, x_min:x_max + 1]

    print(f"    Cropped data shape: {cropped_data.shape}")

    if save_results:
        if output_directory is None:
            raise ValueError("output_directory must be specified if save_results is True.")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        # Copy data back to CPU before saving
        cropped_data_cpu = cropped_data.cpu().numpy()
        export_data(cropped_data_cpu, output_directory, export_as_single_tif=True, file_name="cropped_image_sequence")

    print()
    return cropped_data


def crop_boundaries(data: Union[np.ndarray, torch.Tensor], params: dict):
    """
    Wrapper that dispatches to CPU or GPU version depending on GPU_AVAILABLE flag.
    """
    use_gpu = int(params.get('GPU_AVAILABLE', 0)) == 1
    if use_gpu:
        print("GPU processing requested.")
        if not isinstance(data, torch.Tensor):
            # Convert from numpy to torch.Tensor on GPU
            data = torch.from_numpy(data).float().to("cuda")
        else:
            data = data.to("cuda")
        return crop_boundaries_GPU(data, params)
    else:
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        return crop_boundaries_CPU(data, params)
