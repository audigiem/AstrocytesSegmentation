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

# import cupy as cp
from numba import cuda
import torch


def compute_boundaries_CPU(
    data: np.ndarray, params: dict
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
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
    required_keys = {"preprocessing", "save", "paths"}
    if not required_keys.issubset(params.keys()):
        raise ValueError(
            f"Missing required parameters: {required_keys - params.keys()}"
        )
    pixel_cropped = int(params["preprocessing"]["pixel_cropped"])
    save_results = int(params["save"]["save_boundaries"]) == 1
    output_directory = params["paths"]["output_dir"]

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

            data[t, z, :, crop_start : crop_start + pixel_cropped] = default_value
            data[t, z, :, crop_end - pixel_cropped : crop_end] = default_value

    index_xmin += pixel_cropped
    index_xmax -= pixel_cropped

    print(
        f"    index_xmin = {index_xmin}\n     index_xmax = {index_xmax}\n     default_value = {default_value}"
    )

    if save_results:
        if output_directory is None:
            raise ValueError(
                "output_directory must be specified if save_results is True."
            )
        os.makedirs(output_directory, exist_ok=True)
        export_data(data, output_directory, export_as_single_tif=True, file_name="data")
        save_numpy_tab(index_xmin, output_directory, file_name="index_Xmin.npy")
        save_numpy_tab(index_xmax, output_directory, file_name="index_Xmax.npy")

    print(60 * "=")
    print()
    return index_xmin, index_xmax, default_value, data


def compute_boundaries_GPU(
    data: torch.Tensor, params: dict
) -> Tuple[np.ndarray, np.ndarray, float, torch.Tensor]:
    """
    Compute cropping boundaries in X for each Z slice using PyTorch on GPU.

    @param data: 4D data (T, Z, Y, X), either a numpy array or a torch tensor
    @param params: Dictionary with parameters:
        - pixel_cropped: Number of pixels to crop in X.
        - save_results: Boolean for saving results.
        - output_directory: Where to save results if needed.
    @return: (index_xmin, index_xmax, default_value, data)
    """
    print(" - Computing cropping boundaries in X for each Z slice (PyTorch GPU)...")

    # Extract params
    pixel_cropped = int(params["preprocessing"]["pixel_cropped"])
    save_results = int(params["save"]["save_boundaries"]) == 1
    out_dir = params["paths"]["output_dir"]

    # # Ensure tensor format on GPU
    # if isinstance(data, np.ndarray):
    #     data_gpu = torch.from_numpy(data).float().to('cuda')
    # else:
    #     data_gpu = data.to('cuda') if not data.is_cuda else data

    T, Z, Y, X = data.shape
    default_val = float(data[0, 0, 0, X - 1].item())

    xmin = torch.full((Z,), -1, dtype=torch.int32, device="cuda")
    xmax = torch.full((Z,), X - 1, dtype=torch.int32, device="cuda")

    y_sample_size = max(1, Y // 10)
    y_indices = torch.randperm(Y, device="cuda")[:y_sample_size]

    for z in range(Z):
        found = False
        for x in range(X):
            vals = data[0, z, y_indices, x]
            all_default = (vals == default_val).all()
            if not all_default:
                if not found:
                    xmin[z] = x
                    found = True
            elif found:
                xmax[z] = x - 1
                break

    xmin += pixel_cropped
    xmax -= pixel_cropped

    x_range = torch.arange(X, device="cuda").unsqueeze(0)  # (1, X)
    mask_zx = (x_range < xmin[:, None]) | (x_range >= (xmax[:, None] + 1))  # (Z, X)
    mask = mask_zx.unsqueeze(0).unsqueeze(2)  # (1, Z, 1, X)
    data[mask.expand(T, Z, Y, X)] = default_val

    # Convert to CPU for output
    index_xmin = xmin.cpu().numpy()
    index_xmax = xmax.cpu().numpy()
    data_result = data.cpu().numpy()

    if save_results:
        if out_dir is None:
            raise ValueError(
                "output_directory must be specified if save_results is True."
            )
        os.makedirs(out_dir, exist_ok=True)
        export_data(
            data_result,
            out_dir,
            export_as_single_tif=True,
            file_name="bounded_image_sequence",
        )
        save_numpy_tab(index_xmin, out_dir, file_name="index_Xmin.npy")
        save_numpy_tab(index_xmax, out_dir, file_name="index_Xmax.npy")

    return index_xmin, index_xmax, default_val, data


def compute_boundaries(
    data: np.ndarray | torch.Tensor, params: dict
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray | torch.Tensor]:
    """
    Compute cropping boundaries in X for each Z slice based on background values.
    Sets boundary pixels to default_value and saves results optionally.

    @param data: 4D data (T, Z, Y, X) as a numpy array or torch tensor.
    @param params: Dictionary containing the parameters:
        - pixel_cropped: Number of pixels to crop from the height dimension.
        - save_results: Boolean indicating whether to save the results.
        - output_directory: Directory to save the results if save_results is True.
    @return: (index_xmin, index_xmax, default_value, data)
    """
    required_keys = {"GPU_AVAILABLE"}
    if not required_keys.issubset(params.keys()):
        raise ValueError(
            f"Missing required parameters: {required_keys - params.keys()}"
        )
    _GPU_AVAILABLE = int(params["GPU_AVAILABLE"]) == 1  # Convert to boolean
    if _GPU_AVAILABLE:
        return compute_boundaries_GPU(data, params)
    else:
        return compute_boundaries_CPU(data, params)
