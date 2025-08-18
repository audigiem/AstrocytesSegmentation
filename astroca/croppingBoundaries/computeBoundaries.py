"""
@file computeBoundaries.py
@brief This module provides functionality to compute cropping boundaries for 3D image sequences with time dimension.
@detail Due to the acquisition process, the data is supposed to have empty band(s) for each z-slice.
To speed up the calculations over the images, we spot those empty bands whose value is equal to default_value
(0.0 or 50.0) and compute the cropping boundaries for each z-slice. The cropping boundaries are computed as
the first and last non-empty band in each z-slice.
"""

import numpy as np
from astroca.tools.exportData import (
    export_data,
    save_numpy_tab,
    export_data_GPU_with_memory_optimization as export_data_GPU,
    save_tensor_as_numpy_GPU,
)
import os
from tqdm import tqdm
from typing import List, Dict, Tuple, Any
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
) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """
    Compute cropping boundaries in X for each Z slice using PyTorch on GPU.

    @param data: 4D data (T, Z, Y, X), either a numpy array or a torch tensor
    @param params: Dictionary with parameters:
        - pixel_cropped: Number of pixels to crop in X.
        - save_results: Boolean for saving results.
        - output_directory: Where to save results if needed.
    @return: (index_xmin, index_xmax, default_value, data)
    """
    print(" - [GPU] Computing cropping boundaries (ULTRA OPTIMIZED)...")

    device = data.device
    T, Z, Y, X = data.shape

    # Extraction des paramètres
    pixel_cropped = int(params["preprocessing"]["pixel_cropped"])
    save_results = int(params["save"]["save_boundaries"]) == 1
    out_dir = params["paths"]["output_dir"]

    # Calcul vectorisé de la valeur par défaut
    default_val = data[0, 0, 0, X - 1].item()

    # Échantillonnage vectorisé des indices Y
    y_sample_size = max(1, Y // 10)
    y_indices = torch.randperm(Y, device=device)[:y_sample_size]

    # Extraction des données échantillonnées pour le premier frame
    sampled_data = data[0, :, y_indices, :]  # Shape: (Z, y_sample_size, X)

    # Détection vectorisée des valeurs non-default
    non_default_mask = sampled_data != default_val  # Shape: (Z, y_sample_size, X)

    # Agrégation sur la dimension Y : True si au moins un pixel non-default
    has_signal = non_default_mask.any(dim=1)  # Shape: (Z, X)

    # Calcul vectorisé des indices xmin et xmax
    xmin = torch.full((Z,), -1, dtype=torch.int32, device=device)
    xmax = torch.full((Z,), X - 1, dtype=torch.int32, device=device)

    # Trouver les premiers indices non-default pour chaque Z
    for z in range(Z):
        signal_positions = torch.nonzero(has_signal[z], as_tuple=False).squeeze(-1)
        if len(signal_positions) > 0:
            xmin[z] = signal_positions[0]
            xmax[z] = signal_positions[-1]

    # Ajustement avec pixel_cropped
    xmin += pixel_cropped
    xmax -= pixel_cropped

    # Application du masque vectorisée ultra-rapide
    x_coords = torch.arange(X, device=device).view(1, 1, X)  # Broadcasting shape
    xmin_broadcast = xmin.view(Z, 1, 1)  # (Z, 1, 1)
    xmax_broadcast = xmax.view(Z, 1, 1)  # (Z, 1, 1)

    # Masque vectorisé complet
    boundary_mask = (x_coords < xmin_broadcast) | (
        x_coords > xmax_broadcast
    )  # (Z, 1, X)
    boundary_mask = boundary_mask.unsqueeze(0).expand(T, Z, Y, X)  # (T, Z, Y, X)

    # Application vectorisée du masque sur toutes les données
    data[boundary_mask] = default_val

    print(
        f"    index_xmin = {xmin}\n     index_xmax = {xmax}\n     default_value = {default_val}"
    )
    if save_results:
        if out_dir is None:
            raise ValueError(
                "output_directory must be specified if save_results is True."
            )
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        export_data_GPU(
            data,
            out_dir,
            export_as_single_tif=True,
            file_name="data",
            max_memory_usage_mb=2048,
        )
        save_tensor_as_numpy_GPU(
            xmin, out_dir, file_name="index_Xmin.npy", async_save=True
        )
        save_tensor_as_numpy_GPU(
            xmax, out_dir, file_name="index_Xmax.npy", async_save=True
        )

    print(60 * "=")
    print()
    return xmin, xmax, default_val, data


def compute_boundaries(
    data: np.ndarray | torch.Tensor, params: dict
) -> Tuple[
    np.ndarray | torch.Tensor,
    np.ndarray | torch.Tensor,
    float,
    np.ndarray | torch.Tensor,
]:
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
