"""
@file varianceStabilization.py
@brief This module performs variance stabilization using the Anscombe transform.
@detail It converts Poisson noise to approximately Gaussian noise with stabilized variance (mean = 0, variance = 1).
The transform is applied only within the meaningful X-boundaries [index_xmin[z], index_xmax[z]] for each Z slice.
"""

import numpy as np
import os
from astroca.tools.exportData import export_data
from tqdm import tqdm
import torch
from typing import Union


def compute_variance_stabilization_CPU(
    data: np.ndarray, index_xmin: np.ndarray, index_xmax: np.ndarray, params: dict
) -> np.ndarray:
    """
    @brief Applies the Anscombe variance stabilization transform in-place to the image sequence.
    The Anscombe transform is applied as follows:
        A(x) = sqrt(x + 3/8) * 2
    This transform stabilizes the variance of Poisson-distributed data, making it approximately Gaussian.

    @param data: data (T, Z, Y, X) where T is time, Z is depth, Y is height, and X is width.
    @param index_xmin: 1D array of shape (Z,) with left cropping bounds per z
    @param index_xmax: 1D array of shape (Z,) with right cropping bounds per z
    @param params: Dictionary containing the parameters:
        - save_results: Boolean indicating whether to save the transformed data.
        - output_directory: Directory to save the transformed data if save_results is True.
    @return: The transformed data with stabilized variance.

    """
    print("=== Applying variance stabilization using Anscombe transform... ===")

    # extract necessary parameters
    required_keys = {"save", "paths"}
    if not required_keys.issubset(params.keys()):
        raise ValueError(
            f"Missing required parameters: {required_keys - params.keys()}"
        )

    save_results = int(params["save"]["save_variance_stabilization"]) == 1
    output_directory = params["paths"]["output_dir"]

    T, Z, Y, X = data.shape
    data = data.astype(np.float32)

    for z in tqdm(range(Z), desc="Variance stabilization per Z-slice", unit="slice"):
        x_min, x_max = index_xmin[z], index_xmax[z] + 1
        if x_min >= x_max:
            continue

        sub_volume = data[:, z, :, x_min:x_max]
        # Apply in-place: sqrt + scale
        np.sqrt(sub_volume + 3.0 / 8.0, out=sub_volume)
        sub_volume *= 2.0  # in-place multiplication

    if save_results:
        if output_directory is None:
            raise ValueError(
                "Output directory must be specified when save_results is True."
            )
        os.makedirs(output_directory, exist_ok=True)
        export_data(
            data,
            output_directory,
            export_as_single_tif=True,
            file_name="variance_stabilized_sequence",
        )
    print(60 * "=")
    print()
    return data


def compute_variance_stabilization_GPU(
    data: torch.Tensor, index_xmin: np.ndarray, index_xmax: np.ndarray, params: dict
) -> torch.Tensor:
    """
    Applies Anscombe variance stabilization transform on GPU using PyTorch.

    @param data: torch Tensor of shape (T, Z, Y, X) where T is time, Z is depth, Y is height, and X is width.
    @param index_xmin: 1D numpy array of shape (Z,) with left cropping bounds per z.
    @param index_xmax: 1D numpy array of shape (Z,) with right cropping bounds per z.
    @param params: Dictionary containing:
        - save_results
        - output_directory
    @return: Variance-stabilized data as a NumPy array.
    """
    print("=== Applying variance stabilization on GPU using PyTorch... ===")

    device = torch.device("cuda")
    index_xmin_torch = torch.from_numpy(index_xmin).to(device)
    index_xmax_torch = torch.from_numpy(index_xmax).to(device)

    T, Z, Y, X = data.shape

    for z in tqdm(
        range(Z), desc="Variance stabilization per Z-slice (GPU)", unit="slice"
    ):
        x_min = index_xmin_torch[z].item()
        x_max = index_xmax_torch[z].item() + 1
        if x_min >= x_max:
            continue
        sub_volume = data[:, z, :, x_min:x_max]
        sub_volume.add_(3.0 / 8.0).sqrt_().mul_(2.0)

    result = data.cpu().numpy()

    if int(params["save"]["save_variance_stabilization"]) == 1:
        out_dir = params["paths"]["output_dir"]
        if out_dir is None:
            raise ValueError(
                "Output directory must be specified when save_results is True."
            )
        os.makedirs(out_dir, exist_ok=True)
        export_data(
            result,
            out_dir,
            export_as_single_tif=True,
            file_name="variance_stabilized_sequence",
        )

    print("=" * 60 + "\n")
    return data


def compute_variance_stabilization(
    data: np.ndarray | torch.Tensor,
    index_xmin: np.ndarray,
    index_xmax: np.ndarray,
    params: dict,
) -> np.ndarray | torch.Tensor:
    """
    Dispatcher for variance stabilization, CPU or GPU.
    """
    if int(params.get("GPU_AVAILABLE", 0)) == 1:
        return compute_variance_stabilization_GPU(data, index_xmin, index_xmax, params)
    else:
        return compute_variance_stabilization_CPU(data, index_xmin, index_xmax, params)


def check_variance(
    data: np.ndarray, index_xmin: np.ndarray, index_xmax: np.ndarray
) -> bool:
    """
    @brief Checks if the variance of the transformed data is approximately 1.
    @param data: 4D numpy array of shape (T, Z, Y, X)
    @param index_xmin: 1D array of shape (depth,) with left cropping bounds per z
    @param index_xmax: 1D array of shape (depth,) with right cropping bounds per z
    @return: True if variance is approximately 1, False otherwise
    """
    T, Z, Y, X = data.shape

    for z in range(Z):
        x_min = index_xmin[z]
        x_max = index_xmax[z] + 1  # Python exclusive slicing

        if x_min >= x_max:
            continue  # avoid invalid slices

        sub_volume = data[:, z, :, x_min:x_max]
        variance = np.var(sub_volume)

        if not np.isclose(variance, 1.0):
            return False

    return True


def anscombe_inverse(
    data: Union[np.ndarray, torch.Tensor],
    index_xmin: np.ndarray,
    index_xmax: np.ndarray,
    param_values: dict,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Dispatcher pour la transformation inverse d'Anscombe (CPU ou GPU)
    """
    if param_values.get("GPU_AVAILABLE", 0) == 1:
        return anscombe_inverse_GPU(data, index_xmin, index_xmax, param_values)
    else:
        # Conversion si nécessaire
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        return anscombe_inverse_CPU(data, index_xmin, index_xmax, param_values)




def anscombe_inverse_CPU(
    data: np.ndarray, index_xmin: np.ndarray, index_xmax: np.ndarray, param_values: dict
) -> np.ndarray:
    """
    Compute inverse of Anscombe transform to compute the amplitude of the image
        A⁻¹(x) = (x / 2)^2 - 3/8

    @param data: 4D numpy array of shape (T, Z, Y, X)
    @param index_xmin: 1D array of shape (Z,) with left cropping bounds per z
    @param index_xmax: 1D array of shape (Z,) with right cropping bounds per z
    @param param_values: Dictionary containing the parameters:
        - save_results: If True, saves the result to output_directory
        - output_directory: Directory to save the result if save_results is True
    @return: Transformed image of same shape as input
    """
    print(" - Applying inverse Anscombe transform on 3D volume...")
    required_keys = {"save", "paths"}
    if not required_keys.issubset(param_values.keys()):
        raise ValueError(
            f"Missing required parameters: {required_keys - param_values.keys()}"
        )

    save_results = int(param_values["save"]["save_anscombe_inverse"]) == 1
    output_directory = param_values["paths"]["output_dir"]
    _, Z, Y, X = data.shape
    data = data.astype(np.float32)

    data_out = np.zeros_like(data, dtype=np.float32)

    for z in tqdm(range(Z), desc="Inverse Anscombe per Z-slice", unit="slice"):
        x_min = index_xmin[z]
        x_max = index_xmax[z] + 1

        if x_min >= x_max:
            continue

        slice_section = data[0, z, :, x_min:x_max]
        slice_out = data_out[0, z, :, x_min:x_max]

        # Apply A⁻¹(x) = (x/2)^2 - 3/8
        np.divide(slice_section, 2.0, out=slice_out)
        np.square(slice_out, out=slice_out)
        slice_out -= 3.0 / 8.0

    if save_results:
        if output_directory is None:
            raise ValueError(
                "Output directory must be specified when save_results is True."
            )
        os.makedirs(output_directory, exist_ok=True)
        export_data(
            data_out,
            output_directory,
            export_as_single_tif=True,
            file_name="inverse_anscombe_transformed_volume",
        )

    return data_out


def anscombe_inverse_GPU_optimized(
    data: torch.Tensor,
    index_xmin: np.ndarray,
    index_xmax: np.ndarray,
    param_values: dict
) -> torch.Tensor:
    """
    GPU optimized inverse Anscombe transform avec vectorisation complète
    """
    print(" - Applying inverse Anscombe transform on 3D volume (GPU optimized)...")

    required_keys = {"save", "paths"}
    if not required_keys.issubset(param_values.keys()):
        raise ValueError(f"Missing required parameters: {required_keys - param_values.keys()}")

    save_results = int(param_values["save"]["save_anscombe_inverse"]) == 1
    output_directory = param_values["paths"]["output_dir"]

    _, Z, Y, X = data.shape
    device = data.device

    # Convertir les indices en tenseurs GPU
    index_xmin_torch = torch.from_numpy(index_xmin).to(device)
    index_xmax_torch = torch.from_numpy(index_xmax).to(device)

    # Créer grille de coordonnées
    z_coords = torch.arange(Z, device=device).unsqueeze(1).unsqueeze(2)  # (Z, 1, 1)
    y_coords = torch.arange(Y, device=device).unsqueeze(0).unsqueeze(2)  # (1, Y, 1)
    x_coords = torch.arange(X, device=device).unsqueeze(0).unsqueeze(0)  # (1, 1, X)

    # Étendre les indices pour broadcasting
    index_xmin_expanded = index_xmin_torch.unsqueeze(1).unsqueeze(2)  # (Z, 1, 1)
    index_xmax_expanded = index_xmax_torch.unsqueeze(1).unsqueeze(2)  # (Z, 1, 1)

    # Créer masque vectorisé complet
    valid_mask = (x_coords >= index_xmin_expanded) & (x_coords <= index_xmax_expanded)

    # Initialiser le résultat
    data_out = torch.zeros_like(data, dtype=torch.float32, device=device)

    # Premier frame seulement
    slice_data = data[0]  # Shape: (Z, Y, X)

    # Application vectorisée complète
    transformed = torch.pow(slice_data / 2.0, 2) - 3.0 / 8.0

    # Appliquer le masque
    data_out[0] = torch.where(valid_mask, transformed, torch.zeros_like(transformed))

    if save_results:
        if output_directory is None:
            raise ValueError("Output directory must be specified when save_results is True.")
        os.makedirs(output_directory, exist_ok=True)
        export_data(
            data_out.cpu().numpy(),
            output_directory,
            export_as_single_tif=True,
            file_name="inverse_anscombe_transformed_volume",
        )

    return data_out