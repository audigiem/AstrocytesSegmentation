"""
@file dynamicImage.py
@brief Module for computing the dynamic image (ΔF = F - F0) and the background F0 estimation over time.
"""
from email.contentmanager import raw_data_manager

# from joblib import Parallel, delayed
from astroca.tools.exportData import export_data
import os
import numpy as np
import time
from astroca.varianceStabilization.varianceStabilization import anscombe_inverse
from tqdm import tqdm
import torch
from typing import Union, Tuple


def compute_dynamic_image(
    data: Union[np.ndarray, torch.Tensor],
    F0: Union[np.ndarray, torch.Tensor],
    index_xmin: np.ndarray,
    index_xmax: np.ndarray,
    time_window: int,
    params: dict,
) -> Tuple[Union[np.ndarray, torch.Tensor], float]:
    """
    Wrapper function to compute the dynamic image (dF = F - F0) and estimate the noise level.
    """
    if params.get("GPU_AVAILABLE", 0) == 1:
        return compute_dynamic_image_GPU(
            data, F0, index_xmin, index_xmax, time_window, params
        )
    else:
        return compute_dynamic_image_CPU(
            data, F0, index_xmin, index_xmax, time_window, params
        )


def compute_dynamic_image_CPU(
    data: np.ndarray,
    F0: np.ndarray,
    index_xmin: np.ndarray,
    index_xmax: np.ndarray,
    time_window: int,
    params: dict,
) -> tuple[np.ndarray, float]:
    """
    Compute ΔF = F - F0 and estimate the noise level as the median of ΔF.

    @param image_sequence: 4D image sequence (T, Z, Y, X)
    @param F0: Background array of shape (nbF0, Z, Y, X)
    @param index_xmin: cropping bounds in X for each Z
    @param index_xmax: cropping bounds in X for each Z
    @param time_window: the duration of each background block
    @param params: Dictionary containing the parameters:
        - save_results: If True, saves the result to output_directory
        - output_directory: Directory to save the result if save_results is True
    @return: (dF: array of shape (T, Z, Y, X), mean_noise: float)
    """
    print("=== Computing dynamic image (dF = F - F0) and estimating noise... ===")
    print(" - Computing dynamic image...")

    # Extract necessary parameters
    required_keys = {"save", "paths"}
    if not required_keys.issubset(params.keys()):
        raise ValueError(
            f"Missing required parameters: {required_keys - params.keys()}"
        )
    save_results = int(params["save"]["save_df"]) == 1
    output_directory = params["paths"]["output_dir"]

    T, Z, Y, X = data.shape
    nbF0 = F0.shape[0]

    dF = np.copy(data)

    # Préallocation avec estimation maximale
    width_without_zeros = sum(
        max(0, index_xmax[z] - index_xmin[z] + 1) for z in range(Z)
    )
    flattened_dF = np.empty(T * Y * width_without_zeros, dtype=np.float32)
    k = 0

    for t in tqdm(range(T), desc="Computing ΔF over time", unit="frame"):
        it = min(t // time_window, nbF0 - 1)
        for z in range(Z):
            x_min, x_max = index_xmin[z], index_xmax[z] + 1
            if x_min >= x_max:
                continue
            # Vectorisé sur Y
            delta = data[t, z, :, x_min:x_max] - F0[it, z, :, x_min:x_max]
            dF[t, z, :, x_min:x_max] = delta
            n = x_max - x_min
            flattened_dF[k : k + Y * n] = delta.reshape(-1)
            k += Y * n

    mean_noise = float(np.median(flattened_dF[:k]))
    print(f"    mean_Noise = {mean_noise:.6f}")

    if save_results:
        if output_directory is None:
            raise ValueError(
                "Output directory must be specified when save_results is True."
            )
        os.makedirs(output_directory, exist_ok=True)
        export_data(
            dF,
            output_directory,
            export_as_single_tif=True,
            file_name="dynamic_image_dF",
        )

    print()
    return dF, mean_noise


def compute_dynamic_image_GPU(
    data: torch.Tensor,
    F0: torch.Tensor,
    index_xmin: np.ndarray,
    index_xmax: np.ndarray,
    time_window: int,
    params: dict,
) -> tuple[torch.Tensor, float]:
    """
    Compute ΔF = F - F0 and estimate the noise level as the median of ΔF using PyTorch on GPU.

    @param data: 4D image sequence (T, Z, Y, X) as a PyTorch tensor
    @param F0: Background array of shape (nbF0, Z, Y, X) as a PyTorch tensor
    @param index_xmin: cropping bounds in X for each Z
    @param index_xmax: cropping bounds in X for each Z
    @param time_window: the duration of each background block
    @param params: Dictionary containing the parameters:
        - save_results: If True, saves the result to output_directory
        - output_directory: Directory to save the result if save_results is True
    @return: (dF: tensor of shape (T, Z, Y, X), mean_noise: float)
    """
    print(
        "=== Computing dynamic image (dF = F - F0) and estimating noise on GPU... ==="
    )
    print(" - Computing dynamic image...")

    # Extract necessary parameters
    required_keys = {"save", "paths"}
    if not required_keys.issubset(params.keys()):
        raise ValueError(
            f"Missing required parameters: {required_keys - params.keys()}"
        )
    save_results = int(params["save"]["save_df"]) == 1
    output_directory = params["paths"]["output_dir"]

    T, Z, Y, X = data.shape
    nbF0 = F0.shape[0]

    # CORRECTION 1: Utiliser torch.copy() au lieu de torch.empty_like()
    # pour commencer avec les données originales comme dans la version CPU
    dF = data.clone()

    # Préallocation avec estimation maximale
    width_without_zeros = sum(
        max(0, index_xmax[z] - index_xmin[z] + 1) for z in range(Z)
    )
    flattened_dF = torch.empty(
        T * Y * width_without_zeros, dtype=torch.float32, device=data.device
    )
    k = 0

    for t in tqdm(range(T), desc="Computing ΔF over time", unit="frame"):
        it = min(t // time_window, nbF0 - 1)
        for z in range(Z):
            x_min, x_max = index_xmin[z], index_xmax[z] + 1
            if x_min >= x_max:
                continue
            # CORRECTION 2: Calculer delta exactement comme dans la version CPU
            delta = data[t, z, :, x_min:x_max] - F0[it, z, :, x_min:x_max]
            dF[t, z, :, x_min:x_max] = delta
            n = x_max - x_min
            # CORRECTION 3: Utiliser reshape(-1) au lieu de view(-1)
            # pour être cohérent avec NumPy
            flattened_dF[k : k + Y * n] = delta.reshape(-1)
            k += Y * n

    # CORRECTION 4: S'assurer que le calcul de médiane est identique
    mean_noise = float(torch.median(flattened_dF[:k]))
    print(f"    mean_Noise = {mean_noise:.6f}")

    if save_results:
        if output_directory is None:
            raise ValueError(
                "Output directory must be specified when save_results is True."
            )
        os.makedirs(output_directory, exist_ok=True)
        # CORRECTION 5: Convertir en NumPy avec le bon dtype
        export_data(
            dF.cpu().numpy().astype(np.float32),
            output_directory,
            export_as_single_tif=True,
            file_name="dynamic_image_dF",
        )

    print()
    return dF, mean_noise


def compute_image_amplitude(
        data_cropped: Union[np.ndarray, torch.Tensor],
        F0: Union[np.ndarray, torch.Tensor],
        index_xmin: np.ndarray,
        index_xmax: np.ndarray,
        param_values: dict,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Dispatcher pour le calcul d'amplitude (CPU ou GPU)
    """
    if param_values.get("GPU_AVAILABLE", 0) == 1:
        # Détection automatique GPU si les données sont sur GPU
        if isinstance(data_cropped, torch.Tensor) and data_cropped.is_cuda:
            # Assurer que F0 est aussi sur GPU
            if isinstance(F0, np.ndarray):
                F0 = torch.from_numpy(F0).to(data_cropped.device)
            elif isinstance(F0, torch.Tensor) and not F0.is_cuda:
                F0 = F0.to(data_cropped.device)

            return compute_image_amplitude_GPU(
                data_cropped, F0, index_xmin, index_xmax, param_values
            )
    else:
        # Conversion vers numpy si nécessaire
        if isinstance(data_cropped, torch.Tensor):
            data_cropped = data_cropped.cpu().numpy()
        if isinstance(F0, torch.Tensor):
            F0 = F0.cpu().numpy()
        return compute_image_amplitude_CPU(data_cropped, F0, index_xmin, index_xmax, param_values)


def compute_image_amplitude_CPU(
    data_cropped: np.ndarray,
    F0: np.ndarray,
    index_xmin: np.ndarray,
    index_xmax: np.ndarray,
    param_values: dict,
) -> np.ndarray:
    """
    @brief Compute the amplitude of the image using the Anscombe inverse transform.
    result -> (data_cropped - f0_inv)/f0_inv
    @param data_cropped: 4D numpy array of shape (T, Z, Y, X) representing the cropped raw data.
    @param F0: 4D numpy array of shape (1, Z, Y, X) representing the background estimation to be inversed.
    @param index_xmin: 1D array of shape (depth,) with left cropping bounds per z
    @param index_xmax: 1D array of shape (depth,) with right cropping bounds per z
    @param param_values: Dictionary containing the parameters:
        - save_results: If True, saves the result to output_directory
        - output_directory: Directory to save the result if save_results is True
    @return: 4D numpy array of shape (T, Z, Y, X) with the amplitude values.
    """
    print("=== Computing image amplitude... ===")
    required_keys = {"save", "paths"}
    if not required_keys.issubset(param_values.keys()):
        raise ValueError(
            f"Missing required parameters: {required_keys - param_values.keys()}"
        )
    save_results_amplitude = int(param_values["save"]["save_amplitude"]) == 1
    output_directory = param_values["paths"]["output_dir"]

    f0_inv = anscombe_inverse(F0, index_xmin, index_xmax, param_values=param_values)

    T, Z, Y, X = data_cropped.shape
    image_amplitude = np.zeros_like(data_cropped, dtype=np.float32)

    for z in tqdm(range(Z), desc="Computing amplitude per Z-slice", unit="slice"):
        x_min = index_xmin[z]
        x_max = index_xmax[z] + 1

        if x_min >= x_max:
            continue

        f0_slice = f0_inv[0, z, :, x_min:x_max]
        f0_safe = np.where(f0_slice == 0, 1e-10, f0_slice)

        for t in range(T):
            data_slice = data_cropped[t, z, :, x_min:x_max]
            result_slice = (data_slice - f0_slice) / f0_safe
            image_amplitude[t, z, :, x_min:x_max] = result_slice

    if save_results_amplitude:
        if output_directory is None:
            raise ValueError(
                "Output directory must be specified when save_results is True."
            )
        os.makedirs(output_directory, exist_ok=True)
        export_data(
            image_amplitude,
            output_directory,
            export_as_single_tif=True,
            file_name="amplitude",
        )

    print(60 * "=")
    print()
    return image_amplitude


def compute_image_amplitude_GPU(
        data_cropped: torch.Tensor,
        F0: torch.Tensor,
        index_xmin: np.ndarray,
        index_xmax: np.ndarray,
        param_values: dict,
) -> torch.Tensor:
    """
    GPU optimized image amplitude computation: (data_cropped - f0_inv)/f0_inv
    """
    print("=== Computing image amplitude (GPU)... ===")

    required_keys = {"save", "paths"}
    if not required_keys.issubset(param_values.keys()):
        raise ValueError(f"Missing required parameters: {required_keys - param_values.keys()}")

    save_results_amplitude = int(param_values["save"]["save_amplitude"]) == 1
    output_directory = param_values["paths"]["output_dir"]

    # Calcul de f0_inv sur GPU
    f0_inv = anscombe_inverse(F0, index_xmin, index_xmax, param_values=param_values)

    T, Z, Y, X = data_cropped.shape
    device = data_cropped.device

    # Convertir les indices en tenseurs GPU
    index_xmin_torch = torch.from_numpy(index_xmin).to(device)
    index_xmax_torch = torch.from_numpy(index_xmax).to(device)

    # Créer masque vectorisé
    z_coords = torch.arange(Z, device=device)
    x_coords = torch.arange(X, device=device)
    zz, xx = torch.meshgrid(z_coords, x_coords, indexing='ij')
    valid_mask = (xx >= index_xmin_torch[zz]) & (xx <= index_xmax_torch[zz])

    # Initialiser le résultat
    image_amplitude = torch.zeros_like(data_cropped, dtype=torch.float32, device=device)

    # f0_slice pour tous les z à la fois
    f0_slice = f0_inv[0]  # Shape: (Z, Y, X)
    f0_safe = torch.where(f0_slice == 0, torch.tensor(1e-10, device=device), f0_slice)

    # Calcul vectorisé pour tous les temps
    for t in tqdm(range(T), desc="Computing amplitude GPU", unit="frame"):
        data_slice = data_cropped[t]  # Shape: (Z, Y, X)

        # Calcul vectorisé: (data - f0) / f0_safe
        result = (data_slice - f0_slice) / f0_safe

        # Appliquer le masque pour les zones valides seulement
        result = torch.where(
            valid_mask.unsqueeze(1),  # Étendre pour dimension Y
            result,
            torch.zeros_like(result)
        )

        image_amplitude[t] = result

    if save_results_amplitude:
        if output_directory is None:
            raise ValueError("Output directory must be specified when save_results is True.")
        os.makedirs(output_directory, exist_ok=True)
        export_data(
            image_amplitude.cpu().numpy(),
            output_directory,
            export_as_single_tif=True,
            file_name="amplitude",
        )

    print("=" * 60 + "\n")
    return image_amplitude