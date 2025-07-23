"""
@file parametersNoise_GPU.py
@brief Estimate the standard deviation of the noise in a 3D+time image sequence using GPU.
@detail Computes a per-voxel STD over time and estimates the global noise level as the median of non-zero voxels in the map.
"""
import torch
import numpy as np
from tqdm import tqdm


def compute_pseudo_residuals_GPU(data: torch.Tensor) -> torch.Tensor:
    """
    r(t) = (data[t] - data[t-1]) / sqrt(2)
    """
    # S'assurer que les données sont en float64 pour une précision maximale
    if data.dtype != torch.float64:
        data = data.to(torch.float64)

    # Utiliser torch.diff pour être cohérent avec np.diff
    diff_data = torch.diff(data, dim=0)  # Equivalent à np.diff(data, axis=0)

    # Calculer sqrt(2) avec la même précision
    sqrt_2 = torch.tensor(2.0, dtype=torch.float64, device=data.device).sqrt()
    result = diff_data / sqrt_2

    # Convertir en float32 à la fin comme dans la version CPU
    return result.to(torch.float32)


def mad_with_pseudo_residual_GPU(residuals: torch.Tensor) -> torch.Tensor:
    """
    Robust noise estimation: sigma = 1.4826 * median(|residuals|),
    excluding residuals that are zero.
    """
    abs_res = torch.abs(residuals)

    # Marquer les zéros exacts comme NaN (avec une tolérance très petite)
    zero_mask = torch.abs(abs_res) < 1e-10
    abs_res = torch.where(zero_mask, torch.tensor(float('nan'), device=residuals.device), abs_res)

    # Calculer la médiane avec une approche plus stable
    # Reshape pour traiter chaque voxel séparément
    T_minus_1, Y, X_roi = abs_res.shape
    abs_res_flat = abs_res.reshape(T_minus_1, -1)  # (T-1, Y*X_roi)

    # Calculer la médiane pour chaque voxel
    medians = torch.zeros(Y * X_roi, dtype=torch.float32, device=residuals.device)

    for i in range(Y * X_roi):
        voxel_data = abs_res_flat[:, i]
        # Filtrer les NaN
        valid_data = voxel_data[~torch.isnan(voxel_data)]
        if valid_data.numel() > 0:
            medians[i] = torch.median(valid_data)
        else:
            medians[i] = float('nan')

    # Reshape back et appliquer le facteur MAD
    mad_constant = torch.tensor(1.4826, dtype=torch.float32, device=residuals.device)
    return (mad_constant * medians).reshape(Y, X_roi)


def estimate_std_map_over_time_GPU(data: torch.Tensor, xmin: np.ndarray, xmax: np.ndarray) -> torch.Tensor:
    """
    For each voxel (x,y,z), compute the MAD-based std estimation on GPU.
    @param data: 4D torch tensor of shape (T, Z, Y, X) representing the image sequence.
    @param xmin: 1D array of cropping bounds (left) for each Z slice.
    @param xmax: 1D array of cropping bounds (right) for each Z slice.
    @return: 3D torch tensor of shape (Z, Y, X) containing the estimated standard deviation for each voxel.
    """
    T, Z, Y, X = data.shape
    residuals = compute_pseudo_residuals_GPU(data)  # Shape: (T-1, Z, Y, X)

    # Initialiser avec NaN comme dans la version CPU
    std_map = torch.full((Z, Y, X), float('nan'), dtype=torch.float32, device=data.device)

    for z in tqdm(range(Z), desc="Estimating std over time", unit="slice"):
        x0, x1 = xmin[z], xmax[z] + 1
        if x0 >= x1:
            continue
        res_slice = residuals[:, z, :, x0:x1]  # (T-1, Y, Xroi)
        std_map[z, :, x0:x1] = mad_with_pseudo_residual_GPU(res_slice)

    return std_map


def estimate_std_over_time_GPU(data: torch.Tensor, xmin: np.ndarray, xmax: np.ndarray) -> float:
    """
    Final std estimation as the median of the 3D map excluding zeros (GPU version).
    @param data: 4D torch tensor of shape (T, Z, Y, X) representing the image sequence.
    @param xmin: 1D array of cropping bounds (left) for each Z slice.
    @param xmax: 1D array of cropping bounds (right) for each Z slice.
    @return : Estimated standard deviation of the noise over time.
    """
    print(" - Estimating standard deviation over time ...")
    std_map = estimate_std_map_over_time_GPU(data, xmin, xmax)

    # Filtrer exactement comme dans la version CPU
    mask = ~torch.isnan(std_map) & (std_map > 0)
    valid = std_map[mask]

    if valid.numel() > 0:
        # Trier les données pour un calcul de médiane plus stable
        valid_sorted = torch.sort(valid)[0]
        n = valid_sorted.numel()

        if n % 2 == 1:
            # Nombre impair d'éléments
            std = float(valid_sorted[n // 2])
        else:
            # Nombre pair d'éléments - moyenne des deux éléments centraux
            mid1 = valid_sorted[n // 2 - 1]
            mid2 = valid_sorted[n // 2]
            std = float((mid1 + mid2) / 2.0)
    else:
        std = 0.0

    print(f" std_noise = {std:.7f}")
    print(60 * "=")
    print()
    return std