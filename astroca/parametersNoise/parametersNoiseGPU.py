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
    # Utiliser torch.diff pour être cohérent avec np.diff
    diff_data = torch.diff(data, dim=0)  # Equivalent à np.diff(data, axis=0)
    return (diff_data / torch.sqrt(torch.tensor(2.0, device=data.device))).to(torch.float32)


def mad_with_pseudo_residual_GPU(residuals: torch.Tensor) -> torch.Tensor:
    """
    Robust noise estimation: sigma = 1.4826 * median(|residuals|),
    excluding residuals that are zero.
    """
    abs_res = torch.abs(residuals)
    # Remplacer les zéros par NaN comme dans la version CPU
    abs_res = torch.where(abs_res == 0.0, torch.tensor(float('nan'), device=residuals.device), abs_res)
    # Calculer la médiane en ignorant les NaN (équivalent à np.nanmedian)
    return 1.4826 * torch.nanmedian(abs_res, dim=0).values


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
    # ~torch.isnan(std_map) & (std_map > 0) équivalent à ~np.isnan(std_map) & (std_map > 0)
    mask = ~torch.isnan(std_map) & (std_map > 0)
    valid = std_map[mask]

    if valid.numel() > 0:  # Equivalent à valid.size > 0 en NumPy
        std = float(torch.median(valid))
    else:
        std = 0.0

    print(f" std_noise = {std:.7f}")
    print(60 * "=")
    print()
    return std