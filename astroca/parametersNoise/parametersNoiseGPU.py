"""
GPU version of noise estimation functions for AstroCA
"""

import torch
import numpy as np
from tqdm import tqdm


def compute_pseudo_residuals_GPU(data: torch.Tensor) -> torch.Tensor:
    """
    GPU version: r(t) = (data[t] - data[t-1]) / sqrt(2)
    @param data: 4D tensor of shape (T, Z, Y, X)
    @return: 4D tensor of shape (T-1, Z, Y, X)
    """
    # torch.diff is equivalent to np.diff along axis=0
    return torch.diff(data, dim=0) / torch.sqrt(torch.tensor(2.0, device=data.device, dtype=data.dtype))

def mad_with_pseudo_residual_GPU(residuals: torch.Tensor) -> torch.Tensor:
    """
    GPU version: Robust noise estimation: sigma = 1.4826 * median(|residuals|),
    excluding residuals that are zero.
    @param residuals: tensor of shape (T-1, Y, Xroi)
    @return: tensor of shape (Y, Xroi)
    """
    abs_res = torch.abs(residuals)

    # Replace zeros with NaN (equivalent to np.nan)
    abs_res = torch.where(abs_res == 0.0, torch.tensor(float('nan'), device=abs_res.device), abs_res)

    # Compute nanmedian along time axis (dim=0)
    # PyTorch doesn't have nanmedian, so we need to implement it
    median_vals = torch.nanmedian(abs_res, dim=0).values

    return 1.4826 * median_vals

def estimate_std_map_over_time_GPU(data: torch.Tensor, xmin: np.ndarray, xmax: np.ndarray) -> torch.Tensor:
    """
    GPU version: For each voxel (x,y,z), compute the MAD-based std estimation.
    @param data: 4D tensor of shape (T, Z, Y, X) representing the image sequence.
    @param xmin: 1D array of cropping bounds (left) for each Z slice.
    @param xmax: 1D array of cropping bounds (right) for each Z slice.
    @return: 3D tensor of shape (Z, Y, X) containing the estimated standard deviation for each voxel.
    """
    T, Z, Y, X = data.shape
    residuals = compute_pseudo_residuals_GPU(data)  # Shape: (T-1, Z, Y, X)

    # Initialize std_map with NaN (equivalent to np.full with np.nan)
    std_map = torch.full((Z, Y, X), float('nan'), dtype=torch.float32, device=data.device)

    for z in tqdm(range(Z), desc="Estimating std over time (GPU)", unit="slice"):
        x0, x1 = int(xmin[z]), int(xmax[z]) + 1
        if x0 >= x1:
            continue

        res_slice = residuals[:, z, :, x0:x1]  # (T-1, Y, Xroi)
        std_map[z, :, x0:x1] = mad_with_pseudo_residual_GPU(res_slice)

    return std_map

def estimate_std_over_time_GPU(data: torch.Tensor, xmin: np.ndarray, xmax: np.ndarray) -> float:
    """
    GPU version: Final std estimation as the median of the 3D map excluding zeros.
    @param data: 4D tensor of shape (T, Z, Y, X) representing the image sequence.
    @param xmin: 1D array of cropping bounds (left) for each Z slice.
    @param xmax: 1D array of cropping bounds (right) for each Z slice.
    @return: Estimated standard deviation of the noise over time.
    """
    print(" - Estimating standard deviation over time (GPU) ...")

    # Ensure data is on GPU
    if not data.is_cuda:
        print("   Warning: Data is not on GPU, moving to GPU...")
        data = data.cuda()

    std_map = estimate_std_map_over_time_GPU(data, xmin, xmax)

    # Filter valid values (not NaN and > 0)
    valid_mask = (~torch.isnan(std_map)) & (std_map > 0)
    valid = std_map[valid_mask]

    if valid.numel() > 0:
        std = float(torch.median(valid).item())
    else:
        std = 0.0

    print(f" std_noise = {std:.7f}")
    print(60*"=")
    print()
    return std