"""
@filename: zScore.py
@brief: This module provides functionality to compute the z-score of a 3D image sequence with time dimension.
@detail: Computes a z-score for each voxel in the 3D image sequence across the time dimension to identify significant deviations from the noise level.
"""


import numpy as np
from tqdm import tqdm
import torch

def compute_z_score_CPU(data: np.ndarray, std_noise: float, gaussian_noise_mean: float, threshold: float, index_xmin: list, index_xmax: list) -> np.ndarray:
    """
    @brief Compute the z-score for each voxel in the 3D image sequence across the time dimension.
    Zscore(vox, t) = data(vox, t) - gaussian_noise_mean / std_noise

    @param data: 4D numpy array of shape (T, Z, Y, X) representing the image sequence.
    @param std_noise: Standard deviation of the noise level to normalize the z-score.
    @param gaussian_noise_mean: Mean (or Med depending on the previous code) of the Gaussian noise, used to center the z-score calculation.
    @param threshold: Threshold value to determine significant deviations in the z-score.
    @param index_xmin: 1D array of cropping bounds (left) for each Z slice.
    @param index_xmax: 1D array of cropping bounds (right) for each Z slice.
    @return: 4D numpy array of z-scores with the same shape as input data.
    """
    print(f" - Compute binary z-score...")

    T, Z, Y, X = data.shape

    if std_noise <= 0:
        raise ValueError("std_noise must be > 0")

    processed = np.zeros_like(data, dtype=np.uint8)
    value = data - gaussian_noise_mean

    for z in tqdm(range(Z), desc="Computing z-score for each Z slice"):
        x_min = index_xmin[z]
        x_max = index_xmax[z] + 1  # +1 for python slice inclusivity

        # mask spatial : voxels en X entre xmin et xmax
        # On traite T, Y et X dans la plage [xmin:xmax)
        # Pour X hors de cette plage on garde processed=0

        # Extract the sub-block for this Z plane over all time and Y:
        subblock = value[:, z, :, x_min:x_max]  # shape (T, Y, xwidth)

        # Threshold mask on this subblock
        mask = subblock >= (std_noise * threshold)

        # Affecter 255 dans processed sur la même zone uniquement où mask est True
        processed[:, z, :, x_min:x_max][mask] = 255

    return processed


def compute_z_score_GPU(
    data: torch.Tensor,  # shape (T, Z, Y, X), dtype=torch.float32
    std_noise: float,
    gaussian_noise_mean: float,
    threshold: float,
    index_xmin: torch.Tensor,  # shape (Z,), dtype=torch.int64
    index_xmax: torch.Tensor   # shape (Z,), dtype=torch.int64
) -> torch.Tensor:
    """
    GPU version of compute_z_score_CPU using PyTorch.

    Args:
        data (torch.Tensor): 4D tensor of shape (T, Z, Y, X), float32, on GPU.
        std_noise (float): Std of Gaussian noise.
        gaussian_noise_mean (float): Mean (or median) of Gaussian noise.
        threshold (float): Z-score threshold.
        index_xmin (torch.Tensor): 1D tensor of cropping bounds for each Z slice.
        index_xmax (torch.Tensor): 1D tensor of cropping bounds (right side, inclusive).
        
    Returns:
        torch.Tensor: Binary z-score volume, dtype=torch.uint8, values in {0, 255}.
    """
    if std_noise <= 0:
        raise ValueError("std_noise must be > 0")

    T, Z, Y, X = data.shape
    processed = torch.zeros_like(data, dtype=torch.uint8, device=data.device)

    # Center the data
    value = data - gaussian_noise_mean

    # Loop over Z is preserved due to variable x_min/x_max per Z
    for z in range(Z):
        x_min = index_xmin[z].item()
        x_max = index_xmax[z].item() + 1  # inclusive range

        # Subblock over (T, Y, xrange)
        subblock = value[:, z, :, x_min:x_max]

        # Mask where z-score is over threshold
        mask = subblock >= (std_noise * threshold)

        # Set mask to 255 in the output tensor
        processed[:, z, :, x_min:x_max][mask] = 255

    return processed


def compute_z_score(data: np.ndarray | torch.Tensor, 
                    std_noise: float, 
                    gaussian_noise_mean: float, 
                    threshold: float, 
                    index_xmin: list | torch.Tensor, 
                    index_xmax: list | torch.Tensor,
                    GPU_AVAILABLE: bool = False) -> np.ndarray | torch.Tensor:
    """
    Compute the z-score for each voxel in the 3D image sequence across the time dimension.
    
    Args:
        data (np.ndarray | torch.Tensor): 4D array or tensor of shape (T, Z, Y, X).
        std_noise (float): Standard deviation of the noise level.
        gaussian_noise_mean (float): Mean of the Gaussian noise.
        threshold (float): Threshold value for z-score.
        index_xmin (list | torch.Tensor): Cropping bounds (left) for each Z slice.
        index_xmax (list | torch.Tensor): Cropping bounds (right) for each Z slice.
        GPU_AVAILABLE (bool): Whether to use GPU for computation.
        
    Returns:
        np.ndarray | torch.Tensor: Binary z-score volume with values in {0, 255}.
    """
    if GPU_AVAILABLE:
        if not torch.is_tensor(data):
            raise TypeError("When GPU is available, data must be a torch.Tensor.")
        return compute_z_score_GPU(data, std_noise, gaussian_noise_mean, threshold, index_xmin, index_xmax)
    else:
        if not isinstance(data, np.ndarray):
            raise TypeError("When GPU is not available, data must be a numpy.ndarray.")
        return compute_z_score_CPU(data, std_noise, gaussian_noise_mean, threshold, index_xmin, index_xmax)