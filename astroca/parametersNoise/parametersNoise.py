"""
@file parametersNoise.py
@brief Estimate the standard deviation of the noise in a 3D+time image sequence.
@detail Computes a per-voxel STD over time and estimates the global noise level as the median of non-zero voxels in the map.
"""
import numpy as np
from tqdm import tqdm

def compute_pseudo_residuals(data: np.ndarray) -> np.ndarray:
    """
    r(t) = (data[t] - data[t-1]) / sqrt(2)
    """
    return (np.diff(data, axis=0) / np.sqrt(2)).astype(np.float32)

def mad_with_pseudo_residual(residuals: np.ndarray) -> np.ndarray:
    """
    Robust noise estimation: sigma = 1.4826 * median(|residuals|),
    excluding residuals that are zero.
    """
    abs_res = np.abs(residuals)
    abs_res[abs_res == 0.0] = np.nan  # Ignore zeros using NaN
    return 1.4826 * np.nanmedian(abs_res, axis=0)

def estimate_std_map_over_time(data: np.ndarray, xmin: np.ndarray, xmax: np.ndarray) -> np.ndarray:
    """
    For each voxel (x,y,z), compute the MAD-based std estimation.
    @param data: 4D numpy array of shape (T, Z, Y, X) representing the image sequence.
    @param xmin: 1D array of cropping bounds (left) for each Z slice.
    @param xmax: 1D array of cropping bounds (right) for each Z slice.
    @return: 3D numpy array of shape (Z, Y, X) containing the estimated standard deviation for each voxel.
    """
    T, Z, Y, X = data.shape
    residuals = compute_pseudo_residuals(data)  # Shape: (T-1, Z, Y, X)
    std_map = np.full((Z, Y, X), np.nan, dtype=np.float32)
    
    for z in tqdm(range(Z), desc="Estimating std over time", unit="slice"):
        x0, x1 = xmin[z], xmax[z] + 1
        if x0 >= x1: 
            continue
        res_slice = residuals[:, z, :, x0:x1]  # (T-1, Y, Xroi)
        std_map[z, :, x0:x1] = mad_with_pseudo_residual(res_slice)
    
    return std_map

def estimate_std_over_time(data: np.ndarray, xmin: np.ndarray, xmax: np.ndarray) -> float:
    """
    Final std estimation as the median of the 3D map excluding zeros.
    @param data: 4D numpy array of shape (T, Z, Y, X) representing the image sequence.
    @param xmin: 1D array of cropping bounds (left) for each Z slice.
    @param xmax: 1D array of cropping bounds (right) for each Z slice.
    @return : Estimated standard deviation of the noise over time.
    """
    print(" - Estimating standard deviation over time ...")
    std_map = estimate_std_map_over_time(data, xmin, xmax)
    valid = std_map[~np.isnan(std_map) & (std_map > 0)]
    std = float(np.median(valid)) if valid.size else 0.0
    print(f" std_noise = {std:.7f}")
    print(60*"=")
    print()
    return std