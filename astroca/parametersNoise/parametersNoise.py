"""
@file parametersNoise.py
@brief Estimate the standard deviation of the noise in a 3D+time image sequence.
@detail Computes a per-voxel STD over time and estimates the global noise level as the median of non-zero voxels in the map.
"""
import numpy as np
from tqdm import tqdm
from numba import njit, prange
from astroca.parametersNoise.parametersNoiseGPU import estimate_std_over_time_GPU
from astroca.tools.exportData import export_data
import torch


# @profile
def compute_pseudo_residuals(data: np.ndarray) -> np.ndarray:
    """
    r(t) = (data[t] - data[t-1]) / sqrt(2)
    """
    return (np.diff(data, axis=0) / np.sqrt(2)).astype(np.float32)


# @profile
def mad_with_pseudo_residual(residuals: np.ndarray) -> np.ndarray:
    """
    Robust noise estimation: sigma = 1.4826 * median(|residuals|),
    excluding residuals that are zero.
    """
    abs_res = np.abs(residuals)
    abs_res[abs_res == 0.0] = np.nan  # Ignore zeros using NaN
    return 1.4826 * np.nanmedian(abs_res, axis=0)


# @profile
def estimate_std_map_over_time(
    data: np.ndarray, xmin: np.ndarray, xmax: np.ndarray
) -> np.ndarray:
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


# @profile
def estimate_std_over_time(
    data: np.ndarray, xmin: np.ndarray, xmax: np.ndarray
) -> float:
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
    print(f"    std_noise = {std:.7f}")
    print(60 * "=")
    print()
    return std


@njit
def quickselect_median_exact(arr, n):
    """
    @fn quickselect_median_exact
    @brief Exact median version ensuring same output as np.nanmedian.
    @param arr Array to compute median from
    @param n Number of valid elements
    @return Median value
    """
    if n == 0:
        return np.nan
    if n == 1:
        return arr[0]

    # Créer une copie pour ne pas modifier l'original
    sorted_arr = np.zeros(n, dtype=np.float32)
    for i in range(n):
        sorted_arr[i] = arr[i]

    # Tri complet pour garantir la même précision que NumPy
    for i in range(1, n):
        key = sorted_arr[i]
        j = i - 1
        while j >= 0 and sorted_arr[j] > key:
            sorted_arr[j + 1] = sorted_arr[j]
            j -= 1
        sorted_arr[j + 1] = key

    if n % 2 == 1:
        return sorted_arr[n // 2]
    else:
        return (sorted_arr[n // 2 - 1] + sorted_arr[n // 2]) / 2.0


@njit(parallel=True)
def mad_with_pseudo_residual_numba(residuals):
    """
    @fn mad_with_pseudo_residual_numba
    @brief Numba optimized version of mad_with_pseudo_residual.
    @param residuals Array of shape (T-1, Y, X_roi)
    @return Array of shape (Y, X_roi) with MAD estimations
    """
    T_minus_1, Y, X_roi = residuals.shape
    result = np.zeros((Y, X_roi), dtype=np.float32)

    # Traitement parallèle de chaque pixel
    for y in prange(Y):
        for x in prange(X_roi):
            # Compter d'abord les valeurs non-nulles
            valid_count = 0
            for t in range(T_minus_1):
                abs_val = abs(residuals[t, y, x])
                if abs_val != 0.0:
                    valid_count += 1

            # Si aucune valeur valide, retourner NaN
            if valid_count == 0:
                result[y, x] = np.nan
            else:
                # Créer un array avec seulement les valeurs absolues non-nulles
                valid_array = np.zeros(valid_count, dtype=np.float32)
                idx = 0
                for t in range(T_minus_1):
                    abs_val = abs(residuals[t, y, x])
                    if abs_val != 0.0:
                        valid_array[idx] = abs_val
                        idx += 1

                # Calculer la médiane avec la version exacte
                median_val = quickselect_median_exact(valid_array, valid_count)
                result[y, x] = np.float32(1.4826 * median_val)

    return result


def estimate_std_map_over_time_optimized(
    data: np.ndarray, xmin: np.ndarray, xmax: np.ndarray
) -> np.ndarray:
    """
    @fn estimate_std_map_over_time_optimized
    @brief Optimized version of estimate_std_map_over_time using Numba.
    @param data 4D numpy array of shape (T, Z, Y, X) representing the image sequence.
    @param xmin 1D array of cropping bounds (left) for each Z slice.
    @param xmax 1D array of cropping bounds (right) for each Z slice.
    @return 3D numpy array of shape (Z, Y, X) containing the estimated standard deviation for each voxel.
    """
    T, Z, Y, X = data.shape
    residuals = compute_pseudo_residuals(data)  # Utiliser la fonction originale
    std_map = np.full((Z, Y, X), np.nan, dtype=np.float32)

    for z in tqdm(range(Z), desc="Estimating std over time", unit="slice"):
        x0, x1 = int(xmin[z]), int(xmax[z]) + 1
        if x0 >= x1:
            continue

        res_slice = residuals[:, z, :, x0:x1]  # (T-1, Y, Xroi)

        # S'assurer que res_slice est contigu en mémoire pour Numba
        if not res_slice.flags["C_CONTIGUOUS"]:
            res_slice = np.ascontiguousarray(res_slice)

        # Utiliser la version Numba optimisée
        std_result = mad_with_pseudo_residual_numba(res_slice)
        std_map[z, :, x0:x1] = std_result

    return std_map


def estimate_std_over_time_optimized(
    data: np.ndarray, xmin: np.ndarray, xmax: np.ndarray
) -> float:
    """
    @fn estimate_std_over_time_optimized
    @brief Optimized version of estimate_std_over_time.
    @param data 4D numpy array of shape (T, Z, Y, X) representing the image sequence.
    @param xmin 1D array of cropping bounds (left) for each Z slice.
    @param xmax 1D array of cropping bounds (right) for each Z slice.
    @return Estimated standard deviation of the noise over time.
    """
    print(" - Estimating standard deviation over time (Optim) ...")
    std_map = estimate_std_map_over_time_optimized(data, xmin, xmax)
    valid = std_map[~np.isnan(std_map) & (std_map > 0)]
    std = float(np.median(valid)) if valid.size else 0.0
    print(f"    std_noise = {std:.7f}")
    print(60 * "=")
    print()
    return std


def estimate_std_over_time(
    data: np.ndarray | torch.Tensor,
    xmin: np.ndarray,
    xmax: np.ndarray,
    GPU_AVAILABLE: bool = False,
) -> float:
    """
    Wrapper function to estimate the standard deviation of the noise over time.
    Chooses between CPU and GPU implementations based on the data type.

    @param data: 4D numpy array or torch tensor of shape (T, Z, Y, X) representing the image sequence.
    @param xmin: 1D array of cropping bounds (left) for each Z slice.
    @param xmax: 1D array of cropping bounds (right) for each Z slice.
    @return: Estimated standard deviation of the noise over time.
    """
    if GPU_AVAILABLE:
        if not isinstance(data, torch.Tensor):
            raise TypeError("When GPU is available, data must be a torch.Tensor.")
        return estimate_std_over_time_GPU(data, xmin, xmax)

    else:
        if not isinstance(data, np.ndarray):
            raise TypeError("When GPU is not available, data must be a numpy.ndarray.")
        return estimate_std_over_time_optimized(data, xmin, xmax)
