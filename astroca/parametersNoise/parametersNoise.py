"""
@file parametersNoise.py
@brief Estimate the standard deviation of the noise in a 3D+time image sequence.
@detail Computes a per-voxel STD over time and estimates the global noise level as the median of non-zero voxels in the map.
"""
import numpy as np
from tqdm import tqdm
from numba import njit, prange

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

# @profile
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

@njit
def quickselect_median_exact(arr, n):
    """
    Version exacte de médiane garantissant la même sortie que np.nanmedian.
    Utilise un tri complet pour garantir la précision.
    """
    if n == 0:
        return np.nan
    if n == 1:
        return arr[0]
    
    # Tri complet pour garantir la même précision que NumPy
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    
    if n % 2 == 1:
        return arr[n // 2]
    else:
        return (arr[n // 2 - 1] + arr[n // 2]) / 2.0


@njit(parallel=True)
def mad_with_pseudo_residual_numba(residuals):
    """
    Version Numba optimisée de mad_with_pseudo_residual.
    EXACTEMENT équivalente à la version NumPy originale.
    
    @param residuals: Array de shape (T-1, Y, X_roi)
    @return: Array de shape (Y, X_roi) avec les estimations MAD
    """
    T_minus_1, Y, X_roi = residuals.shape
    result = np.zeros((Y, X_roi), dtype=np.float32)
    
    # Traitement parallèle de chaque pixel
    for y in prange(Y):
        for x in prange(X_roi):
            # Créer un array temporaire pour stocker les valeurs absolues
            temp_array = np.zeros(T_minus_1, dtype=np.float32)
            
            # Remplir avec les valeurs absolues (comme np.abs)
            for t in range(T_minus_1):
                temp_array[t] = abs(residuals[t, y, x])
            
            # Compter les valeurs non-nulles pour reproduire exactement np.nanmedian
            valid_count = 0
            for t in range(T_minus_1):
                if temp_array[t] != 0.0:
                    valid_count += 1
            
            # Si aucune valeur valide, retourner NaN
            if valid_count == 0:
                result[y, x] = np.nan
            else:
                # Créer un array compact avec seulement les valeurs non-nulles
                valid_array = np.zeros(valid_count, dtype=np.float32)
                idx = 0
                for t in range(T_minus_1):
                    if temp_array[t] != 0.0:
                        valid_array[idx] = temp_array[t]
                        idx += 1
                
                # Calculer la médiane avec la version exacte
                median_val = quickselect_median_exact(valid_array, valid_count)
                result[y, x] = np.float32(1.4826 * median_val)
    
    return result

def compute_pseudo_residuals_optimized(data: np.ndarray) -> np.ndarray:
    """
    Version optimisée du calcul des résidus pseudo.
    Utilise une approche in-place quand possible.
    """
    # La fonction originale est déjà assez optimisée
    # Le coût principal vient de l'allocation mémoire et de la division
    return (np.diff(data, axis=0) / np.sqrt(2)).astype(np.float32)

def estimate_std_map_over_time_optimized(data: np.ndarray, xmin: np.ndarray, xmax: np.ndarray) -> np.ndarray:
    """
    Version optimisée de estimate_std_map_over_time utilisant Numba.
    
    @param data: 4D numpy array of shape (T, Z, Y, X) representing the image sequence.
    @param xmin: 1D array of cropping bounds (left) for each Z slice.
    @param xmax: 1D array of cropping bounds (right) for each Z slice.
    @return: 3D numpy array of shape (Z, Y, X) containing the estimated standard deviation for each voxel.
    """
    T, Z, Y, X = data.shape
    residuals = compute_pseudo_residuals_optimized(data)  # Shape: (T-1, Z, Y, X)
    std_map = np.full((Z, Y, X), np.nan, dtype=np.float32)
    
    for z in tqdm(range(Z), desc="Estimating std over time (OPTIMIZED)", unit="slice"):
        x0, x1 = int(xmin[z]), int(xmax[z]) + 1
        if x0 >= x1: 
            continue
        
        res_slice = residuals[:, z, :, x0:x1]  # (T-1, Y, Xroi)
        
        # Utiliser la version Numba optimisée
        std_result = mad_with_pseudo_residual_numba(res_slice)
        std_map[z, :, x0:x1] = std_result
    
    return std_map

def estimate_std_over_time_optimized(data: np.ndarray, xmin: np.ndarray, xmax: np.ndarray) -> float:
    """
    Version optimisée de estimate_std_over_time.
    
    @param data: 4D numpy array of shape (T, Z, Y, X) representing the image sequence.
    @param xmin: 1D array of cropping bounds (left) for each Z slice.
    @param xmax: 1D array of cropping bounds (right) for each Z slice.
    @return : Estimated standard deviation of the noise over time.
    """
    print(" - Estimating standard deviation over time (OPTIMIZED) ...")
    std_map = estimate_std_map_over_time_optimized(data, xmin, xmax)
    valid = std_map[~np.isnan(std_map) & (std_map > 0)]
    std = float(np.median(valid)) if valid.size else 0.0
    print(f" std_noise = {std:.7f}")
    print(60*"=")
    print()
    return std