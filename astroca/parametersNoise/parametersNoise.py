"""
@file parametersNoise.py
@brief Estimate the standard deviation of the noise in a 3D+time image sequence.
@detail Computes a per-voxel STD over time and estimates the global noise level as the median of non-zero voxels in the map.
"""

import numpy as np
import time
from astroca.init.scene import ImageSequence3DPlusTime


# import tifffile or np.save for optional saving

def estimate_std(image_sequence: ImageSequence3DPlusTime,
                 name: str,
                 index_xmin: np.ndarray,
                 index_xmax: np.ndarray,
                 save: bool = False,
                 save_path: str = None) -> float:
    """
    @brief Estimate the standard deviation of Gaussian noise in a 3D+time image sequence.

    @param image_sequence: 4D image of shape (T, Z, Y, X)
    @param name: Name to display with the result (e.g., 'stdNoise')
    @param index_xmin: Array of cropping bounds (left) for each Z
    @param index_xmax: Array of cropping bounds (right) for each Z
    @param save: If True, saves the noise map
    @param save_path: Path for saving the 3D noise map
    @return: Estimated noise standard deviation
    """
    start = time.time()
    data = image_sequence.get_data()  # shape (T, Z, Y, X)
    T, Z, Y, X = data.shape

    # Step 1: compute std over time for each voxel
    std_map = estimate_std_over_time(data, index_xmin, index_xmax)  # expected shape: (Z, Y, X)

    # Step 2: keep only valid (non-zero) values for median
    valid_voxels = std_map[std_map > 0.0]
    std_noise = float(np.median(valid_voxels)) if valid_voxels.size > 0 else 0.0

    print(f"{name} = {std_noise:.6f}")

    if save and save_path is not None:
        np.savez_compressed(save_path, std_map=std_map)

    duration = (time.time() - start) / 60
    print(f"Duration estimateSTD() = {duration:.6f} min")

    return std_noise


def estimate_std_over_time(data: np.ndarray, xmin: np.ndarray, xmax: np.ndarray) -> float:
    """
    Compute the std over time for each voxel, excluding empty bands in X.

    @param data: 4D array of shape (T, Z, Y, X)
    @param xmin: Array of cropping bounds (left) for each Z
    @param xmax: Array of cropping bounds (right) for each Z
    @return: float median of the standard deviation map across all valid voxels.
    """
    print("Estimating standard deviation over time...")
    T, Z, Y, X = data.shape
    std_map = np.zeros((Z, Y, X), dtype=np.float32)

    for z in range(Z):
        x_min, x_max = xmin[z], xmax[z] + 1
        if x_min >= x_max:
            continue
        for y in range(Y):
            std_map[z, y, x_min:x_max] = np.std(data[:, z, y, x_min:x_max], axis=0)

    # find the median of non-zero voxels
    valid_voxels = std_map[std_map > 0.0]
    std_noise = float(np.median(valid_voxels)) if valid_voxels.size > 0 else 0.0
    print(f"Estimated std over time: {std_noise:.6f}")
    return std_noise

