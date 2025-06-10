"""
@file parametersNoise.py
@brief Estimate the standard deviation of the noise in a 3D+time image sequence.
@detail Computes a per-voxel STD over time and estimates the global noise level as the median of non-zero voxels in the map.
"""

import numpy as np
import time

from astroca.tools.scene import ImageSequence3DPlusTime


# import tifffile or np.save for optional saving


def estimate_std_over_time(data: np.ndarray, xmin: np.ndarray, xmax: np.ndarray) -> float:
    """
    Compute the std over time for each voxel, excluding empty bands in X.

    @param data: 4D array of shape (T, Z, Y, X)
    @param xmin: Array of cropping bounds (left) for each Z
    @param xmax: Array of cropping bounds (right) for each Z
    @return: float median of the standard deviation map across all valid voxels.
    """
    print("Estimating standard deviation over time...")

    # for z in range(Z):
    #     x_min, x_max = xmin[z], xmax[z] + 1
    #     if x_min >= x_max:
    #         continue
    #     for y in range(Y):
    #         std_map[z, y, x_min:x_max] = np.std(data[:, z, y, x_min:x_max], axis=0)

    # find the median of non-zero voxels
    valid_voxels = data[data > 0.0]
    std_noise = float(np.median(valid_voxels)) if valid_voxels.size > 0 else 0.0
    print(f"Estimated std over time: {std_noise:.6f}")
    return std_noise

