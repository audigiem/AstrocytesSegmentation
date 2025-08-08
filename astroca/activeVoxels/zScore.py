"""
@filename: zScore.py
@brief: This module provides functionality to compute the z-score of a 3D
image sequence with time dimension.
@detail: Computes a z-score for each voxel in the 3D image sequence across
the time dimension to identify significant deviations from the noise level.
"""


import numpy as np
from tqdm import tqdm


def compute_z_score(
    data: np.ndarray,
    std_noise: float,
    gaussian_noise_mean: float,
    threshold: float,
    index_xmin: list,
    index_xmax: list,
) -> np.ndarray:
    """
    @brief Compute the z-score for each voxel in the 3D image sequence across
    the time dimension.
    Zscore(vox, t) = data(vox, t) - gaussian_noise_mean / std_noise

    @param data: 4D numpy array of shape (T, Z, Y, X) representing the image sequence.
    @param std_noise: Standard deviation of the noise level to normalize the z-score.
    @param gaussian_noise_mean: Mean (or Med depending on the previous code)
    of the Gaussian noise, used to center the z-score calculation.
    @param threshold: Threshold value to determine significant
    deviations in the z-score.
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

    for z in tqdm(range(Z), desc="Computing z-score for each Z slice", unit="slice"):
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
