"""
@filename: zScore.py
@brief: This module provides functionality to compute the z-score of a 3D image sequence with time dimension.
@detail: Computes a z-score for each voxel in the 3D image sequence across the time dimension to identify significant deviations from the noise level.
"""


import numpy as np
from astroca.tools.scene import ImageSequence3DPlusTime

def compute_z_score(data: np.ndarray, std_noise: float, gaussian_noise_mean: float, threshold: float) -> np.ndarray:
    """
    @brief Compute the z-score for each voxel in the 3D image sequence across the time dimension.
    Zscore(vox, t) = data(vox, t) - gaussian_noise_mean / std_noise

    @param data: 4D numpy array of shape (T, Z, Y, X) representing the image sequence.
    @param std_noise: Standard deviation of the noise level to normalize the z-score.
    @param gaussian_noise_mean: Mean (or Med depending on the previous code) of the Gaussian noise, used to center the z-score calculation.
    @param threshold: Threshold value to determine significant deviations in the z-score.
    @return: 4D numpy array of z-scores with the same shape as input data.
    """
    if std_noise <= 0:
        raise ValueError("Standard deviation of noise must be greater than zero.")

    z_scores = (data - gaussian_noise_mean) / std_noise

    # Apply thresholding (values below threshold become 0)
    thresholded_z_scores = np.where(z_scores >= threshold, 255, 0)

    print(
        f"Computed z-scores with shape: {thresholded_z_scores.shape}, "
        f"threshold: {threshold}, std_noise: {std_noise}, "
        f"gaussian_noise_mean: {gaussian_noise_mean}"
    )

    return thresholded_z_scores.astype(np.uint8)  # Convert to uint8 for image representation
