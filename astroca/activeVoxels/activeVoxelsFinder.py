"""
@file activeVoxelsFinder.py
@brief This module provides functionality to find active voxels in a 3D+time image sequence.
"""

import numpy as np
from astroca.activeVoxels.zScore import compute_z_score
from astroca.activeVoxels.spaceMorphology import fill_space_morphology, apply_median_filter
import os
from astroca.tools.exportData import export_data

def find_active_voxels(dF: np.ndarray, std_noise: float, gaussian_noise_mean: float, threshold: float, radius: int = 1, size_median_filter: int = 2, save_results: bool = False, output_directory: str = None) -> np.ndarray:
    """
    @brief Find active voxels in a 3D+time image sequence based on z-score thresholding.

    @param dF: 4D numpy array of shape (T, Z, Y, X) representing the image sequence.
    @param std_noise: Standard deviation of the noise level to normalize the z-score.
    @param gaussian_noise_mean: Mean (or median) of the Gaussian noise, used to center the z-score calculation.
    @param threshold: Threshold value to determine significant deviations in the z-score.
    @param radius: Radius of the ball-like morphology to use for filling.
    @param size_median_filter: Size of the median filter to apply for smoothing the data.
    @param save_results: Boolean flag to indicate whether to save the results.
    @param output_directory: Directory to save the results if save_results is True.
    @return: 4D numpy array of active voxels with the same shape as input data, where active voxels are marked as dF value and inactive as 0.
    @raise ValueError: If the input data is not a 4D numpy array or if the standard deviation of noise is not greater than zero.
    """
    if dF.ndim != 4:
        raise ValueError("Input must be a 4D numpy array of shape (T, Z, Y, X).")

    data = compute_z_score(dF, std_noise, gaussian_noise_mean, threshold)
    if save_results:
        if output_directory is None:
            raise ValueError("Output directory must be specified when save_results is True.")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        export_data(data, output_directory, export_as_single_tif=True, file_name="zScore")
    # data = apply_median_filter(data, size=size_median_filter)
    # if save_results:
    #     export_data(data, output_directory, export_as_single_tif=True, file_name="medianFiltered_1")
    data = fill_space_morphology(data, radius)
    if save_results:
        export_data(data, output_directory, export_as_single_tif=True, file_name="filledSpaceMorphology")
    data = apply_median_filter(data, size=size_median_filter)
    if save_results:
        export_data(data, output_directory, export_as_single_tif=True, file_name="medianFiltered_2")
    active_voxels = np.where(data > 0, dF, 0)  # Keep original dF values for active voxels, set inactive to 0
    if save_results:
        export_data(active_voxels, output_directory, export_as_single_tif=True, file_name="activeVoxels")


    return active_voxels