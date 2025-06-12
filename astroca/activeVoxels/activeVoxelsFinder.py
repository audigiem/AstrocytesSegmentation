"""
@file activeVoxelsFinder.py
@brief This module provides functionality to find active voxels in a 3D+time image sequence.
"""

import numpy as np
from astroca.activeVoxels.zScore import compute_z_score
from astroca.activeVoxels.spaceMorphology import fill_space_morphology, apply_median_filter_3d_per_time, apply_median_filter_spherical, apply_median_filter_spherical_fast, apply_median_filter_spherical_numba
import os
from astroca.tools.exportData import export_data

def find_active_voxels(dF: np.ndarray, std_noise: float, gaussian_noise_mean: float, threshold: float, index_xmin: list, index_xmax: list, radius: tuple, size_median_filter: int = 2, save_results: bool = False, output_directory: str = None) -> np.ndarray:
    """
    @brief Find active voxels in a 3D+time image sequence based on z-score thresholding.

    @param dF: 4D numpy array of shape (T, Z, Y, X) representing the image sequence.
    @param std_noise: Standard deviation of the noise level to normalize the z-score.
    @param gaussian_noise_mean: Mean (or median) of the Gaussian noise, used to center the z-score calculation.
    @param threshold: Threshold value to determine significant deviations in the z-score.
    @param index_xmin: 1D array of cropping bounds (left) for each Z slice.
    @param index_xmax: 1D array of cropping bounds (right) for each Z slice.
    @param radius: Tuple specifying the radius for morphological operations (e.g., (1, 1, 1) for 3D).
    @param size_median_filter: Size of the median filter to apply for smoothing the data.
    @param save_results: Boolean flag to indicate whether to save the results.
    @param output_directory: Directory to save the results if save_results is True.
    @return: 4D numpy array of active voxels with the same shape as input data, where active voxels are marked as dF value and inactive as 0.
    @raise ValueError: If the input data is not a 4D numpy array or if the standard deviation of noise is not greater than zero.
    """
    if dF.ndim != 4:
        raise ValueError("Input must be a 4D numpy array of shape (T, Z, Y, X).")

    data = compute_z_score(dF, std_noise, gaussian_noise_mean, threshold, index_xmin, index_xmax)
    if save_results:
        if output_directory is None:
            raise ValueError("Output directory must be specified when save_results is True.")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        export_data(data, output_directory, export_as_single_tif=True, file_name="zScore")
    print()
    # data = apply_median_filter(data, size=size_median_filter)
    # if save_results:
    #     export_data(data, output_directory, export_as_single_tif=True, file_name="medianFiltered_1")
    data = fill_space_morphology(data, radius)
    if save_results:
        export_data(data, output_directory, export_as_single_tif=True, file_name="filledSpaceMorphology")
    print()
    # data = apply_median_filter_3d_per_time(data, size=size_median_filter)
    # data = apply_median_filter_spherical(data)
    # data = apply_median_filter_spherical_fast(data)
    data = apply_median_filter_spherical_numba(data)
    if save_results:
        export_data(data, output_directory, export_as_single_tif=True, file_name="medianFiltered_2")
    print()
    # vox(x,t) > 0 -> active_vox(x,t) = dF(x,t)
    # vox(x,t) = 0 -> active_vox(x,t) = 0
    # vox(x,t) < 0 -> active_vox(x,t) = std_noise
    active_voxels = voxels_finder(data, dF, std_noise, index_xmin, index_xmax)
    if save_results:
        export_data(active_voxels, output_directory, export_as_single_tif=True, file_name="activeVoxels")
    print()
    return active_voxels


def voxels_finder(filtered_data: np.ndarray, dF: np.ndarray, std_noise: float, index_xmin: list, index_xmax: list) -> np.ndarray:
    """
    @brief Determine active voxels based on the value of the filtered data. If data(x,t) > 0, then active_voxels(x,t) = dF(x,t); if data(x,t) < 0, then active_voxels(x,t) = std_noise; otherwise, active_voxels(x,t) = 0.
    @param filtered_data: 4D numpy array of shape (T, Z, Y, X) representing the filtered data.
    @param dF: 4D numpy array of shape (T, Z, Y, X) representing the dynamic image.
    @param std_noise: Standard deviation of the noise level.
    @param index_xmin: 1D array of cropping bounds (left) for each Z slice.
    @param index_xmax: 1D array of cropping bounds (right) for each Z slice.
    @return: 4D numpy array of active voxels with the same shape as input data, where active voxels are marked as dF value and inactive as 0.
    """
    print("Finding active voxels...")
    if filtered_data.ndim != 4 or dF.ndim != 4:
        raise ValueError("Input must be a 4D numpy array of shape (T, Z, Y, X).")

    active_voxels = np.zeros_like(dF)

    T, Z, Y, X = filtered_data.shape

    for z in range(Z):
        x_min = index_xmin[z]
        x_max = index_xmax[z] + 1

        filtered_slice = filtered_data[:, z, :, x_min:x_max]  # shape (T, Y, x_max-x_min)
        dF_slice = dF[:, z, :, x_min:x_max]
        active_slice = active_voxels[:, z, :, x_min:x_max]

        mask_nonzero = filtered_slice != 0  # condition processedAV[i][index] != 0
        mask_pos = (filtered_slice > 0) & mask_nonzero
        mask_neg = (filtered_slice < 0) & mask_nonzero

        active_slice[mask_pos] = dF_slice[mask_pos]
        active_slice[mask_neg] = std_noise
    return active_voxels
