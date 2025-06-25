"""
@file activeVoxelsFinder.py
@brief This module provides functionality to find active voxels in a 3D+time image sequence.
"""

import numpy as np
from astroca.activeVoxels.zScore import compute_z_score
from astroca.activeVoxels.spaceMorphology import *
from astroca.activeVoxels.testMedianFilter import *
import os
from astroca.tools.exportData import export_data

def find_active_voxels(dF: np.ndarray, std_noise: float, gaussian_noise_mean: float, index_xmin: list, index_xmax: list, params_values: dict, save_results: bool = False, output_directory: str = None) -> np.ndarray:
    """
    @brief Find active voxels in a 3D+time image sequence based on z-score thresholding.

    @param dF: 4D numpy array of shape (T, Z, Y, X) representing the image sequence.
    @param std_noise: Standard deviation of the noise level to normalize the z-score.
    @param gaussian_noise_mean: Mean (or median) of the Gaussian noise, used to center the z-score calculation.
    @param index_xmin: 1D array of cropping bounds (left) for each Z slice.
    @param index_xmax: 1D array of cropping bounds (right) for each Z slice.
    @param params_values: Dictionary containing parameters for feature computation.
    @param save_results: Boolean flag to indicate whether to save the results.
    @param output_directory: Directory to save the results if save_results is True.
    @return: 4D numpy array of active voxels with the same shape as input data, where active voxels are marked as dF value and inactive as 0.
    @raise ValueError: If the input data is not a 4D numpy array or if the standard deviation of noise is not greater than zero.
    """
    print("=== Finding active voxels in the 3D+time image sequence ===")
    if dF.ndim != 4:
        raise ValueError("Input must be a 4D numpy array of shape (T, Z, Y, X).")

    if len(params_values) != 4:
        raise ValueError("params_values must contain exactly 4 parameters: 'size_median_filter', 'border_condition', 'threshold', and 'radius'.")
    threshold = float(params_values['threshold_zscore'])
    radius = int(params_values['radius_closing_morphology'])
    size_median_filter = float(params_values['median_size'])
    border_condition = params_values['border_condition']

    data = compute_z_score(dF, std_noise, gaussian_noise_mean, threshold, index_xmin, index_xmax)
    if save_results:
        if output_directory is None:
            raise ValueError("Output directory must be specified when save_results is True.")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        export_data(data, output_directory, export_as_single_tif=True, file_name="zScore")
    print()

    data = closing_morphology_in_space(data, radius, border_condition)
    if save_results:
        export_data(data, output_directory, export_as_single_tif=True, file_name="filledSpaceMorphology")
    print()
    # data = apply_median_filter_3d_per_time(data, size=size_median_filter)
    # data = apply_median_filter_spherical_numba(data, radius=size_median_filter, border_condition=border_condition)
    # data = median_filter_3d(data, 1.5, border_condition)
    # data = median_3d_for_4d_stack(data, size_median_filter, n_workers=8)
    data = unified_median_filter_3d(data, size_median_filter)

    # data = apply_median_filter_4d(data)   nul
    # data = apply_median_filter_4d_parallel(data)
    if save_results:
        export_data(data, output_directory, export_as_single_tif=True, file_name="medianFiltered_2")
    print()
    # vox(x,t) > 0 -> active_vox(x,t) = dF(x,t)
    # vox(x,t) = 0 -> active_vox(x,t) = 0
    # vox(x,t) < 0 -> active_vox(x,t) = std_noise
    active_voxels = voxels_finder(data, dF, std_noise, index_xmin, index_xmax)
    if save_results:
        export_data(active_voxels, output_directory, export_as_single_tif=True, file_name="activeVoxels")

    print(60 * "=")
    print()
    return active_voxels


def voxels_finder(filtered_data: np.ndarray, dF: np.ndarray, std_noise: float, index_xmin: list, index_xmax: list) -> np.ndarray:
    """
    @brief Determine active voxels based on the value of the filtered data. 
    If data(x,t) > 0, then active_voxels(x,t) = dF(x,t); 
    if data(x,t) < 0, then active_voxels(x,t) = std_noise; 
    otherwise, active_voxels(x,t) = 0.
    @param filtered_data: 4D numpy array of shape (T, Z, Y, X) representing the filtered data.
    @param dF: 4D numpy array of shape (T, Z, Y, X) representing the dynamic image.
    @param std_noise: Standard deviation of the noise level.
    @param index_xmin: 1D array of cropping bounds (left) for each Z slice.
    @param index_xmax: 1D array of cropping bounds (right) for each Z slice.
    @return: 4D numpy array of active voxels with the same shape as input data, where active voxels are marked as dF value and inactive as 0.
    """
    print(" - Finding active voxels...")
    if filtered_data.ndim != 4 or dF.ndim != 4:
        raise ValueError("Input must be a 4D numpy array of shape (T, Z, Y, X).")

    active_voxels = np.zeros_like(dF)

    # positive_mask: dF(x,t) > 0 and filtered_data(x,t) > 0
    positive_mask = (filtered_data > 0) & (dF >= 0)
    # negative_mask: dF(x,t) < 0 and filtered_data(x,t) > 0
    negative_mask = (filtered_data > 0) & (dF < 0)
    # null_mask: remaining voxels where filtered_data(x,t) <= 0
    null_mask = filtered_data <= 0

    T, Z, Y, X = dF.shape

    active_voxels[positive_mask] = dF[positive_mask]
    active_voxels[negative_mask] = std_noise
    active_voxels[null_mask] = 0
    # Cropping the active voxels based on index_xmin and index_xmax
    for z in tqdm(range(Z), desc="Cropping active voxels", unit="slice"):
        active_voxels[:, z, :, index_xmin[z]:index_xmax[z]] = active_voxels[:, z, :, index_xmin[z]:index_xmax[z]]

    return active_voxels
