"""
@file activeVoxelsFinder.py
@brief This module provides functionality to find active voxels in a 3D+time image sequence.
"""

import numpy as np
from astroca.activeVoxels.zScore import compute_z_score
from astroca.activeVoxels.spaceMorphology import *
from astroca.activeVoxels.medianFilter import *
import os
from astroca.tools.exportData import export_data

def find_active_voxels(dF: np.ndarray | torch.Tensor, std_noise: float, gaussian_noise_mean: float, index_xmin: list, index_xmax: list, params_values: dict) -> np.ndarray | torch.Tensor:
    """
    @brief Find active voxels in a 3D+time image sequence based on z-score thresholding.

    @param dF: 4D numpy array of shape (T, Z, Y, X) representing the image sequence.
    @param std_noise: Standard deviation of the noise level to normalize the z-score.
    @param gaussian_noise_mean: Mean (or median) of the Gaussian noise, used to center the z-score calculation.
    @param index_xmin: 1D array of cropping bounds (left) for each Z slice.
    @param index_xmax: 1D array of cropping bounds (right) for each Z slice.
    @param params_values: Dictionary containing the parameters:
        - 'size_median_filter': Size of the median filter to apply.
        - 'border_condition': Border condition for the median filter.
        - 'threshold_zscore': Z-score threshold for determining active voxels.
        - 'radius_closing_morphology': Radius for the closing morphology operation.
        - 'save_results': Boolean indicating whether to save the results.
        - 'output_directory': Directory to save the results if save_results is True.
    @return: 4D numpy array of active voxels with the same shape as input data, where active voxels are marked as dF value and inactive as 0.
    @raise ValueError: If the input data is not a 4D numpy array or if the standard deviation of noise is not greater than zero.
    """
    print("=== Finding active voxels in the 3D+time image sequence ===")
    if dF.ndim != 4:
        raise ValueError("Input must be a 4D numpy array of shape (T, Z, Y, X).")

    required_keys = {'active_voxels', 'save', 'paths'}
    if not required_keys.issubset(params_values.keys()):
        raise ValueError(f"Missing required parameters: {required_keys - params_values.keys()}")
    save_results = int(params_values['save']['save_av']) == 1
    output_directory = params_values['paths']['output_dir']
    threshold = float(params_values['active_voxels']['threshold_zscore'])
    radius = int(params_values['active_voxels']['radius_closing_morphology'])
    size_median_filter = float(params_values['active_voxels']['median_size'])
    border_condition = params_values['active_voxels']['border_condition']

    if int(params_values['GPU_AVAILABLE']) == 1:
        GPU_AVAILABLE = True
    else:
        GPU_AVAILABLE = False

    data = compute_z_score(dF, std_noise, gaussian_noise_mean, threshold, index_xmin, index_xmax, GPU_AVAILABLE)
    if save_results:
        if output_directory is None:
            raise ValueError("Output directory must be specified when save_results is True.")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        # convert data to np.ndarray if it's a torch.Tensor
        if isinstance(data, torch.Tensor):
            data_to_export = data.cpu().numpy()
        else:
            data_to_export = data
        export_data(data_to_export, output_directory, export_as_single_tif=True, file_name="zScore")
    print()

    data = closing_morphology_in_space(data, radius, border_condition, GPU_AVAILABLE)
    if save_results:
        if isinstance(data, torch.Tensor):
            data_to_export = data.cpu().numpy()
        else:
            data_to_export = data
        export_data(data_to_export, output_directory, export_as_single_tif=True, file_name="filledSpaceMorphology")
    print()

    data = unified_median_filter_3d(data, size_median_filter, border_condition, use_gpu=GPU_AVAILABLE)
    if save_results:
        if isinstance(data, torch.Tensor):
            data_to_export = data.cpu().numpy()
        else:
            data_to_export = data
        export_data(data_to_export, output_directory, export_as_single_tif=True, file_name="medianFiltered_2")
    print()

    active_voxels = voxels_finder(data, dF, std_noise, index_xmin, index_xmax, GPU_AVAILABLE)
    if save_results:
        if isinstance(active_voxels, torch.Tensor):
            active_voxels_to_export = active_voxels.cpu().numpy()
        else:
            active_voxels_to_export = active_voxels
        export_data(active_voxels_to_export, output_directory, export_as_single_tif=True, file_name="activeVoxels")

    print(60 * "=")
    print()
    return active_voxels



def voxels_finder(filtered_data: np.ndarray | torch.Tensor, dF: np.ndarray | torch.Tensor, std_noise: float, index_xmin: np.ndarray, index_xmax: np.ndarray, use_gpu: bool=False) -> np.ndarray | torch.Tensor:
    """
    @brief Determine active voxels based on the value of the filtered data.
    If data(x,t) > 0, then active_voxels(x,t) = dF(x,t);
    if data(x,t) < 0, then active_voxels(x,t) = std_noise;
    otherwise, active_voxels(x,t) = 0.

    This version delegates to CPU or GPU based on `use_gpu`.
    """
    if use_gpu:
        return voxels_finder_GPU(filtered_data, dF, std_noise, index_xmin, index_xmax)
    else:
        return voxels_finder_CPU(filtered_data, dF, std_noise, index_xmin, index_xmax)



def voxels_finder_CPU(filtered_data: np.ndarray, dF: np.ndarray, std_noise: float, index_xmin: list, index_xmax: list) -> np.ndarray:
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
    positive_mask = (filtered_data != 0) & (dF > 0)
    # negative_mask: dF(x,t) < 0 and filtered_data(x,t) > 0
    negative_mask = (filtered_data != 0) & (dF <= 0)
    # null_mask: remaining voxels where filtered_data(x,t) <= 0

    T, Z, Y, X = dF.shape

    active_voxels[positive_mask] = dF[positive_mask]
    active_voxels[negative_mask] = std_noise
    # Cropping the active voxels based on index_xmin and index_xmax
    for z in tqdm(range(Z), desc="Cropping active voxels", unit="slice"):
        active_voxels[:, z, :, :index_xmin[z]] = 0
        active_voxels[:, z, :, index_xmax[z]+1:] = 0
    return active_voxels


def voxels_finder_GPU(filtered_data: torch.Tensor, dF: torch.Tensor, std_noise: float, index_xmin: list, index_xmax: list) -> torch.Tensor:
    print(" - Finding active voxels (GPU)...")
    if filtered_data.ndim != 4 or dF.ndim != 4:
        raise ValueError("Input must be a 4D torch.Tensor of shape (T, Z, Y, X).")

    T, Z, Y, X = dF.shape
    device = dF.device

    filtered_data = filtered_data.to(device)

    # std_noise en tensor GPU
    std_noise_tensor = torch.tensor(std_noise, dtype=dF.dtype, device=device)

    # Masques
    positive_mask = (filtered_data != 0) & (dF > 0)
    negative_mask = (filtered_data != 0) & (dF <= 0)

    active_voxels = torch.zeros_like(dF)

    active_voxels[positive_mask] = dF[positive_mask]
    active_voxels[negative_mask] = std_noise_tensor

    # Crop en GPU
    for z in tqdm(range(Z), desc="Cropping active voxels (GPU)", unit="slice"):
        if index_xmin[z] > 0:
            active_voxels[:, z, :, :index_xmin[z]] = 0
        if index_xmax[z] < X - 1:
            active_voxels[:, z, :, index_xmax[z] + 1:] = 0

    return active_voxels


