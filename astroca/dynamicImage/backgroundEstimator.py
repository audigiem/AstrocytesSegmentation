"""
@file backgroundEstimator
@brief Module fot computing the background F0 estimation over time
@detail Applies a moving mean or median filter to estimate the fluorescence baseline (F0) and supports min or percentile aggregation.
"""

from astroca.tools.exportData import export_data
import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
import torch

def background_estimation_single_block(data: np.ndarray | torch.Tensor, index_xmin: np.ndarray, index_xmax: np.ndarray, params_values: dict) -> np.ndarray | torch.Tensor:
    """
    Estimate the background F0 using the entire time sequence as a single block.

    @param data: 4D image sequence (T, Z, Y, X)
    @param index_xmin: Array of cropping bounds (left) for each Z
    @param index_xmax: Array of cropping bounds (right) for each Z
    @param params_values: Dictionary containing the parameters:
        - moving_window: Size of the moving window for aggregation
        - method: Aggregation method, either 'min' or 'percentile'
        - method2: Secondary aggregation method, either 'Mean' or 'Med'
        - percentile: Percentile value for 'percentile' method (default 10.0)
    @return: Background array of shape (1, Z, Y, X)
    """
    if int(params_values.get("GPU_AVAILABLE", 0)) == 1:
        return background_estimation_GPU(data, index_xmin, index_xmax, params_values)
    else:
        return background_estimation_CPU(data, index_xmin, index_xmax, params_values)




def background_estimation_CPU(data: np.ndarray,
                                       index_xmin: np.ndarray,
                                       index_xmax: np.ndarray,
                                       params_values: dict) -> np.ndarray:
    """
    Estimate the background F0 using the entire time sequence as a single block.

    @param data: 4D image sequence (T, Z, Y, X)
    @param index_xmin: Array of cropping bounds (left) for each Z
    @param index_xmax: Array of cropping bounds (right) for each Z
    @param params_values: Dictionary containing the parameters:
        - moving_window: Size of the moving window for aggregation
        - method: Aggregation method, either 'min' or 'percentile'
        - method2: Secondary aggregation method, either 'Mean' or 'Med'
        - percentile: Percentile value for 'percentile' method (default 10.0)
    @return: Background array of shape (1, Z, Y, X)
    """
    print("=== Fluorescence baseline F0 estimation ===")
    required_keys = {'background_estimation', 'save', 'paths'}
    if not required_keys.issubset(params_values.keys()):
        raise ValueError(f"Missing required parameters: {required_keys - params_values.keys()}")

    moving_window = int(params_values['background_estimation']['moving_window'])
    method = params_values['background_estimation']['method']
    method2 = params_values['background_estimation']['method2']
    percentile = float(params_values['background_estimation']['percentile'])
    save_results = int(params_values['save']['save_background_estimation']) == 1
    output_directory = params_values['paths']['output_dir']

    if method not in {'min', 'percentile'}:
        raise ValueError("method must be 'min' or 'percentile'")
    if method2 not in {'Mean', 'Med'}:
        raise ValueError("method2 must be 'Mean' or 'Med'")
    if not (0 <= percentile <= 100):
        raise ValueError("percentile must be between 0 and 100")

    T, Z, Y, X = data.shape

    if len(index_xmin) != Z or len(index_xmax) != Z:
        raise ValueError("index_xmin and index_xmax must have length Z")
    if T < moving_window:
        raise ValueError(f"Time sequence too short: T={T}, window={moving_window}")

    num_iter = T - moving_window + 1
    F0 = np.zeros((1, Z, Y, X), dtype=np.float32)

    for z in tqdm(range(Z), desc="Estimating background per Z-slice", unit="slice"):
        x_min = int(index_xmin[z])
        x_max = int(index_xmax[z])

        if not (0 <= x_min < x_max < X):
            continue

        roi = data[:, z, :, x_min:x_max + 1]  # shape: (T, Y, X_roi)
        windowed = sliding_window_view(roi, window_shape=moving_window, axis=0)  # shape: (num_iter, Y, X_roi, moving_window)

        # Move the window dimension to the last axis for easier reduction
        windowed = np.moveaxis(windowed, -1, 0)  # shape: (moving_window, num_iter, Y, X_roi)

        if method2 == "Mean":
            moving_vals = np.mean(windowed, axis=0)  # shape: (num_iter, Y, X_roi)
        else:  # Med
            moving_vals = np.median(windowed, axis=0)

        if method == "min":
            result = np.min(moving_vals, axis=0)
        else:  # percentile
            k = int(np.ceil((percentile / 100.0) * num_iter))
            k = np.clip(k, 1, num_iter)
            sorted_vals = np.sort(moving_vals, axis=0)
            result = sorted_vals[k - 1]  # kth smallest

        F0[0, z, :, x_min:x_max + 1] = result


    if save_results:
        if output_directory is None:
            raise ValueError("Output directory must be specified.")
        os.makedirs(output_directory, exist_ok=True)
        export_data(F0, output_directory, export_as_single_tif=True, file_name="F0_estimated")

    print(60*"=")
    print()

    return F0



def background_estimation_GPU(data: torch.Tensor,
                              index_xmin: np.ndarray,
                              index_xmax: np.ndarray,
                              params_values: dict) -> torch.Tensor:
    """
    Estimate the background F0 using the entire time sequence as a single block on GPU.

    @param data: 4D image sequence (T, Z, Y, X) as a PyTorch tensor
    @param index_xmin: Array of cropping bounds (left) for each Z
    @param index_xmax: Array of cropping bounds (right) for each Z
    @param params_values: Dictionary containing the parameters:
        - moving_window: Size of the moving window for aggregation
        - method: Aggregation method, either 'min' or 'percentile'
        - method2: Secondary aggregation method, either 'Mean' or 'Med'
        - percentile: Percentile value for 'percentile' method (default 10.0)
    @return: Background tensor of shape (1, Z, Y, X)
    """
    print("=== Fluorescence baseline F0 estimation (GPU) ===")

    required_keys = {'background_estimation', 'save', 'paths'}
    if not required_keys.issubset(params_values.keys()):
        raise ValueError(f"Missing required parameters: {required_keys - params_values.keys()}")

    device = data.device
    moving_window = int(params_values['background_estimation']['moving_window'])
    method = params_values['background_estimation']['method']
    method2 = params_values['background_estimation']['method2']
    percentile = float(params_values['background_estimation']['percentile'])
    save_results = int(params_values['save']['save_background_estimation']) == 1
    output_directory = params_values['paths']['output_dir']

    if method not in {'min', 'percentile'}:
        raise ValueError("method must be 'min' or 'percentile'")
    if method2 not in {'Mean', 'Med'}:
        raise ValueError("method2 must be 'Mean' or 'Med'")
    if not (0 <= percentile <= 100):
        raise ValueError("percentile must be between 0 and 100")

    T, Z, Y, X = data.shape
    if len(index_xmin) != Z or len(index_xmax) != Z:
        raise ValueError("index_xmin and index_xmax must have length Z")
    if T < moving_window:
        raise ValueError(f"Time sequence too short: T={T}, window={moving_window}")

    num_iter = T - moving_window + 1
    F0 = torch.zeros((1, Z, Y, X), dtype=torch.float32, device=device)

    for z in tqdm(range(Z), desc="Estimating background per Z-slice (GPU)", unit="slice"):
        x_min = int(index_xmin[z])
        x_max = int(index_xmax[z])

        if not (0 <= x_min < x_max < X):
            continue

        roi = data[:, z, :, x_min:x_max + 1]  # shape: (T, Y, X_roi)

        # Unfold over time (dim 0): (T - w + 1, w, Y, X_roi)
        windowed = roi.unfold(0, moving_window, 1)  # (num_iter, moving_window, Y, X_roi)
        windowed = windowed.permute(1, 0, 2, 3)  # shape: (moving_window, num_iter, Y, X_roi)

        if method2 == "Mean":
            moving_vals = torch.mean(windowed, dim=0)  # (num_iter, Y, X_roi)
        else:  # Med
            moving_vals = torch.median(windowed, dim=0).values  # (num_iter, Y, X_roi)

        if method == "min":
            result = torch.min(moving_vals, dim=0).values  # (Y, X_roi)
        else:  # percentile
            k = int(np.ceil((percentile / 100.0) * num_iter))
            k = np.clip(k, 1, num_iter)
            sorted_vals, _ = torch.sort(moving_vals, dim=0)
            result = sorted_vals[k - 1]  # kth smallest (Y, X_roi)

        # CORRECTION : Assigner directement result qui a la forme (Y, X_roi)
        # à la slice correspondante dans F0
        F0[0, z, :, x_min:x_max + 1] = result

    if save_results:
        if output_directory is None:
            raise ValueError("Output directory must be specified.")
        os.makedirs(output_directory, exist_ok=True)
        F0_cpu = F0.detach().cpu().numpy()
        export_data(F0_cpu, output_directory, export_as_single_tif=True, file_name="F0_estimated")

    print("=" * 60)
    print()
    return F0