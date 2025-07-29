"""
@file backgroundEstimator
@brief Module fot computing the background F0 estimation over time
@detail Applies a moving mean or median filter to estimate the fluorescence baseline (F0) and supports min or percentile aggregation.
"""



# from joblib import Parallel, delayed
from astroca.tools.exportData import export_data
from astroca.tools.medianComputationTools import quickselect_median, quickselect_kth
import os
import numpy as np
import time
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
from numba import njit, prange



# @profile
def background_estimation_single_block(data: np.ndarray,
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


@njit(parallel=True)
def process_windowed_data_numba(windowed, moving_window, num_iter, Y, X_roi, method2, method, k):
    """
    @brief Numba-optimized core computation for windowed data processing
    @param windowed: Shape (moving_window, num_iter, Y, X_roi)
    @param moving_window: Size of moving window
    @param num_iter: Number of iterations
    @param Y: Height dimension
    @param X_roi: Width of ROI
    @param method2: Aggregation method ('Mean' = 0, 'Med' = 1)
    @param method: Final method ('min' = 0, 'percentile' = 1)
    @param k: k-th element for percentile (0-indexed)
    @return: Result array of shape (Y, X_roi)
    """
    result = np.zeros((Y, X_roi), dtype=np.float32)
    
    # Process each pixel independently with parallel loops
    for y in prange(Y):
        for x in prange(X_roi):
            # Extract time series for this pixel across all windows
            temp_array = np.zeros(moving_window, dtype=np.float32)
            moving_vals = np.zeros(num_iter, dtype=np.float32)
            
            # For each window position
            for iter_idx in range(num_iter):
                # Extract values for this window
                for w in range(moving_window):
                    temp_array[w] = windowed[w, iter_idx, y, x]
                
                # Apply method2 (Mean or Median)
                if method2 == 0:  # Mean
                    moving_vals[iter_idx] = np.mean(temp_array)
                else:  # Median
                    # Create a copy for quickselect (it modifies the array)
                    temp_copy = temp_array.copy()
                    moving_vals[iter_idx] = quickselect_median(temp_copy, moving_window)
            
            # Apply final method (min or percentile)
            if method == 0:  # min
                result[y, x] = np.min(moving_vals)
            else:  # percentile
                # Create a copy for quickselect
                moving_copy = moving_vals.copy()
                result[y, x] = quickselect_kth(moving_copy, num_iter, k)
    
    return result

def background_estimation_single_block_numba(data: np.ndarray,
                                           index_xmin: np.ndarray,
                                           index_xmax: np.ndarray,
                                           params_values: dict) -> np.ndarray:
    """
    Version ultra-optimisée avec Numba pour l'estimation du background F0.
    Utilise les fonctions quickselect personnalisées pour des gains de performance maximaux.
    
    @param data: 4D image sequence (T, Z, Y, X)
    @param index_xmin: Array of cropping bounds (left) for each Z
    @param index_xmax: Array of cropping bounds (right) for each Z
    @param params_values: Dictionary containing the parameters
    @return: Background array of shape (1, Z, Y, X)
    """
    print("=== Fluorescence baseline F0 estimation (NUMBA OPTIMIZED) ===")
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

    # Convert string parameters to integers for Numba
    method2_int = 0 if method2 == "Mean" else 1  # 0 = Mean, 1 = Med
    method_int = 0 if method == "min" else 1     # 0 = min, 1 = percentile
    
    # Pre-calculate k for percentiles (convert to 0-indexed)
    k = 0
    if method == "percentile":
        k = int(np.ceil((percentile / 100.0) * num_iter))
        k = np.clip(k, 1, num_iter) - 1  # Convert to 0-indexed

    for z in tqdm(range(Z), desc="Estimating background per Z-slice", unit="slice"):
        x_min = int(index_xmin[z])
        x_max = int(index_xmax[z])

        if not (0 <= x_min < x_max < X):
            continue

        roi = data[:, z, :, x_min:x_max + 1].astype(np.float32)  # shape: (T, Y, X_roi)
        X_roi = roi.shape[2]
        
        # Create sliding windows
        windowed = sliding_window_view(roi, window_shape=moving_window, axis=0)  # shape: (num_iter, Y, X_roi, moving_window)
        
        # Move the window dimension to the first axis for Numba processing
        windowed = np.moveaxis(windowed, -1, 0)  # shape: (moving_window, num_iter, Y, X_roi)
        
        # Use Numba-optimized processing
        result = process_windowed_data_numba(
            windowed, moving_window, num_iter, Y, X_roi, 
            method2_int, method_int, k
        )
        
        F0[0, z, :, x_min:x_max + 1] = result

    if save_results:
        if output_directory is None:
            raise ValueError("Output directory must be specified.")
        os.makedirs(output_directory, exist_ok=True)
        export_data(F0, output_directory, export_as_single_tif=True, file_name="F0_estimated_numba")

    print(60*"=")
    print()

    return F0