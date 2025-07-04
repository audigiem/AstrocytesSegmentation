"""
@file backgroundEstimator
@brief Module fot computing the background F0 estimation over time
@detail Applies a moving mean or median filter to estimate the fluorescence baseline (F0) and supports min or percentile aggregation.
"""



# from joblib import Parallel, delayed
from astroca.tools.exportData import export_data
from numba import njit, prange
import os
import numpy as np
import time
from numpy.lib.stride_tricks import sliding_window_view
from astroca.varianceStabilization.varianceStabilization import anscombe_inverse
from tqdm import tqdm




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
    required_keys = {'background_estimation', 'files', 'paths'}
    if not required_keys.issubset(params_values.keys()):
        raise ValueError(f"Missing required parameters: {required_keys - params_values.keys()}")

    moving_window = int(params_values['background_estimation']['moving_window'])
    method = params_values['background_estimation']['method']
    method2 = params_values['background_estimation']['method2']
    percentile = float(params_values['background_estimation']['percentile'])
    save_results = int(params_values['files']['save_results']) == 1
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


def background_estimation_single_block_ultra_optimized(image_sequence: 'ImageSequence3DPlusTime',
                                                       index_xmin: np.ndarray,
                                                       index_xmax: np.ndarray,
                                                       moving_window: int,
                                                       method: str = "percentile",
                                                       method2: str = "Mean",
                                                       percentile: float = 10.0) -> np.ndarray:
    """
    Version ultra-optimisée qui traite tous les Z valides en une seule opération vectorisée.
    """
    start_time = time.time()
    print("Estimating background F0 (ultra-optimized mode)...")

    # Parameters validation
    if method not in ["min", "percentile"]:
        raise ValueError("method must be 'min' or 'percentile'")
    if method2 not in ["Mean", "Med"]:
        raise ValueError("method2 must be 'Mean' or 'Med'")
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")

    data = image_sequence.get_data()  # (T, Z, Y, X)
    T, Z, Y, X = data.shape

    if T < moving_window:
        raise ValueError(f"Time sequence length {T} is less than moving_window {moving_window}")

    F0 = np.zeros((1, Z, Y, X), dtype=np.float32)
    num_iter = T - moving_window + 1

    print(f"Processing entire sequence of {T} frames with moving window {moving_window}")

    # Créer un masque pour les régions valides
    mask = np.zeros((Z, Y, X), dtype=bool)
    for z in range(Z):
        x_min = int(index_xmin[z])
        x_max = int(index_xmax[z])
        if x_min < x_max and x_min >= 0 and x_max < X:
            mask[z, :, x_min:x_max + 1] = True

    # Appliquer le masque aux données
    if np.any(mask):
        # Fenêtre glissante sur toutes les données d'un coup
        windowed_data = np.lib.stride_tricks.sliding_window_view(
            data, window_shape=moving_window, axis=0
        )  # Shape: (num_iter, Z, Y, X, moving_window)

        # Calcul vectorisé de la moyenne/médiane mobile
        if method2 == "Mean":
            moving_values = np.mean(windowed_data, axis=-1)  # Shape: (num_iter, Z, Y, X)
        else:  # method2 == "Med"
            moving_values = np.median(windowed_data, axis=-1)  # Shape: (num_iter, Z, Y, X)

        # Agrégation finale
        if method == "min":
            result = np.min(moving_values, axis=0)  # Shape: (Z, Y, X)
        else:  # method == "percentile"
            sorted_values = np.sort(moving_values, axis=0)  # Shape: (num_iter, Z, Y, X)
            index_percentile = max(1, int(np.ceil(percentile / 100.0 * num_iter)))
            result = sorted_values[index_percentile - 1]  # Shape: (Z, Y, X)

        # Appliquer le masque au résultat
        F0[0] = np.where(mask, result, 0)

    elapsed_time = time.time() - start_time
    print(f"Background F0 estimated in {elapsed_time:.2f} seconds.")
    print()
    return F0


@njit(parallel=True)
def background_estimation_numba(data: np.ndarray,
                                 index_xmin: np.ndarray,
                                 index_xmax: np.ndarray,
                                 moving_window: int,
                                 time_window: int,
                                 method: str = "percentile",
                                 method2: str = "Mean",
                                 percentile: float = 10.0) -> np.ndarray:
    """
    @brief Estimate the background F0 using a moving window approach with Numba for performance.
    @param data: 4D image sequence (T, Z, Y, X)
    @param index_xmin: Array of cropping bounds (left) for each Z
    @param index_xmax: Array of cropping bounds (right) for each Z
    @param moving_window: Size of the moving window for aggregation
    @param time_window: Duration of each background block in time
    @param method: Aggregation method, either 'min' or 'percentile'
    @param method2: Secondary aggregation method, either 'Mean' or 'Med'
    @param percentile: Percentile value for 'percentile' method (default 10.0)
    @return: Background array of shape (nbF0, Z, Y, X)
    """
    start_time = time.time()
    print("Estimating background F0 with Numba...")
    time_length, depth, height, width = data.shape
    nbF0 = max(time_length // time_window, 1)
    size = width * height * depth
    F0 = np.zeros((nbF0, size), dtype=np.float32)

    num_iter = np.full(nbF0, time_window - moving_window + 1)
    num_iter[-1] = time_length - (nbF0 - 1) * time_window - moving_window + 1

    for z in prange(depth):
        for y in range(height):
            for x in range(index_xmin[z], index_xmax[z] + 1):
                index = z + (x + y * width) * depth

                for it in range(nbF0):
                    t_offset = it * time_window
                    val3 = 0.0
                    tabMovingVal = np.empty(num_iter[it], dtype=np.float32)

                    for j in range(num_iter[it]):
                        if method2 == "Med":
                            tabMed = np.empty(moving_window, dtype=np.float32)
                            for i in range(moving_window):
                                tabMed[i] = data[t_offset + j + i, index]
                            val = np.median(tabMed)
                            if method == "min":
                                tabMovingVal[j] = val
                            else:
                                tabMovingVal[j] = val

                        elif method2 == "Mean":
                            if j == 0:
                                val3 = 0.0
                                for i in range(moving_window):
                                    val3 += data[t_offset + i, index]
                            else:
                                val3 = val3 - data[t_offset + j - 1, index] + data[t_offset + j - 1 + moving_window, index]

                            if method == "min":
                                tabMovingVal[j] = val3 / moving_window
                            else:
                                tabMovingVal[j] = val3 / moving_window

                    if method == "min":
                        F0[it, index] = np.min(tabMovingVal)
                    elif method == "percentile":
                        k = int(np.ceil(percentile / 100.0 * num_iter[it])) - 1
                        F0[it, index] = np.partition(tabMovingVal, k)[k]
    print(f"Background F0 estimated with Numba in {time.time() - start_time:.2f} seconds.")
    return F0
