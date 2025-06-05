"""
@file dynamicImage.py
@brief Module for computing the dynamic image (ΔF = F - F0) and the background F0 estimation over time.
@detail Applies a moving mean or median filter to estimate the fluorescence baseline (F0) and supports min or percentile aggregation.
"""

import numpy as np
from astroca.init.scene import ImageSequence3DPlusTime


from joblib import Parallel, delayed
import time

from numba import njit, prange
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def background_estimation_numpy(image_sequence: ImageSequence3DPlusTime,
                                 index_xmin: np.ndarray,
                                 index_xmax: np.ndarray,
                                 moving_window: int,
                                 time_window: int,
                                 method: str = "percentile",
                                 method2: str = "Mean",
                                 percentile: float = 10.0) -> np.ndarray:
    """
    @brief Estimate the background F0 using a moving window approach.
    @param image_sequence: 4D image sequence (T, Z, Y, X)
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
    print("Estimating background F0...")
    data = image_sequence.get_data()  # (T, Z, Y, X)
    T, Z, Y, X = data.shape
    nbF0 = max(T // time_window, 1)
    F0 = np.zeros((nbF0, Z, Y, X), dtype=np.float32)

    for it in range(nbF0):
        t_start = it * time_window
        block = data[t_start:t_start + time_window]  # shape (time_window, Z, Y, X)

        for z in range(Z):
            x_min, x_max = index_xmin[z], index_xmax[z] + 1
            if x_min >= x_max:
                continue

            sub = block[:, z, :, x_min:x_max]  # shape (tw, Y, X')

            # sliding windows on time axis
            sw = sliding_window_view(sub, window_shape=(moving_window,), axis=0)  # shape (n_iter, moving_window, Y, X')

            if method2 == "Mean":
                values = sw.mean(axis=1)  # shape (n_iter, Y, X')
            elif method2 == "Med":
                values = np.median(sw, axis=1)
            else:
                raise ValueError("method2 must be 'Mean' or 'Med'")

            # Assure alignement Y, X'
            if method == "min":
                F0[it, z, :, x_min:x_max] = np.min(values, axis=0)
            elif method == "percentile":
                k = int(np.ceil(percentile / 100.0 * values.shape[0])) - 1
                F0[it, z, :, x_min:x_max] = np.partition(values, k, axis=0)[k]
            else:
                raise ValueError("method must be 'min' or 'percentile'")
            
    
    print(f"Background F0 estimated in {time.time() - start_time:.2f} seconds.")
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


def compute_dynamic_image(image_sequence: ImageSequence3DPlusTime,
                          F0: np.ndarray,
                          index_xmin: np.ndarray,
                          index_xmax: np.ndarray,
                          time_window: int) -> tuple[np.ndarray, float]:
    """
    @brief Compute ΔF = F - F0 and estimate the noise level as the median of ΔF.

    @param image_sequence: 4D image sequence (T, Z, Y, X)
    @param F0: Background array of shape (nbF0, Z, Y, X)
    @param index_xmin: cropping bounds in X for each Z
    @param index_xmax: cropping bounds in X for each Z
    @param time_window: the duration of each background block
    @return: (dF: array of shape (T, Z, Y, X), mean_noise: float)
    """
    print("Computing dynamic image (dF = F - F0) and estimating noise...")
    data = image_sequence.get_data()  # shape (T, Z, Y, X)
    T, Z, Y, X = data.shape
    nbF0 = F0.shape[0]

    width_without_zeros = sum(index_xmax[z] - index_xmin[z] + 1 for z in range(Z))

    flattened_dF = np.empty((T * Y * width_without_zeros), dtype=np.float32)
    k = 0

    dF = np.copy(data)

    for t in range(T):
        it = min(t // time_window, nbF0 - 1)
        for z in range(Z):
            x_min, x_max = index_xmin[z], index_xmax[z] + 1
            if x_min >= x_max:
                continue
            for y in range(Y):
                # slice (X,)
                delta = dF[t, z, y, x_min:x_max] - F0[it, z, y, x_min:x_max]
                dF[t, z, y, x_min:x_max] = delta
                flattened_dF[k:k + x_max - x_min] = delta
                k += x_max - x_min

    mean_noise = float(np.median(flattened_dF[:k]))
    print(f"mean_Noise = {mean_noise}")
    print(f"Dynamic image computed.")
    print()

    return dF, mean_noise

