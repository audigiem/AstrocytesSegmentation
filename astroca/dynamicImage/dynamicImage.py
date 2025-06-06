"""
@file dynamicImage.py
@brief Module for computing the dynamic image (ΔF = F - F0) and the background F0 estimation over time.
@detail Applies a moving mean or median filter to estimate the fluorescence baseline (F0) and supports min or percentile aggregation.
"""

from astroca.tools.scene import ImageSequence3DPlusTime

# from joblib import Parallel, delayed

from numba import njit, prange

import numpy as np
import time
from numpy.lib.stride_tricks import sliding_window_view

def background_estimation_numpy(image_sequence: 'ImageSequence3DPlusTime',
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
    @param method: Aggregation method ('min' or 'percentile')
    @param method2: Secondary aggregation method ('Mean' or 'Med')
    @param percentile: Percentile value for 'percentile' method (default 10.0)
    @return: Background array of shape (nbF0, Z, Y, X)
    """
    start_time = time.time()
    print("Estimating background F0...")

    # Parameters validation
    if method not in ["min", "percentile"]:
        raise ValueError("method must be 'min' or 'percentile'")
    if method2 not in ["Mean", "Med"]:
        raise ValueError("method2 must be 'Mean' or 'Med'")
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")

    data = image_sequence.get_data()  # (T, Z, Y, X)
    T, Z, Y, X = data.shape

    # Dimension validation
    if len(index_xmin) != Z or len(index_xmax) != Z:
        raise ValueError("index_xmin and index_xmax must have length Z")

    nbF0 = max(T // time_window, 1)
    F0 = np.zeros((nbF0, Z, Y, X), dtype=np.float32)

    # Precompute valid z indices and crop bounds to avoid repeated checks
    valid_z_indices = []
    crop_bounds = []
    for z in range(Z):
        x_min, x_max = int(index_xmin[z]), int(index_xmax[z]) + 1
        if x_min < x_max and x_min >= 0 and x_max <= X:
            valid_z_indices.append(z)
            crop_bounds.append((x_min, x_max))

    print(f"Processing {nbF0} temporal blocks of size {time_window} with moving window {moving_window}")

    # Process temporal blocks
    for it in range(nbF0):
        t_start = it * time_window
        t_end = min(t_start + time_window, T)  # Handle the last block
        block = data[t_start:t_end]  # shape (actual_time_window, Z, Y, X)

        print(f"Processing block {it + 1}/{nbF0}, time range [{t_start}:{t_end}], block shape: {block.shape}")

        # Special case: if block is too small for sliding window
        if block.shape[0] < moving_window:
            print(f"Warning: Block {it} has only {block.shape[0]} frames, "
                  f"less than moving_window={moving_window}")
            # Use all available frames in the block
            for idx, z in enumerate(valid_z_indices):
                x_min, x_max = crop_bounds[idx]
                sub = block[:, z, :, x_min:x_max]

                if method2 == "Mean":
                    F0[it, z, :, x_min:x_max] = np.mean(sub, axis=0)
                else:  # method2 == "Med"
                    F0[it, z, :, x_min:x_max] = np.median(sub, axis=0)
            continue

        # Normal processing with sliding window
        for idx, z in enumerate(valid_z_indices):
            x_min, x_max = crop_bounds[idx]
            sub = block[:, z, :, x_min:x_max]  # shape (tw, Y, X')

            # Sliding window on temporal axis (axis=0)
            sw = sliding_window_view(sub, window_shape=(moving_window,), axis=0)
            # sw.shape: (n_windows, Y, X', moving_window)

            # Temporal aggregation within each window (on last axis)
            if method2 == "Mean":
                values = np.mean(sw, axis=-1)  # shape (n_windows, Y, X')
            else:  # method2 == "Med"
                values = np.median(sw, axis=-1)  # shape (n_windows, Y, X')

            # Final aggregation over windows (axis=0)
            if method == "min":
                result = np.min(values, axis=0)  # shape (Y, X')
            else:  # method == "percentile"
                result = np.percentile(values, percentile, axis=0)  # shape (Y, X')

            # Dimension check before assignment
            expected_shape = (Y, x_max - x_min)
            if result.shape != expected_shape:
                print(f"Warning: result shape {result.shape} != expected {expected_shape}")
                print(f"sub.shape: {sub.shape}, sw.shape: {sw.shape}, values.shape: {values.shape}")
                continue

            F0[it, z, :, x_min:x_max] = result

    elapsed_time = time.time() - start_time
    print(f"Background F0 estimated in {elapsed_time:.2f} seconds.")
    print()
    return F0


def background_estimation_single_block(image_sequence: 'ImageSequence3DPlusTime',
                                       index_xmin: np.ndarray,
                                       index_xmax: np.ndarray,
                                       moving_window: int,
                                       method: str = "percentile",
                                       method2: str = "Mean",
                                       percentile: float = 10.0) -> np.ndarray:
    """
    @brief Estimate the background F0 using entire time sequence as single block.
    Version optimisée NumPy avec le même comportement que Java.
    @param image_sequence: 4D image sequence (T, Z, Y, X)
    @param index_xmin: Array of cropping bounds (left) for each Z
    @param index_xmax: Array of cropping bounds (right) for each Z
    @param moving_window: Size of the moving window for aggregation
    @param method: Aggregation method ('min' or 'percentile')
    @param method2: Secondary aggregation method ('Mean' or 'Med')
    @param percentile: Percentile value for 'percentile' method (default 10.0)
    @return: Background array of shape (1, Z, Y, X)
    """
    start_time = time.time()
    print("Estimating background F0 (single block mode, optimized)...")

    # Parameters validation
    if method not in ["min", "percentile"]:
        raise ValueError("method must be 'min' or 'percentile'")
    if method2 not in ["Mean", "Med"]:
        raise ValueError("method2 must be 'Mean' or 'Med'")
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")

    data = image_sequence.get_data()  # (T, Z, Y, X)
    T, Z, Y, X = data.shape

    # Dimension validation
    if len(index_xmin) != Z or len(index_xmax) != Z:
        raise ValueError("index_xmin and index_xmax must have length Z")

    if T < moving_window:
        raise ValueError(f"Time sequence length {T} is less than moving_window {moving_window}")

    F0 = np.zeros((1, Z, Y, X), dtype=np.float32)

    # Calcul du nombre d'itérations comme en Java
    time_window = T
    num_iter = time_window - moving_window + 1

    print(f"Processing entire sequence of {T} frames with moving window {moving_window}")
    print(f"Number of iterations: {num_iter}")

    # Traitement optimisé par slice Z
    for z in range(Z):
        x_min = int(index_xmin[z])
        x_max = int(index_xmax[z])

        # Vérification des bornes comme en Java
        if x_min >= x_max or x_min < 0 or x_max >= X:
            continue

        # Extraire la région d'intérêt pour ce Z
        # Inclure x_max comme en Java (+1)
        roi_data = data[:, z, :, x_min:x_max + 1]  # Shape: (T, Y, X_roi)

        # Créer une vue de fenêtre glissante sur l'axe temporel
        # roi_data.shape = (T, Y, X_roi)
        # On veut une fenêtre glissante sur l'axe 0 (temps)
        windowed_data = np.lib.stride_tricks.sliding_window_view(
            roi_data, window_shape=moving_window, axis=0
        )  # Shape: (num_iter, Y, X_roi, moving_window)

        # Calcul de la moyenne ou médiane mobile
        if method2 == "Mean":
            # Moyenne sur la dernière dimension (fenêtre temporelle)
            moving_values = np.mean(windowed_data, axis=-1)  # Shape: (num_iter, Y, X_roi)
        else:  # method2 == "Med"
            # Médiane sur la dernière dimension (fenêtre temporelle)
            moving_values = np.median(windowed_data, axis=-1)  # Shape: (num_iter, Y, X_roi)

        # Agrégation finale selon la méthode
        if method == "min":
            # Minimum sur l'axe des itérations
            result = np.min(moving_values, axis=0)  # Shape: (Y, X_roi)
        else:  # method == "percentile"
            # Tri sur l'axe des itérations pour chaque pixel
            sorted_values = np.sort(moving_values, axis=0)  # Shape: (num_iter, Y, X_roi)

            # Calcul de l'index du percentile comme en Java avec Math.ceil
            index_percentile = int(np.ceil(percentile / 100.0 * num_iter))
            # Assurer que l'index est au moins 1 (comme le ceil en Java)
            index_percentile = max(1, index_percentile)

            # Sélectionner le percentile (indexation 0-based donc -1)
            result = sorted_values[index_percentile - 1]  # Shape: (Y, X_roi)

        # Assigner le résultat à la région correspondante
        F0[0, z, :, x_min:x_max + 1] = result

    elapsed_time = time.time() - start_time
    print(f"Background F0 estimated in {elapsed_time:.2f} seconds.")
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

