"""
@file medianFilter.py
@brief 3D median filter for 4D stacks (T,Z,Y,X) with spherical neighborhood and border handling
GPU and CPU versions with identical results
"""
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from astroca.activeVoxels.medianFilterGPU import unified_median_filter_3d_gpu
from tqdm import tqdm
from scipy.ndimage import median_filter
from numba import njit, prange
import torch


def unified_median_filter_3d(
        data: np.ndarray | torch.Tensor,
        radius: float = 1.5,
        border_mode: str = 'reflect',
        n_workers: int = None,
        use_gpu: bool = False,
) -> torch.Tensor | np.ndarray:
    """
    @brief Unified 3D median filter for 4D stacks (T,Z,Y,X)

    @details Applies a 3D median filter to each frame of a 4D stack using spherical neighborhood.
    Supports both CPU (NumPy) and GPU (PyTorch) implementations with identical results.

    @param data: Input 4D stack (T,Z,Y,X) - np.ndarray for CPU, torch.Tensor for GPU
    @param radius: Radius of the spherical neighborhood (1.5 → 3×3×7 neighborhood)
    @param border_mode: Border handling mode: 'reflect', 'nearest', 'constant', 'ignore', etc.
    @param n_workers: Number of threads to use for CPU version (None for automatic)
    @param use_gpu: If True, use GPU implementation, else CPU

    @return Filtered 4D stack as torch.Tensor
    """
    if use_gpu:
        return unified_median_filter_3d_gpu(data, radius, border_mode)
    else:
        return unified_median_filter_3d_cpu(data, radius, border_mode, n_workers)


def unified_median_filter_3d_cpu(
        data: np.ndarray,
        radius: float = 1.5,
        border_mode: str = 'reflect',
        n_workers: int = None
) -> np.ndarray:
    """
    @brief CPU version of 3D median filter (original implementation)
    """
    print(f" - Apply 3D median filter (CPU) with radius={radius}, border mode='{border_mode}'")
    
    if border_mode == 'ignore':
        T, Z, Y, X = data.shape
        data_3D = data.reshape(T * Z, Y, X)  # Reshape to treat as 3D
        offsets = generate_spherical_offsets(radius)
        median_filtered = apply_median_filter_3d_ignore_border(data_3D, offsets)
        data_filtered_4D = median_filtered.reshape(T, Z, Y, X)
        return data_filtered_4D
    
    r = int(np.ceil(radius))

    # Create spherical mask
    shape = (2 * r + 1, 2 * r + 1, 2 * r + 1)
    mask = np.zeros(shape, dtype=bool)
    center = np.array([r, r, r])
    for idx in np.ndindex(shape):
        if np.linalg.norm(np.array(idx) - center) <= radius:
            mask[idx] = True

    # Manual padding: only on Z, Y, X axes
    pad_width = [(0, 0), (r, r), (r, r), (r, r)]
    padded = np.pad(data, pad_width=pad_width, mode=border_mode)

    filtered = np.empty_like(data)

    def process_frame(t):
        """Process a single frame with median filter"""
        result = median_filter(
            padded[t], footprint=mask, mode=border_mode
        )
        # Remove padding
        filtered[t] = result[r:-r, r:-r, r:-r]

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        list(tqdm(
            executor.map(process_frame, range(data.shape[0])),
            total=data.shape[0], desc=f"Processing frames (CPU) with median filter and {border_mode} border condition",
            unit="frame"
        ))

    return filtered




# Original CPU functions (unchanged)
@njit
def quickselect_median(arr, n):
    """
    @brief Optimized median calculation for small arrays

    @details Uses quickselect to find median element without full sorting.
    Falls back to insertion sort for very small arrays (n ≤ 20).

    @param arr Input array (will be modified)
    @param n Number of elements to consider in array

    @return Median value
    """
    if n == 0:
        return 0
    if n == 1:
        return arr[0]

    # For small arrays, insertion sort is faster
    if n <= 20:
        # Insertion sort
        for i in range(1, n):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key

        if n % 2 == 1:
            return arr[n // 2]
        else:
            return (arr[n // 2 - 1] + arr[n // 2]) / 2.0

    # For larger arrays, use quickselect
    # But in practice, our spherical neighborhoods are small
    return quickselect_kth(arr, n, n // 2)


@njit
def quickselect_kth(arr, n, k):
    """
    @brief Finds the k-th smallest element (0-indexed)

    @param arr Input array (will be modified)
    @param n Number of elements to consider
    @param k Index of desired element (0 = smallest)

    @return The k-th smallest element
    """
    left = 0
    right = n - 1

    while left < right:
        pivot_idx = partition(arr, left, right)
        if pivot_idx == k:
            return arr[k]
        elif pivot_idx < k:
            left = pivot_idx + 1
        else:
            right = pivot_idx - 1

    return arr[k]


@njit
def partition(arr, left, right):
    """
    @brief Partition function for quickselect

    @param arr Array to partition
    @param left Left index
    @param right Right index (pivot position)

    @return Final pivot position
    """
    pivot = arr[right]
    i = left - 1

    for j in range(left, right):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[right] = arr[right], arr[i + 1]
    return i + 1


@njit(parallel=True)
def apply_median_filter_3d_ignore_border(frame: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """
    @brief Applies 3D median filter while ignoring borders

    @details Optimized version using Numba parallel processing. Only considers valid
    neighbors within image bounds when computing median.

    @param frame Input 3D frame (Z,Y,X)
    @param offsets Array of (dz,dy,dx) offsets defining spherical neighborhood

    @return Filtered 3D frame
    """
    Z, Y, X = frame.shape
    result = np.empty((Z, Y, X), dtype=frame.dtype)
    max_neighbors = offsets.shape[0]

    # for z in range(Z):
    for z in prange(Z):
        for y in range(Y):
            for x in range(X):
                # Temporary array for each thread
                tmp_values = np.empty(max_neighbors, dtype=frame.dtype)
                count = 0

                # Collect valid values in spherical neighborhood
                for i in range(offsets.shape[0]):
                    dz, dy, dx = offsets[i]
                    zz, yy, xx = z + dz, y + dy, x + dx

                    if 0 <= zz < Z and 0 <= yy < Y and 0 <= xx < X:
                        tmp_values[count] = frame[zz, yy, xx]
                        count += 1

                # Compute median over valid values
                if count > 0:
                    result[z, y, x] = quickselect_median(tmp_values, count)
                else:
                    # Fallback: keep original value if no valid neighbors
                    result[z, y, x] = frame[z, y, x]

    return result


def generate_spherical_offsets(radius: float):
    """
    @brief Generates offsets for spherical neighborhood of given radius

    @details Uses exact algorithm from Java code with normalized ellipsoid.
    Returns array of (dz,dy,dx) offsets where distance ≤ radius.

    @param radius Filter radius (spherical)

    @return Array of integer offsets defining spherical neighborhood
    """
    radx = rady = radz = radius  # Sphere = ellipsoid with equal radii

    vx = int(np.ceil(radx))
    vy = int(np.ceil(rady))
    vz = int(np.ceil(radz))

    # Calculate inverse squared radii (as in Java)
    rx2 = 1.0 / (radx * radx) if radx != 0.0 else 0.0
    ry2 = 1.0 / (rady * rady) if rady != 0.0 else 0.0
    rz2 = 1.0 / (radz * radz) if radz != 0.0 else 0.0

    offsets = []

    # Loops in same order as Java: k(z), j(y), i(x)
    for k in range(-vz, vz + 1):  # dz
        for j in range(-vy, vy + 1):  # dy
            for i in range(-vx, vx + 1):  # dx
                # Normalized distance from Java
                dist = (i * i) * rx2 + (j * j) * ry2 + (k * k) * rz2

                if dist <= 1.0:  # Exact condition from Java
                    offsets.append((k, j, i))  # (dz, dy, dx)

    offsets_array = np.array(offsets, dtype=np.int32)
    return offsets_array