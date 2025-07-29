"""
@file medianFilter.py
@brief 3D median filter for 4D stacks (T,Z,Y,X) with spherical neighborhood and border handling
"""
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from astroca.tools.medianComputationTools import quickselect_median
from tqdm import tqdm
from scipy.ndimage import median_filter
from numba import njit, prange


def unified_median_filter_3d(
        data: np.ndarray,
        radius: float = 1.5,
        border_mode: str = 'reflect',
        n_workers: int = None
) -> np.ndarray:
    """
    @brief Unified 3D median filter for 4D stacks (T,Z,Y,X)

    @details Applies a 3D median filter to each frame of a 4D stack using spherical neighborhood.
    Supports multi-threading for improved performance on large datasets.

    @param data: Input 4D stack (T,Z,Y,X)
    @param radius: Radius of the spherical neighborhood (1.5 → 3×3×7 neighborhood)
    @param border_mode: Border handling mode: 'reflect', 'nearest', 'constant', 'ignore', etc.
    @param n_workers: Number of threads to use (None for automatic)

    @return Filtered 4D stack with same dimensions as input
    """
    print(f" - Apply 3D median filter with radius={radius}, border mode='{border_mode}'")
    if border_mode == 'ignore':
        T, Z, Y, X = data.shape
        data_3D = data.reshape(T * Z, Y, X)  # Reshape to treat as 3D
        offsets = generate_spherical_offsets(radius)
        median_filtered = apply_median_filter_3d_ignore_border(data_3D, offsets)
        data_filtered_4D = median_filtered.reshape(T, Z, Y, X)
        return data_filtered_4D
    print(f" - Apply 3D median filter with radius={radius}, border mode='{border_mode}'")
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
        """
        @brief Process a single frame with median filter

        @param t Frame index to process
        """
        result = median_filter(
            padded[t], footprint=mask, mode=border_mode
        )
        # Remove padding
        filtered[t] = result[r:-r, r:-r, r:-r]

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        list(tqdm(
            executor.map(process_frame, range(data.shape[0])),
            total=data.shape[0], desc=f"Processing frames with median filter and {border_mode} border condition",
            unit="frame"
        ))

    return filtered





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
    # print(f"Generated {len(offsets)} offsets for radius {radius}")
    # print(f"Integer bounds: vx={vx}, vy={vy}, vz={vz}")
    # print(f"Normalization factors: rx2={rx2:.6f}, ry2={ry2:.6f}, rz2={rz2:.6f}")
    # print(f"Some offsets: {offsets_array[:10]}")  # Show first few offsets
    return offsets_array