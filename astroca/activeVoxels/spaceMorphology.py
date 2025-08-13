"""
@file spaceMorphology.py
@brief This module provides functions to fill/connect structures in space using a ball-shaped morphology of radius 1.
"""
from scipy.ndimage import binary_closing
import numpy as np
from skimage.morphology import ball
from numba import njit, prange
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from astroca.tools.medianComputationTools import generate_spherical_offsets
import torch


def closing_morphology_in_space(
    data: np.ndarray | torch.Tensor,
    radius: int,
    border_mode: str = "constant",
    GPU_AVAILABLE: bool = False,
) -> np.ndarray | torch.Tensor:
    if GPU_AVAILABLE:
        return closing_morphology_in_space_GPU(data, radius, border_mode)
    else:
        return closing_morphology_in_space_CPU(data, radius, border_mode)


def closing_morphology_in_space_CPU(
    data: np.ndarray, radius: int, border_mode: str = "constant"
) -> np.ndarray:
    """
    @fn closing_morphology_in_space
    @brief Apply 3D morphological closing with a spherical structuring element to each time frame of a 4D sequence (T, Z, Y, X).
    @param data 4D numpy array (T, Z, Y, X), binary (0/255 or 0/1)
    @param radius Radius of the spherical structuring element
    @param border_mode 'constant', 'reflect', etc. OR 'ignore' to skip voxels near borders
    @return 4D numpy array (T, Z, Y, X) after closing
    """
    if border_mode == "ignore":
        return closing_morphology_in_space_ignore_border_CPU(data, radius)

    print(
        f" - Apply morphological closing with radius={radius} and border mode='{border_mode}'"
    )
    struct_elem = ball(radius)
    result = np.zeros_like(data, dtype=np.uint8)

    # Ensure binary input
    binary = data > 0

    for t in tqdm(range(binary.shape[0]), desc="Morphological closing", unit="frame"):
        pad_width = ((radius, radius), (radius, radius), (radius, radius))
        padded = np.pad(binary[t], pad_width=pad_width, mode=border_mode)
        closed = binary_closing(padded, structure=struct_elem)
        closed = closed[radius:-radius, radius:-radius, radius:-radius]
        result[t] = closed.astype(np.uint8) * 255

    return result


@njit(parallel=True)
def morphological_erosion_ignore_border(
    frame: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    """
    @fn morphological_erosion_ignore_border
    @brief Morphological erosion: take the minimum in the spherical neighborhood. Only considers valid voxels (inside the volume).
    @param frame 3D numpy array (Z, Y, X), binary
    @param offsets Array of spherical offsets
    @return 3D numpy array (Z, Y, X) after erosion
    """
    Z, Y, X = frame.shape
    eroded = np.zeros((Z, Y, X), dtype=np.uint8)

    for z in prange(Z):
        for y in range(Y):
            for x in range(X):
                min_val = 1  # Maximum possible value

                for i in range(offsets.shape[0]):
                    dz, dy, dx = offsets[i]
                    zz, yy, xx = z + dz, y + dy, x + dx

                    if 0 <= zz < Z and 0 <= yy < Y and 0 <= xx < X:
                        if frame[zz, yy, xx] < min_val:
                            min_val = frame[zz, yy, xx]
                    # Voxels outside the volume are ignored

                eroded[z, y, x] = min_val
    return eroded


@njit(parallel=True)
def morphological_dilation_ignore_border(
    frame: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    """
    @fn morphological_dilation_ignore_border
    @brief Morphological dilation: take the maximum in the spherical neighborhood. Only considers valid voxels (inside the volume).
    @param frame 3D numpy array (Z, Y, X), binary
    @param offsets Array of spherical offsets
    @return 3D numpy array (Z, Y, X) after dilation
    """
    Z, Y, X = frame.shape
    dilated = np.zeros((Z, Y, X), dtype=np.uint8)

    for z in prange(Z):
        for y in range(Y):
            for x in range(X):
                max_val = 0  # Minimum possible value

                for i in range(offsets.shape[0]):
                    dz, dy, dx = offsets[i]
                    zz, yy, xx = z + dz, y + dy, x + dx

                    if 0 <= zz < Z and 0 <= yy < Y and 0 <= xx < X:
                        if frame[zz, yy, xx] > max_val:
                            max_val = frame[zz, yy, xx]
                    # Voxels outside the volume are ignored

                dilated[z, y, x] = max_val
    return dilated


def closing_morphology_in_space_ignore_border_CPU(
    data: np.ndarray, radius: int, n_workers: int = None
) -> np.ndarray:
    """
    @fn closing_morphology_in_space_ignore_border
    @brief Fast 3D morphological closing on a 4D binary array (T, Z, Y, X), with border_mode='ignore'.
    @param data 4D numpy array (T, Z, Y, X), binary (0/255 or 0/1)
    @param radius Radius of the spherical structuring element
    @param n_workers Number of parallel workers (default: None)
    @return 4D numpy array (T, Z, Y, X) after closing
    """
    print(
        f" - Fast morphological closing with radius={radius} and border mode='ignore'"
    )

    binary = (data > 0).astype(np.uint8)
    T, Z, Y, X = binary.shape

    result = np.empty((T, Z, Y, X), dtype=np.uint8)
    offsets = generate_spherical_offsets(radius)

    def process(t):
        frame = binary[t]

        closed = morphological_dilation_ignore_border(frame, offsets)

        eroded = morphological_erosion_ignore_border(closed, offsets)

        result[t] = eroded.astype(np.uint8) * 255  # Convert to binary (0/255)

    # Process all frames
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        list(
            tqdm(
                executor.map(process, range(T)),
                total=T,
                desc="Morphological closing (ignore border)",
                unit="frame",
            )
        )

    return result


def closing_morphology_in_space_GPU(
    data: torch.Tensor, radius: int, border_mode: str = "ignore"
) -> torch.Tensor:
    """
    Apply 3D morphological closing (dilation followed by erosion) using PyTorch on GPU.
    Parameters:
        data: 4D torch tensor (T, Z, Y, X), binary (0/1)
        radius: Radius of spherical structuring element
        border_mode: 'constant', 'reflect', 'nearest'
    Returns:
        4D numpy array (T, Z, Y, X), dtype uint8 with values 0 or 255
    """
    if border_mode == "ignore":
        return closing_morphology_in_space_ignore_border_GPU(data, radius)

    print(
        f" - [GPU] Apply morphological closing with radius={radius} and border mode='{border_mode}'"
    )

    T, Z, Y, X = data.shape
    # Create structuring element
    struct_elem_np = ball(radius).astype(np.uint8)
    struct_elem = (
        torch.from_numpy(struct_elem_np).to(torch.float32).to("cuda")
    )  # shape (Dz, Dy, Dx)
    se_sum = struct_elem.sum()

    pad = (radius, radius, radius, radius, radius, radius)  # Pad (Dx, Dy, Dz)

    mode_map = {"constant": "constant", "reflect": "reflect", "nearest": "replicate"}
    if border_mode not in mode_map:
        raise ValueError(
            f"Unsupported border_mode '{border_mode}' for GPU. Use one of: {list(mode_map.keys())}"
        )

    # Prepare output
    result = torch.zeros_like(data, dtype=torch.uint8)

    # Apply per-frame
    for t in tqdm(range(T), desc="[GPU] Morphological closing", unit="frame"):
        frame = data[t]  # (Z, Y, X)
        frame = frame[None, None].float()  # shape (1, 1, Z, Y, X)

        # Padding
        padded = torch.nn.functional.pad(frame, pad, mode=mode_map[border_mode])

        # Dilation = conv > 0
        dilated = (
            torch.nn.functional.conv3d(padded, struct_elem[None, None], bias=None) > 0
        )

        # Pad dilated volume again
        dilated = dilated.float()
        padded_dilated = torch.nn.functional.pad(
            dilated, pad, mode=mode_map[border_mode]
        )

        # Erosion = conv == sum(struct_elem)
        eroded = (
            torch.nn.functional.conv3d(
                padded_dilated, struct_elem[None, None], bias=None
            )
            == se_sum
        )

        result[t] = eroded[0, 0].to(torch.uint8) * 255

    return result


def closing_morphology_in_space_ignore_border_GPU(
    data: torch.Tensor, radius: int
) -> torch.Tensor:
    """
    Fast 3D morphological closing on a 4D binary array (T, Z, Y, X), with border_mode='ignore' using PyTorch on GPU.
    Only neighbors within bounds are considered (no padding).
    Parameters:
        data: 4D numpy array (T, Z, Y, X), binary (0/255 or 0/1)
        radius: radius of the spherical structuring element
    Returns:
        4D numpy array (T, Z, Y, X), uint8, values in {0, 255}
    """
    print(
        f" - [GPU] Fast morphological closing with radius={radius} and border mode='ignore'"
    )

    T, Z, Y, X = data.shape
    result = torch.empty((T, Z, Y, X), dtype=torch.uint8)

    # Offsets for spherical neighborhood
    offsets = generate_spherical_offsets(radius)
    offsets_torch = torch.from_numpy(offsets).to(torch.int32).to("cuda")  # (N, 3)

    for t in tqdm(range(T), desc="[GPU] Morph closing (ignore border)", unit="frame"):
        frame = data[t]  # (Z, Y, X)
        frame_padded = frame  # no padding

        dilated = torch.zeros_like(frame, dtype=torch.uint8)
        eroded = torch.ones_like(frame, dtype=torch.uint8)

        # Apply dilation (max in neighborhood)
        for offset in offsets_torch:
            dz, dy, dx = offset.tolist()

            z1, z2 = max(0, -dz), min(Z, Z - dz)
            y1, y2 = max(0, -dy), min(Y, Y - dy)
            x1, x2 = max(0, -dx), min(X, X - dx)

            zz1, zz2 = z1 + dz, z2 + dz
            yy1, yy2 = y1 + dy, y2 + dy
            xx1, xx2 = x1 + dx, x2 + dx

            dilated[z1:z2, y1:y2, x1:x2] = torch.maximum(
                dilated[z1:z2, y1:y2, x1:x2], frame_padded[zz1:zz2, yy1:yy2, xx1:xx2]
            )

        # Apply erosion (min in neighborhood)
        for offset in offsets_torch:
            dz, dy, dx = offset.tolist()

            z1, z2 = max(0, -dz), min(Z, Z - dz)
            y1, y2 = max(0, -dy), min(Y, Y - dy)
            x1, x2 = max(0, -dx), min(X, X - dx)

            zz1, zz2 = z1 + dz, z2 + dz
            yy1, yy2 = y1 + dy, y2 + dy
            xx1, xx2 = x1 + dx, x2 + dx

            eroded[z1:z2, y1:y2, x1:x2] = torch.minimum(
                eroded[z1:z2, y1:y2, x1:x2], dilated[zz1:zz2, yy1:yy2, xx1:xx2]
            )

        result[t] = eroded * 255

    return result
