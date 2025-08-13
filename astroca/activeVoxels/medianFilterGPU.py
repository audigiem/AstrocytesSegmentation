"""
@file medianFilterGPU.py
@brief 3D median filter for 4D stacks (T,Z,Y,X) with spherical neighborhood and border handling on GPU
"""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from astroca.tools.medianComputationTools import generate_spherical_offsets
from astroca.activeVoxels.medianFilter import apply_median_filter_3d_ignore_border


def unified_median_filter_3d_gpu(
    data: torch.Tensor,
    radius: float = 1.5,
    border_mode: str = "reflect",
) -> torch.Tensor | np.ndarray:
    """
    @brief GPU version of 3D median filter using PyTorch
    """
    print(
        f" - Apply 3D median filter (GPU) with radius={radius}, border mode='{border_mode}'"
    )

    if border_mode == "ignore":
        # fallback on CPU implementation if ignore mode is used
        print(" - Using CPU implementation for 'ignore' border mode")
        # convert to numpy for CPU processing
        data_np = data.cpu().numpy()
        T, Z, Y, X = data_np.shape
        data_3D = data_np.reshape(T * Z, Y, X)  # Reshape to treat as 3D
        offsets = generate_spherical_offsets(radius)
        median_filtered = apply_median_filter_3d_ignore_border(data_3D, offsets)
        data_filtered_4D = median_filtered.reshape(T, Z, Y, X)
        return data_filtered_4D

    else:
        return apply_median_filter_3d_gpu_with_padding(data, radius, border_mode)





def apply_median_filter_3d_gpu_with_padding(
    data: torch.Tensor,
    radius: float,
    border_mode: str,
) -> torch.Tensor:
    """
    @brief GPU implementation with padding for standard border modes
    """
    T, Z, Y, X = data.shape
    r = int(np.ceil(radius))

    # Convert border_mode to PyTorch padding mode
    if border_mode == "reflect":
        padding_mode = "reflect"
    elif border_mode == "nearest" or border_mode == "edge":
        padding_mode = "replicate"
    elif border_mode == "constant":
        padding_mode = "constant"
    else:
        padding_mode = "reflect"  # fallback

    # Create spherical offsets
    offsets = generate_spherical_offsets(radius)
    offsets = torch.tensor(offsets, dtype=torch.long, device=data.device)

    # Pad the data (pad format for 3D: [X_left, X_right, Y_left, Y_right, Z_left, Z_right])
    padded = F.pad(data, (r, r, r, r, r, r), mode=padding_mode)

    # Process in batches
    batch_size = min(T, 4)  # Smaller batch for padded version due to memory
    result = torch.empty_like(data)

    for i in tqdm(
        range(0, T, batch_size),
        desc=f"Processing frames (GPU) with {border_mode} border",
    ):
        end_idx = min(i + batch_size, T)
        batch_padded = padded[i:end_idx]
        batch_result = median_filter_batch_gpu_padded(batch_padded, offsets, r)
        result[i:end_idx] = batch_result

    return result


def median_filter_batch_gpu_padded(
    batch_padded: torch.Tensor, offsets: torch.Tensor, r: int
) -> torch.Tensor:
    """
    @brief Process a batch of padded frames with median filter
    """
    batch_size, Z_pad, Y_pad, X_pad = batch_padded.shape
    Z, Y, X = Z_pad - 2 * r, Y_pad - 2 * r, X_pad - 2 * r
    n_offsets = offsets.shape[0]

    # Create output tensor
    result = torch.empty(
        (batch_size, Z, Y, X), dtype=batch_padded.dtype, device=batch_padded.device
    )

    # Process each position in the original (unpadded) space
    for b in range(batch_size):
        for z in range(Z):
            for y in range(Y):
                for x in range(X):
                    # Center position in padded coordinates
                    z_center, y_center, x_center = z + r, y + r, x + r

                    # Collect all neighbors in spherical region
                    values = []
                    for i in range(n_offsets):
                        dz, dy, dx = offsets[i]
                        zz = z_center + dz
                        yy = y_center + dy
                        xx = x_center + dx
                        values.append(batch_padded[b, zz, yy, xx])

                    values_tensor = torch.stack(values)
                    if values:
                        values_tensor = torch.stack(values)
                        median_val = torch.median(values_tensor)
                        # Handle different PyTorch versions
                        if hasattr(median_val, "values"):
                            result[b, z, y, x] = median_val.values
                        else:
                            result[b, z, y, x] = median_val

    return result



