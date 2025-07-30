"""
@file medianFilterGPU.py
@brief 3D median filter for 4D stacks (T,Z,Y,X) with spherical neighborhood and border handling on GPU
"""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from astroca.activeVoxels.medianFilter import generate_spherical_offsets


def unified_median_filter_3d_gpu(
        data: torch.Tensor,
        radius: float = 1.5,
        border_mode: str = 'reflect',
) -> torch.Tensor:
    """
    @brief GPU version of 3D median filter using PyTorch
    """
    print(f" - Apply 3D median filter (GPU) with radius={radius}, border mode='{border_mode}'")

    if border_mode == 'ignore':
        T, Z, Y, X = data.shape
        data3D = data.reshape(T * Z, Y, X)  # Reshape to treat as 3D
        radius = int(np.ceil(radius))
        offsets = generate_spherical_offsets(radius)
        # convert to torch.Tensor
        offsets = torch.tensor(offsets, dtype=torch.long, device=data.device)

        median_data = apply_median_filter_3d_gpu_ignore_border(data3D, offsets)

        result = median_data.reshape(T, Z, Y, X)  # Reshape back to 4D
        return result
    else:
        return apply_median_filter_3d_gpu_with_padding(data, radius, border_mode)


def apply_median_filter_3d_gpu_ignore_border(data_3D: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """
    Applique un filtre médian 3D sur GPU, en ignorant les bords comme en version CPU.

    data_3D : (TZ, Y, X) torch.Tensor (sur GPU)
    offsets : (N, 3) torch.LongTensor (dz, dy, dx)
    return : même forme que data_3D, avec médiane appliquée uniquement sur voisins valides
    """
    device = data_3D.device
    dtype = data_3D.dtype
    TZ, Y, X = data_3D.shape
    N = offsets.shape[0]

    # Créer une grille des coordonnées (tz, y, x)
    tz_grid, y_grid, x_grid = torch.meshgrid(
        torch.arange(TZ, device=device),
        torch.arange(Y, device=device),
        torch.arange(X, device=device),
        indexing='ij'
    )  # shape: (TZ, Y, X)

    # Initialiser le tenseur de voisinages valides : (N, TZ, Y, X)
    neighborhoods = torch.empty((N, TZ, Y, X), device=device, dtype=dtype)
    mask = torch.ones((N, TZ, Y, X), device=device, dtype=torch.bool)

    for i in range(N):
        dz, dy, dx = offsets[i]

        zz = tz_grid + dz
        yy = y_grid + dy
        xx = x_grid + dx

        # Vérifie si les indices sont dans les bornes
        valid = (zz >= 0) & (zz < TZ) & (yy >= 0) & (yy < Y) & (xx >= 0) & (xx < X)
        mask[i] = valid

        # Remplit avec 0 temporairement ; remplacé plus tard
        zz = torch.clamp(zz, 0, TZ - 1)
        yy = torch.clamp(yy, 0, Y - 1)
        xx = torch.clamp(xx, 0, X - 1)

        neighborhoods[i] = data_3D[zz, yy, xx]

    # Remplacer les valeurs invalides par un marqueur (ex : NaN ou très grand)
    neighborhoods[~mask] = float('nan')  # On ignore NaNs dans la médiane

    # Calcul médiane en ignorant les NaNs
    median = torch.nanmedian(neighborhoods, dim=0).values

    # Fallback : si aucun voisin valide, on garde la valeur originale
    all_invalid = (~mask).all(dim=0)
    median[all_invalid] = data_3D[all_invalid]

    return median


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
    if border_mode == 'reflect':
        padding_mode = 'reflect'
    elif border_mode == 'nearest' or border_mode == 'edge':
        padding_mode = 'replicate'
    elif border_mode == 'constant':
        padding_mode = 'constant'
    else:
        padding_mode = 'reflect'  # fallback

    # Create spherical offsets
    offsets = generate_spherical_offsets(radius)
    offsets = torch.tensor(offsets, dtype=torch.long, device=data.device)

    # Pad the data (pad format for 3D: [X_left, X_right, Y_left, Y_right, Z_left, Z_right])
    padded = F.pad(data, (r, r, r, r, r, r), mode=padding_mode)

    # Process in batches
    batch_size = min(T, 4)  # Smaller batch for padded version due to memory
    result = torch.empty_like(data)

    for i in tqdm(range(0, T, batch_size), desc=f"Processing frames (GPU) with {border_mode} border"):
        end_idx = min(i + batch_size, T)
        batch_padded = padded[i:end_idx]
        batch_result = median_filter_batch_gpu_padded(batch_padded, offsets, r)
        result[i:end_idx] = batch_result

    return result


def median_filter_batch_gpu_padded(batch_padded: torch.Tensor, offsets: torch.Tensor, r: int) -> torch.Tensor:
    """
    @brief Process a batch of padded frames with median filter
    """
    batch_size, Z_pad, Y_pad, X_pad = batch_padded.shape
    Z, Y, X = Z_pad - 2 * r, Y_pad - 2 * r, X_pad - 2 * r
    n_offsets = offsets.shape[0]

    # Create output tensor
    result = torch.empty((batch_size, Z, Y, X), dtype=batch_padded.dtype, device=batch_padded.device)

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
                        if hasattr(median_val, 'values'):
                            result[b, z, y, x] = median_val.values
                        else:
                            result[b, z, y, x] = median_val

    return result


def median_filter_batch_gpu_vectorized(batch_data: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """
    @brief Vectorized GPU implementation for better performance (ignore border mode)
    """
    batch_size, Z, Y, X = batch_data.shape
    n_offsets = offsets.shape[0]
    device = batch_data.device

    # Create coordinate grids
    z_coords = torch.arange(Z, device=device).view(-1, 1, 1).expand(Z, Y, X)
    y_coords = torch.arange(Y, device=device).view(1, -1, 1).expand(Z, Y, X)
    x_coords = torch.arange(X, device=device).view(1, 1, -1).expand(Z, Y, X)

    result = torch.empty_like(batch_data)

    for b in range(batch_size):
        # For each offset, create the shifted coordinates
        neighbor_values = torch.full((Z, Y, X, n_offsets), float('nan'), device=device)
        valid_mask = torch.zeros((Z, Y, X, n_offsets), dtype=torch.bool, device=device)

        for i, (dz, dy, dx) in enumerate(offsets):
            # Calculate neighbor coordinates
            neighbor_z = z_coords + dz
            neighbor_y = y_coords + dy
            neighbor_x = x_coords + dx

            # Create validity mask
            valid = (
                    (neighbor_z >= 0) & (neighbor_z < Z) &
                    (neighbor_y >= 0) & (neighbor_y < Y) &
                    (neighbor_x >= 0) & (neighbor_x < X)
            )

            # Clamp coordinates to valid range for indexing
            neighbor_z_clamped = torch.clamp(neighbor_z, 0, Z - 1)
            neighbor_y_clamped = torch.clamp(neighbor_y, 0, Y - 1)
            neighbor_x_clamped = torch.clamp(neighbor_x, 0, X - 1)

            # Extract values
            values = batch_data[b, neighbor_z_clamped, neighbor_y_clamped, neighbor_x_clamped]

            # Apply validity mask
            neighbor_values[:, :, :, i] = torch.where(valid, values, torch.tensor(float('nan'), device=device))
            valid_mask[:, :, :, i] = valid

        # Compute median for each position
        for z in range(Z):
            for y in range(Y):
                for x in range(X):
                    valid_values = neighbor_values[z, y, x, valid_mask[z, y, x]]
                    if len(valid_values) > 0:
                        result[b, z, y, x] = torch.median(valid_values).values
                    else:
                        result[b, z, y, x] = batch_data[b, z, y, x]

    return result



