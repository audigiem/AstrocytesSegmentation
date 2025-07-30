"""
@file medianFilterGPU.py
@brief 3D median filter for 4D stacks (T,Z,Y,X) with spherical neighborhood and border handling on GPU
"""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from astroca.tools.medianComputationTools import generate_spherical_offsets


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
        offsets = generate_spherical_offsets_torch(radius, device=data.device)

        median_data = apply_median_filter_3d_gpu_optimized(data3D, offsets)

        result = median_data.reshape(T, Z, Y, X)  # Reshape back to 4D
        return result
    else:
        return apply_median_filter_3d_gpu_with_padding(data, radius, border_mode)


def generate_spherical_offsets_torch(radius: float, device: torch.device = None) -> torch.Tensor:
    """
    Génère les offsets pour un voisinage sphérique avec PyTorch

    Args:
        radius: Rayon du filtre sphérique
        device: Device PyTorch (GPU/CPU)

    Returns:
        Tensor d'offsets (N, 3) avec (dz, dy, dx)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    radx = rady = radz = radius

    vx = int(np.ceil(radx))
    vy = int(np.ceil(rady))
    vz = int(np.ceil(radz))

    # Calcul des rayons inverses au carré
    rx2 = 1.0 / (radx * radx) if radx != 0.0 else 0.0
    ry2 = 1.0 / (rady * rady) if rady != 0.0 else 0.0
    rz2 = 1.0 / (radz * radz) if radz != 0.0 else 0.0

    offsets = []

    # Même ordre que dans le code Java/CPU
    for k in range(-vz, vz + 1):  # dz
        for j in range(-vy, vy + 1):  # dy
            for i in range(-vx, vx + 1):  # dx
                dist = (i * i) * rx2 + (j * j) * ry2 + (k * k) * rz2

                if dist <= 1.0:
                    offsets.append([k, j, i])  # (dz, dy, dx)

    return torch.tensor(offsets, dtype=torch.long, device=device)




def apply_median_filter_3d_gpu_memory_efficient(data_3d: torch.Tensor,
                                                offsets: torch.Tensor,
                                                chunk_size: int = None,
                                                batch_size: int = None) -> torch.Tensor:
    """
    Version mémoire-efficiente pour très gros volumes avec double chunking

    Args:
        data_3d: Tensor 3D (Z, Y, X) sur GPU
        offsets: Tensor d'offsets (N, 3) avec (dz, dy, dx)
        chunk_size: Taille des chunks Z pour traitement par batch (défaut: auto)
        batch_size: Taille des mini-batches dans chaque chunk (défaut: auto)

    Returns:
        Tensor filtré de même forme que l'entrée
    """
    device = data_3d.device
    dtype = data_3d.dtype
    Z, Y, X = data_3d.shape
    N = offsets.shape[0]

    # Calculer les tailles automatiquement si non spécifiées
    if chunk_size is None or batch_size is None:
        memory_per_element = 4 if dtype == torch.float32 else 8
        gpu_memory = torch.cuda.get_device_properties(device).total_memory
        available_memory = gpu_memory * 0.6  # 60% pour être sûr

        if chunk_size is None:
            # Chunk size basé sur la mémoire disponible
            estimated_memory_per_z = N * Y * X * memory_per_element
            chunk_size = max(1, min(Z, int(available_memory * 0.7 / estimated_memory_per_z)))

        if batch_size is None:
            # Batch size pour les mini-chunks dans le plan XY
            target_memory = available_memory * 0.3  # 30% pour les mini-batches
            elements_per_batch = int(target_memory / (N * memory_per_element))
            batch_size = max(1, min(Y * X, elements_per_batch))

    result = torch.empty_like(data_3d)

    # Traitement par chunks le long de l'axe Z
    for z_start in range(0, Z, chunk_size):
        z_end = min(z_start + chunk_size, Z)
        z_slice = slice(z_start, z_end)

        # Extraire le chunk avec padding pour les voisins
        max_offset_z = max(abs(offsets[:, 0].max().item()), abs(offsets[:, 0].min().item()))
        z_start_padded = max(0, z_start - max_offset_z)
        z_end_padded = min(Z, z_end + max_offset_z)

        data_chunk = data_3d[z_start_padded:z_end_padded]
        z_offset_in_chunk = z_start - z_start_padded

        # Appliquer le filtre sur le chunk avec mini-batching
        chunk_result = apply_median_filter_3d_gpu_optimized_chunk_batched(
            data_chunk, offsets, z_slice, z_offset_in_chunk, batch_size
        )

        result[z_slice] = chunk_result

    return result


def apply_median_filter_3d_gpu_optimized_chunk_batched(data_chunk: torch.Tensor,
                                                       offsets: torch.Tensor,
                                                       target_z_slice: slice,
                                                       z_offset: int,
                                                       batch_size: int) -> torch.Tensor:
    """
    Applique le filtre médian sur un chunk avec mini-batching dans le plan XY
    """
    device = data_chunk.device
    dtype = data_chunk.dtype
    Z_chunk, Y, X = data_chunk.shape
    N = offsets.shape[0]

    z_start, z_end = target_z_slice.start, target_z_slice.stop
    z_target_size = z_end - z_start

    result = torch.empty((z_target_size, Y, X), device=device, dtype=dtype)

    # Traitement par mini-batches dans le plan XY
    total_pixels = Y * X

    for batch_start in range(0, total_pixels, batch_size):
        batch_end = min(batch_start + batch_size, total_pixels)

        # Convertir les indices linéaires en coordonnées 2D
        pixel_indices = torch.arange(batch_start, batch_end, device=device)
        y_indices = pixel_indices // X
        x_indices = pixel_indices % X

        # Traiter chaque plan Z du chunk cible
        for z_local in range(z_target_size):
            z_global = z_local + z_offset

            # Créer les coordonnées pour ce mini-batch
            batch_neighborhoods = torch.empty((N, batch_end - batch_start),
                                              device=device, dtype=dtype)
            batch_valid_mask = torch.empty((N, batch_end - batch_start),
                                           device=device, dtype=torch.bool)

            # Pour chaque offset, calculer les voisins
            for i, (dz, dy, dx) in enumerate(offsets):
                neighbor_z = z_global + dz.item()
                neighbor_y = y_indices + dy.item()
                neighbor_x = x_indices + dx.item()

                # Masque de validité
                valid = ((neighbor_z >= 0) & (neighbor_z < Z_chunk) &
                         (neighbor_y >= 0) & (neighbor_y < Y) &
                         (neighbor_x >= 0) & (neighbor_x < X))

                batch_valid_mask[i] = valid

                # Clamping des coordonnées
                neighbor_z_safe = torch.clamp(neighbor_z, 0, Z_chunk - 1)
                neighbor_y_safe = torch.clamp(neighbor_y, 0, Y - 1)
                neighbor_x_safe = torch.clamp(neighbor_x, 0, X - 1)

                # Récupérer les valeurs
                values = data_chunk[neighbor_z_safe, neighbor_y_safe, neighbor_x_safe]
                batch_neighborhoods[i] = torch.where(valid, values,
                                                     torch.tensor(float('nan'), device=device, dtype=dtype))

            # Calculer la médiane
            batch_medians = torch.nanmedian(batch_neighborhoods, dim=0).values

            # Gérer les cas sans voisins valides
            no_valid = (~batch_valid_mask).all(dim=0)
            original_values = data_chunk[z_global, y_indices, x_indices]
            batch_medians = torch.where(no_valid, original_values, batch_medians)

            # Stocker les résultats
            result[z_local, y_indices, x_indices] = batch_medians

    return result


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



