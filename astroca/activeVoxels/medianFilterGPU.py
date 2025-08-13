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
    border_mode: str = "reflect",
) -> torch.Tensor:
    """
    @brief GPU version of 3D median filter using PyTorch
    """
    print(
        f" - Apply 3D median filter (GPU) with radius={radius}, border mode='{border_mode}'"
    )

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
    @brief Process a batch of padded frames with median filter (optimized)
    """
    batch_size, Z_pad, Y_pad, X_pad = batch_padded.shape
    Z, Y, X = Z_pad - 2 * r, Y_pad - 2 * r, X_pad - 2 * r
    n_offsets = offsets.shape[0]

    # Create output tensor
    result = torch.empty(
        (batch_size, Z, Y, X), dtype=batch_padded.dtype, device=batch_padded.device
    )

    # Vectorisation pour améliorer les performances
    for b in range(batch_size):
        # Préparer toutes les positions en une fois
        for z in range(Z):
            for y in range(Y):
                for x in range(X):
                    # Center position in padded coordinates
                    z_center, y_center, x_center = z + r, y + r, x + r

                    # Collecter tous les voisins
                    neighbor_coords = offsets + torch.tensor([z_center, y_center, x_center],
                                                             device=offsets.device)

                    # Extraire les valeurs des voisins
                    values = batch_padded[b, neighbor_coords[:, 0],
                    neighbor_coords[:, 1],
                    neighbor_coords[:, 2]]

                    # Calculer la médiane
                    median_val = torch.median(values)
                    if isinstance(median_val, tuple):
                        result[b, z, y, x] = median_val[0]
                    else:
                        result[b, z, y, x] = median_val

    return result


# def median_filter_batch_gpu_padded(
#         batch_padded: torch.Tensor, offsets: torch.Tensor, r: int
# ) -> torch.Tensor:
#     """
#     @brief Process a batch of padded frames with median filter (conservative)
#     """
#     batch_size, Z_pad, Y_pad, X_pad = batch_padded.shape
#     Z, Y, X = Z_pad - 2 * r, Y_pad - 2 * r, X_pad - 2 * r
#
#     result = torch.empty(
#         (batch_size, Z, Y, X), dtype=batch_padded.dtype, device=batch_padded.device
#     )
#
#     # Traiter par tranches pour éviter les problèmes de mémoire
#     chunk_size = 1000  # Traiter 1000 voxels à la fois
#
#     for b in range(batch_size):
#         total_voxels = Z * Y * X
#
#         for start_idx in range(0, total_voxels, chunk_size):
#             end_idx = min(start_idx + chunk_size, total_voxels)
#
#             # Convertir les indices linéaires en coordonnées 3D
#             linear_indices = torch.arange(start_idx, end_idx, device=batch_padded.device)
#             z_coords = linear_indices // (Y * X) + r
#             y_coords = (linear_indices % (Y * X)) // X + r
#             x_coords = linear_indices % X + r
#
#             center_coords = torch.stack([z_coords, y_coords, x_coords], dim=1)
#
#             # Coordonnées des voisins
#             neighbor_coords = center_coords.unsqueeze(1) + offsets.unsqueeze(0)
#
#             # Extraire les valeurs
#             neighbors = batch_padded[
#                 b,
#                 neighbor_coords[:, :, 0],
#                 neighbor_coords[:, :, 1],
#                 neighbor_coords[:, :, 2]
#             ]
#
#             # Calculer les médianes
#             medians = torch.median(neighbors, dim=1).values
#
#             # Assigner les résultats
#             z_out = (linear_indices // (Y * X)).long()
#             y_out = ((linear_indices % (Y * X)) // X).long()
#             x_out = (linear_indices % X).long()
#
#             result[b, z_out, y_out, x_out] = medians
#
#     return result