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
    @brief Optimized GPU version with better batch management
    """
    print(f" - Apply 3D median filter (GPU) with radius={radius}, border mode='{border_mode}'")

    T, Z, Y, X = data.shape
    r = int(np.ceil(radius))

    # Ajuster la taille du batch selon la mémoire GPU disponible
    gpu_memory = torch.cuda.get_device_properties(data.device).total_memory
    frame_memory = Z * Y * X * data.element_size() * 8  # Estimation avec padding
    optimal_batch_size = max(1, min(T, gpu_memory // frame_memory // 4))  # Factor de sécurité

    # Convertir le mode de bordure
    padding_mode = {
        "reflect": "reflect",
        "nearest": "replicate",
        "edge": "replicate",
        "constant": "constant"
    }.get(border_mode, "reflect")

    # Créer les offsets sphériques une seule fois
    offsets = generate_spherical_offsets(radius)
    offsets = torch.tensor(offsets, dtype=torch.long, device=data.device)

    result = torch.empty_like(data)

    # Traitement par batch optimisé
    for i in tqdm(range(0, T, optimal_batch_size), desc=f"GPU median filter ({border_mode})"):
        end_idx = min(i + optimal_batch_size, T)
        batch = data[i:end_idx]

        # Padding
        padded = F.pad(batch, (r, r, r, r, r, r), mode=padding_mode)

        # Filtrage
        batch_result = median_filter_batch_gpu_padded(padded, offsets, r)
        result[i:end_idx] = batch_result

    return result


def median_filter_batch_gpu_padded(
        batch_padded: torch.Tensor, offsets: torch.Tensor, r: int
) -> torch.Tensor:
    """
    @brief Process a batch of padded frames with median filter (fully optimized)
    """
    batch_size, Z_pad, Y_pad, X_pad = batch_padded.shape
    Z, Y, X = Z_pad - 2 * r, Y_pad - 2 * r, X_pad - 2 * r

    # Créer toutes les coordonnées de sortie en une fois
    z_coords = torch.arange(Z, device=batch_padded.device, dtype=torch.long) + r
    y_coords = torch.arange(Y, device=batch_padded.device, dtype=torch.long) + r
    x_coords = torch.arange(X, device=batch_padded.device, dtype=torch.long) + r

    # Grille 3D de tous les centres
    zz, yy, xx = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    centers = torch.stack([zz.flatten(), yy.flatten(), xx.flatten()], dim=1)
    # Shape: (Z*Y*X, 3)

    # Calculer toutes les coordonnées de voisins en une fois
    all_neighbors = centers.unsqueeze(1) + offsets.unsqueeze(0)
    # Shape: (Z*Y*X, n_offsets, 3)

    result = torch.empty((batch_size, Z, Y, X), dtype=batch_padded.dtype, device=batch_padded.device)

    # Traiter par batch (parallélisation massive)
    for b in range(batch_size):
        # Extraire toutes les valeurs de voisins en une seule opération
        neighbor_values = batch_padded[
            b,
            all_neighbors[:, :, 0],
            all_neighbors[:, :, 1],
            all_neighbors[:, :, 2]
        ]
        # Shape: (Z*Y*X, n_offsets)

        # Calculer toutes les médianes en parallèle
        medians = torch.median(neighbor_values, dim=1).values

        # Reshape vers la forme de sortie
        result[b] = medians.view(Z, Y, X)

    return result