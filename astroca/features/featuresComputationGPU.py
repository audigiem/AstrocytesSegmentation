"""
@file featuresComputationGPU.py
@brief GPU-optimized module for computing features from 3D image sequences with time dimension.
"""

import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from typing import Union, Dict, Tuple, Optional
from astroca.features.coactive import compute_coactive_csv
from astroca.features.hotspots import (
    compute_hot_spots_from_features,
    write_csv_hot_spots,
)


def save_features_from_events_GPU(
        calcium_events: torch.Tensor,
        events_ids: int,
        image_amplitude: torch.Tensor,
        params_values: dict = None,
) -> None:
    """
    GPU-optimized version for computing features from calcium events
    """
    print("=== Computing features from calcium events (GPU) ===")
    required_keys = {"paths", "save", "features_extraction"}
    if not required_keys.issubset(params_values.keys()):
        raise ValueError(
            f"Missing required parameters: {required_keys - params_values.keys()}"
        )

    features = compute_features_GPU(
        calcium_events, events_ids, image_amplitude, params_values
    )

    save_result = int(params_values["save"]["save_features"]) == 1
    output_directory = params_values["paths"]["output_dir"]

    # Convertir en CPU pour les calculs de hot spots (si nécessaire)
    features_cpu = {k: {kk: vv.cpu().item() if isinstance(vv, torch.Tensor) else vv
                        for kk, vv in v.items()} for k, v in features.items()}

    # hot_spots = compute_hot_spots_from_features(features_cpu, params_values)

    if save_result:
        if output_directory is None:
            raise ValueError(
                "Output directory must be specified when save_result is True."
            )
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        write_csv_features_GPU(features_cpu, output_directory)
        # compute_coactive_csv(features_cpu, output_directory)
        # write_csv_hot_spots(hot_spots, output_directory)

    print(60 * "=")


def precompute_event_voxel_indices_GPU(calcium_events: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-optimized precomputation of indices and event_ids for all non-zero voxels
    """
    # Trouver tous les indices non-zéro en une seule opération
    coords = torch.nonzero(calcium_events > 0, as_tuple=False)  # Shape: (N, 4)
    event_ids = calcium_events[calcium_events > 0]  # Shape: (N,)

    return coords, event_ids


def compute_features_GPU(
    calcium_events: torch.Tensor,
    events_ids: int,
    image_amplitude: torch.Tensor,
    params_values: dict = None,
) -> dict:
    """
    Version optimisée mémoire avec libération explicite
    """
    if calcium_events.ndim != 4 or image_amplitude.ndim != 4:
        raise ValueError("Inputs must be 4D tensors (T, Z, Y, X)")

    device = calcium_events.device

    # Paramètres
    voxel_size_x = float(params_values["features_extraction"]["voxel_size_x"])
    voxel_size_y = float(params_values["features_extraction"]["voxel_size_y"])
    voxel_size_z = float(params_values["features_extraction"]["voxel_size_z"])
    voxel_size = voxel_size_x * voxel_size_y * voxel_size_z

    threshold_median_localized = float(params_values["features_extraction"]["threshold_median_localized"])
    volume_localized = float(params_values["features_extraction"]["volume_localized"])

    # Précomputation optimisée
    coords_all, event_ids_all = precompute_event_voxel_indices_GPU(calcium_events)

    if len(coords_all) == 0:
        return {}

    unique_event_ids = torch.unique(event_ids_all)
    features = {}

    # Batch size adaptatif selon la mémoire disponible
    if torch.cuda.is_available():
        mem_free = torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)
        batch_size = min(200, max(10, int(mem_free / (1024**3))))  # Adaptatif selon la mémoire libre
    else:
        batch_size = 50

    for i in tqdm(range(0, len(unique_event_ids), batch_size), desc="Computing features (GPU)"):
        batch_event_ids = unique_event_ids[i:i + batch_size]

        # Traitement du batch
        batch_features = compute_features_batch_GPU(
            batch_event_ids, coords_all, event_ids_all, image_amplitude,
            voxel_size, voxel_size_x, voxel_size_y, voxel_size_z,
            threshold_median_localized, volume_localized, device
        )

        features.update(batch_features)

        # Libération mémoire explicite
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return features


def compute_features_batch_GPU(
        event_ids: torch.Tensor,
        coords_all: torch.Tensor,
        event_ids_all: torch.Tensor,
        image_amplitude: torch.Tensor,
        voxel_size: float,
        voxel_size_x: float,
        voxel_size_y: float,
        voxel_size_z: float,
        threshold_median_localized: float,
        volume_localized: float,
        device: torch.device
) -> dict:
    """
    Version ultra-optimisée avec vectorisation avancée
    """
    batch_features = {}

    # Précalcul des masques pour tous les événements du batch
    event_masks = event_ids_all.unsqueeze(0) == event_ids.unsqueeze(1)  # (batch_size, n_coords)

    for i, event_id in enumerate(event_ids):
        event_id_val = event_id.item()
        mask = event_masks[i]

        if not torch.any(mask):
            continue

        coords = coords_all[mask]
        t, z, y, x = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
        nb_voxels = len(t)

        if nb_voxels == 0:
            continue

        # Calculs vectorisés optimisés
        centroid_x = torch.mean(x.float())
        centroid_y = torch.mean(y.float())
        centroid_z = torch.mean(z.float())
        centroid_t = torch.mean(t.float())

        t0, t1 = torch.min(t), torch.max(t)
        duration = t1 - t0 + 1
        volume = nb_voxels * voxel_size

        # Amplitude optimisée avec indexing avancé
        amplitude = torch.max(image_amplitude[t, z, y, x])

        # Classification optimisée
        if duration > 1:
            class_result = is_localized_GPU(
                coords, t0, duration, volume,
                voxel_size_x, voxel_size_y, voxel_size_z,
                threshold_median_localized, volume_localized, device
            )
        else:
            class_label = "Localized" if volume <= volume_localized else "Localized but not in a microdomain"
            class_result = {
                "class": class_label,
                "confidence": torch.tensor(0.0, device=device),
                "mean_displacement": torch.tensor(0.0, device=device),
                "median_displacement": torch.tensor(0.0, device=device),
            }

        batch_features[event_id_val] = {
            "T0 [frame]": t0,
            "Duration [frame]": duration,
            "CentroidX [voxel]": centroid_x.int(),
            "CentroidY [voxel]": centroid_y.int(),
            "CentroidZ [voxel]": centroid_z.int(),
            "CentroidT [voxel]": centroid_t.int(),
            "Volume [µm^3]": volume,
            "Amplitude": amplitude,
            "Class": class_result["class"],
            "STD displacement [µm]": class_result["confidence"],
            "Mean displacement [µm]": class_result["mean_displacement"],
            "Median displacement [µm]": class_result["median_displacement"],
        }

    return batch_features


def is_localized_GPU(
        coords: torch.Tensor,
        t0: torch.Tensor,
        duration: torch.Tensor,
        volume: float,
        voxel_size_x: float,
        voxel_size_y: float,
        voxel_size_z: float,
        threshold_median_localized: float,
        volume_localized: float,
        device: torch.device
) -> dict:
    """
    GPU-optimized classification avec vectorisation complète et gestion des cas limites
    """
    duration_val = duration.item()
    t0_val = t0.item()

    if duration_val <= 1:
        # Cas trivial - pas de déplacement à calculer
        return {
            "class": "Localized" if volume <= volume_localized else "Localized but not in a microdomain",
            "confidence": torch.tensor(0.0, device=device),
            "mean_displacement": torch.tensor(0.0, device=device),
            "median_displacement": torch.tensor(0.0, device=device),
        }

    # Créer un tenseur de temps pour la vectorisation
    time_range = torch.arange(t0_val, t0_val + duration_val, device=device)

    # Initialiser les centroïdes avec NaN pour détecter les frames vides
    centroids_x = torch.full((duration_val,), float('nan'), device=device)
    centroids_y = torch.full((duration_val,), float('nan'), device=device)
    centroids_z = torch.full((duration_val,), float('nan'), device=device)

    # Vectorisation du calcul des centroïdes
    coords_t = coords[:, 0]
    coords_x = coords[:, 3].float() * voxel_size_x
    coords_y = coords[:, 2].float() * voxel_size_y
    coords_z = coords[:, 1].float() * voxel_size_z

    # Calcul vectorisé des centroïdes par frame
    for i, t_val in enumerate(time_range):
        frame_mask = coords_t == t_val
        if torch.any(frame_mask):
            centroids_x[i] = torch.mean(coords_x[frame_mask])
            centroids_y[i] = torch.mean(coords_y[frame_mask])
            centroids_z[i] = torch.mean(coords_z[frame_mask])

    # Filtrer les centroïdes valides (non-NaN)
    valid_mask = ~torch.isnan(centroids_x)
    if torch.sum(valid_mask) < 2:
        # Pas assez de points pour calculer des distances
        return {
            "class": "Localized" if volume <= volume_localized else "Localized but not in a microdomain",
            "confidence": torch.tensor(0.0, device=device),
            "mean_displacement": torch.tensor(0.0, device=device),
            "median_displacement": torch.tensor(0.0, device=device),
        }

    valid_centroids_x = centroids_x[valid_mask]
    valid_centroids_y = centroids_y[valid_mask]
    valid_centroids_z = centroids_z[valid_mask]
    valid_frames = torch.arange(len(valid_centroids_x), device=device)

    # Calcul vectorisé des distances entre toutes les paires de frames
    distances = []
    n_valid = len(valid_centroids_x)

    for tau in range(1, min(n_valid, duration_val)):
        for i in range(n_valid - tau):
            j = i + tau

            x_dist = valid_centroids_x[j] - valid_centroids_x[i]
            y_dist = valid_centroids_y[j] - valid_centroids_y[i]
            z_dist = valid_centroids_z[j] - valid_centroids_z[i]

            distance = torch.sqrt(x_dist ** 2 + y_dist ** 2 + z_dist ** 2)
            distances.append(distance)

    if len(distances) == 0:
        median_displacement = torch.tensor(0.0, device=device)
        mean_displacement = torch.tensor(0.0, device=device)
        std_displacement = torch.tensor(0.0, device=device)
    elif len(distances) == 1:
        distances_tensor = torch.stack(distances)
        median_displacement = distances_tensor[0]
        mean_displacement = distances_tensor[0]
        std_displacement = torch.tensor(0.0, device=device)  # Éviter le warning
    else:
        distances_tensor = torch.stack(distances)
        median_displacement = torch.median(distances_tensor)
        mean_displacement = torch.mean(distances_tensor)
        # Utiliser unbiased=False pour éviter le warning avec peu d'échantillons
        std_displacement = torch.std(distances_tensor, unbiased=False)

    # Classification
    localized = median_displacement <= threshold_median_localized

    if localized:
        if volume <= volume_localized:
            class_label = "Localized"
        else:
            class_label = "Localized but not in a microdomain"
    else:
        class_label = "Wave"

    return {
        "class": class_label,
        "confidence": std_displacement,
        "mean_displacement": mean_displacement,
        "median_displacement": median_displacement,
    }


def write_csv_features_GPU(features: dict, output_directory: str) -> None:
    """
    Write GPU-computed features to CSV file
    """
    # Convertir les tenseurs en valeurs Python
    converted_features = {}
    for event_id, feature_dict in features.items():
        converted_dict = {}
        for key, value in feature_dict.items():
            if isinstance(value, torch.Tensor):
                converted_dict[key] = value.cpu().item()
            else:
                converted_dict[key] = value
        converted_features[event_id] = converted_dict

    df = pd.DataFrame.from_dict(converted_features, orient="index")

    # Réorganiser les colonnes
    columns_order = [
        "T0 [frame]",
        "Duration [frame]",
        "CentroidX [voxel]",
        "CentroidY [voxel]",
        "CentroidZ [voxel]",
        "CentroidT [voxel]",
        "Volume [µm^3]",
        "Amplitude",
        "Class",
        "STD displacement [µm]",
        "Mean displacement [µm]",
        "Median displacement [µm]",
    ]
    df = df[columns_order]

    output_file = os.path.join(output_directory, "Features.csv")
    df.to_csv(output_file, index_label="Label", sep=";")
    print(f"Features saved to {output_file}")


# Fonction dispatcher unifiée
def save_features_from_events(
        calcium_events: Union[np.ndarray, torch.Tensor],
        events_ids: int,
        image_amplitude: Union[np.ndarray, torch.Tensor],
        params_values: dict = None,
) -> None:
    """
    Dispatcher function for CPU or GPU feature computation
    """
    if isinstance(calcium_events, torch.Tensor) and calcium_events.is_cuda:
        # Assurer que image_amplitude est aussi sur GPU
        if isinstance(image_amplitude, np.ndarray):
            image_amplitude = torch.from_numpy(image_amplitude).to(calcium_events.device)
        elif isinstance(image_amplitude, torch.Tensor) and not image_amplitude.is_cuda:
            image_amplitude = image_amplitude.to(calcium_events.device)

        return save_features_from_events_GPU(calcium_events, events_ids, image_amplitude, params_values)
    else:
        # Import de la version CPU
        from astroca.features.featuresComputation import save_features_from_events as save_features_CPU

        # Conversion vers numpy si nécessaire
        if isinstance(calcium_events, torch.Tensor):
            calcium_events = calcium_events.cpu().numpy()
        if isinstance(image_amplitude, torch.Tensor):
            image_amplitude = image_amplitude.cpu().numpy()

        return save_features_CPU(calcium_events, events_ids, image_amplitude, params_values)


def compute_features(
        calcium_events: Union[np.ndarray, torch.Tensor],
        events_ids: int,
        image_amplitude: Union[np.ndarray, torch.Tensor],
        params_values: dict = None,
) -> dict:
    """
    Dispatcher function for feature computation
    """
    if isinstance(calcium_events, torch.Tensor) and calcium_events.is_cuda:
        return compute_features_GPU(calcium_events, events_ids, image_amplitude, params_values)
    else:
        # Import de la version CPU
        from astroca.features.featuresComputation import compute_features as compute_features_CPU

        # Conversion vers numpy si nécessaire
        if isinstance(calcium_events, torch.Tensor):
            calcium_events = calcium_events.cpu().numpy()
        if isinstance(image_amplitude, torch.Tensor):
            image_amplitude = image_amplitude.cpu().numpy()

        return compute_features_CPU(calcium_events, events_ids, image_amplitude, params_values)