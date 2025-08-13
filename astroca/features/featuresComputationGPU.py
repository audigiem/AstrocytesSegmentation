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
    GPU-optimized feature computation with advanced vectorization and batching.
    """
    if calcium_events.ndim != 4 or image_amplitude.ndim != 4:
        raise ValueError("Inputs must be 4D tensors (T, Z, Y, X)")

    device = calcium_events.device

    # Paramètres
    voxel_size_x = float(params_values["features_extraction"]["voxel_size_x"])
    voxel_size_y = float(params_values["features_extraction"]["voxel_size_y"])
    voxel_size_z = float(params_values["features_extraction"]["voxel_size_z"])
    voxel_size = voxel_size_x * voxel_size_y * voxel_size_z

    threshold_median_localized = float(
        params_values["features_extraction"]["threshold_median_localized"]
    )
    volume_localized = float(params_values["features_extraction"]["volume_localized"])

    # Précomputation des indices
    coords_all, event_ids_all = precompute_event_voxel_indices_GPU(calcium_events)

    if len(coords_all) == 0:
        return {}

    # Vectorisation pour tous les événements
    unique_event_ids = torch.unique(event_ids_all)
    features = {}

    # Traitement par batch d'événements
    batch_size = min(100, len(unique_event_ids))  # Ajustable selon la mémoire GPU

    for i in tqdm(range(0, len(unique_event_ids), batch_size), desc="Computing features (GPU)"):
        batch_event_ids = unique_event_ids[i:i + batch_size]

        # Masque pour les événements du batch
        batch_mask = torch.isin(event_ids_all, batch_event_ids)
        batch_coords = coords_all[batch_mask]
        batch_event_ids_all = event_ids_all[batch_mask]

        # Calcul vectorisé des caractéristiques
        batch_features = compute_features_batch_GPU(
            batch_event_ids, batch_coords, batch_event_ids_all, image_amplitude,
            voxel_size, voxel_size_x, voxel_size_y, voxel_size_z,
            threshold_median_localized, volume_localized, device
        )

        features.update(batch_features)

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
    Traitement vectorisé d'un batch d'événements
    """
    batch_features = {}

    for event_id in event_ids:
        event_id_val = event_id.item()

        # Masque pour cet événement
        mask = event_ids_all == event_id
        if not torch.any(mask):
            continue

        coords = coords_all[mask]  # Shape: (N, 4) où N = nombre de voxels pour cet événement
        t, z, y, x = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
        nb_voxels = len(t)

        if nb_voxels == 0:
            continue

        # Calculs vectorisés des caractéristiques de base
        centroid_x = torch.mean(x.float()).int()
        centroid_y = torch.mean(y.float()).int()
        centroid_z = torch.mean(z.float()).int()
        centroid_t = torch.mean(t.float()).int()

        t0, t1 = torch.min(t), torch.max(t)
        duration = t1 - t0 + 1
        volume = nb_voxels * voxel_size

        # Amplitude de l'événement
        amplitude = torch.max(image_amplitude[t, z, y, x])

        # Classification
        if duration > 1:
            class_result = is_localized_GPU(
                coords, t0, duration, volume,
                voxel_size_x, voxel_size_y, voxel_size_z,
                threshold_median_localized, volume_localized, device
            )
            class_label = class_result["class"]
            confidence = class_result["confidence"]
            mean_displacement = class_result["mean_displacement"]
            median_displacement = class_result["median_displacement"]
        else:
            if volume <= volume_localized:
                class_label = "Localized"
            else:
                class_label = "Localized but not in a microdomain"
            confidence = torch.tensor(0.0, device=device)
            mean_displacement = torch.tensor(0.0, device=device)
            median_displacement = torch.tensor(0.0, device=device)

        batch_features[event_id_val] = {
            "T0 [frame]": t0,
            "Duration [frame]": duration,
            "CentroidX [voxel]": centroid_x,
            "CentroidY [voxel]": centroid_y,
            "CentroidZ [voxel]": centroid_z,
            "CentroidT [voxel]": centroid_t,
            "Volume [µm^3]": volume,
            "Amplitude": amplitude,
            "Class": class_label,
            "STD displacement [µm]": confidence,
            "Mean displacement [µm]": mean_displacement,
            "Median displacement [µm]": median_displacement,
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
    GPU-optimized classification with vectorized centroid computation
    """
    duration_val = duration.item()
    t0_val = t0.item()

    # Initialiser les centroïdes
    centroids_x = torch.zeros(duration_val, device=device)
    centroids_y = torch.zeros(duration_val, device=device)
    centroids_z = torch.zeros(duration_val, device=device)

    # Calcul vectorisé des centroïdes par frame
    for i, t in enumerate(range(t0_val, t0_val + duration_val)):
        frame_mask = coords[:, 0] == t
        if torch.any(frame_mask):
            frame_coords = coords[frame_mask]
            centroids_x[i] = torch.mean(frame_coords[:, 3].float()) * voxel_size_x
            centroids_y[i] = torch.mean(frame_coords[:, 2].float()) * voxel_size_y
            centroids_z[i] = torch.mean(frame_coords[:, 1].float()) * voxel_size_z

    # Calcul vectorisé des distances
    distances = []
    for tau in range(1, duration_val):
        for i in range(duration_val - tau):
            x_dist = centroids_x[i + tau] - centroids_x[i]
            y_dist = centroids_y[i + tau] - centroids_y[i]
            z_dist = centroids_z[i + tau] - centroids_z[i]

            distance = torch.sqrt(x_dist ** 2 + y_dist ** 2 + z_dist ** 2)
            distances.append(distance)

    if distances:
        distances_tensor = torch.stack(distances)
        median_displacement = torch.median(distances_tensor)
        mean_displacement = torch.mean(distances_tensor)
        std_displacement = torch.std(distances_tensor)
    else:
        median_displacement = torch.tensor(0.0, device=device)
        mean_displacement = torch.tensor(0.0, device=device)
        std_displacement = torch.tensor(0.0, device=device)

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