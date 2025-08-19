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
    CORRECTION: Inclut maintenant hotspots et coactive events
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
    save_quantifications = int(params_values["save"]["save_quantification"]) == 1
    output_directory = params_values["paths"]["output_dir"]

    features_cpu = convert_features_to_cpu(features)

    if save_result:
        if output_directory is None:
            raise ValueError(
                "Output directory must be specified when save_result is True."
            )
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Sauvegarder toutes les données comme en CPU
        write_csv_features_GPU(features_cpu, output_directory)

    if save_quantifications:
        hot_spots = compute_hot_spots_from_features(features_cpu, params_values)

        # CORRECTION: Inclure les événements coactifs
        compute_coactive_csv(features_cpu, output_directory)

        # CORRECTION: Inclure les hotspots
        write_csv_hot_spots(hot_spots, output_directory)

    print("=" * 60 + "\n")


def convert_features_to_cpu(features: dict) -> dict:
    """
    Convertit les features GPU vers CPU de manière robuste
    """
    features_cpu = {}
    for event_id, feature_dict in features.items():
        features_cpu[event_id] = {}
        for key, value in feature_dict.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:  # Scalaire
                    features_cpu[event_id][key] = value.cpu().item()
                else:  # Array
                    features_cpu[event_id][key] = value.cpu().numpy()
            else:
                features_cpu[event_id][key] = value
    return features_cpu


def precompute_event_voxel_indices_GPU(
    calcium_events: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPU version of precompute_event_voxel_indices - EXACT match with CPU
    """
    # Utiliser exactement la même logique que la version CPU
    coords = torch.nonzero(calcium_events > 0, as_tuple=False)  # Équivalent np.argwhere
    event_ids = calcium_events[calcium_events > 0]  # Équivalent direct
    return coords, event_ids


def compute_features_GPU(
    calcium_events: torch.Tensor,
    events_ids: int,
    image_amplitude: torch.Tensor,
    params_values: dict = None,
) -> dict:
    """
    CORRECTION: Version GPU qui reproduit EXACTEMENT la logique CPU
    """
    if calcium_events.ndim != 4 or image_amplitude.ndim != 4:
        raise ValueError("Inputs must be 4D arrays (T, Z, Y, X)")

    device = calcium_events.device

    voxel_size_x = float(params_values["features_extraction"]["voxel_size_x"])
    voxel_size_y = float(params_values["features_extraction"]["voxel_size_y"])
    voxel_size_z = float(params_values["features_extraction"]["voxel_size_z"])
    voxel_size = voxel_size_x * voxel_size_y * voxel_size_z

    threshold_median_localized = float(
        params_values["features_extraction"]["threshold_median_localized"]
    )
    volume_localized = float(params_values["features_extraction"]["volume_localized"])

    features = {}
    coords_all, event_ids_all = precompute_event_voxel_indices_GPU(calcium_events)

    for event_id in tqdm(
        range(1, events_ids + 1), desc="Computing features per event (GPU)"
    ):
        mask = event_ids_all == event_id
        if not torch.any(mask):
            continue

        coords = coords_all[mask]
        t, z, y, x = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
        nb_voxels = len(t)
        if nb_voxels == 0:
            continue

        # CORRECTION: Centroid - reproduction exacte de la version CPU
        centroid_x = int(torch.mean(x.float()).item())  # Identique à int(np.mean(x))
        centroid_y = int(torch.mean(y.float()).item())
        centroid_z = int(torch.mean(z.float()).item())
        centroid_t = int(torch.mean(t.float()).item())

        t0, t1 = t.min().item(), t.max().item()
        duration = t1 - t0 + 1
        volume = nb_voxels * voxel_size

        # CORRECTION: Amplitude - même calcul exact que CPU
        amplitude_values = image_amplitude[t, z, y, x]
        amplitude = torch.max(amplitude_values).item()

        # CORRECTION: Classification - utiliser la même logique EXACTE que CPU
        if duration > 1:
            # CRUCIAL: Convertir en numpy pour utiliser exactement la même fonction
            coords_np = coords.cpu().numpy()

            class_result = is_localized_GPU_exact(
                coords_np,  # Maintenant numpy comme en CPU
                t0,
                duration,
                volume,
                voxel_size_x,
                voxel_size_y,
                voxel_size_z,
                threshold_median_localized,
                volume_localized,
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
            confidence = 0.0
            mean_displacement = 0.0
            median_displacement = 0.0

        features[event_id] = {
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

    return features


def is_localized_GPU_exact(
    coords: np.ndarray,  # CORRECTION: Utiliser numpy pour exactitude
    t0: int,
    duration: int,
    volume: float,
    voxel_size_x: float,
    voxel_size_y: float,
    voxel_size_z: float,
    threshold_median_localized: float,
    volume_localized: float,
) -> dict:
    """
    CORRECTION: Reproduction EXACTE de la fonction CPU is_localized
    Utilise numpy pour garantir les mêmes calculs numériques
    """
    # Compute centroids per time frame (EXACTEMENT comme Java/CPU)
    centroids_x = np.zeros(duration)
    centroids_y = np.zeros(duration)
    centroids_z = np.zeros(duration)

    for i, t in enumerate(range(t0, t0 + duration)):
        frame_coords = coords[coords[:, 0] == t]
        if len(frame_coords) > 0:
            # CORRECTION: Utiliser exactement les mêmes indices que CPU
            centroids_x[i] = np.mean(frame_coords[:, 3]) * voxel_size_x  # x = index 3
            centroids_y[i] = np.mean(frame_coords[:, 2]) * voxel_size_y  # y = index 2
            centroids_z[i] = np.mean(frame_coords[:, 1]) * voxel_size_z  # z = index 1

    # Compute distances like Java/CPU (all tau combinations)
    distances = []
    for tau in range(1, duration):
        for i in range(duration - tau):
            x_dist = centroids_x[i + tau] - centroids_x[i]
            y_dist = centroids_y[i + tau] - centroids_y[i]
            z_dist = centroids_z[i + tau] - centroids_z[i]

            distance = np.sqrt(x_dist**2 + y_dist**2 + z_dist**2)
            distances.append(distance)

    distances = np.array(distances)

    # CORRECTION: Utiliser numpy pour les statistiques (même précision)
    median_displacement = np.median(distances)
    mean_displacement = np.mean(distances)
    std_displacement = np.std(distances)

    # Classification logic like Java/CPU
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
    Version GPU de write_csv_features - identique à la version CPU
    """
    df = pd.DataFrame.from_dict(features, orient="index")

    # CORRECTION: Même ordre de colonnes que CPU
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
    df.to_csv(output_file, index_label="Label", sep=";")  # Même séparateur
    print(f"Features saved to {output_file}")
