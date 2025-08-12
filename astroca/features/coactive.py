import csv
from typing import List, Set, Dict, Tuple
import os
import numpy as np
import math


def compute_coactive_csv(
    features: dict, path_output_dir: str, params_values: dict = None
) -> None:
    """
    Processes features dictionary to identify coactive labels and generates a new CSV.

    Args:
        features: Dictionary containing features data
        path_output_dir: Output directory path
        params_values: Dictionary containing voxel size parameters
    """

    # Extract voxel sizes if provided
    voxel_size_x = 1.0
    voxel_size_y = 1.0
    voxel_size_z = 1.0

    if params_values and "features_extraction" in params_values:
        voxel_size_x = float(params_values["features_extraction"]["voxel_size_x"])
        voxel_size_y = float(params_values["features_extraction"]["voxel_size_y"])
        voxel_size_z = float(params_values["features_extraction"]["voxel_size_z"])

    # Extract localized events from features dictionary
    localized_events = _extract_localized_events(features)
    if not localized_events:
        raise ValueError("No localized events found in features data")

    # Optimized data structures
    list_list_label: List[List[int]] = []
    list_t0: List[int] = []
    list_coactive_stats: List[Dict] = []
    checked_labels: Set[int] = set()  # Set for O(1) lookup

    nb_events = len(localized_events)

    for i in range(nb_events):
        event_id, event_data = localized_events[i]

        # Skip if already processed
        if event_id in checked_labels:
            continue

        t0_int = event_data["T0 [frame]"]

        # Initialize list of coactive labels and their data
        current_labels = [event_id]
        current_events_data = [event_data]
        checked_labels.add(event_id)

        # Search for coactive labels with the same T0
        for j in range(i + 1, nb_events):
            event_id_j, event_data_j = localized_events[j]

            if event_id_j in checked_labels:
                continue

            t0_int_j = event_data_j["T0 [frame]"]

            if t0_int_j == t0_int:
                current_labels.append(event_id_j)
                current_events_data.append(event_data_j)
                checked_labels.add(event_id_j)

        # Calculate coactive statistics
        coactive_stats = _calculate_coactive_statistics(
            current_events_data, voxel_size_x, voxel_size_y, voxel_size_z
        )

        list_list_label.append(current_labels)
        list_t0.append(t0_int)
        list_coactive_stats.append(coactive_stats)

    # Write results
    write_csv_coactive(path_output_dir, list_list_label, list_t0, list_coactive_stats)
    print(f"COMPLETED - Found {len(list_list_label)} coactive groups")


def _extract_localized_events(features: dict) -> List[Tuple[int, Dict]]:
    """
    Extracts localized events from features dictionary.

    Args:
        features: Dictionary with feature data

    Returns:
        List of tuples containing (event_id, event_data) for localized events
    """
    localized_events = []

    for event_id, event_data in features.items():
        if event_data["Class"] == "Localized":
            localized_events.append((event_id, event_data))

    return localized_events


def _calculate_coactive_statistics(
    events_data: List[Dict],
    voxel_size_x: float,
    voxel_size_y: float,
    voxel_size_z: float,
) -> Dict:
    """
    Calculate statistics for coactive events including distances between all pairs.

    Args:
        events_data: List of event data dictionaries
        voxel_size_x, voxel_size_y, voxel_size_z: Voxel sizes for distance calculation

    Returns:
        Dictionary with coactive statistics
    """
    n_events = len(events_data)

    if n_events == 1:
        return {
            "event_count": 1,
            "mean_distance": 0.0,
            "median_distance": 0.0,
            "std_distance": 0.0,
            "min_distance": 0.0,
            "max_distance": 0.0,
            "all_distances": [],
            "mean_centroid_x": events_data[0]["CentroidX [voxel]"],
            "mean_centroid_y": events_data[0]["CentroidY [voxel]"],
            "mean_centroid_z": events_data[0]["CentroidZ [voxel]"],
            "spatial_span_x": 0.0,
            "spatial_span_y": 0.0,
            "spatial_span_z": 0.0,
        }

    # Extract centroids
    centroids = []
    for event in events_data:
        centroids.append(
            [
                event["CentroidX [voxel]"],
                event["CentroidY [voxel]"],
                event["CentroidZ [voxel]"],
            ]
        )

    centroids = np.array(centroids)

    # Calculate all pairwise distances
    distances = []
    for i in range(n_events):
        for j in range(i + 1, n_events):
            x_dist = (centroids[i, 0] - centroids[j, 0]) * voxel_size_x
            y_dist = (centroids[i, 1] - centroids[j, 1]) * voxel_size_y
            z_dist = (centroids[i, 2] - centroids[j, 2]) * voxel_size_z

            distance = math.sqrt(x_dist**2 + y_dist**2 + z_dist**2)
            distances.append(distance)

    distances = np.array(distances)

    # Calculate spatial statistics
    mean_centroid = np.mean(centroids, axis=0)
    spatial_spans = np.max(centroids, axis=0) - np.min(centroids, axis=0)

    return {
        "event_count": n_events,
        "mean_distance": float(np.mean(distances)),
        "median_distance": float(np.median(distances)),
        "std_distance": float(np.std(distances)),
        "min_distance": float(np.min(distances)),
        "max_distance": float(np.max(distances)),
        "all_distances": distances.tolist(),
        "mean_centroid_x": float(mean_centroid[0]),
        "mean_centroid_y": float(mean_centroid[1]),
        "mean_centroid_z": float(mean_centroid[2]),
        "spatial_span_x": float(spatial_spans[0] * voxel_size_x),
        "spatial_span_y": float(spatial_spans[1] * voxel_size_y),
        "spatial_span_z": float(spatial_spans[2] * voxel_size_z),
    }


def write_csv_coactive(
    path_output_dir: str,
    list_list_label: List[List[int]],
    list_t0: List[int],
    list_coactive_stats: List[Dict],
) -> None:
    """
    Writes coactive data to a CSV file with distance statistics.

    Args:
        path_output_dir: Output directory path
        list_list_label: List of coactive label groups
        list_t0: List of corresponding T0 values
        list_coactive_stats: List of coactive statistics
    """
    try:
        # Ensure directory exists
        os.makedirs(path_output_dir, exist_ok=True)

        output_path = os.path.join(path_output_dir, "coactive.csv")

        with open(output_path, "w", newline="", encoding="utf-8") as file:
            csv_writer = csv.writer(file, delimiter=";")

            # Header with units
            csv_writer.writerow(
                [
                    "T0 [frame]",
                    "Coactive Labels [IDs]",
                    "Event Count [count]",
                    "All Distances [µm]",
                    "Mean Distance [µm]",
                    "Median Distance [µm]",
                    "Std Distance [µm]",
                    "Min Distance [µm]",
                    "Max Distance [µm]",
                    "Mean Centroid X [voxel]",
                    "Mean Centroid Y [voxel]",
                    "Mean Centroid Z [voxel]",
                    "Spatial Span X [µm]",
                    "Spatial Span Y [µm]",
                    "Spatial Span Z [µm]",
                ]
            )

            # Write data (reste inchangé)
            for t0, labels, stats in zip(list_t0, list_list_label, list_coactive_stats):
                if len(labels) > 1:
                    labels_str = "[" + "; ".join(f"{i}" for i in labels) + "]"
                    distances_str = (
                        "["
                        + "; ".join(f"{d:.3f}" for d in stats["all_distances"])
                        + "]"
                    )
                else:
                    labels_str = labels
                    distances_str = "None"
                csv_writer.writerow(
                    [
                        t0,
                        labels_str,
                        stats["event_count"],
                        distances_str,
                        f"{stats['mean_distance']}",
                        f"{stats['median_distance']}",
                        f"{stats['std_distance']}",
                        f"{stats['min_distance']}",
                        f"{stats['max_distance']}",
                        f"{stats['mean_centroid_x']}",
                        f"{stats['mean_centroid_y']}",
                        f"{stats['mean_centroid_z']}",
                        f"{stats['spatial_span_x']}",
                        f"{stats['spatial_span_y']}",
                        f"{stats['spatial_span_z']}",
                    ]
                )

        print(f"Coactive events written to: {output_path}")

        # Créer le fichier de documentation
        _create_documentation_file(path_output_dir)

    except IOError as e:
        print(f"Writing error: {e}")


def _create_documentation_file(path_output_dir: str) -> None:
    """
    Creates a documentation file explaining the coactive CSV columns.

    Args:
        path_output_dir: Output directory path
    """
    doc_path = os.path.join(path_output_dir, "coactive_columns_documentation.txt")

    documentation = """DOCUMENTATION - FICHIER COACTIVE.CSV
=====================================

Ce fichier contient l'analyse des événements co-actifs (événements qui se produisent au même instant T0).

DESCRIPTION DES COLONNES :
-------------------------

T0 [frame] :
- Description : Instant temporel où se produisent les événements co-actifs
- Unité : frame (numéro de frame dans la séquence temporelle)
- Calcul : Valeur directe extraite des données d'événements localisés

Coactive Labels [IDs] :
- Description : Liste des identifiants (labels) des événements co-actifs
- Unité : IDs (identifiants numériques)
- Format : [ID1; ID2; ID3] pour plusieurs événements, ou ID unique pour un événement isolé
- Calcul : Regroupement de tous les événements ayant le même T0

Event Count [count] :
- Description : Nombre d'événements co-actifs à cet instant T0
- Unité : count (nombre d'événements)
- Calcul : Longueur de la liste des labels co-actifs

All Distances [µm] :
- Description : Liste de toutes les distances entre paires d'événements co-actifs
- Unité : µm (micromètres)
- Format : [dist1; dist2; dist3] ou "None" pour les événements uniques
- Calcul : Distance euclidienne 3D entre tous les couples de centroïdes d'événements

Mean Distance [µm] :
- Description : Distance moyenne entre tous les événements co-actifs
- Unité : µm (micromètres)
- Calcul : Moyenne arithmétique de toutes les distances par paires
- Valeur : 0.0 pour les événements uniques

Median Distance [µm] :
- Description : Distance médiane entre tous les événements co-actifs
- Unité : µm (micromètres)
- Calcul : Médiane de toutes les distances par paires
- Valeur : 0.0 pour les événements uniques

Std Distance [µm] :
- Description : Écart-type des distances entre événements co-actifs
- Unité : µm (micromètres)
- Calcul : Écart-type de toutes les distances par paires
- Valeur : 0.0 pour les événements uniques

Min Distance [µm] :
- Description : Distance minimale entre événements co-actifs
- Unité : µm (micromètres)
- Calcul : Minimum de toutes les distances par paires
- Valeur : 0.0 pour les événements uniques

Max Distance [µm] :
- Description : Distance maximale entre événements co-actifs
- Unité : µm (micromètres)
- Calcul : Maximum de toutes les distances par paires
- Valeur : 0.0 pour les événements uniques

Mean Centroid X [voxel] :
- Description : Position moyenne des centroïdes sur l'axe X
- Unité : voxel (coordonnée en voxels)
- Calcul : Moyenne des coordonnées X de tous les centroïdes des événements co-actifs

Mean Centroid Y [voxel] :
- Description : Position moyenne des centroïdes sur l'axe Y
- Unité : voxel (coordonnée en voxels)
- Calcul : Moyenne des coordonnées Y de tous les centroïdes des événements co-actifs

Mean Centroid Z [voxel] :
- Description : Position moyenne des centroïdes sur l'axe Z
- Unité : voxel (coordonnée en voxels)
- Calcul : Moyenne des coordonnées Z de tous les centroïdes des événements co-actifs

Spatial Span X [µm] :
- Description : Étendue spatiale des événements co-actifs sur l'axe X
- Unité : µm (micromètres)
- Calcul : (Max_X - Min_X) * voxel_size_x des centroïdes
- Valeur : 0.0 pour les événements uniques

Spatial Span Y [µm] :
- Description : Étendue spatiale des événements co-actifs sur l'axe Y
- Unité : µm (micromètres)
- Calcul : (Max_Y - Min_Y) * voxel_size_y des centroïdes
- Valeur : 0.0 pour les événements uniques

Spatial Span Z [µm] :
- Description : Étendue spatiale des événements co-actifs sur l'axe Z
- Unité : µm (micromètres)
- Calcul : (Max_Z - Min_Z) * voxel_size_z des centroïdes
- Valeur : 0.0 pour les événements uniques

NOTES IMPORTANTES :
------------------
- Les distances sont calculées en utilisant les tailles de voxels spécifiées dans les paramètres
- Les événements uniques (Event Count = 1) ont des valeurs de distance et d'étendue spatiale de 0.0
- Les coordonnées de centroïdes sont en voxels, les distances et étendues spatiales en micromètres
- Le délimiteur du CSV est le point-virgule (;)

FORMULE DE CALCUL DES DISTANCES :
--------------------------------
Distance = √[(X₁-X₂)²×voxel_size_x² + (Y₁-Y₂)²×voxel_size_y² + (Z₁-Z₂)²×voxel_size_z²]

Généré automatiquement par le module coactive.py
"""

    try:
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(documentation)
        print(f"Documentation créée : {doc_path}")
    except IOError as e:
        print(f"Erreur lors de la création de la documentation : {e}")


def analyze_coactive_patterns(list_coactive_stats: List[Dict]) -> Dict:
    """
    Analyze patterns in coactive events.

    Args:
        list_coactive_stats: List of coactive statistics

    Returns:
        Dictionary with pattern analysis
    """
    if not list_coactive_stats:
        return {}

    # Filter out single events for distance analysis
    multi_event_stats = [
        stats for stats in list_coactive_stats if stats["event_count"] > 1
    ]

    if not multi_event_stats:
        return {
            "total_coactive_groups": len(list_coactive_stats),
            "single_event_groups": len(list_coactive_stats),
            "multi_event_groups": 0,
        }

    # Extract metrics for analysis
    mean_distances = [stats["mean_distance"] for stats in multi_event_stats]
    event_counts = [stats["event_count"] for stats in list_coactive_stats]
    spatial_spans_x = [stats["spatial_span_x"] for stats in multi_event_stats]
    spatial_spans_y = [stats["spatial_span_y"] for stats in multi_event_stats]
    spatial_spans_z = [stats["spatial_span_z"] for stats in multi_event_stats]

    return {
        "total_coactive_groups": len(list_coactive_stats),
        "single_event_groups": len(
            [s for s in list_coactive_stats if s["event_count"] == 1]
        ),
        "multi_event_groups": len(multi_event_stats),
        "average_events_per_group": float(np.mean(event_counts)),
        "median_events_per_group": float(np.median(event_counts)),
        "max_events_per_group": max(event_counts),
        "average_mean_distance": float(np.mean(mean_distances)),
        "median_mean_distance": float(np.median(mean_distances)),
        "std_mean_distance": float(np.std(mean_distances)),
        "average_spatial_span_x": float(np.mean(spatial_spans_x)),
        "average_spatial_span_y": float(np.mean(spatial_spans_y)),
        "average_spatial_span_z": float(np.mean(spatial_spans_z)),
        "compact_groups": len(
            [s for s in multi_event_stats if s["mean_distance"] < 10.0]
        ),  # < 10µm
        "dispersed_groups": len(
            [s for s in multi_event_stats if s["mean_distance"] >= 10.0]
        ),  # >= 10µm
    }


def get_coactive_summary(
    features: dict, path_output_dir: str, params_values: dict = None
) -> Dict:
    """
    Generate a complete summary of coactive analysis.

    Args:
        features: Dictionary containing features data
        path_output_dir: Output directory path
        params_values: Dictionary containing parameters

    Returns:
        Dictionary with complete analysis summary
    """
    # Run coactive analysis
    localized_events = _extract_localized_events(features)

    if not localized_events:
        return {"error": "No localized events found"}

    # Extract voxel sizes
    voxel_size_x = 1.0
    voxel_size_y = 1.0
    voxel_size_z = 1.0

    if params_values and "features_extraction" in params_values:
        voxel_size_x = float(params_values["features_extraction"]["voxel_size_x"])
        voxel_size_y = float(params_values["features_extraction"]["voxel_size_y"])
        voxel_size_z = float(params_values["features_extraction"]["voxel_size_z"])

    # Process coactive events
    list_coactive_stats = []
    unique_t0_values = set()

    for event_id, event_data in localized_events:
        unique_t0_values.add(event_data["T0 [frame]"])

    # Group by T0 and calculate statistics
    for t0 in unique_t0_values:
        t0_events = [
            (eid, ed) for eid, ed in localized_events if ed["T0 [frame]"] == t0
        ]
        events_data = [ed for _, ed in t0_events]

        coactive_stats = _calculate_coactive_statistics(
            events_data, voxel_size_x, voxel_size_y, voxel_size_z
        )
        list_coactive_stats.append(coactive_stats)

    # Analyze patterns
    pattern_analysis = analyze_coactive_patterns(list_coactive_stats)

    return {
        "total_localized_events": len(localized_events),
        "unique_time_points": len(unique_t0_values),
        "pattern_analysis": pattern_analysis,
        "processing_summary": {
            "voxel_sizes": {"x": voxel_size_x, "y": voxel_size_y, "z": voxel_size_z},
            "output_file": os.path.join(path_output_dir, "coactive.csv"),
        },
    }
