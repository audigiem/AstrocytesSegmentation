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
        print("No localized events found in features dictionary")
        return

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

            # Header
            csv_writer.writerow(
                [
                    "T0",
                    "Coactive Labels",
                    "Event Count",
                    "All Distances [µm]",
                    "Mean Distance [µm]",
                    "Median Distance [µm]",
                    "Std Distance [µm]",
                    "Min Distance [µm]",
                    "Max Distance [µm]",
                    "Mean Centroid X",
                    "Mean Centroid Y",
                    "Mean Centroid Z",
                    "Spatial Span X [µm]",
                    "Spatial Span Y [µm]",
                    "Spatial Span Z [µm]",
                ]
            )

            # Write data
            for t0, labels, stats in zip(list_t0, list_list_label, list_coactive_stats):
                # Convert label list to string
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

    except IOError as e:
        print(f"Writing error: {e}")


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
