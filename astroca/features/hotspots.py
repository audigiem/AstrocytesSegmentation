import math
from typing import List, Dict, Tuple
import numpy as np
import csv
import os


def compute_hot_spots_from_features(features_dict: Dict, params: Dict) -> List[Dict]:
    """
    Identifies hot spots directly from features dictionary based on Euclidean distance
    between centroids of localized events.

    Args:
        features_dict: Dictionary containing event features with keys like event_id and feature values
        params: Dictionary containing parameters for hot spot detection

    Returns:
        List of hot spot groups with temporal analysis
    """
    required_keys = {'features_extraction'}
    if not required_keys.issubset(params.keys()):
        raise ValueError(f"Missing required keys in params: {required_keys - set(params.keys())}")

    # Extract parameters
    voxel_size_x = float(params['features_extraction']['voxel_size_x'])
    voxel_size_y = float(params['features_extraction']['voxel_size_y'])
    voxel_size_z = float(params['features_extraction']['voxel_size_z'])
    threshold_hot_spot = float(params['features_extraction']['threshold_hot_spots'])

    # Extract localized events directly from features dictionary
    localized_data = _extract_localized_events(features_dict)
    if not localized_data:
        print("No localized events found in features dictionary")
        return []

    # Extract coordinates, labels and time information
    event_data = []
    for event_id, event_features in localized_data.items():
        event_data.append({
            'label': event_id,
            'x': event_features['CentroidX [voxel]'],
            'y': event_features['CentroidY [voxel]'],
            'z': event_features['CentroidZ [voxel]'],
            't0': event_features['T0 [frame]']
        })

    # Sort by time for temporal analysis
    event_data.sort(key=lambda x: x['t0'])

    # Extract arrays for processing
    labels = np.array([item['label'] for item in event_data], dtype=np.int32)
    coordinates = np.array([[item['x'], item['y'], item['z']] for item in event_data], dtype=np.float32)
    time_points = np.array([item['t0'] for item in event_data], dtype=np.int32)

    # Scale coordinates by voxel sizes
    voxel_scales = np.array([voxel_size_x, voxel_size_y, voxel_size_z], dtype=np.float32)
    scaled_coords = coordinates * voxel_scales

    # Find hot spots using optimized clustering with temporal analysis
    hot_spot_groups = _find_hot_spots_with_temporal_analysis(
        labels, coordinates, scaled_coords, time_points, threshold_hot_spot
    )

    return hot_spot_groups


def _extract_localized_events(features_dict: Dict) -> Dict:
    """
    Extracts localized events from features dictionary.

    Args:
        features_dict: Dictionary with feature data

    Returns:
        Dictionary containing only localized event data
    """
    localized_events = {}

    for event_id, event_features in features_dict.items():
        if event_features['Class'] == 'Localized':
            localized_events[event_id] = event_features

    return localized_events


def _find_hot_spots_with_temporal_analysis(labels: np.ndarray, coordinates: np.ndarray,
                                           scaled_coords: np.ndarray, time_points: np.ndarray,
                                           threshold: float) -> List[Dict]:
    """
    Optimized hot spot detection with temporal interval analysis.

    Args:
        labels: Array of event labels
        coordinates: Original coordinates (unscaled)
        scaled_coords: Coordinates scaled by voxel sizes
        time_points: Array of T0 time points for each event
        threshold: Distance threshold for clustering

    Returns:
        List of hot spot groups with temporal statistics
    """
    n_events = len(labels)
    if n_events == 0:
        return []

    # Track processed events
    processed = np.zeros(n_events, dtype=bool)
    hot_spot_groups = []

    for i in range(n_events):
        if processed[i]:
            continue

        # Calculate distances from current event to all others
        distances = np.linalg.norm(scaled_coords - scaled_coords[i], axis=1)

        # Find events within threshold (including current event)
        within_threshold = distances <= threshold
        group_indices = np.where(within_threshold & ~processed)[0]

        if len(group_indices) > 0:
            # Create hot spot group
            group_labels = labels[group_indices].tolist()
            group_coords = coordinates[group_indices]
            group_times = time_points[group_indices]

            # Calculate temporal statistics
            temporal_stats = _calculate_temporal_intervals(group_times)

            hot_spot_group = {
                'CentroidX [voxel]': float(np.mean(group_coords[:, 0])),
                'CentroidY [voxel]': float(np.mean(group_coords[:, 1])),
                'CentroidZ [voxel]': float(np.mean(group_coords[:, 2])),
                'Nb localized events': len(group_indices),
                'Label(s) event(s)': group_labels,
                'time_points': group_times.tolist(),
                'temporal_span': int(group_times.max() - group_times.min()) if len(group_times) > 1 else 0,
                'mean_time_interval': temporal_stats['mean_interval'],
                'median_time_interval': temporal_stats['median_interval'],
                'std_time_interval': temporal_stats['std_interval'],
                'min_time_interval': temporal_stats['min_interval'],
                'max_time_interval': temporal_stats['max_interval']
            }

            hot_spot_groups.append(hot_spot_group)

            # Mark as processed
            processed[group_indices] = True

    return hot_spot_groups


def _calculate_temporal_intervals(time_points: np.ndarray) -> Dict:
    """
    Calculate temporal interval statistics for events in a hot spot.

    Args:
        time_points: Array of T0 time points

    Returns:
        Dictionary with temporal statistics
    """
    if len(time_points) <= 1:
        return {
            'mean_interval': 0.0,
            'median_interval': 0.0,
            'std_interval': 0.0,
            'min_interval': 0,
            'max_interval': 0
        }

    # Sort time points
    sorted_times = np.sort(time_points)

    # Calculate intervals between consecutive events
    intervals = np.diff(sorted_times)

    return {
        'mean_interval': float(np.mean(intervals)),
        'median_interval': float(np.median(intervals)),
        'std_interval': float(np.std(intervals)),
        'min_interval': int(intervals.min()),
        'max_interval': int(intervals.max())
    }

def write_csv_hot_spots(hot_spot_groups: List[Dict], path_output_dir: str) -> None:
    """
    Writes hot spot data to a CSV file with temporal analysis.

    Args:
        hot_spot_groups: List of hot spot groups
        path_output_dir: Output directory path
    """
    if not hot_spot_groups:
        print("No hot spots to write")
        return

    # Ensure output directory exists
    os.makedirs(path_output_dir, exist_ok=True)
    output_file = os.path.join(path_output_dir, "HotSpots.csv")

    try:
        with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'CentroidX [voxel]',
                'CentroidY [voxel]',
                'CentroidZ [voxel]',
                'Nb localized events',
                'Label(s) event(s)',
                'time_points',
                'temporal_span',
                'mean_time_interval',
                'median_time_interval',
                'std_time_interval',
                'min_time_interval',
                'max_time_interval'
            ]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()

            for i, group in enumerate(hot_spot_groups, 1):
                # Format labels and time_points as lists for multiple events
                if group['Nb localized events'] > 1:
                    labels_str = "[" + "; ".join(f"{i}" for i in group['Label(s) event(s)']) + "]"
                    time_points_str = "[" + "; ".join(f"{i}" for i in group['time_points']) + "]"
                else:
                    labels_str = str(group['Label(s) event(s)'][0])
                    time_points_str = str(group['time_points'][0])

                writer.writerow({
                    'CentroidX [voxel]': group['CentroidX [voxel]'],
                    'CentroidY [voxel]': group['CentroidY [voxel]'],
                    'CentroidZ [voxel]': group['CentroidZ [voxel]'],
                    'Nb localized events': group['Nb localized events'],
                    'Label(s) event(s)': labels_str,
                    'time_points': time_points_str,
                    'temporal_span': group['temporal_span'],
                    'mean_time_interval': group['mean_time_interval'],
                    'median_time_interval': group['median_time_interval'],
                    'std_time_interval': group['std_time_interval'],
                    'min_time_interval': group['min_time_interval'],
                    'max_time_interval': group['max_time_interval']
                })

        print(f"Hot spots written to {output_file}")
        print(f"COMPLETED - Found {len(hot_spot_groups)} hot spots")


    except IOError as e:
        print(f"Error writing hot spots file: {e}")


def get_hot_spot_statistics(hot_spot_groups: List[Dict]) -> Dict:
    """
    Analyzes hot spot statistics including temporal analysis.

    Returns:
        Dictionary with comprehensive statistics
    """
    if not hot_spot_groups:
        return {'total_hot_spots': 0}

    event_counts = [group['event_count'] for group in hot_spot_groups]
    temporal_spans = [group['temporal_span'] for group in hot_spot_groups]
    mean_intervals = [group['mean_time_interval'] for group in hot_spot_groups if group['mean_time_interval'] > 0]

    stats = {
        'total_hot_spots': len(hot_spot_groups),
        'total_events_in_hot_spots': sum(event_counts),
        'average_events_per_hot_spot': float(np.mean(event_counts)),
        'median_events_per_hot_spot': float(np.median(event_counts)),
        'max_events_per_hot_spot': max(event_counts),
        'min_events_per_hot_spot': min(event_counts),
        'hot_spot_sizes': event_counts,
        'average_temporal_span': float(np.mean(temporal_spans)),
        'median_temporal_span': float(np.median(temporal_spans)),
        'max_temporal_span': max(temporal_spans),
        'min_temporal_span': min(temporal_spans)
    }

    if mean_intervals:
        stats.update({
            'average_mean_time_interval': float(np.mean(mean_intervals)),
            'median_mean_time_interval': float(np.median(mean_intervals)),
            'overall_temporal_regularity': float(np.std(mean_intervals))
        })

    return stats


def analyze_hot_spot_temporal_patterns(hot_spot_groups: List[Dict]) -> Dict:
    """
    Advanced temporal pattern analysis for hot spots.

    Returns:
        Dictionary with temporal pattern analysis
    """
    if not hot_spot_groups:
        return {}

    patterns = {
        'single_event_spots': 0,
        'regular_intervals': 0,  # CV < 0.5
        'irregular_intervals': 0,  # CV >= 0.5
        'burst_patterns': 0,  # Very short intervals followed by long ones
        'periodic_candidates': []  # Hot spots with potentially periodic behavior
    }

    for group in hot_spot_groups:
        if group['event_count'] == 1:
            patterns['single_event_spots'] += 1
        elif group['event_count'] > 1:
            # Coefficient of variation for interval regularity
            if group['std_time_interval'] > 0 and group['mean_time_interval'] > 0:
                cv = group['std_time_interval'] / group['mean_time_interval']
                if cv < 0.5:
                    patterns['regular_intervals'] += 1
                    if group['event_count'] >= 3:  # Minimum for periodicity analysis
                        patterns['periodic_candidates'].append({
                            'hotspot_id': group['representative_label'],
                            'event_count': group['event_count'],
                            'mean_interval': group['mean_time_interval'],
                            'cv': cv
                        })
                else:
                    patterns['irregular_intervals'] += 1

            # Detect burst patterns (mix of very short and long intervals)
            if (group['min_time_interval'] <= 2 and
                    group['max_time_interval'] >= 10 and
                    group['event_count'] >= 3):
                patterns['burst_patterns'] += 1

    return patterns