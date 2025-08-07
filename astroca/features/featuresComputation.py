"""
@file featuresComputation.py
@brief This module provides functionality to compute features from a 3D image sequence with time dimension.
"""

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from astroca.features.coactive import compute_coactive_csv
from astroca.features.hotspots import compute_hot_spots_from_features, write_csv_hot_spots


# @profile
def save_features_from_events(calcium_events: np.ndarray, events_ids: int, image_amplitude: np.ndarray, params_values: dict=None) -> None:
    """
    @brief Compute features from calcium events in a 3D image sequence with time dimension and save them to an Excel file.
        - duration: nb of frames of the event
        - t0: time of the first frame of the event
        - amplitude: amplitude if the intensity of the event
        - volume: volume of the event in micrometers^3
        - centroïd: coordinates of the center of mass of the event
        - classification: classification of the event (wave, local, etc.)
    @param calcium_events: 4D numpy array of shape (T, Z, Y, X) representing the calcium events.
    @param events_ids: Number of unique event IDs from the calcium events data.
    @param image_amplitude: 4D numpy array of shape (T, Z, Y, X) representing the amplitude of the image.
    @param params_values: Dictionary containing parameters for feature computation:
        - voxel_size_x, voxel_size_y, voxel_size_z: size of a voxel in micrometers.
        - threshold_median_localized: threshold for median distance to classify as localized.
        - volume_localized: threshold for volume to classify as localized.
        - save_result: Boolean flag to indicate whether to save the results.
        - output_directory: Directory to save the results if save_result is True.
    @return:
    """
    print("=== Computing features from calcium events ===")
    required_keys = {'paths', 'save', 'features_extraction'}
    if not required_keys.issubset(params_values.keys()):
        raise ValueError(f"Missing required parameters: {required_keys - params_values.keys()}")
    features = compute_features(calcium_events, events_ids, image_amplitude, params_values)

    save_result = int(params_values['save']['save_features']) == 1
    output_directory = params_values['paths']['output_dir']

    hot_spots = compute_hot_spots_from_features(features, params_values)

    if save_result:
        if output_directory is None:
            raise ValueError("Output directory must be specified when save_result is True.")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        # write_excel_features(features, output_directory)
        write_csv_features(features, output_directory)
        compute_coactive_csv(features, output_directory)
        write_csv_hot_spots(hot_spots, output_directory)

    print(60*"=")
    
# @profile
def precompute_event_voxel_indices(calcium_events):
    """
    Precompute indices and event_ids for all non-zero voxels in the 4D calcium_events array.
    Returns:
        coords: ndarray of shape (N, 4), with each row being (t, z, y, x)
        event_ids: ndarray of shape (N,), corresponding to the event ID at each coord
    """
    coords = np.argwhere(calcium_events > 0)
    event_ids = calcium_events[calcium_events > 0]
    return coords, event_ids

# @profile
def compute_features(calcium_events: np.ndarray, events_ids: int, image_amplitude: np.ndarray, params_values: dict=None) -> dict:
    """
    @brief Compute features from calcium events in a 3D image sequence with time dimension.
        - duration: nb of frames of the event
        - t0: time of the first frame of the event
        - amplitude: amplitude if the intensity of the event
        - volume: volume of the event in micrometers^3
        - centroïd: coordinates of the center of mass of the event
        - classification: classification of the event (wave, local, etc.)
    @param calcium_events: 4D numpy array of shape (T, Z, Y, X) representing the calcium events.
    @param events_ids: Number of unique event IDs from the calcium events data.
    @param image_amplitude: 4D numpy array of shape (T, Z, Y, X) representing the amplitude of the image.
    @param params_values: Dictionary containing parameters for feature computation:
        - voxel_size_x, voxel_size_y, voxel_size_z: size of a voxel in micrometers.
        - threshold_median_localized: threshold for median distance to classify as localized.
        - volume_localized: threshold for volume to classify as localized.
        - save_result: Boolean flag to indicate whether to save the results.
        - output_directory: Directory to save the results if save_result is True.
    @return: Dictionary containing computed features.
    """
    if calcium_events.ndim != 4 or image_amplitude.ndim != 4:
        raise ValueError("Inputs must be 4D arrays (T, Z, Y, X)")

    voxel_size_x = float(params_values['features_extraction']['voxel_size_x'])
    voxel_size_y = float(params_values['features_extraction']['voxel_size_y'])
    voxel_size_z = float(params_values['features_extraction']['voxel_size_z'])
    voxel_size = voxel_size_x * voxel_size_y * voxel_size_z
    
    threshold_median_localized = float(params_values['features_extraction']['threshold_median_localized'])
    volume_localized = float(params_values['features_extraction']['volume_localized'])

    features = {}
    coords_all, event_ids_all = precompute_event_voxel_indices(calcium_events)

    for event_id in tqdm(range(1, events_ids + 1), desc="Computing features per event"):
        mask = (event_ids_all == event_id)
        if not np.any(mask):
            continue

        coords = coords_all[mask]
        t, z, y, x = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
        nb_voxels = len(t)
        if nb_voxels == 0:
            continue

        # Centroid
        centroid_x = int(np.mean(x))
        centroid_y = int(np.mean(y))
        centroid_z = int(np.mean(z))
        centroid_t = int(np.mean(t))
        t0, t1 = t.min(), t.max()
        duration = t1 - t0 + 1
        volume = nb_voxels * voxel_size

        # Event amplitude
        amplitude = np.max(image_amplitude[t, z, y, x])

        # Classification
        if duration > 1:
            coords_stacked = np.column_stack((t, z, y, x))
            class_result = is_localized(coords_stacked, t0, duration, volume,
                                        voxel_size_x, voxel_size_y, voxel_size_z,
                                        threshold_median_localized, volume_localized)
            class_label = class_result['class']
            confidence = class_result['confidence']
            mean_displacement = class_result['mean_displacement']
            median_displacement = class_result['median_displacement']
        else:
            if volume <= volume_localized:
                class_label = "Localized"
            else:
                class_label = "Localized but not in a microdomain"
            confidence = 0.0
            mean_displacement = 0.0
            median_displacement = 0.0

        features[event_id] = {
            'T0 [frame]': t0,
            'Duration [frame]': duration,
            'CentroidX [voxel]': centroid_x,
            'CentroidY [voxel]': centroid_y,
            'CentroidZ [voxel]': centroid_z,
            'CentroidT [voxel]': centroid_t,
            'Volume [µm^3]': volume,
            'Amplitude': amplitude,
            'Class': class_label,
            'STD displacement [µm]': confidence,
            'Mean displacement [µm]': mean_displacement,
            'Median displacement [µm]': median_displacement
        }

    return features


# @profile
def is_localized(coords, t0: int, duration: int, volume: float, 
                voxel_size_x: float, voxel_size_y: float, voxel_size_z: float,
                threshold_median_localized: float, volume_localized: float) -> dict:
    """
    Classify event as 'Wave', 'Localized', or 'Localized but not in a microdomain'
    based on centroid dynamics and volume - following Java logic exactly.
    """
    
    # Compute centroids per time frame (like Java)
    centroids_x = np.zeros(duration)
    centroids_y = np.zeros(duration)
    centroids_z = np.zeros(duration)
    
    for i, t in enumerate(range(t0, t0 + duration)):
        frame_coords = coords[coords[:, 0] == t]
        if len(frame_coords) > 0:
            centroids_x[i] = np.mean(frame_coords[:, 3]) * voxel_size_x
            centroids_y[i] = np.mean(frame_coords[:, 2]) * voxel_size_y
            centroids_z[i] = np.mean(frame_coords[:, 1]) * voxel_size_z
    
    # Compute distances like Java (all tau combinations)
    distances = []
    for tau in range(1, duration):
        for i in range(duration - tau):
            x_dist = centroids_x[i + tau] - centroids_x[i]
            y_dist = centroids_y[i + tau] - centroids_y[i]
            z_dist = centroids_z[i + tau] - centroids_z[i]
            
            distance = np.sqrt(x_dist**2 + y_dist**2 + z_dist**2)
            distances.append(distance)
    
    distances = np.array(distances)
    median_displacement = np.median(distances)
    mean_displacement = np.mean(distances)
    std_displacement = np.std(distances)
    
    # Classification logic like Java
    localized = median_displacement <= threshold_median_localized
    
    if localized:
        if volume <= volume_localized:
            class_label = "Localized"
        else:
            class_label = "Localized but not in a microdomain"
    else:
        class_label = "Wave"
    
    return {
        'class': class_label,
        'confidence': std_displacement,
        'mean_displacement': mean_displacement,
        'median_displacement': median_displacement
    }

# @profile
def write_excel_features(features: dict, output_directory: str) -> None:
    """
    Write the computed features to an Excel file.

    @param features: Dictionary containing computed features.
    @param output_directory: Directory to save the Excel file.
    """
    df = pd.DataFrame.from_dict(features, orient='index')
    # Réorganiser les colonnes pour correspondre à l'ordre Java
    columns_order = ['T0 [frame]', 'Duration [frame]', 'CentroidX [voxel]', 'CentroidY [voxel]', 'CentroidZ [voxel]', 'CentroidT [voxel]',
                    'Volume [µm^3]', 'Amplitude', 'Class', 'STD displacement [µm]', 'Mean displacement [µm]', 'Median displacement [µm]']
    df = df[columns_order]
    
    output_file = os.path.join(output_directory, "Features.xlsx")
    df.to_excel(output_file, index_label='Label')
    print(f"Features saved to {output_file}")

# @profile
def write_csv_features(features: dict, output_directory: str) -> None:
    """
    Write the computed features to a CSV file.

    @param features: Dictionary containing computed features.
    @param output_directory: Directory to save the CSV file.
    """
    df = pd.DataFrame.from_dict(features, orient='index')
    # Réorganiser les colonnes pour correspondre à l'ordre Java
    columns_order = ['T0 [frame]', 'Duration [frame]', 'CentroidX [voxel]', 'CentroidY [voxel]', 'CentroidZ [voxel]', 'CentroidT [voxel]',
                    'Volume [µm^3]', 'Amplitude', 'Class', 'STD displacement [µm]', 'Mean displacement [µm]', 'Median displacement [µm]']
    df = df[columns_order]
    
    output_file = os.path.join(output_directory, "Features.csv")
    df.to_csv(output_file, index_label='Label', sep=';')
    print(f"Features saved to {output_file}")