"""
@file featuresComputation.py
@brief This module provides functionality to compute features from a 3D image sequence with time dimension.
"""

import numpy as np
import pandas as pd
import os
from tqdm import tqdm


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
        - threshold_distance_localized: threshold for distance to classify as wave.
        - volume_localized: threshold for volume to classify as localized.
        - save_result: Boolean flag to indicate whether to save the results.
        - output_directory: Directory to save the results if save_result is True.
    @return:
    """
    print("=== Computing features from calcium events ===")
    required_keys = {'paths', 'files', 'features_extraction'}
    if not required_keys.issubset(params_values.keys()):
        raise ValueError(f"Missing required parameters: {required_keys - params_values.keys()}")
    features = compute_features(calcium_events, events_ids, image_amplitude, params_values)

    save_result = int(params_values['files']['save_results']) == 1
    output_directory = params_values['paths']['output_dir']

    if save_result:
        if output_directory is None:
            raise ValueError("Output directory must be specified when save_result is True.")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        write_excel_features(features, output_directory)
        write_csv_features(features, output_directory)
    print(60*"=")


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
        - threshold_distance_localized: threshold for distance to classify as wave.
        - volume_localized: threshold for volume to classify as localized.
        - save_result: Boolean flag to indicate whether to save the results.
        - output_directory: Directory to save the results if save_result is True.
    @return: Dictionary containing computed features.
    """
    if calcium_events.ndim != 4 or image_amplitude.ndim != 4:
        raise ValueError("Inputs must be 4D arrays (T, Z, Y, X)")

    voxel_size = np.array([
        float(params_values['features_extraction']['voxel_size_x']),
        float(params_values['features_extraction']['voxel_size_y']),
        float(params_values['features_extraction']['voxel_size_z']),
    ])
    volume_localized = float(params_values['features_extraction']['volume_localized'])

    features = {}

    for event_id in tqdm(range(1, events_ids + 1), desc="Computing features per event"):
        event_mask = (calcium_events == event_id)
        if not np.any(event_mask):
            continue

        coords = np.argwhere(event_mask)
        nb_voxels = coords.shape[0]
        t0 = coords[:, 0].min()
        t1 = coords[:, 0].max()
        duration = t1 - t0 + 1
        centroid_x = coords[:, 3].mean().astype(int)
        centroid_y = coords[:, 2].mean().astype(int)
        centroid_z = coords[:, 1].mean().astype(int)
        centroid_t = coords[:, 0].mean().astype(int)
        volume = nb_voxels * np.prod(voxel_size)
        amplitude = np.max(image_amplitude[event_mask])

        # Classification
        if duration > 1:
            class_label, confidence = classify_event(coords, t0, duration, volume, params_values)
        else:
            if volume <= volume_localized:
                class_label = "Localized"
                confidence = 100
            else:
                class_label = "Localized but no microdomain"
                confidence = 100

        features[event_id] = {
            'T0': t0,
            'Duration': duration,
            'CentroidX': centroid_x,
            'CentroidY': centroid_y,
            'CentroidZ': centroid_z,
            'CentroidT': centroid_t,
            'Volume': volume,
            'Amplitude': amplitude,
            'Class': class_label,
            'Class confidence [%]': confidence
        }

    return features

def classify_event(coords , t0: int, duration: int, volume: float, params_values: dict) -> tuple:
    """
    Classify event as 'Wave', 'Localized', or 'Localized but no microdomain'
    based on centroid dynamics and volume.
    """
    threshold_dist = float(params_values['features_extraction']['threshold_distance_localized'])
    threshold_med = float(params_values['features_extraction']['threshold_median_localized'])
    volume_localized = float(params_values['features_extraction']['volume_localized'])

    # Compute centroids per time frame
    centroids = np.full((duration, 3), np.nan)
    for i, t in enumerate(range(t0, t0 + duration)):
        frame_coords = coords[coords[:, 0] == t]
        if frame_coords.size > 0:
            centroids[i] = frame_coords[:, 1:4].mean(axis=0)

    dists = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
    if np.any(dists > threshold_dist):
        idx = np.argmax(dists > threshold_dist)
        confidence = min(100, dists[idx] * 80 / threshold_dist)
        return "Wave", confidence
    else:
        median_val = np.nanmedian(dists)
        confidence = min(100, median_val * 50 / threshold_med)
        if volume <= volume_localized:
            return "Localized", confidence
        else:
            return "Localized but no microdomain", confidence

def write_excel_features(features: dict, output_directory: str) -> None:
    """
    Write the computed features to an Excel file.

    @param features: Dictionary containing computed features.
    @param output_directory: Directory to save the Excel file.
    """
    df = pd.DataFrame.from_dict(features, orient='index')
    output_file = f"{output_directory}calcium_events_features.xlsx"
    df.to_excel(output_file, index_label='Event ID')
    print(f"Features saved to {output_file}")


def write_csv_features(features: dict, output_directory: str) -> None:
    """
    Write the computed features to a CSV file.

    @param features: Dictionary containing computed features.
    @param output_directory: Directory to save the CSV file.
    """
    df = pd.DataFrame.from_dict(features, orient='index')
    output_file = f"{output_directory}calcium_events_features.csv"
    df.to_csv(output_file, index_label='Label')
    print(f"Features saved to {output_file}")