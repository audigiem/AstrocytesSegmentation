"""
@file featuresComputation.py
@brief This module provides functionality to compute features from a 3D image sequence with time dimension.
"""

import numpy as np
import pandas as pd
import os


def save_features_from_events(calcium_events: np.ndarray, events_ids: list, image_amplitude: np.ndarray, params_values: dict=None, save_result: bool=False, output_directory: str=None) -> None:
    """
    @brief Compute features from calcium events in a 3D image sequence with time dimension and save them to an Excel file.
        - duration: nb of frames of the event
        - t0: time of the first frame of the event
        - amplitude: amplitude if the intensity of the event
        - volume: volume of the event in micrometers^3
        - centroïd: coordinates of the center of mass of the event
        - classification: classification of the event (wave, local, etc.)
    @param calcium_events: 4D numpy array of shape (T, Z, Y, X) representing the calcium events.
    @param events_ids: List of unique event IDs from the calcium events data.
    @param image_amplitude: 4D numpy array of shape (T, Z, Y, X) representing the amplitude of the image.
    @param params_values: Dictionary containing parameters for feature computation.
    @param save_result: Boolean flag to indicate whether to save the results.
    @param output_directory: Directory to save the results if save_result is True.
    @return:
    """
    features = compute_features(calcium_events, events_ids, image_amplitude, params_values, save_result, output_directory)

    if save_result:
        if output_directory is None:
            raise ValueError("Output directory must be specified when save_result is True.")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        write_excel_features(features, output_directory)
        write_csv_features(features, output_directory)


def compute_features(calcium_events: np.ndarray, events_ids: list, image_amplitude: np.ndarray, params_values: dict=None, save_result: bool=False, output_directory: str=None) -> dict:
    """
    @brief Compute features from calcium events in a 3D image sequence with time dimension.
        - duration: nb of frames of the event
        - t0: time of the first frame of the event
        - amplitude: amplitude if the intensity of the event
        - volume: volume of the event in micrometers^3
        - centroïd: coordinates of the center of mass of the event
        -classification: classification of the event (wave, local, etc.)

    @param calcium_events: 4D numpy array of shape (T, Z, Y, X) representing the calcium events.
    @param events_ids: List of unique event IDs from the calcium events data.
    @param image_amplitude: 4D numpy array of shape (T, Z, Y, X) representing the amplitude of the image.
    @param params_values: dictionary containing parameters for feature computation.
    @param save_result: Boolean flag to indicate whether to save the results.
    @param output_directory: Directory to save the results if save_result is True.
    @return: Dictionary containing computed features.
    """
    if calcium_events.ndim != 4:
        raise ValueError("Input must be a 4D numpy array of shape (T, Z, Y, X).")
    if image_amplitude.ndim != 4:
        raise ValueError("Image amplitude must be a 4D numpy array of shape (T, Z, Y, X).")

    features = {}
    voxel_size_x = float(params_values['voxel_size_x'])
    voxel_size_y = float(params_values['voxel_size_y'])
    voxel_size_z = float(params_values['voxel_size_z'])
    voxel_size = np.array([voxel_size_x, voxel_size_y, voxel_size_z])
    volume_localized = float(params_values['volume_localized'])


    for event_id in events_ids:
        # voxels belonging to the event
        event_mask = (calcium_events == event_id)
        if not np.any(event_mask):
            continue
        coords = np.argwhere(event_mask)
        nb_voxels = coords.shape[0]
        t0 = coords[:, 0].min()  # First frame of the event
        t1 = coords[:, 0].max()  # Last frame of the event
        duration = t1 - t0 + 1

        # centroïd calculation
        centroid = np.mean(coords, axis=0)

        # volume calculation
        volume = nb_voxels * np.prod(voxel_size)

        # mean amplitude calculation (or max ??)
        amplitude = np.max(image_amplitude[event_mask])

        # classification
        if duration > 1:
            class_label, confidence = classify_event(coords, t0, duration, volume, params_values)
        else:
            if volume <= volume_localized:
                class_label = "localized"
                confidence = 1.0
            else:
                class_label = "localized but not microdomain"
                confidence = 1.0

        # Store features
        features[event_id] = {
            'duration': duration,
            't0': t0,
            'amplitude': amplitude,
            'volume': volume,
            'centroid': centroid,
            'classification': class_label,
            'confidence': confidence
        }
    return features

def classify_event(coords , t0: int, duration: int, volume: float, params_values: dict) -> tuple:
    """
    Determine the classification of a calcium event based on its spatial characteristics (localized or wave) and compute its confidence level.

    """
    threshold_distance_localized = float(params_values['threshold_distance_localized'])
    threshold_median_localized = float(params_values['threshold_median_localized'])
    volume_localized = float(params_values['volume_localized'])

    # Centroid calculation for each frame in the event
    centroids = []
    for t in range(t0, t0 + duration):
        frame_coords = coords[coords[:, 0] == t]
        if frame_coords.shape[0] == 0:
            centroids.append([np.nan, np.nan, np.nan])
        else:
            centroids.append(frame_coords[:, 1:4].mean(axis=0))
    centroids = np.array(centroids)

    # Calculate distances between consecutive centroids
    dists = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
    big_distance = np.any(dists > threshold_distance_localized)
    confidence = 0.0

    if big_distance:
        idx = np.argmax(dists > threshold_distance_localized)
        confidence = min(100, dists[idx] * 80 / threshold_distance_localized)
        return "Wave", confidence
    else:
        median_val = np.nanmedian(dists)
        if median_val < threshold_distance_localized:
            confidence = min(100, median_val * 50 / threshold_median_localized)
            if volume <= volume_localized:
                return "Localized", confidence
            else:
                return "Localized but no microdomain", confidence
        else:
            confidence = min(100, median_val * 50 / threshold_median_localized)
            return "Wave", confidence


def write_excel_features(features: dict, output_directory: str) -> None:
    """
    Write the computed features to an Excel file.

    @param features: Dictionary containing computed features.
    @param output_directory: Directory to save the Excel file.
    """
    df = pd.DataFrame.from_dict(features, orient='index')
    output_file = f"{output_directory}/calcium_events_features.xlsx"
    df.to_excel(output_file, index_label='Event ID')
    print(f"Features saved to {output_file}")


def write_csv_features(features: dict, output_directory: str) -> None:
    """
    Write the computed features to a CSV file.

    @param features: Dictionary containing computed features.
    @param output_directory: Directory to save the CSV file.
    """
    df = pd.DataFrame.from_dict(features, orient='index')
    output_file = f"{output_directory}/calcium_events_features.csv"
    df.to_csv(output_file, index_label='Event ID')
    print(f"Features saved to {output_file}")