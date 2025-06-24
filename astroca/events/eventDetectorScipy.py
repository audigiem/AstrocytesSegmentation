"""
@file: eventDetectorScipy.py
@brief: This module provides functionality to detect calcium events from active voxels using scipy's tools.
"""
import numpy as np
from scipy.ndimage import label, binary_dilation, generate_binary_structure, distance_transform_edt
from numba import njit
from astroca.tools.loadData import load_data
from tqdm import tqdm
import time
from astroca.events.eventMergerOptimized import process_small_groups_optimized

@njit
def fast_bincount_nonzero(labeled_flat, max_label):
    """Version optimisée de bincount pour éviter les zéros inutiles"""
    counts = np.zeros(max_label + 1, dtype=np.int32)
    for i in range(labeled_flat.size):
        if labeled_flat[i] > 0:
            counts[labeled_flat[i]] += 1
    return counts

@njit
def assign_small_groups(labeled_array, small_labels, large_labels, distance_to_large):
    """Assign small groups to their nearest large group using precomputed distances"""
    output = labeled_array.copy()
    for label_val in small_labels:
        # Find all positions with this label
        positions = np.argwhere(labeled_array == label_val)
        if len(positions) > 0:
            # Find minimal distance to large group for this small group
            min_dist = np.inf
            for pos in positions:
                dist = distance_to_large[pos[0], pos[1], pos[2], pos[3]]
                if dist < min_dist:
                    min_dist = dist

            if min_dist < np.inf:  # If there's a reachable large group
                # Find all large group labels at this minimal distance
                candidate_labels = set()
                for pos in positions:
                    if distance_to_large[pos[0], pos[1], pos[2], pos[3]] == min_dist:
                        candidate = labeled_array[pos[0], pos[1], pos[2], pos[3]]
                        if candidate in large_labels:
                            candidate_labels.add(candidate)

                if len(candidate_labels) > 0:
                    # Assign to the first candidate (could also choose the largest)
                    chosen_label = list(candidate_labels)[0]
                    for pos in positions:
                        output[pos[0], pos[1], pos[2], pos[3]] = chosen_label
    return output

def process_small_groups(labeled, sizes, threshold_size_3d, threshold_size_3d_removed):
    """Process small groups according to the specified logic"""
    # Identify small and large groups
    small_mask = (sizes < threshold_size_3d) & (sizes > 0)
    large_mask = sizes >= threshold_size_3d

    small_labels = np.where(small_mask)[0]
    large_labels = np.where(large_mask)[0]

    if len(small_labels) == 0:
        return labeled

    print(f"[INFO] Processing {len(small_labels)} small groups...")

    # Create binary masks for small and large groups
    small_groups_mask = np.isin(labeled, small_labels)
    large_groups_mask = np.isin(labeled, large_labels)

    # Step 1: Group neighboring small groups together
    print("[INFO] Merging neighboring small groups...")
    structure = generate_binary_structure(4, 1)
    merged_small, num_merged = label(small_groups_mask, structure)
    merged_small = merged_small * small_groups_mask  # Keep only the original small groups

    # Update small labels after merging
    new_small_labels = np.unique(merged_small)
    new_small_labels = new_small_labels[new_small_labels != 0]  # Remove background

    # Create output array with large groups and merged small groups
    output = np.where(large_groups_mask, labeled, 0)
    output = np.where(merged_small > 0, merged_small + labeled.max(), output)

    # Calculate sizes of new merged groups
    new_sizes = fast_bincount_nonzero(output.ravel(), output.max())

    # Identify which merged groups are still small
    still_small_mask = (new_sizes < threshold_size_3d) & (new_sizes > 0)
    still_small_labels = np.where(still_small_mask)[0]

    if len(still_small_labels) == 0:
        return output

    print(f"[INFO] {len(still_small_labels)} small groups remain after merging")

    # Step 2: Assign remaining small groups to nearest large group
    print("[INFO] Assigning small groups to nearest large groups...")

    # Create mask of large groups for distance calculation
    large_mask_for_dist = np.isin(output, large_labels)

    # Compute distance to nearest large group
    distance_to_large = distance_transform_edt(~large_mask_for_dist)

    # Assign small groups to nearest large group
    output = assign_small_groups(output, still_small_labels, large_labels, distance_to_large)

    # Step 3: Remove isolated groups that are too small
    print("[INFO] Removing isolated small groups...")
    final_sizes = fast_bincount_nonzero(output.ravel(), output.max())
    too_small_mask = (final_sizes < threshold_size_3d_removed) & (final_sizes > 0)
    too_small_labels = np.where(too_small_mask)[0]

    if len(too_small_labels) > 0:
        output[np.isin(output, too_small_labels)] = 0

    return output

def detect_events(active_voxels: np.ndarray, params_values: dict) -> np.ndarray:
    """
    @brief Detect calcium events from active voxels using scipy's find_peaks function.
    Optimized version with significant performance improvements.

    @param active_voxels: 4D numpy array of active voxels (T, Z, Y, X).
    @param params_values: Dictionary containing parameters for event detection.
    @return: 4D numpy array of detected events.
    """
    print("\n[INFO] Starting event detection...")
    start_time = time.time()

    if active_voxels.ndim != 4:
        raise ValueError("Input must be a 4D numpy array of shape (T, Z, Y, X).")

    # Conversion en types optimaux
    threshold_size_3d = int(params_values['events_extraction']['threshold_size_3d'])
    threshold_size_3d_removed = int(params_values['events_extraction']['threshold_size_3d_removed'])

    print(f"[PARAMS] Threshold size 3D: {threshold_size_3d}")
    print(f"[PARAMS] Threshold size 3D removed: {threshold_size_3d_removed}")

    # Structure pré-calculée (réutilisable)
    structure = generate_binary_structure(4, 1)
    print("[INFO] Structure for labeling generated")

    # Labellisation initiale
    print("[INFO] Performing initial labeling...")
    labeled, num_features = label(active_voxels, structure)
    print(f"[INFO] Initial labeling complete. Found {num_features} features")

    if num_features == 0:
        print("[WARNING] No features found in input data")
        return np.zeros_like(labeled)


    processed_result = process_small_groups_optimized(labeled, threshold_size_3d, threshold_size_3d_removed)

    return processed_result


def show_results(final_labels: np.ndarray):
    """
    @brief Display the results of the event detection.

    @param final_labels: 4D numpy array of detected events.
    """
    print("\n[RESULTS] Event detection summary:")
    unique, counts = np.unique(final_labels, return_counts=True)
    event_sizes = dict(zip(unique, counts))

    print("Number of detected events:", len(event_sizes) - 1)  # Exclude background
    for event_id, size in event_sizes.items():
        if event_id == 0:
            continue  # Skip background
        frames = np.unique(np.argwhere(final_labels == event_id)[:, 0])
        print(f"Event ID {event_id}: Size = {size}, Frames = {frames}, Duration = {len(frames)} frames")

if __name__ == "__main__":
    # Example usage
    print("=== Event Detection Demo ===")
    print("Loading data...")
    active_voxels = load_data("/home/matteo/Bureau/INRIA/codeJava/outputdirFewerTime/AV.tif")
    print(f"Loaded data with shape: {active_voxels.shape}")

    params_values = {
        'events_extraction': {'threshold_size_3d': 400, 'threshold_size_3d_removed': 20, 'threshold_corr': 0.5}
    }

    print("\nStarting detection process...")
    detected_events = detect_events(active_voxels, params_values)

    print("\nDetection complete. Showing results...")
    show_results(detected_events)
    print("\n=== Process completed ===")