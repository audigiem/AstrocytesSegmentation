"""
@file: eventDetectorScipy.py
@brief: This module provides functionality to detect calcium events from active voxels using scipy's tools.
"""
import numpy as np
from scipy.ndimage import label, binary_dilation, generate_binary_structure
from numba import njit
from astroca.tools.loadData import load_data


@njit
def fast_bincount_nonzero(labeled_flat, max_label):
    """Version optimisée de bincount pour éviter les zéros inutiles"""
    counts = np.zeros(max_label + 1, dtype=np.int32)
    for i in range(labeled_flat.size):
        if labeled_flat[i] > 0:
            counts[labeled_flat[i]] += 1
    return counts


@njit
def find_neighbor_labels(contact_flat, labeled_flat):
    """Version optimisée pour trouver les labels voisins"""
    neighbors = []
    for i in range(contact_flat.size):
        if contact_flat[i] and labeled_flat[i] > 0:
            neighbors.append(labeled_flat[i])
    return np.array(neighbors, dtype=np.int32)


def detect_events(active_voxels: np.ndarray, params_values: dict) -> np.ndarray:
    """
    @brief Detect calcium events from active voxels using scipy's find_peaks function.
    Optimized version with significant performance improvements.

    @param active_voxels: 4D numpy array of active voxels (T, Z, Y, X).
    @param params_values: Dictionary containing parameters for event detection.
    @return: 4D numpy array of detected events.
    """

    if active_voxels.ndim != 4:
        raise ValueError("Input must be a 4D numpy array of shape (T, Z, Y, X).")

    # Conversion en types optimaux
    threshold_size_3d = int(params_values['threshold_size_3d'])
    threshold_size_3d_removed = int(params_values['threshold_size_3d_removed'])

    # Structure pré-calculée (réutilisable)
    structure = generate_binary_structure(4, 1)

    # Labellisation initiale
    labeled, num_features = label(active_voxels, structure)

    if num_features == 0:
        return np.zeros_like(labeled)

    # Optimisation 1: Utiliser des vues plates pour éviter les copies
    labeled_flat = labeled.ravel()
    max_label = labeled_flat.max()

    # Optimisation 2: Bincount optimisé avec numba
    sizes = fast_bincount_nonzero(labeled_flat, max_label)

    # Optimisation 3: Masques booléens plus efficaces
    small_mask = (sizes < threshold_size_3d) & (sizes > 0)
    large_mask = sizes >= threshold_size_3d

    if not np.any(small_mask):
        # Pas de petits groupes à traiter
        return labeled

    small_ids = np.where(small_mask)[0]
    large_ids = np.where(large_mask)[0]

    # Optimisation 4: Utiliser broadcasting pour les masques
    small_mask_4d = np.isin(labeled, small_ids)
    large_mask_4d = np.isin(labeled, large_ids)

    # Optimisation 5: Éviter les copies inutiles dans la dilatation
    dilated_small = binary_dilation(small_mask_4d, structure=structure)
    merged_small_mask = dilated_small & small_mask_4d

    # Re-labellisation des petits groupes
    small_labeled, num_small = label(merged_small_mask, structure)

    if num_small == 0:
        return labeled

    # Optimisation 6: Pré-allouer le résultat final
    final_labels = labeled.copy()  # Commencer avec les labels existants

    # Optimisation 7: Traitement vectorisé quand possible
    large_mask_flat = large_mask_4d.ravel()
    labeled_flat = labeled.ravel()

    next_label = max_label + 1

    for i in range(1, num_small + 1):
        group_mask = (small_labeled == i)

        # Optimisation 8: Calcul de taille plus direct
        group_size = np.count_nonzero(group_mask)

        # Dilatation sur le groupe spécifique
        dilated = binary_dilation(group_mask, structure=structure)
        contact = dilated & large_mask_4d

        if np.any(contact):
            # Optimisation 9: Fonction numba pour trouver les voisins
            contact_flat = contact.ravel()
            neighbor_labels = find_neighbor_labels(contact_flat, labeled_flat)

            if len(neighbor_labels) > 0:
                # Trouver le voisin le plus fréquent
                unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
                target = unique_labels[np.argmax(counts)]
                final_labels[group_mask] = target
            else:
                # Cas rare: contact détecté mais pas de labels valides
                if group_size >= threshold_size_3d_removed:
                    final_labels[group_mask] = next_label
                    next_label += 1
        else:
            # Groupe isolé
            if group_size >= threshold_size_3d_removed:
                final_labels[group_mask] = next_label
                next_label += 1
            else:
                # Optimisation 10: Marquer explicitement comme background
                final_labels[group_mask] = 0

    return final_labels


def show_results(final_labels: np.ndarray):
    """
    @brief Display the results of the event detection.

    @param final_labels: 4D numpy array of detected events.
    """
    unique, counts = np.unique(final_labels, return_counts=True)
    event_sizes = dict(zip(unique, counts))

    print("Number of detected events:", len(event_sizes) - 1)  # Exclude background
    for event_id, size in event_sizes.items():
        if event_id == 0:
            continue  # Skip background
        frames = np.unique(np.argwhere(final_labels == event_id)[:, 0])
        print(f"Event ID {event_id}: Size = {size}, Frames = {frames}, Number of frames = {len(frames)}")

if __name__ == "__main__":
    # Example usage
    active_voxels = load_data("/home/matteo/Bureau/INRIA/codeJava/outputdirFewerTime/AV.tif")
    params_values = {
        'threshold_size_3d': 400,
        'threshold_size_3d_removed': 20,
        'threshold_corr': 0.5
    }

    detected_events = detect_events(active_voxels, params_values)

    # show number of detected events, their size and the number of frames
    unique, counts = np.unique(detected_events, return_counts=True)
    event_sizes = dict(zip(unique, counts))
    print("Number of detected events:", len(event_sizes) - 1)  # Exclude background
    for event_id, size in event_sizes.items():
        if event_id == 0:
            continue  # Skip background
        # show which frames the event is present in
        frames = np.unique(np.argwhere(detected_events == event_id)[:, 0])
        print(f"Event ID {event_id}: Size = {size}, Frames = {frames}, Number of frames = {len(frames)}")




