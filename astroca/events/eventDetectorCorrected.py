"""
@file eventDetectorCorrected.py
@brief Optimized event detector for calcium events in 4D data.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import time
import os
from astroca.tools.exportData import export_data
from astroca.events.tools import find_seed_fast, get_valid_neighbors, batch_check_conditions, compute_max_ncc_fast, find_nonzero_pattern_bounds, compute_max_ncc_strict
from tqdm import tqdm
from collections import defaultdict


class EventDetectorOptimized:
    """
    @class EventDetectorOptimized
    @brief Optimized event detector for calcium events in 4D data.
    """

    def __init__(self, av_data: np.ndarray, threshold_size_3d: int = 10,
                 threshold_size_3d_removed: int = 5, threshold_corr: float = 0.5,
                 plot: bool = False):
        """
        @fn __init__
        @brief Initialize the EventDetector with optimizations.
        @param av_data 4D numpy array of input data
        @param threshold_size_3d Minimum size for event retention
        @param threshold_size_3d_removed Minimum size for not removing small events
        @param threshold_corr Correlation threshold
        @param plot Enable plotting (default: False)
        """
        print("=== Finding events in 4D data ===")
        print(f"Input data range: [{av_data.min():.3f}, {av_data.max():.3f}]")
        print(f"Non-zero voxels density: {np.count_nonzero(av_data) / av_data.size:.5f}")

        # Convertir en float32 pour économiser la mémoire et améliorer les performances
        self.av_ = av_data.astype(np.float32)
        self.time_length_, self.depth_, self.height_, self.width_ = av_data.shape

        self.threshold_size_3d_ = threshold_size_3d
        self.threshold_size_3d_removed_ = threshold_size_3d_removed
        self.threshold_corr_ = threshold_corr
        self.plot_ = plot

        # Masque pré-calculé pour les voxels non-zéro
        self.nonzero_mask_ = self.av_ != 0

        # Utiliser int16 si possible pour économiser la mémoire
        max_events_estimate = np.count_nonzero(self.av_) // threshold_size_3d + 1000
        dtype = np.int16 if max_events_estimate < 32000 else np.int32
        self.id_connected_voxel_ = np.zeros_like(self.av_, dtype=dtype)

        self.final_id_events_ = []

        # Cache optimisé avec limite de taille
        self.pattern_cache_size_limit_ = 10000
        self.pattern_: Dict[Tuple[int, int, int, int], np.ndarray] = {}
        self.frame_cache_limit = 1000
        self._frame_cache = {}

        # Offsets pré-calculés
        self._neighbor_offsets = self._generate_neighbor_offsets()
        self._neighbor_offsets_4d = self._generate_neighbor_offsets_4d()

        # Statistiques
        self.stats_ = {
            "patterns_computed": 0,
            "correlations_computed": 0,
            "events_retained": 0,
            "events_merged": 0,
            "events_removed": 0,
            "cache_hits": 0
        }

        self.small_av_group_set_ = set()    # Cache pour les ids des petits groupes

        # Structures optimisées pour le traitement en batch
        self._batch_size = 1000
        self._reusable_arrays = self._initialize_reusable_arrays()

    def _initialize_reusable_arrays(self) -> Dict[str, np.ndarray]:
        """
        @fn _initialize_reusable_arrays
        @brief Initialize reusable arrays to avoid repeated allocations.
        @return Dictionary of reusable arrays
        """
        return {
            'temp_coords': np.zeros((self._batch_size, 3), dtype=np.int32),
            'temp_patterns': np.zeros((self._batch_size, 50), dtype=np.float32),  # Assume max pattern length 50
            'temp_correlations': np.zeros(self._batch_size, dtype=np.float32)
        }

    def _generate_neighbor_offsets(self) -> np.ndarray:
        """
        @fn _generate_neighbor_offsets
        @brief Pre-generate 26-neighbor offsets for 3D spatial connectivity.
        @return Array of neighbor offsets (N, 3)
        """
        offsets = []
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == dy == dz == 0:
                        continue
                    offsets.append([dz, dy, dx])
        return np.array(offsets, dtype=np.int32)

    def _generate_neighbor_offsets_4d(self) -> np.ndarray:
        """
        @fn _generate_neighbor_offsets_4d
        @brief Pre-generate neighbor offsets for 4D connectivity.
        @return Array of neighbor offsets (N, 4)
        """
        offsets = []
        for dt in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == dy == dz == dt == 0:
                            continue
                        offsets.append([dt, dz, dy, dx])
        return np.array(offsets, dtype=np.int32)

    # @profile 
    def find_events(self) -> None:
        """
        @fn find_events
        @brief Main optimized event finding method.
        @return None
        """
        print(
            f"Thresholds -> size: {self.threshold_size_3d_}, removed: {self.threshold_size_3d_removed_}, corr: {self.threshold_corr_}")
        start_time = time.time()

        # Vérification préliminaire
        if not np.any(self.nonzero_mask_):
            print("No non-zero voxels found!")
            return

        event_id = 1
        # Utiliser des structures optimisées
        waiting_for_processing = []
        small_AV_groups = []
        id_small_AV_groups = []

        # for t in range(self.time_length_):
        for t in tqdm(range(self.time_length_), desc="Processing time frames", unit="frame"):
            # Traitement optimisé des seeds
            while True:
                seed = self._find_seed_point_fast(t)
                if seed is None:
                    break

                x, y, z = seed

                # Extraction optimisée du profil d'intensité
                intensity_profile = self.av_[:, z, y, x]
                pattern = self._detect_pattern_optimized(intensity_profile, t)
                if pattern is None:
                    break

                # Initialisation de la région
                pattern_key = (t, z, y, x)
                self.pattern_[pattern_key] = pattern
                self.id_connected_voxel_[t, z, y, x] = event_id
                waiting_for_processing = [[t, z, y, x]]

                # Ajout des points temporels du pattern
                start_t = t
                while start_t > 0 and intensity_profile[start_t - 1] != 0:
                    start_t -= 1

                for i in range(1, len(pattern)):
                    t0 = start_t + i
                    if t0 < self.time_length_ and self.id_connected_voxel_[t0, z, y, x] == 0:
                        self.id_connected_voxel_[t0, z, y, x] = event_id
                        pattern_key = (t0, z, y, x)
                        self.pattern_[pattern_key] = pattern
                        waiting_for_processing.append([t0, z, y, x])

                # Croissance de région optimisée
                self._grow_region_optimized(waiting_for_processing, event_id)

                group_size = len(waiting_for_processing)

                if group_size < self.threshold_size_3d_:
                    small_AV_groups.append(waiting_for_processing.copy())
                    id_small_AV_groups.append(event_id)
                    # print(f"Small {event_id} {group_size}")
                else:
                    # print(f"Large {event_id} {group_size}")
                    self.final_id_events_.append(event_id)
                    self.stats_["events_retained"] += 1

                event_id += 1

                # Gestion de la mémoire du cache
                if len(self.pattern_) > self.pattern_cache_size_limit_:
                    self._cleanup_pattern_cache()
        print(f" - Found {event_id - 1} events with {len(self.final_id_events_)} retained and {len(small_AV_groups)} small groups.")
        # Traitement des petits groupes
        if small_AV_groups:
            self._process_small_groups(small_AV_groups, id_small_AV_groups)

        # print(f"Size of each final event:")
        # for event_id in self.final_id_events_:
        #     size = np.sum(self.id_connected_voxel_ == event_id)
        #     print(f"    Event ID={event_id}: {size} voxels")
        self._compute_final_id_events()
        # test pour voir si le renommage fonctionne
        # for event in range(1,len(self.final_id_events_)+1):
        #     size = np.sum(self.id_connected_voxel_ == event)
        #     print(f"{size}")
        stats = self.get_statistics()
        for key, value in stats.items():
            if key == "event_sizes":
                print(" - Event sizes:")
                for size in value:
                    print(f"  {size}")
            else:
                print(f" - {key}: {value}")
                    

        print(f"Total time taken: {time.time() - start_time:.2f} seconds")

    def _cleanup_pattern_cache(self) -> None:
        """
        @fn _cleanup_pattern_cache
        @brief Clean up pattern cache to prevent memory overflow.
        @return None
        """
        # Garder seulement les 50% les plus récents
        items = list(self.pattern_.items())
        keep_count = len(items) // 2
        self.pattern_ = dict(items[-keep_count:])

    def _get_frame_cache(self, t: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        @fn _get_frame_cache
        @brief Get cached frames or cache them if not present.
        @param t Time index
        @return Tuple of (av_frame, id_frame)
        """
        if t not in self._frame_cache:
            # Nettoyage du cache si trop plein
            if len(self._frame_cache) >= self.frame_cache_limit:
                # Supprimer les plus anciens
                oldest_key = min(self._frame_cache.keys())
                del self._frame_cache[oldest_key]

            # Mettre en cache les frames actuelles
            self._frame_cache[t] = (self.av_[t], self.id_connected_voxel_[t])
        else:
            self.stats_["cache_hits"] += 1

        return self._frame_cache[t]

    # @profile 
    def _grow_region_optimized(self, waiting_for_processing: List, event_id: int) -> None:
        """
        @fn _grow_region_optimized
        @brief Optimized region growing with batch processing.
        @param waiting_for_processing List of voxels to process
        @param event_id Current event ID
        @return None
        """
        index_waiting = 0

        while index_waiting < len(waiting_for_processing):
            seed = waiting_for_processing[index_waiting]
            pattern_key = tuple(seed)

            if pattern_key not in self.pattern_:
                index_waiting += 1
                continue

            pattern = self.pattern_[pattern_key]

            # Traitement optimisé des voisins
            self._find_connected_AV_optimized(seed, pattern, event_id, waiting_for_processing)
            index_waiting += 1

    # @profile
    def _find_connected_AV_optimized(self, seed: List[int], pattern: np.ndarray,
                                     event_id: int, waiting_list: List[List[int]]) -> None:
        """
        @fn _find_connected_AV_optimized
        @brief Ultra-optimized search for connected neighbors.
        @param seed Seed voxel [t, z, y, x]
        @param pattern Reference pattern
        @param event_id Current event ID
        @param waiting_list List of voxels to process
        @return None
        """
        t, z, y, x = seed

        # Utiliser le cache de frames pour éviter les accès répétés
        av_frame, id_frame = self._get_frame_cache(t)

        # Obtenir les coordonnées des voisins valides
        valid_coords = get_valid_neighbors(z, y, x, self.depth_, self.height_, self.width_, self._neighbor_offsets)

        if len(valid_coords) == 0:
            return

        # Batch check des conditions - évite les accès mémoire répétés
        valid_mask = batch_check_conditions(av_frame, id_frame, valid_coords)

        # Filtrer uniquement les voisins valides
        valid_neighbors = valid_coords[valid_mask]

        if len(valid_neighbors) == 0:
            return

        # Pré-calculer les patterns pour tous les voisins valides
        neighbor_patterns = []
        neighbor_coords_filtered = []

        for coord in valid_neighbors:
            nz, ny, nx = coord

            # Extraction du profil d'intensité
            intensity_profile = self.av_[:, nz, ny, nx]
            neighbor_pattern = self._detect_pattern_optimized(intensity_profile, t)

            if neighbor_pattern is not None:
                neighbor_patterns.append(neighbor_pattern)
                neighbor_coords_filtered.append((nz, ny, nx))

        # Traitement en batch des corrélations
        for i, neighbor_pattern in enumerate(neighbor_patterns):
            nz, ny, nx = neighbor_coords_filtered[i]

            # Calcul de corrélation optimisé - SEULEMENT le maximum
            # max_corr = compute_max_ncc_fast(pattern, neighbor_pattern)
            max_corr = compute_max_ncc_strict(pattern, neighbor_pattern)
            self.stats_["correlations_computed"] += 1

            if max_corr > self.threshold_corr_:
                self.id_connected_voxel_[t, nz, ny, nx] = event_id
                pattern_key = (t, nz, ny, nx)
                self.pattern_[pattern_key] = neighbor_pattern
                waiting_list.append([t, nz, ny, nx])

                # Ajout des points temporels - optimisé
                intensity_profile = self.av_[:, nz, ny, nx]
                start_t = t
                while start_t > 0 and intensity_profile[start_t - 1] != 0:
                    start_t -= 1

                # Traitement vectorisé des points temporels
                temporal_points = []
                for p in range(len(neighbor_pattern)):
                    tp = start_t + p
                    if (tp < self.time_length_ and
                            self.id_connected_voxel_[tp, nz, ny, nx] == 0):
                        temporal_points.append([tp, nz, ny, nx])

                # Assignation en batch
                if temporal_points:
                    for tp_point in temporal_points:
                        tp, _, _, _ = tp_point
                        self.id_connected_voxel_[tp, nz, ny, nx] = event_id
                        pattern_key = (tp, nz, ny, nx)
                        self.pattern_[pattern_key] = neighbor_pattern
                        waiting_list.append(tp_point)

    def _find_seed_point_fast(self, t0: int) -> Optional[Tuple[int, int, int]]:
        """
        @fn _find_seed_point_fast
        @brief Optimized seed search.
        @param t0 Time index
        @return Tuple (x, y, z) or None if not found
        """
        x, y, z, max_val = find_seed_fast(self.av_[t0], self.id_connected_voxel_[t0])

        if x == -1:
            return None
        return (x, y, z)

    # @profile
    def _detect_pattern_optimized(self, intensity_profile: np.ndarray, t: int) -> Optional[np.ndarray]:
        """
        @fn _detect_pattern_optimized
        @brief Optimized pattern detection.
        @param intensity_profile 1D numpy array of intensity values
        @param t Time index
        @return Pattern as 1D numpy array or None
        """
        if intensity_profile[t] == 0:
            return None

        start, end = find_nonzero_pattern_bounds(intensity_profile, t)
        pattern = intensity_profile[start:end].copy()

        self.stats_["patterns_computed"] += 1
        return pattern

    # @profile
    def _process_small_groups(self, small_av_groups: List, id_small_av_groups: List) -> None:
        self.small_av_groups_set_ = set(id_small_av_groups)

        self._group_small_neighborhood_regions(small_av_groups, id_small_av_groups)

        # Optimisation: traitement avec tqdm pour le suivi
        with tqdm(total=len(small_av_groups), desc="Processing small groups", unit="group") as pbar:
            i = 0
            while i < len(small_av_groups):
                group = small_av_groups[i]
                group_id = id_small_av_groups[i]

                change_id = self._change_id_small_regions(group, id_small_av_groups)
                if change_id:
                    self.stats_["events_merged"] += 1
                    del small_av_groups[i]
                    del id_small_av_groups[i]
                    self.small_av_groups_set_.discard(group_id)
                    pbar.total -= 1  # Réduire le total car on a supprimé un élément
                else:
                    if len(group) >= self.threshold_size_3d_removed_:
                        self.final_id_events_.append(group_id)
                    else:
                        coords = np.array(group)
                        self.id_connected_voxel_[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]] = 0
                        self.stats_["events_removed"] += 1

                    del small_av_groups[i]
                    del id_small_av_groups[i]
                    self.small_av_groups_set_.discard(group_id)
                    pbar.total -= 1  # Réduire le total car on a supprimé un élément
                
                pbar.update(1)
    
    # @profile
    def _group_small_neighborhood_regions(self, small_av_groups: List, list_ids_small_av_group: List) -> None:
        """Version optimisée qui préserve la logique exacte de l'original"""
        
        # Optimisation: pré-calculer les limites et offsets
        bounds = np.array([self.time_length_, self.depth_, self.height_, self.width_])
        neighbor_offsets = np.array(self._neighbor_offsets_4d)
        
        with tqdm(total=len(small_av_groups), desc="Grouping neighborhoods", unit="group") as pbar:
            id_ = 0
            while id_ < len(small_av_groups):
                list_av = small_av_groups[id_]
                group_id = list_ids_small_av_group[id_]

                # Optimisation: utiliser defaultdict
                neighbor_id_counts = defaultdict(int)

                # Optimisation: conversion en numpy array une seule fois
                coords_array = np.array(list_av)
                
                # Optimisation: calcul vectorisé des voisins
                coords_expanded = coords_array[:, None, :] + neighbor_offsets[None, :, :]
                coords_flat = coords_expanded.reshape(-1, 4)
                
                # Optimisation: vérification vectorisée des limites
                valid_mask = np.all(
                    (coords_flat >= 0) & (coords_flat < bounds), 
                    axis=1
                )
                
                if np.any(valid_mask):
                    valid_coords = coords_flat[valid_mask]
                    
                    # Optimisation: accès vectorisé aux neighbor_ids
                    neighbor_ids = self.id_connected_voxel_[
                        valid_coords[:, 0], 
                        valid_coords[:, 1], 
                        valid_coords[:, 2], 
                        valid_coords[:, 3]
                    ]
                    
                    # Optimisation: filtrage vectorisé
                    valid_neighbors_mask = (
                        (neighbor_ids != 0) & 
                        (neighbor_ids != group_id) & 
                        np.array([nid in self.small_av_groups_set_ for nid in neighbor_ids])
                    )
                    
                    if np.any(valid_neighbors_mask):
                        valid_neighbors = neighbor_ids[valid_neighbors_mask]
                        unique_ids, counts = np.unique(valid_neighbors, return_counts=True)
                        
                        for nid, count in zip(unique_ids, counts):
                            neighbor_id_counts[nid] += count

                if neighbor_id_counts:
                    # Optimisation: calcul du maximum en une seule passe
                    max_count = max(neighbor_id_counts.values())
                    # Optimisation: filtrage et tri plus efficace
                    candidates = [(nid, count) for nid, count in neighbor_id_counts.items() 
                                if count == max_count]
                    new_id = min(candidates)[0]  # Plus petit ID en cas d'égalité
                    
                    new_id_index = list_ids_small_av_group.index(new_id)
                    max_neighbor_count = neighbor_id_counts[new_id]

                    # Même logique de comparaison que l'original
                    if max_neighbor_count >= len(list_av):
                        # Optimisation: utiliser le array déjà créé
                        self.id_connected_voxel_[coords_array[:, 0], coords_array[:, 1], 
                                            coords_array[:, 2], coords_array[:, 3]] = new_id
                        small_av_groups[new_id_index].extend(list_av)

                        self.small_av_groups_set_.discard(group_id)
                        del small_av_groups[id_]
                        del list_ids_small_av_group[id_]
                        pbar.total -= 1  # Réduire le total
                        pbar.update(1)
                        continue  # Ne pas incrémenter id_ - CRUCIAL pour la logique
                
                id_ += 1
                pbar.update(1)

    # @profile
    def _change_id_small_regions(self, list_av: List, list_ids_small_av_group: List) -> bool:
        """Version optimisée avec calculs vectorisés"""
        
        # Optimisation: pré-calculer les éléments nécessaires
        small_av_set = set(list_ids_small_av_group)
        bounds = np.array([self.time_length_, self.depth_, self.height_, self.width_])
        neighbor_offsets = np.array(self._neighbor_offsets_4d)

        # Optimisation: conversion en numpy array une seule fois
        coords_array = np.array(list_av)
        
        # Optimisation: calcul vectorisé des voisins
        coords_expanded = coords_array[:, None, :] + neighbor_offsets[None, :, :]
        coords_flat = coords_expanded.reshape(-1, 4)
        
        # Optimisation: vérification vectorisée des limites
        valid_mask = np.all(
            (coords_flat >= 0) & (coords_flat < bounds), 
            axis=1
        )
        
        if not np.any(valid_mask):
            return False
        
        valid_coords = coords_flat[valid_mask]
        
        # Optimisation: accès vectorisé aux neighbor_ids
        neighbor_ids = self.id_connected_voxel_[
            valid_coords[:, 0], 
            valid_coords[:, 1], 
            valid_coords[:, 2], 
            valid_coords[:, 3]
        ]
        
        # Optimisation: filtrage vectorisé
        valid_neighbors_mask = (
            (neighbor_ids != 0) & 
            np.array([nid not in small_av_set for nid in neighbor_ids])
        )
        
        if not np.any(valid_neighbors_mask):
            return False
        
        valid_neighbors = neighbor_ids[valid_neighbors_mask]
        unique_ids, counts = np.unique(valid_neighbors, return_counts=True)
        
        # Optimisation: calcul du maximum en une seule passe
        max_count = np.max(counts)
        max_indices = counts == max_count
        candidates = unique_ids[max_indices]
        new_id = np.min(candidates)  # Plus petit ID en cas d'égalité

        # Optimisation: utiliser le array déjà créé
        self.id_connected_voxel_[coords_array[:, 0], coords_array[:, 1], 
                            coords_array[:, 2], coords_array[:, 3]] = new_id
        return True

    # @profile 
    def _compute_final_id_events(self) -> None:
        """Calcul optimisé des IDs finaux."""
        if not self.final_id_events_:
            return

        max_id = self.id_connected_voxel_.max()
        if max_id == 0:
            return

        # Remappage vectorisé
        id_map = np.zeros(max_id + 1, dtype=self.id_connected_voxel_.dtype)

        final_ids = sorted(self.final_id_events_)
        for new_id, old_id in enumerate(final_ids, start=1):
            id_map[old_id] = new_id

        self.id_connected_voxel_ = id_map[self.id_connected_voxel_]

    def get_results(self) -> Tuple[np.ndarray, List[int]]:
        """Return the final results."""
        return self.id_connected_voxel_, self.final_id_events_

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about detected events."""
        stats = {
            'nb_events': len(self.final_id_events_),
            'event_sizes': [],
            'total_event_voxels': 0,
            'cache_hits': 0
        }

        # for event in range(1,len(self.final_id_events_)+1):
        #     size = np.sum(self.id_connected_voxel_ == event)
        #     stats['event_sizes'].append(size)
        #     stats['total_event_voxels'] += size


        stats.update(self.stats_)
        return stats


def detect_calcium_events_opti(av_data: np.ndarray, params_values: dict = None) -> Tuple[np.ndarray, List[int]]:
    """
    Fonction optimisée pour détecter les événements calcium dans des données 4D.
    """
    required_keys = {'events_extraction', 'save', 'paths'}
    if not required_keys.issubset(params_values.keys()):
        raise ValueError(f"Missing required parameters: {required_keys - params_values.keys()}")
    threshold_size_3d = int(params_values['events_extraction']['threshold_size_3d'])
    threshold_size_3d_removed = int(params_values['events_extraction']['threshold_size_3d_removed'])
    threshold_corr = float(params_values['events_extraction']['threshold_corr'])
    save_results = int(params_values['save']['save_events']) == 1
    output_directory = params_values['paths']['output_dir']

    detector = EventDetectorOptimized(av_data, threshold_size_3d,
                                      threshold_size_3d_removed, threshold_corr)

    detector.find_events()
    id_connections, id_events = detector.get_results()

    if save_results:
        if output_directory is None:
            raise ValueError("Output directory must be specified if save_results is True.")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        id_connections = id_connections.astype(np.float32)
        export_data(id_connections, output_directory, export_as_single_tif=True, file_name="ID_calciumEvents")

    print(60 * "=")
    print()
    return id_connections, len(id_events)

