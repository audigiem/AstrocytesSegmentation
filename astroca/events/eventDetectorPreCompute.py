"""
@file eventDetector_ultra_optimized.py
@brief Version ultra-optimisée avec pré-calculs massifs et Numba
"""

import numpy as np
import numba as nb
from numba import njit, prange, types
from numba.typed import Dict, List as NumbaList
from typing import List, Tuple, Optional, Any
import time
from collections import deque
from tqdm import tqdm
import os
from astroca.tools.exportData import export_data
from astroca.events.eventMergerOptimized import process_small_groups_optimized
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components
from collections import defaultdict


VOXEL_TYPE = types.UniTuple(types.int64, 4)


# ============= FONCTIONS NUMBA ULTRA-OPTIMISÉES =============

@njit
def precompute_all_patterns(av_data):
    """Pré-calcule TOUS les patterns possibles en une seule passe"""
    T, Z, Y, X = av_data.shape

    # Dictionnaire pour stocker les patterns (hash -> pattern)
    patterns = Dict.empty(
        key_type=types.int64,
        value_type=types.float32[:]
    )

    # Mapping voxel -> hash du pattern
    voxel_to_pattern_hash = np.zeros((T, Z, Y, X), dtype=np.int64)

    for z in prange(Z):
        for y in range(Y):
            for x in range(X):
                # Extraire le profil temporel complet
                profile = av_data[:, z, y, x]

                # Trouver toutes les séquences non-nulles
                in_sequence = False
                start_t = 0

                for t in range(T):
                    if profile[t] != 0:
                        if not in_sequence:
                            start_t = t
                            in_sequence = True
                    else:
                        if in_sequence:
                            # Fin de séquence
                            pattern = profile[start_t:t]
                            pattern_hash = hash_pattern(pattern)
                            patterns[pattern_hash] = pattern.copy()

                            # Marquer tous les voxels de cette séquence
                            for tt in range(start_t, t):
                                voxel_to_pattern_hash[tt, z, y, x] = pattern_hash

                            in_sequence = False

                # Gérer la séquence qui se termine à la fin
                if in_sequence:
                    pattern = profile[start_t:T]
                    pattern_hash = hash_pattern(pattern)
                    patterns[pattern_hash] = pattern.copy()

                    for tt in range(start_t, T):
                        voxel_to_pattern_hash[tt, z, y, x] = pattern_hash

    return patterns, voxel_to_pattern_hash


@njit
def hash_pattern(pattern):
    """Hash rapide d'un pattern pour indexation"""
    hash_val = np.int64(0)
    for i in range(len(pattern)):
        # Simple polynomial hash
        hash_val = hash_val * 31 + np.int64(pattern[i] * 1000)
    return hash_val




@njit
def compute_max_ncc_fast(p1, p2):
    """Version ultra-rapide qui ne calcule que le max de la corrélation"""
    n1, n2 = len(p1), len(p2)

    # Normalisations pré-calculées
    norm1 = 0.0
    norm2 = 0.0
    for i in range(n1):
        norm1 += p1[i] * p1[i]
    for i in range(n2):
        norm2 += p2[i] * p2[i]

    if norm1 == 0 or norm2 == 0:
        return 0.0

    max_corr = 0.0

    # Calculer seulement les positions les plus prometteuses
    max_overlap = min(n1, n2)

    for lag in range(-max_overlap // 2, max_overlap // 2 + 1):
        corr_val = 0.0
        count = 0

        for i in range(n1):
            j = i - lag
            if 0 <= j < n2:
                corr_val += p1[i] * p2[j]
                count += 1

        if count > 0:
            normalized_corr = corr_val / np.sqrt(norm1 * norm2)
            if normalized_corr > max_corr:
                max_corr = normalized_corr

    return max_corr


@njit
def get_valid_neighbors_batch(av_data, id_connected, t, z, y, x):
    """Récupère tous les voisins valides d'un coup"""
    T, Z, Y, X = av_data.shape
    neighbors = []

    for dz in [-1, 0, 1]:
        nz = z + dz
        if nz < 0 or nz >= Z:
            continue
        for dy in [-1, 0, 1]:
            ny = y + dy
            if ny < 0 or ny >= Y:
                continue
            for dx in [-1, 0, 1]:
                nx = x + dx
                if dx == dy == dz == 0:
                    continue
                if nx < 0 or nx >= X:
                    continue

                if av_data[t, nz, ny, nx] != 0 and id_connected[t, nz, ny, nx] == 0:
                    neighbors.append((nz, ny, nx))

    return neighbors


@njit
def propagate_temporal_fast(voxel_to_pattern_hash, id_connected, event_id, t, z, y, x):
    """Propagation temporelle ultra-rapide"""
    T = id_connected.shape[0]
    pattern_hash = voxel_to_pattern_hash[t, z, y, x]

    # Définir le type de retour explicitement
    temporal_voxels = NumbaList.empty_list(VOXEL_TYPE)

    if pattern_hash == 0:
        return temporal_voxels

    # Trouver tous les voxels temporels avec le même pattern
    for tt in range(T):
        if (voxel_to_pattern_hash[tt, z, y, x] == pattern_hash and
                id_connected[tt, z, y, x] == 0):
            id_connected[tt, z, y, x] = event_id
            temporal_voxels.append((tt, z, y, x))

    return temporal_voxels


# ============= CLASSE ULTRA-OPTIMISÉE =============

class EventDetectorUltraOptimized:
    """Version ultra-optimisée avec pré-calculs massifs"""

    def __init__(self, av_data: np.ndarray, threshold_size_3d: int = 10,
                 threshold_size_3d_removed: int = 5, threshold_corr: float = 0.5,
                 plot: bool = False):

        print("=== Event Detection optimized with pre-computated patterns ===")
        print(f"Input shape: {av_data.shape}")
        print(f"Non-zero voxels: {np.count_nonzero(av_data)}/{av_data.size}")

        self.av_ = av_data.astype(np.float32)
        self.time_length_, self.depth_, self.height_, self.width_ = av_data.shape

        self.threshold_size_3d_ = threshold_size_3d
        self.threshold_size_3d_removed_ = threshold_size_3d_removed
        self.threshold_corr_ = threshold_corr
        self.plot_ = plot

        self.id_connected_voxel_ = np.zeros_like(self.av_, dtype=np.int32)
        self.final_id_events_ = []

        # ÉTAPE 1: Pré-calcul massif de tous les patterns
        print("Pré-calcul des patterns...")
        start = time.time()
        self.patterns_dict_, self.voxel_to_pattern_hash_ = precompute_all_patterns(self.av_)
        print(f"✓ {len(self.patterns_dict_)} patterns uniques en {time.time() - start:.2f}s")

        # # ÉTAPE 2: Pré-calcul de toutes les corrélations
        # print("Pré-calcul des corrélations...")
        # start = time.time()
        # self.correlation_matrix_, self.pattern_hashes_list_ = precompute_correlations(self.patterns_dict_)
        # print(f"✓ Matrice {self.correlation_matrix_.shape} en {time.time() - start:.2f}s")
        self.correlation_cache_ = Dict.empty(
            key_type=types.UniTuple(types.int64, 2),
            value_type=types.float32
        )

        self.stats_ = {"events_retained": 0, "events_merged": 0, "events_removed": 0, "events_assigned": 0}

    def find_events(self) -> None:
        """Version ultra-optimisée de la recherche d'événements"""
        print(
            f"Thresholds: size={self.threshold_size_3d_}, removed={self.threshold_size_3d_removed_}, corr={self.threshold_corr_}")

        if np.count_nonzero(self.av_) == 0:
            print("No non-zero voxels found!")
            return

        event_id = 1
        all_events = {}

        for t in tqdm(range(self.time_length_), desc="Processing frames"):
            seed = self._find_seed_point_fast(t)

            while seed is not None:
                x, y, z = seed

                # Récupération ultra-rapide du pattern
                pattern_hash = self.voxel_to_pattern_hash_[t, z, y, x]
                if pattern_hash == 0:
                    break

                # BFS ultra-optimisé
                current_group = self._bfs_ultra_fast(t, z, y, x, pattern_hash, event_id)

                all_events[event_id] = {
                    'voxels': current_group,
                    'size': len(current_group),
                    'is_large': len(current_group) >= self.threshold_size_3d_,
                }

                event_id += 1
                seed = self._find_seed_point_fast(t)

        print(f"Phase 2: Processing {sum(1 for e in all_events.values() if not e['is_large'])} small groups...")
        self._process_all_small_groups_optimized(all_events)

        print(f"\nTotal events: {len(self.final_id_events_)}")

    @staticmethod
    @njit()
    def get_correlation_cached(cache, patterns_dict, h1, h2):
        key = (min(h1, h2), max(h1, h2))
        if key in cache:
            return cache[key]

        p1 = patterns_dict[h1].astype(np.float32)
        p2 = patterns_dict[h2].astype(np.float32)
        corr = np.float32(compute_max_ncc_fast(p1, p2))
        corr = np.float32(corr)
        cache[key] = corr
        return corr



    def _bfs_ultra_fast(self, seed_t, seed_z, seed_y, seed_x, seed_pattern_hash, event_id):
        """BFS ultra-rapide avec toutes les optimisations"""
        queue = deque()
        visited = set()
        group_voxels = []

        # Propagation temporelle immédiate du seed
        temporal_voxels = propagate_temporal_fast(
            self.voxel_to_pattern_hash_, self.id_connected_voxel_,
            event_id, seed_t, seed_z, seed_y, seed_x
        )

        for voxel in temporal_voxels:
            queue.append(voxel)
            visited.add(voxel)
            group_voxels.append(voxel)

        while queue:
            t, z, y, x = queue.popleft()

            # Récupération des voisins en batch
            neighbors = get_valid_neighbors_batch(
                self.av_, self.id_connected_voxel_, t, z, y, x
            )

            for nz, ny, nx in neighbors:
                neighbor_hash = self.voxel_to_pattern_hash_[t, nz, ny, nx]
                if neighbor_hash == 0:
                    continue

                correlation = self.get_correlation_cached(
                    self.correlation_cache_, self.patterns_dict_,
                    seed_pattern_hash, neighbor_hash
                )
                if correlation > self.threshold_corr_:
                    # Propagation temporelle du voisin
                    neighbor_temporal = propagate_temporal_fast(
                        self.voxel_to_pattern_hash_, self.id_connected_voxel_,
                        event_id, t, nz, ny, nx
                    )

                    for nvoxel in neighbor_temporal:
                        if nvoxel not in visited:
                            queue.append(nvoxel)
                            visited.add(nvoxel)
                            group_voxels.append(nvoxel)

        return group_voxels

    def _find_seed_point_fast(self, t):
        """Recherche de seed ultra-rapide"""
        frame = self.av_[t]
        unprocessed = (frame > 0) & (self.id_connected_voxel_[t] == 0)

        if not np.any(unprocessed):
            return None

        # Argmax vectorisé
        masked_frame = np.where(unprocessed, frame, -1)
        flat_idx = np.argmax(masked_frame)
        z, y, x = np.unravel_index(flat_idx, frame.shape)

        if masked_frame[z, y, x] <= 0:
            return None

        return (x, y, z)

    def _process_all_small_groups_optimized(self, all_events: dict) -> None:
        """Traitement optimisé de TOUS les petits groupes selon la logique de fusion"""

        # Séparer petits et grands événements
        small_events = {eid: info for eid, info in all_events.items() if not info['is_large']}
        large_events = {eid: info for eid, info in all_events.items() if info['is_large']}

        if not small_events:
            print("No small groups to process")
            return

        print(f"Processing {len(small_events)} small groups against {len(large_events)} large groups")

        # Utiliser la classe EventMergerOptimized adaptée
        processed_result = self._merge_small_groups_with_context(small_events, large_events)

        # Mettre à jour les statistiques
        self.stats_["events_merged"] = processed_result['merged']
        self.stats_["events_assigned"] = processed_result['assigned']
        self.stats_["events_removed"] = processed_result['removed']

        # Ajouter les événements conservés à la liste finale
        for event_id in processed_result['kept_events']:
            if event_id not in self.final_id_events_:
                self.final_id_events_.append(event_id)
                self.stats_["events_retained"] += 1

    def _merge_small_groups_with_context(self, small_events: dict, large_events: dict) -> dict:
        """Version adaptée du merger pour travailler avec le contexte existant"""

        # Créer les structures nécessaires pour le merger
        small_groups_info = {}

        for event_id, event_info in small_events.items():
            # Calculer le centroïde
            voxels_array = np.array(event_info['voxels'])
            centroid = np.mean(voxels_array, axis=0) if len(voxels_array) > 0 else np.zeros(4)

            small_groups_info[event_id] = {
                'size': event_info['size'],
                'centroid': centroid,
                'voxels': voxels_array
            }

        large_groups_info = {}
        for event_id, event_info in large_events.items():
            voxels_array = np.array(event_info['voxels'])
            centroid = np.mean(voxels_array, axis=0) if len(voxels_array) > 0 else np.zeros(4)

            large_groups_info[event_id] = {
                'size': event_info['size'],
                'centroid': centroid,
                'voxels': voxels_array
            }

        # Appliquer la logique de fusion optimisée
        result = self._apply_merging_logic(small_groups_info, large_groups_info)

        return result

    def _apply_merging_logic(self, small_groups: dict, large_groups: dict) -> dict:
        """Application de la logique de fusion selon vos spécifications"""


        result = {
            'kept_events': [],
            'merged': 0,
            'assigned': 0,
            'removed': 0
        }

        if not small_groups:
            return result

        # ÉTAPE 1: Grouper les petits groupes voisins
        print("Step 1: Merging neighboring small groups...")
        merged_groups = self._merge_neighboring_groups(small_groups)
        result['merged'] = len(small_groups) - len(merged_groups)

        # ÉTAPE 2: Assigner aux grands groupes proches
        if large_groups:
            print("Step 2: Assigning to closest large groups...")
            assigned_count = self._assign_to_large_groups(merged_groups, large_groups)
            result['assigned'] = assigned_count

        # ÉTAPE 3: Traiter les groupes restants
        print("Step 3: Processing remaining groups...")
        for group_data in merged_groups:
            if not group_data.get('assigned', False):
                if group_data['size'] >= self.threshold_size_3d_removed_:
                    # Conserver le groupe
                    for original_id in group_data['original_ids']:
                        result['kept_events'].append(original_id)
                else:
                    # Supprimer le groupe
                    result['removed'] += len(group_data['original_ids'])
                    for original_id in group_data['original_ids']:
                        # Marquer pour suppression dans l'array principal
                        for t, z, y, x in small_groups[original_id]['voxels']:
                            self.id_connected_voxel_[t, z, y, x] = 0

        return result

    def _merge_neighboring_groups(self, small_groups: dict) -> list:
        """Fusionne les petits groupes voisins"""
        if len(small_groups) <= 1:
            return [{'original_ids': list(small_groups.keys()),
                     'size': sum(g['size'] for g in small_groups.values()),
                     'centroid': list(small_groups.values())[0]['centroid'] if small_groups else np.zeros(4)}]

        # Créer matrice de distances
        group_ids = list(small_groups.keys())
        centroids = np.array([small_groups[gid]['centroid'] for gid in group_ids])

        # Seuil de voisinage adaptatif
        T, Z, Y, X = self.id_connected_voxel_.shape
        spatial_threshold = max(5.0, np.sqrt(np.mean([g['size'] for g in small_groups.values()])))

        nn = NearestNeighbors(radius=spatial_threshold, metric='euclidean')
        nn.fit(centroids)
        adjacency = nn.radius_neighbors_graph(centroids)

        # Composantes connexes
        n_components, labels = connected_components(adjacency, directed=False)

        # Créer les groupes fusionnés
        merged_groups = []
        groups_by_component = defaultdict(list)

        for i, label in enumerate(labels):
            groups_by_component[label].append(group_ids[i])

        for component_group_ids in groups_by_component.values():
            # Fusionner les groupes de cette composante
            total_size = sum(small_groups[gid]['size'] for gid in component_group_ids)

            # Centroïde pondéré par la taille
            weighted_centroids = []
            total_weight = 0
            for gid in component_group_ids:
                weight = small_groups[gid]['size']
                weighted_centroids.append(small_groups[gid]['centroid'] * weight)
                total_weight += weight

            merged_centroid = np.sum(weighted_centroids, axis=0) / total_weight if total_weight > 0 else np.zeros(4)

            merged_groups.append({
                'original_ids': component_group_ids,
                'size': total_size,
                'centroid': merged_centroid,
                'assigned': False
            })

        return merged_groups

    def _assign_to_large_groups(self, merged_groups: list, large_groups: dict) -> int:
        """Assigne les groupes fusionnés aux grands groupes proches"""
        if not large_groups:
            return 0

        # Préparer les centroïdes des grands groupes
        large_ids = list(large_groups.keys())
        large_centroids = np.array([large_groups[lid]['centroid'] for lid in large_ids])

        nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn.fit(large_centroids)

        assigned_count = 0

        for group_data in merged_groups:
            # Calculer seuil adaptatif
            max_distance = max(10.0, np.power(group_data['size'], 1 / 3) * 3)

            # Trouver le plus proche
            distances, indices = nn.kneighbors([group_data['centroid']], n_neighbors=1)
            closest_distance = distances[0][0]

            if closest_distance <= max_distance:
                # Assigner au groupe proche
                target_large_id = large_ids[indices[0][0]]

                # Réassigner physiquement les voxels
                for original_small_id in group_data['original_ids']:
                    mask = (self.id_connected_voxel_ == original_small_id)
                    self.id_connected_voxel_[mask] = target_large_id

                group_data['assigned'] = True
                assigned_count += 1

        return assigned_count

    def get_results(self) -> Tuple[np.ndarray, List[int]]:
        """Remapping final et retour des résultats"""
        if self.final_id_events_:
            # Remapping vectorisé ultra-rapide
            print("Remapping final des IDs...")
            max_id = max(self.final_id_events_) if self.final_id_events_ else 0
            id_map = np.zeros(max_id + 1, dtype=np.int32)

            for new_id, old_id in enumerate(sorted(self.final_id_events_), 1):
                id_map[old_id] = new_id

            self.id_connected_voxel_ = id_map[self.id_connected_voxel_]

        return self.id_connected_voxel_, list(range(1, len(self.final_id_events_) + 1))

    def get_statistics(self) -> dict:
        """Statistiques rapides"""
        unique_ids = np.unique(self.id_connected_voxel_[self.id_connected_voxel_ > 0])
        event_sizes = [np.sum(self.id_connected_voxel_ == eid) for eid in unique_ids]

        return {
            'nb_events': len(unique_ids),
            'event_sizes': event_sizes,
            'total_event_voxels': sum(event_sizes),
            'mean_event_size': np.mean(event_sizes) if event_sizes else 0,
            'max_event_size': np.max(event_sizes) if event_sizes else 0,
            'min_event_size': np.min(event_sizes) if event_sizes else 0,
        }


def detect_calcium_events_ultra_optimized(av_data: np.ndarray, params_values: dict = None,
                                          save_results: bool = False,
                                          output_directory: str = None) -> Tuple[np.ndarray, List[int]]:
    """
    Version ultra-optimisée de la détection d'événements
    """
    # Gestion des paramètres
    if params_values is None:
        threshold_size_3d = 10
        threshold_size_3d_removed = 5
        threshold_corr = 0.5
        save_results = False
        output_directory = None
    else:
        threshold_size_3d = int(params_values['events_extraction']['threshold_size_3d'])
        threshold_size_3d_removed = int(params_values['events_extraction']['threshold_size_3d_removed'])
        threshold_corr = float(params_values['events_extraction']['threshold_corr'])
        save_results = int(params_values['files']['save_results']) == 1
        output_directory = params_values['paths']['output_dir']

    total_start = time.time()

    detector = EventDetectorUltraOptimized(
        av_data, threshold_size_3d, threshold_size_3d_removed, threshold_corr
    )

    # Détection
    detection_start = time.time()
    detector.find_events()
    detection_time = time.time() - detection_start

    # Résultats
    id_connections, id_events = detector.get_results()

    total_time = time.time() - total_start

    print(f"\n PERFORMANCE SUMMARY:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Detection time: {detection_time:.2f}s")
    print(f"   Events found: {len(id_events)}")
    print(f"   Speed: {av_data.size / total_time:.0f} voxels/second")
    # show event id, size, frames and duration
    for event_id in id_events:
        print(f"   Event ID: {event_id}, Size: {np.sum(id_connections == event_id)}, Frames: {np.unique(np.argwhere(id_connections == event_id)[:, 0])}, Duration: {len(np.unique(np.argwhere(id_connections == event_id)[:, 0]))} frames")

    # Sauvegarde si nécessaire
    if save_results:
        if output_directory is None:
            raise ValueError("Output directory must be specified if save_results is True.")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        id_connections = id_connections.astype(np.float32)  # Ensure the data is in float32 format
        export_data(id_connections, output_directory, export_as_single_tif=True, file_name="ID_calciumEvents")

    print("=" * 60)
    print()
    return id_connections, id_events


# Test function
def test_ultra_optimized():
    """Test avec données synthétiques"""
    print("=== TESTING ULTRA-OPTIMIZED VERSION ===")

    # Données plus grandes pour tester la performance
    shape = (20, 64, 256, 256)  # Plus grand pour vraiment tester
    av_data = np.zeros(shape, dtype=np.float32)

    # Plusieurs événements synthétiques
    np.random.seed(42)

    # Événement 1
    av_data[2:8, 10:20, 50:80, 50:80] = np.random.rand(6, 10, 30, 30) * 0.8 + 0.2

    # Événement 2
    av_data[5:12, 30:40, 150:180, 100:130] = np.random.rand(7, 10, 30, 30) * 0.6 + 0.4

    # Événement 3
    av_data[10:15, 50:55, 200:210, 200:210] = np.random.rand(5, 5, 10, 10) * 0.9 + 0.1

    print(f"Created test data: {shape}")
    print(f"Non-zero voxels: {np.count_nonzero(av_data):,}")

    results = detect_calcium_events_ultra_optimized(av_data)
    return results


if __name__ == "__main__":
    test_ultra_optimized()