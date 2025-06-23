"""
@file eventDetectorPreComputeOptimized.py
@brief Version optimisée mémoire et performance
"""

import numpy as np
import numba as nb
from numba import njit, prange, types
from typing import List, Tuple, Optional, Any
import time
from collections import deque
from tqdm import tqdm
import os
from astroca.tools.exportData import export_data


# ============= FONCTIONS NUMBA OPTIMISÉES =============

@njit
def extract_temporal_profile(av_data, z, y, x):
    """Extrait le profil temporel d'un voxel"""
    T = av_data.shape[0]
    profile = np.zeros(T, dtype=np.float32)
    for t in range(T):
        profile[t] = av_data[t, z, y, x]
    return profile


@njit
def has_nonzero_values(av_data, z, y, x):
    """Vérifie rapidement si un voxel a des valeurs non-nulles"""
    T = av_data.shape[0]
    for t in range(T):
        if av_data[t, z, y, x] != 0:
            return True
    return False


@njit
def find_main_sequence(profile, min_length=3):
    """Trouve la séquence principale (la plus longue) dans un profil"""
    T = len(profile)
    max_start = 0
    max_end = 0
    max_length = 0

    in_sequence = False
    start_t = 0

    for t in range(T):
        if profile[t] != 0:
            if not in_sequence:
                start_t = t
                in_sequence = True
        else:
            if in_sequence:
                end_t = t
                seq_length = end_t - start_t
                if seq_length > max_length and seq_length >= min_length:
                    max_start = start_t
                    max_end = end_t
                    max_length = seq_length
                in_sequence = False

    # Gérer la séquence qui se termine à la fin
    if in_sequence:
        end_t = T
        seq_length = end_t - start_t
        if seq_length > max_length and seq_length >= min_length:
            max_start = start_t
            max_end = end_t
            max_length = seq_length

    if max_length >= min_length:
        return (max_start, max_end, max_length)
    else:
        return None


@njit
def simple_pattern_hash(arr):
    """Hash très simple pour éviter les collisions"""
    if len(arr) == 0:
        return np.int64(0)

    # Normaliser d'abord
    arr_mean = np.mean(arr)
    arr_std = np.std(arr)

    if arr_std == 0:
        return np.int64(int(arr_mean * 1000))

    # Hash basé sur les moments statistiques
    hash_val = np.int64(int(arr_mean * 1000) + int(arr_std * 1000) * 13 + len(arr) * 17)
    return hash_val


@njit
def compute_simple_correlation(p1, p2):
    """Corrélation très simplifiée pour l'efficacité"""
    if len(p1) != len(p2) or len(p1) == 0:
        return 0.0

    # Normalisation simple
    mean1, mean2 = np.mean(p1), np.mean(p2)
    std1, std2 = np.std(p1), np.std(p2)

    if std1 == 0 or std2 == 0:
        return 1.0 if abs(mean1 - mean2) < 0.1 else 0.0

    # Corrélation de Pearson simple
    corr_sum = 0.0
    for i in range(len(p1)):
        corr_sum += ((p1[i] - mean1) / std1) * ((p2[i] - mean2) / std2)

    return corr_sum / len(p1)


@njit
def get_neighbors_3d(z, y, x, Z, Y, X):
    """Récupère les voisins 3D valides"""
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
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                if nx < 0 or nx >= X:
                    continue
                neighbors.append((nz, ny, nx))
    return neighbors


def find_adjacent_groups(group_voxels, all_voxels, group_ids):
    """Version vectorisée pour trouver les groupes adjacents"""
    adjacency = {gid: set() for gid in group_ids}

    # Créer un dictionnaire de voxels par groupe sous forme de set de tuples
    voxel_dict = {gid: set() for gid in group_ids}
    for gid, voxels in zip(group_ids, group_voxels):
        voxel_dict[gid] = set(tuple(voxel) for voxel in voxels)

    # Vérifier les adjacences
    for i, (gid1, voxels1) in enumerate(zip(group_ids, group_voxels)):
        for voxel in voxels1:
            t, z, y, x = voxel

            # Générer tous les voisins possibles
            neighbors = [
                (t, z + dz, y + dy, x + dx)
                for dz in [-1, 0, 1]
                for dy in [-1, 0, 1]
                for dx in [-1, 0, 1]
                if not (dz == 0 and dy == 0 and dx == 0)
                   and 0 <= z + dz < all_voxels.shape[1]
                   and 0 <= y + dy < all_voxels.shape[2]
                   and 0 <= x + dx < all_voxels.shape[3]
            ]

            # Vérifier chaque groupe
            for gid2 in group_ids:
                if gid2 != gid1:
                    # Vérifier si un des voisins est dans l'autre groupe
                    if any(neighbor in voxel_dict[gid2] for neighbor in neighbors):
                        adjacency[gid1].add(gid2)

    return adjacency


def merge_groups(adjacency, group_dict, group_ids):
    """Fusion des groupes version vectorisée"""
    merged = {}
    visited = set()
    new_id = np.max(group_ids) + 1 if group_ids.size > 0 else 1

    for gid in group_ids:
        if gid not in visited:
            current_group = set(group_dict[gid])
            visited.add(gid)

            # BFS pour trouver tous les groupes connectés
            queue = list(adjacency[gid])
            while queue:
                neighbor_gid = queue.pop()
                if neighbor_gid not in visited:
                    current_group.update(group_dict[neighbor_gid])
                    visited.add(neighbor_gid)
                    queue.extend(adjacency[neighbor_gid])

            # Convertir en array numpy
            merged[new_id] = np.array(list(current_group), dtype=np.int32)
            new_id += 1

    return merged

# ============= CLASSE OPTIMISÉE MÉMOIRE =============

class EventDetectorOptimized:
    """Version optimisée pour la mémoire et la performance"""

    def __init__(self, av_data: np.ndarray, threshold_size_3d: int = 10,
                 threshold_size_3d_removed: int = 5, threshold_corr: float = 0.5,
                 plot: bool = False):

        print("=== EVENT DETECTOR OPTIMIZED VERSION ===")
        print(f"Input shape: {av_data.shape}")

        self.av_ = av_data.astype(np.float32)
        self.time_length_, self.depth_, self.height_, self.width_ = av_data.shape

        self.threshold_size_3d_ = threshold_size_3d
        self.threshold_size_3d_removed_ = threshold_size_3d_removed
        self.threshold_corr_ = threshold_corr
        self.plot_ = plot

        self.id_connected_voxel_ = np.zeros_like(self.av_, dtype=np.int32)
        self.final_id_events_ = []

        # Structures minimalistes
        self.voxel_patterns_ = {}  # (z,y,x) -> (start_t, end_t, hash, pattern)
        self.active_voxels_ = []  # Liste simple des voxels actifs

        # Pré-calcul ultra-optimisé
        self._precompute_minimal()

        self.stats_ = {"events_retained": 0, "events_merged": 0, "events_removed": 0}

    def _precompute_minimal(self):
        """Pré-calcul minimal - ne garde que l'essentiel"""
        print("Pré-calcul minimal des patterns...")
        start = time.time()

        non_zero_count = 0
        active_count = 0

        # Parcourir seulement les voxels non-vides
        for z in range(self.depth_):
            for y in range(self.height_):
                for x in range(self.width_):
                    # Check rapide si le voxel a des valeurs
                    if not has_nonzero_values(self.av_, z, y, x):
                        continue

                    non_zero_count += 1

                    # Extraire le profil
                    profile = extract_temporal_profile(self.av_, z, y, x)

                    # Trouver la séquence principale seulement
                    main_seq = find_main_sequence(profile, min_length=3)

                    if main_seq is not None:
                        start_t, end_t, length = main_seq
                        pattern = profile[start_t:end_t]
                        pattern_hash = simple_pattern_hash(pattern)

                        # Stocker de manière compacte
                        self.voxel_patterns_[(z, y, x)] = (start_t, end_t, pattern_hash, pattern)
                        self.active_voxels_.append((z, y, x))
                        active_count += 1

        print(f"✓ {active_count}/{non_zero_count} voxels actifs en {time.time() - start:.2f}s")
        print(f"  Mémoire patterns: ~{len(self.voxel_patterns_) * 100 / 1024:.1f} KB")

    def _get_pattern_for_voxel(self, t, z, y, x):
        """Récupère le pattern pour un voxel à un temps donné"""
        if (z, y, x) not in self.voxel_patterns_:
            return None, None

        start_t, end_t, pattern_hash, pattern = self.voxel_patterns_[(z, y, x)]

        if start_t <= t < end_t:
            return pattern_hash, pattern

        return None, None

    def find_events(self) -> None:
        """Recherche d'événements optimisée"""
        print(
            f"Thresholds: size={self.threshold_size_3d_}, removed={self.threshold_size_3d_removed_}, corr={self.threshold_corr_}")

        if len(self.active_voxels_) == 0:
            print("No active voxels found!")
            return

        event_id = 1
        small_groups = []
        small_group_ids = []

        for t in tqdm(range(self.time_length_), desc="Processing frames"):
            seed = self._find_seed_point(t)

            while seed is not None:
                x, y, z = seed

                # BFS optimisé
                current_group = self._bfs_optimized(t, z, y, x, event_id)

                if len(current_group) == 0:
                    # Marquer ce voxel comme traité
                    self.id_connected_voxel_[t, z, y, x] = -1
                else:
                    # Classification
                    if len(current_group) < self.threshold_size_3d_:
                        small_groups.append(current_group)
                        small_group_ids.append(event_id)
                    else:
                        self.final_id_events_.append(event_id)
                        self.stats_["events_retained"] += 1

                    event_id += 1

                seed = self._find_seed_point(t)

        # Nettoyer les marquages temporaires
        self.id_connected_voxel_[self.id_connected_voxel_ == -1] = 0

        # Traitement des petits groupes
        self._process_small_groups(small_groups, small_group_ids)

        print(f"\nTotal events: {len(self.final_id_events_)}")

    def _bfs_optimized(self, seed_t, seed_z, seed_y, seed_x, event_id):
        """BFS ultra-optimisé"""
        # Vérifier que le seed est valide
        if (self.av_[seed_t, seed_z, seed_y, seed_x] == 0 or
                self.id_connected_voxel_[seed_t, seed_z, seed_y, seed_x] != 0):
            return []

        # Pattern du seed
        seed_hash, seed_pattern = self._get_pattern_for_voxel(seed_t, seed_z, seed_y, seed_x)
        if seed_hash is None:
            return []

        queue = deque()
        visited = set()
        group_voxels = []

        # Ajouter le seed
        seed_voxel = (seed_t, seed_z, seed_y, seed_x)
        queue.append(seed_voxel)
        visited.add(seed_voxel)
        group_voxels.append(seed_voxel)
        self.id_connected_voxel_[seed_t, seed_z, seed_y, seed_x] = event_id

        # Propagation temporelle du seed
        if (seed_z, seed_y, seed_x) in self.voxel_patterns_:
            start_t, end_t, _, _ = self.voxel_patterns_[(seed_z, seed_y, seed_x)]
            for t in range(start_t, end_t):
                voxel_key = (t, seed_z, seed_y, seed_x)
                if (voxel_key not in visited and
                        self.id_connected_voxel_[t, seed_z, seed_y, seed_x] == 0):
                    self.id_connected_voxel_[t, seed_z, seed_y, seed_x] = event_id
                    queue.append(voxel_key)
                    visited.add(voxel_key)
                    group_voxels.append(voxel_key)

        # BFS avec optimisations
        processed_spatial = set()

        while queue:
            t, z, y, x = queue.popleft()

            # Voisins spatiaux
            neighbors = get_neighbors_3d(z, y, x, self.depth_, self.height_, self.width_)

            for nz, ny, nx in neighbors:
                spatial_key = (nz, ny, nx)

                if spatial_key in processed_spatial:
                    continue

                if ((self.av_[t, nz, ny, nx] != 0) and
                        (self.id_connected_voxel_[t, nz, ny, nx] == 0) and
                        (spatial_key in [vox for vox in self.active_voxels_])):

                    neighbor_hash, neighbor_pattern = self._get_pattern_for_voxel(t, nz, ny, nx)
                    if neighbor_hash is None:
                        continue

                    # Corrélation simplifiée
                    if neighbor_hash == seed_hash:
                        correlation = 1.0
                    else:
                        correlation = compute_simple_correlation(seed_pattern, neighbor_pattern)

                    if correlation > self.threshold_corr_:
                        processed_spatial.add(spatial_key)

                        # Propagation temporelle du voisin
                        start_t, end_t, _, _ = self.voxel_patterns_[(nz, ny, nx)]
                        for tt in range(start_t, end_t):
                            voxel_key = (tt, nz, ny, nx)
                            if (voxel_key not in visited and
                                    self.id_connected_voxel_[tt, nz, ny, nx] == 0):
                                self.id_connected_voxel_[tt, nz, ny, nx] = event_id
                                queue.append(voxel_key)
                                visited.add(voxel_key)
                                group_voxels.append(voxel_key)

        return group_voxels

    def _find_seed_point(self, t):
        """Recherche de seed point dans les voxels actifs seulement"""
        max_val = -1
        best_seed = None

        # Chercher seulement dans les voxels actifs
        for z, y, x in self.active_voxels_:
            if (self.av_[t, z, y, x] > 0 and
                    self.id_connected_voxel_[t, z, y, x] == 0 and
                    self.av_[t, z, y, x] > max_val):
                max_val = self.av_[t, z, y, x]
                best_seed = (x, y, z)

        return best_seed

    def _process_small_groups(self, small_groups, small_group_ids):
        """Version optimisée du traitement des petits groupes"""
        if not small_groups:
            return

        # Convertir en listes de tableaux numpy
        group_voxels_np = [np.array(group, dtype=np.int32) for group in small_groups]
        group_ids_np = np.array(small_group_ids, dtype=np.int32)

        # Étape 1: Fusion des petits groupes voisins
        adjacency = find_adjacent_groups(group_voxels_np, self.id_connected_voxel_, group_ids_np)

        # Préparer group_dict
        group_dict = {gid: [tuple(voxel) for voxel in group]
                      for gid, group in zip(group_ids_np, group_voxels_np)}

        merged_groups = merge_groups(adjacency, group_dict, group_ids_np)

        # Étape 2: Assignation aux groupes proches (version vectorisée)
        final_groups = self._assign_to_closest_group_vectorized(merged_groups)

        # Étape 3: Application des résultats
        self._apply_group_results(final_groups)

    def _assign_to_closest_group_vectorized(self, small_groups):
        """Version vectorisée de l'assignation aux groupes proches"""
        if not small_groups:
            return {}

        # Trouver les grands groupes (vectorisé)
        small_group_keys = np.array(list(small_groups.keys()), dtype=np.int32)
        mask = (self.id_connected_voxel_ > 0) & (~np.isin(self.id_connected_voxel_, small_group_keys))
        large_group_ids = np.unique(self.id_connected_voxel_[mask])

        if len(large_group_ids) == 0:
            return small_groups

        # Calcul des centroïdes des grands groupes (vectorisé)
        large_group_centroids = {}
        for gid in large_group_ids:
            positions = np.argwhere(self.id_connected_voxel_ == gid)
            if len(positions) > 0:
                large_group_centroids[gid] = np.mean(positions, axis=0)

        final_groups = {}

        for gid, group in small_groups.items():
            if len(group) == 0:
                continue

            # Calcul du centroïde du petit groupe
            group_arr = np.array(group)
            centroid = np.mean(group_arr, axis=0)

            # Calcul des distances (vectorisé)
            if large_group_centroids:
                lgids = np.array(list(large_group_centroids.keys()))
                centroids = np.array(list(large_group_centroids.values()))
                distances = np.sum(np.abs(centroid - centroids), axis=1)
                closest_idx = np.argmin(distances)
                closest_lgid = lgids[closest_idx]

                if closest_lgid not in final_groups:
                    final_groups[closest_lgid] = []
                final_groups[closest_lgid].extend(group)
                self.stats_["events_merged"] += 1
            else:
                final_groups[gid] = group

        return final_groups

    def _apply_group_results(self, final_groups):
        """Application des résultats"""
        events_retained = 0
        events_removed = 0

        for group_id, group in final_groups.items():
            group_size = len(group)

            if group_size >= self.threshold_size_3d_removed_:
                self.final_id_events_.append(group_id)
                events_retained += 1
                # Vectoriser l'assignation des voxels
                voxels = np.array(group, dtype=np.int32)
                if len(voxels) > 0:
                    self.id_connected_voxel_[voxels[:, 0], voxels[:, 1], voxels[:, 2], voxels[:, 3]] = group_id
            else:
                events_removed += 1
                voxels = np.array(group, dtype=np.int32)
                if len(voxels) > 0:
                    self.id_connected_voxel_[voxels[:, 0], voxels[:, 1], voxels[:, 2], voxels[:, 3]] = 0

        self.stats_["events_retained"] += events_retained
        self.stats_["events_removed"] += events_removed






    def get_results(self) -> Tuple[np.ndarray, List[int]]:
        """Remapping final et retour des résultats"""
        if self.final_id_events_:
            # Remapping efficace
            unique_old_ids = sorted(self.final_id_events_)
            for new_id, old_id in enumerate(unique_old_ids, 1):
                self.id_connected_voxel_[self.id_connected_voxel_ == old_id] = new_id

        return self.id_connected_voxel_, list(range(1, len(self.final_id_events_) + 1))

    def get_statistics(self) -> dict:
        """Statistiques"""
        unique_ids = np.unique(self.id_connected_voxel_[self.id_connected_voxel_ > 0])
        event_sizes = [np.sum(self.id_connected_voxel_ == eid) for eid in unique_ids]

        return {
            'nb_events': len(unique_ids),
            'event_sizes': event_sizes,
            'total_event_voxels': sum(event_sizes) if event_sizes else 0,
            'mean_event_size': np.mean(event_sizes) if event_sizes else 0,
            'max_event_size': np.max(event_sizes) if event_sizes else 0,
            'min_event_size': np.min(event_sizes) if event_sizes else 0,
        }


def detect_calcium_events_safe(av_data: np.ndarray, params_values: dict = None) -> Tuple[np.ndarray, List[int]]:
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

    print("⚡ ULTRA-OPTIMIZED EVENT DETECTION ⚡")
    total_start = time.time()

    detector = EventDetectorOptimized(
        av_data, threshold_size_3d, threshold_size_3d_removed, threshold_corr
    )

    # Détection
    detection_start = time.time()
    detector.find_events()
    detection_time = time.time() - detection_start

    # Résultats
    id_connections, id_events = detector.get_results()

    total_time = time.time() - total_start

    print(f"\n⚡ OPTIMIZED PERFORMANCE SUMMARY:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Detection time: {detection_time:.2f}s")
    print(f"   Events found: {len(id_events)}")
    print(f"   Speed: {av_data.size / total_time:.0f} voxels/second")

    # Statistiques
    stats = detector.get_statistics()
    print(f"   Event statistics: {stats}")

    # Sauvegarde si nécessaire
    if save_results:
        if output_directory is None:
            raise ValueError("Output directory must be specified if save_results is True.")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        id_connections = id_connections.astype(np.float32)
        export_data(id_connections, output_directory, export_as_single_tif=True, file_name="ID_calciumEvents")

    print("=" * 60)
    return id_connections, id_events


# Test function
def test_optimized_version():
    """Test avec données synthétiques optimisé"""
    print("=== TESTING OPTIMIZED VERSION ===")

    # Données de test plus petites
    shape = (10, 16, 64, 64)
    av_data = np.zeros(shape, dtype=np.float32)

    # Événements plus simples
    np.random.seed(42)

    # Événement 1
    av_data[2:6, 8:12, 20:25, 20:25] = 0.5 + np.random.normal(0, 0.1, (4, 4, 5, 5))

    # Événement 2
    av_data[5:8, 4:8, 40:45, 30:35] = 0.3 + np.random.normal(0, 0.05, (3, 4, 5, 5))

    print(f"Created test data: {shape}")
    print(f"Non-zero voxels: {np.count_nonzero(av_data):,}")

    results = detect_calcium_events_safe(av_data)
    return results


if __name__ == "__main__":
    test_optimized_version()