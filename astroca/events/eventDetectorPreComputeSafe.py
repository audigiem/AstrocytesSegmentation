"""
@file eventDetectorPreCompute_Fixed.py
@brief Version corrigée sans segmentation fault
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


# ============= FONCTIONS NUMBA CORRIGÉES =============

@njit
def extract_temporal_profile(av_data, z, y, x):
    """Extrait le profil temporel d'un voxel"""
    T = av_data.shape[0]
    profile = np.zeros(T, dtype=np.float32)
    for t in range(T):
        profile[t] = av_data[t, z, y, x]
    return profile


@njit
def find_nonzero_sequences(profile):
    """Trouve toutes les séquences non-nulles dans un profil"""
    sequences = []
    T = len(profile)
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
                end_t = t
                seq_length = end_t - start_t
                sequences.append((start_t, end_t, seq_length))
                in_sequence = False

    # Gérer la séquence qui se termine à la fin
    if in_sequence:
        end_t = T
        seq_length = end_t - start_t
        sequences.append((start_t, end_t, seq_length))

    return sequences


@njit
def simple_hash(arr):
    """Hash simple et sécurisé pour un array"""
    hash_val = np.int64(5381)
    for i in range(len(arr)):
        val = np.int64(arr[i] * 10000)  # Conversion sécurisée
        hash_val = ((hash_val << 5) + hash_val) + val
        hash_val = hash_val & 0x7FFFFFFFFFFFFFFF  # Éviter overflow
    return hash_val


@njit
def compute_ncc_simple(p1, p2):
    """Calcul NCC simplifié et sécurisé"""
    n1, n2 = len(p1), len(p2)

    if n1 == 0 or n2 == 0:
        return 0.0

    # Normalisation
    mean1 = np.mean(p1)
    mean2 = np.mean(p2)

    std1 = np.std(p1)
    std2 = np.std(p2)

    if std1 == 0 or std2 == 0:
        return 0.0

    # NCC pour différents lags (simplifié)
    max_corr = 0.0
    max_lag = min(n1, n2) // 2

    for lag in range(-max_lag, max_lag + 1):
        corr_sum = 0.0
        count = 0

        for i in range(n1):
            j = i - lag
            if 0 <= j < n2:
                corr_sum += ((p1[i] - mean1) / std1) * ((p2[j] - mean2) / std2)
                count += 1

        if count > 0:
            corr = corr_sum / count
            if corr > max_corr:
                max_corr = corr

    return max_corr


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


# ============= CLASSE CORRIGÉE =============

class EventDetectorSafe:
    """Version sécurisée sans segmentation fault"""

    def __init__(self, av_data: np.ndarray, threshold_size_3d: int = 10,
                 threshold_size_3d_removed: int = 5, threshold_corr: float = 0.5,
                 plot: bool = False):

        print("=== EVENT DETECTOR SAFE VERSION ===")
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

        # Structures pour les patterns (en Python, pas Numba)
        self.patterns_cache_ = {}  # hash -> pattern
        self.voxel_patterns_ = {}  # (z,y,x) -> list of (start_t, end_t, hash)

        # Pré-calcul sécurisé
        self._precompute_patterns_safe()

        self.stats_ = {"events_retained": 0, "events_merged": 0, "events_removed": 0}

    def _precompute_patterns_safe(self):
        """Pré-calcul sécurisé des patterns"""
        print("Pré-calcul des patterns (version sécurisée)...")
        start = time.time()

        pattern_count = 0

        for z in range(self.depth_):
            for y in range(self.height_):
                for x in range(self.width_):
                    # Extraire le profil temporel
                    profile = extract_temporal_profile(self.av_, z, y, x)

                    # Trouver les séquences
                    sequences = find_nonzero_sequences(profile)

                    if len(sequences) > 0:
                        self.voxel_patterns_[(z, y, x)] = []

                        for start_t, end_t, length in sequences:
                            if length > 1:  # Éviter les patterns trop courts
                                pattern = profile[start_t:end_t]
                                pattern_hash = simple_hash(pattern)

                                # Stocker le pattern
                                if pattern_hash not in self.patterns_cache_:
                                    self.patterns_cache_[pattern_hash] = pattern.copy()
                                    pattern_count += 1

                                # Associer le voxel au pattern
                                self.voxel_patterns_[(z, y, x)].append((start_t, end_t, pattern_hash))

        print(f"✓ {pattern_count} patterns uniques en {time.time() - start:.2f}s")

    def _get_pattern_for_voxel(self, t, z, y, x):
        """Récupère le pattern actif pour un voxel à un temps donné"""
        if (z, y, x) not in self.voxel_patterns_:
            return None

        for start_t, end_t, pattern_hash in self.voxel_patterns_[(z, y, x)]:
            if start_t <= t < end_t:
                return pattern_hash

        return None

    def _compute_pattern_correlation(self, hash1, hash2):
        """Calcule la corrélation entre deux patterns"""
        if hash1 == hash2:
            return 1.0

        if hash1 not in self.patterns_cache_ or hash2 not in self.patterns_cache_:
            return 0.0

        pattern1 = self.patterns_cache_[hash1]
        pattern2 = self.patterns_cache_[hash2]

        return compute_ncc_simple(pattern1, pattern2)

    def find_events(self) -> None:
        """Recherche d'événements sécurisée"""
        print(
            f"Thresholds: size={self.threshold_size_3d_}, removed={self.threshold_size_3d_removed_}, corr={self.threshold_corr_}")

        if np.count_nonzero(self.av_) == 0:
            print("No non-zero voxels found!")
            return

        event_id = 1
        small_groups = []
        small_group_ids = []

        for t in tqdm(range(self.time_length_), desc="Processing frames"):
            seed = self._find_seed_point(t)

            while seed is not None:
                x, y, z = seed

                # BFS sécurisé
                current_group = self._bfs_safe(t, z, y, x, event_id)

                # Classification
                if len(current_group) < self.threshold_size_3d_:
                    small_groups.append(current_group)
                    small_group_ids.append(event_id)
                else:
                    self.final_id_events_.append(event_id)
                    self.stats_["events_retained"] += 1

                event_id += 1
                seed = self._find_seed_point(t)

        # Traitement des petits groupes
        self._process_small_groups(small_groups, small_group_ids)

        print(f"\nTotal events: {len(self.final_id_events_)}")

    def _bfs_safe(self, seed_t, seed_z, seed_y, seed_x, event_id):
        """BFS sécurisé"""
        queue = deque()
        visited = set()
        group_voxels = []

        # Pattern du seed
        seed_pattern_hash = self._get_pattern_for_voxel(seed_t, seed_z, seed_y, seed_x)
        if seed_pattern_hash is None:
            return []

        # Propagation temporelle du seed
        if (seed_z, seed_y, seed_x) in self.voxel_patterns_:
            for start_t, end_t, pattern_hash in self.voxel_patterns_[(seed_z, seed_y, seed_x)]:
                if pattern_hash == seed_pattern_hash:
                    for t in range(start_t, end_t):
                        if self.id_connected_voxel_[t, seed_z, seed_y, seed_x] == 0:
                            self.id_connected_voxel_[t, seed_z, seed_y, seed_x] = event_id
                            queue.append((t, seed_z, seed_y, seed_x))
                            visited.add((t, seed_z, seed_y, seed_x))
                            group_voxels.append((t, seed_z, seed_y, seed_x))

        while queue:
            t, z, y, x = queue.popleft()

            # Voisins spatiaux
            neighbors = get_neighbors_3d(z, y, x, self.depth_, self.height_, self.width_)

            for nz, ny, nx in neighbors:
                if (self.av_[t, nz, ny, nx] != 0 and
                        self.id_connected_voxel_[t, nz, ny, nx] == 0):

                    neighbor_pattern_hash = self._get_pattern_for_voxel(t, nz, ny, nx)
                    if neighbor_pattern_hash is None:
                        continue

                    # Vérifier la corrélation
                    correlation = self._compute_pattern_correlation(seed_pattern_hash, neighbor_pattern_hash)

                    if correlation > self.threshold_corr_:
                        # Propagation temporelle du voisin
                        if (nz, ny, nx) in self.voxel_patterns_:
                            for start_t, end_t, pattern_hash in self.voxel_patterns_[(nz, ny, nx)]:
                                if pattern_hash == neighbor_pattern_hash:
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
        """Recherche de seed point"""
        frame = self.av_[t]
        unprocessed = (frame > 0) & (self.id_connected_voxel_[t] == 0)

        if not np.any(unprocessed):
            return None

        # Trouver le maximum
        masked_frame = np.where(unprocessed, frame, -1)
        flat_idx = np.argmax(masked_frame)
        z, y, x = np.unravel_index(flat_idx, frame.shape)

        if masked_frame[z, y, x] <= 0:
            return None

        return (x, y, z)

    def _process_small_groups(self, small_groups, small_group_ids):
        """Traitement des petits groupes"""
        for i, group in enumerate(small_groups):
            if len(group) >= self.threshold_size_3d_removed_:
                self.final_id_events_.append(small_group_ids[i])
                self.stats_["events_retained"] += 1
            else:
                # Supprimer le groupe
                for t, z, y, x in group:
                    self.id_connected_voxel_[t, z, y, x] = 0
                self.stats_["events_removed"] += 1

    def get_results(self) -> Tuple[np.ndarray, List[int]]:
        """Remapping final et retour des résultats"""
        if self.final_id_events_:
            # Remapping
            max_id = max(self.final_id_events_) if self.final_id_events_ else 0
            id_map = np.zeros(max_id + 1, dtype=np.int32)

            for new_id, old_id in enumerate(sorted(self.final_id_events_), 1):
                id_map[old_id] = new_id

            # Vectorisation sécurisée
            old_ids = self.id_connected_voxel_.copy()
            for old_id, new_id in enumerate(id_map):
                if new_id > 0:
                    self.id_connected_voxel_[old_ids == old_id] = new_id

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
    Version sécurisée de la détection d'événements
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

    print("🛡️ SAFE EVENT DETECTION 🛡️")
    total_start = time.time()

    detector = EventDetectorSafe(
        av_data, threshold_size_3d, threshold_size_3d_removed, threshold_corr
    )

    # Détection
    detection_start = time.time()
    detector.find_events()
    detection_time = time.time() - detection_start

    # Résultats
    id_connections, id_events = detector.get_results()

    total_time = time.time() - total_start

    print(f"\n✅ SAFE PERFORMANCE SUMMARY:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Detection time: {detection_time:.2f}s")
    print(f"   Events found: {len(id_events)}")
    print(f"   Speed: {av_data.size / total_time:.0f} voxels/second")

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
def test_safe_version():
    """Test avec données synthétiques"""
    print("=== TESTING SAFE VERSION ===")

    # Données de test
    shape = (10, 32, 128, 128)
    av_data = np.zeros(shape, dtype=np.float32)

    # Plusieurs événements synthétiques
    np.random.seed(42)

    # Événement 1
    av_data[2:6, 10:15, 30:45, 30:45] = np.random.rand(4, 5, 15, 15) * 0.8 + 0.2

    # Événement 2
    av_data[5:8, 20:25, 80:95, 60:75] = np.random.rand(3, 5, 15, 15) * 0.6 + 0.4

    print(f"Created test data: {shape}")
    print(f"Non-zero voxels: {np.count_nonzero(av_data):,}")

    results = detect_calcium_events_safe(av_data)
    return results


if __name__ == "__main__":
    test_safe_version()