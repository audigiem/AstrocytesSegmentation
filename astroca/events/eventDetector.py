import numpy as np
import numba as nb
from numba import njit, prange
from typing import List, Tuple, Optional
import time
from scipy import ndimage
from skimage.measure import label

class EventDetectorOptimized:
    """
    Détecteur d'événements calcium ultra-optimisé avec NumPy et Numba.
    """

    def __init__(self, av_data: np.ndarray, threshold_size_3d: int = 10,
                 threshold_size_3d_removed: int = 5, threshold_corr: float = 0.5,
                 plot: bool = False):
        """
        Initialise le détecteur d'événements.

        Args:
            av_data: Données 4D (temps, profondeur, hauteur, largeur)
            threshold_size_3d: Seuil de taille minimale pour les groupes
            threshold_size_3d_removed: Seuil de taille en dessous duquel les groupes sont supprimés
            threshold_corr: Seuil de corrélation croisée
            plot: Activer les graphiques
        """
        print(f"Input data shape: {av_data.shape}")
        print(f"Input data range: [{av_data.min():.3f}, {av_data.max():.3f}]")
        print(f"Non-zero voxels: {np.count_nonzero(av_data)}/{av_data.size}")

        self.av_ = av_data.astype(np.float32)
        self.time_length_, self.depth_, self.height_, self.width_ = av_data.shape

        self.threshold_size_3d_ = threshold_size_3d
        self.threshold_size_3d_removed_ = threshold_size_3d_removed
        self.threshold_corr_ = threshold_corr
        self.plot_ = plot

        # Pré-calculer le masque des voxels non-zéro pour optimisation
        self.nonzero_mask_ = self.av_ != 0
        self.nonzero_coords_ = np.where(self.nonzero_mask_)
        print(f"Found {len(self.nonzero_coords_[0])} non-zero voxels")

        # Structure de données optimisée
        self.id_connected_voxel_ = np.zeros_like(self.av_, dtype=np.int32)
        self.final_id_events_ = []

        # Cache pour les patterns
        self.pattern_cache_ = {}

        # TRACE: compteurs de debug
        self.stats_ = {
            "patterns_computed": 0,
            "regions_grown": 0,
            "correlations_computed": 0,
            "events_retained": 0,
            "events_merged": 0
        }

    def find_events(self) -> None:
        """
        Fonction principale optimisée pour trouver les événements calcium.
        """
        start_time = time.time()
        print("\n=== Finding events ===")
        print(f"Thresholds -> size: {self.threshold_size_3d_}, removed: {self.threshold_size_3d_removed_}, corr: {self.threshold_corr_}")

        if len(self.nonzero_coords_[0]) == 0:
            print("No non-zero voxels found!")
            return

        event_id = 1
        processed_mask = np.zeros_like(self.av_, dtype=bool)

        # Traiter par frame temporelle pour optimiser
        for t in range(self.time_length_):
            frame_time = time.time()
            # Obtenir tous les voxels non-zéro pour cette frame
            frame_nonzero = np.where(self.nonzero_mask_[t])

            if len(frame_nonzero[0]) == 0:
                continue

            print(f"Processing time frame {t}: {len(frame_nonzero[0])} non-zero voxels")

            for i in range(len(frame_nonzero[0])):
                z, y, x = frame_nonzero[0][i], frame_nonzero[1][i], frame_nonzero[2][i]

                if processed_mask[t, z, y, x] or self.id_connected_voxel_[t, z, y, x] != 0:
                    continue

                # Détection du pattern temporal
                pattern = self._detect_pattern_optimized(x, y, z, t)
                if pattern is None or len(pattern) < 2:
                    continue

                # Croissance de région 3D+T
                event_voxels = self._grow_region_optimized(x, y, z, t, pattern, processed_mask)

                if len(event_voxels) >= self.threshold_size_3d_:
                    # Assigner l'ID à tous les voxels de l'événement
                    for vx, vy, vz, vt in event_voxels:
                        self.id_connected_voxel_[vt, vz, vy, vx] = event_id

                    self.final_id_events_.append(event_id)
                    print(f"Event {event_id}: {len(event_voxels)} voxels")
                    event_id += 1
                    self.stats_["events_retained"] += 1
                elif len(event_voxels) >= self.threshold_size_3d_removed_:
                    # Traiter les petits groupes
                    if self._try_merge_small_group(event_voxels):
                        for vx, vy, vz, vt in event_voxels:
                            neighbor_id = self._find_best_neighbor_id(vx, vy, vz, vt)
                            if neighbor_id > 0:
                                self.id_connected_voxel_[vt, vz, vy, vx] = neighbor_id
                                break
                        print(f"   Small group merged with ID {neighbor_id} ({len(event_voxels)} voxels)")
                        self.stats_["events_merged"] += 1
                print(f"   Frame at time {t} nbr {i} processed in {(time.time() - frame_time):.2f}s")  # TRACE

        nb_events = len(self.final_id_events_)
        print(f"\n=== Found {nb_events} calcium events! ===")
        print("Stats:", self.stats_)  # TRACE
        end_time = time.time()
        duration = (end_time - start_time) / 60
        print(f"Duration: {duration:.2f}min")

    def _detect_pattern_optimized(self, x: int, y: int, z: int, t: int) -> Optional[np.ndarray]:
        """
        Détection optimisée de pattern temporal.
        """
        self.stats_["patterns_computed"] += 1  # TRACE
        key = (x, y, z)
        if key in self.pattern_cache_:
            return self.pattern_cache_[key]

        profile = self.av_[:, z, y, x]

        # Trouver toutes les séquences continues non-zéro
        nonzero_indices = np.where(profile != 0)[0]
        if len(nonzero_indices) == 0:
            return None

        # Trouver la séquence qui contient t
        pattern_start = None
        pattern_end = None

        for i in range(len(nonzero_indices)):
            if nonzero_indices[i] <= t:
                if i == len(nonzero_indices) - 1 or nonzero_indices[i + 1] != nonzero_indices[i] + 1:
                    # Fin d'une séquence
                    if pattern_start is None:
                        pattern_start = nonzero_indices[i]
                    pattern_end = nonzero_indices[i]
                else:
                    # Début ou continuation d'une séquence
                    if pattern_start is None:
                        pattern_start = nonzero_indices[i]
            elif nonzero_indices[i] == t + 1 and pattern_end == t:
                # Extension de la séquence
                pattern_end = nonzero_indices[i]
            else:
                break

        if pattern_start is None or pattern_end is None:
            return None

        # Extraire le pattern
        pattern = profile[pattern_start:pattern_end + 1]

        # Vérifier que le pattern est valide
        if np.all(pattern != 0) and len(pattern) >= 2:
            self.pattern_cache_[key] = pattern
            return pattern

        return None

    def _grow_region_optimized(self, seed_x: int, seed_y: int, seed_z: int, seed_t: int,
                              seed_pattern: np.ndarray, processed_mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Croissance de région optimisée avec corrélation.
        """
        self.stats_["regions_grown"] += 1  # TRACE
        event_voxels = []
        to_process = [(seed_x, seed_y, seed_z, seed_t)]
        visited = set()

        while to_process:
            x, y, z, t = to_process.pop()

            if (x, y, z, t) in visited:
                continue
            visited.add((x, y, z, t))

            if (x < 0 or x >= self.width_ or y < 0 or y >= self.height_ or
                z < 0 or z >= self.depth_ or t < 0 or t >= self.time_length_):
                continue

            if processed_mask[t, z, y, x] or self.av_[t, z, y, x] == 0:
                continue

            # Obtenir le pattern de ce voxel
            current_pattern = self._detect_pattern_optimized(x, y, z, t)
            if current_pattern is None:
                continue

            # Calculer la corrélation avec le pattern de référence
            correlation = self._compute_correlation_fast(seed_pattern, current_pattern)
            self.stats_["correlations_computed"] += 1  # TRACE
            if correlation >= self.threshold_corr_:
                event_voxels.append((x, y, z, t))
                processed_mask[t, z, y, x] = True

                # Ajouter les voisins (6-connectivité pour commencer, plus rapide)
                neighbors = [
                    (x + 1, y, z, t), (x - 1, y, z, t),
                    (x, y + 1, z, t), (x, y - 1, z, t),
                    (x, y, z + 1, t), (x, y, z - 1, t),
                    (x, y, z, t + 1), (x, y, z, t - 1)
                ]

                for nx, ny, nz, nt in neighbors:
                    if (nx, ny, nz, nt) not in visited:
                        to_process.append((nx, ny, nz, nt))

        return event_voxels

    @staticmethod
    @njit(fastmath=True)
    def _compute_correlation_fast(pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """
        Calcul rapide de corrélation normalisée.
        """
        if len(pattern1) == 0 or len(pattern2) == 0:
            return 0.0

        # Utiliser la corrélation de Pearson simplifiée
        min_len = min(len(pattern1), len(pattern2))
        p1 = pattern1[:min_len]
        p2 = pattern2[:min_len]

        mean1 = np.mean(p1)
        mean2 = np.mean(p2)

        num = np.sum((p1 - mean1) * (p2 - mean2))
        den1 = np.sqrt(np.sum((p1 - mean1) ** 2))
        den2 = np.sqrt(np.sum((p2 - mean2) ** 2))

        if den1 == 0 or den2 == 0:
            return 1.0 if np.allclose(p1, p2) else 0.0

        return abs(num / (den1 * den2))

    def _try_merge_small_group(self, event_voxels: List[Tuple[int, int, int, int]]) -> bool:
        """
        Essaie de fusionner un petit groupe avec des groupes voisins.
        """
        neighbor_ids = {}

        for x, y, z, t in event_voxels:
            # Vérifier le voisinage 6-connecté
            for dx, dy, dz, dt in [(1,0,0,0), (-1,0,0,0), (0,1,0,0), (0,-1,0,0),
                                   (0,0,1,0), (0,0,-1,0), (0,0,0,1), (0,0,0,-1)]:
                nx, ny, nz, nt = x + dx, y + dy, z + dz, t + dt

                if (0 <= nx < self.width_ and 0 <= ny < self.height_ and
                    0 <= nz < self.depth_ and 0 <= nt < self.time_length_):

                    neighbor_id = self.id_connected_voxel_[nt, nz, ny, nx]
                    if neighbor_id > 0:
                        neighbor_ids[neighbor_id] = neighbor_ids.get(neighbor_id, 0) + 1

        return len(neighbor_ids) > 0

    def _find_best_neighbor_id(self, x: int, y: int, z: int, t: int) -> int:
        """
        Trouve le meilleur ID de voisin pour la fusion.
        """
        neighbor_ids = {}

        for dx, dy, dz, dt in [(1,0,0,0), (-1,0,0,0), (0,1,0,0), (0,-1,0,0),
                               (0,0,1,0), (0,0,-1,0), (0,0,0,1), (0,0,0,-1)]:
            nx, ny, nz, nt = x + dx, y + dy, z + dz, t + dt

            if (0 <= nx < self.width_ and 0 <= ny < self.height_ and
                0 <= nz < self.depth_ and 0 <= nt < self.time_length_):

                neighbor_id = self.id_connected_voxel_[nt, nz, ny, nx]
                if neighbor_id > 0:
                    neighbor_ids[neighbor_id] = neighbor_ids.get(neighbor_id, 0) + 1

        if neighbor_ids:
            return max(neighbor_ids.keys(), key=lambda k: neighbor_ids[k])
        return 0

    def get_results(self) -> Tuple[np.ndarray, List[int]]:
        """
        Retourne les résultats de la détection d'événements.

        Returns:
            id_connected_voxel: Array des IDs des voxels connectés
            final_id_events: Liste des IDs des événements finaux
        """
        return self.id_connected_voxel_, self.final_id_events_

    def get_statistics(self) -> dict:
        """
        Retourne des statistiques sur la détection.
        """
        stats = {
            'nb_events': len(self.final_id_events_),
            'event_sizes': [],
            'total_event_voxels': 0
        }

        for event_id in self.final_id_events_:
            size = np.sum(self.id_connected_voxel_ == event_id)
            stats['event_sizes'].append(size)
            stats['total_event_voxels'] += size

        if stats['event_sizes']:
            stats['mean_event_size'] = np.mean(stats['event_sizes'])
            stats['median_event_size'] = np.median(stats['event_sizes'])
            stats['max_event_size'] = np.max(stats['event_sizes'])
            stats['min_event_size'] = np.min(stats['event_sizes'])

        return stats


def detect_calcium_events_optimized(av_data: np.ndarray, threshold_size_3d: int = 10,
                                   threshold_size_3d_removed: int = 5,
                                   threshold_corr: float = 0.5) -> Tuple[np.ndarray, List[int], dict]:
    """
    Fonction optimisée pour détecter les événements calcium.

    Args:
        av_data: Données 4D (temps, profondeur, hauteur, largeur)
        threshold_size_3d: Seuil de taille minimale pour les groupes
        threshold_size_3d_removed: Seuil de suppression
        threshold_corr: Seuil de corrélation

    Returns:
        id_connected_voxel: Array des IDs des voxels connectés
        final_id_events: Liste des IDs des événements finaux
        statistics: Dictionnaire avec les statistiques
    """
    detector = EventDetectorOptimized(av_data, threshold_size_3d,
                                    threshold_size_3d_removed, threshold_corr)
    detector.find_events()
    results = detector.get_results()
    stats = detector.get_statistics()

    print(f"\n=== STATISTICS ===")
    print(f"Number of events: {stats['nb_events']}")
    if stats['nb_events'] > 0:
        print(f"Event sizes: {stats['event_sizes']}")
        print(f"Mean event size: {stats['mean_event_size']:.1f}")
        print(f"Max event size: {stats['max_event_size']}")
        print(f"Total event voxels: {stats['total_event_voxels']}")

    return results[0], results[1], stats


# Fonction de test avec données synthétiques
def test_with_synthetic_data():
    """
    Teste le détecteur avec des données synthétiques.
    """
    print("=== TESTING WITH SYNTHETIC DATA ===")

    # Créer des données de test
    shape = (8, 32, 512, 320)
    av_data = np.zeros(shape, dtype=np.float32)

    # Ajouter quelques événements synthétiques
    # Événement 1: pattern temporel dans une petite région
    av_data[2:5, 10:15, 100:120, 50:70] = np.random.rand(3, 5, 20, 20) * 0.5 + 0.5

    # Événement 2: pattern temporel différent
    av_data[1:4, 20:25, 200:230, 100:130] = np.random.rand(3, 5, 30, 30) * 0.3 + 0.7

    # Événement 3: petit événement
    av_data[5:7, 5:8, 400:405, 200:205] = np.random.rand(2, 3, 5, 5) * 0.8 + 0.2

    print(f"Created synthetic data with shape {shape}")
    print(f"Non-zero voxels: {np.count_nonzero(av_data)}")

    # Tester la détection
    results = detect_calcium_events_optimized(av_data,
                                            threshold_size_3d=50,
                                            threshold_size_3d_removed=10,
                                            threshold_corr=0.3)

    return results

if __name__ == "__main__":
    # Test avec données synthétiques
    test_results = test_with_synthetic_data()