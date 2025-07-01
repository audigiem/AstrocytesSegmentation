"""
@file event_merger_optimized.py
@brief Fusion/suppression optimisée des petits groupes d'événements calciques
"""

import numpy as np
import numba as nb
from numba import njit, prange, types
from numba.typed import Dict, List as NumbaList
from scipy import ndimage
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import time
from typing import List, Tuple, Set, Dict as PyDict
from collections import defaultdict


@njit
def get_event_voxels_fast(id_connected, event_id):
    """Récupère rapidement tous les voxels d'un événement"""
    coords = []
    T, Z, Y, X = id_connected.shape

    for t in range(T):
        for z in range(Z):
            for y in range(Y):
                for x in range(X):
                    if id_connected[t, z, y, x] == event_id:
                        coords.append((t, z, y, x))

    return coords


@njit
def compute_event_centroid(voxel_coords):
    """Calcule le centroïde d'un événement"""
    if len(voxel_coords) == 0:
        return (0.0, 0.0, 0.0, 0.0)

    sum_t, sum_z, sum_y, sum_x = 0.0, 0.0, 0.0, 0.0
    for t, z, y, x in voxel_coords:
        sum_t += t
        sum_z += z
        sum_y += y
        sum_x += x

    n = len(voxel_coords)
    return (sum_t / n, sum_z / n, sum_y / n, sum_x / n)


@njit
def euclidean_distance_4d(p1, p2):
    """Distance euclidienne 4D optimisée"""
    dt = p1[0] - p2[0]
    dz = p1[1] - p2[1]
    dy = p1[2] - p2[2]
    dx = p1[3] - p2[3]
    return np.sqrt(dt * dt + dz * dz + dy * dy + dx * dx)


class EventMergerOptimized:
    """Classe optimisée pour la fusion/suppression des petits groupes"""

    def __init__(self, id_connected: np.ndarray, threshold_keep: int = 400,
                 threshold_remove: int = 20):
        """
        Args:
            id_connected: Array 4D (T,Z,Y,X) avec les IDs des événements
            threshold_keep: Seuil au-dessus duquel un groupe est gardé (400)
            threshold_remove: Seuil en-dessous duquel un groupe isolé est supprimé (20)
        """
        self.id_connected = id_connected.copy()
        self.threshold_keep = threshold_keep
        self.threshold_remove = threshold_remove

        # Statistiques
        self.stats = {
            'original_events': 0,
            'small_groups_found': 0,
            'groups_merged': 0,
            'groups_assigned': 0,
            'groups_removed': 0,
            'final_events': 0
        }

    def process_small_groups(self) -> np.ndarray:
        """Traite tous les petits groupes selon la logique spécifiée"""
        print("=== PROCESSING SMALL GROUPS ===")
        start_time = time.time()

        # 1. Analyser tous les événements existants
        event_info = self._analyze_all_events()

        # 2. Identifier les petits et grands groupes
        small_groups, large_groups = self._classify_events(event_info)

        # 3. Grouper les petits groupes voisins
        merged_small_groups = self._merge_neighboring_small_groups(small_groups)

        # 4. Assigner les groupes fusionnés aux grands groupes proches
        self._assign_to_closest_large_groups(merged_small_groups, large_groups)

        # 5. Nettoyer les groupes restants trop petits
        self._remove_isolated_small_groups()

        print(f"Processing completed in {time.time() - start_time:.2f}s")
        self._print_statistics()

        return self.id_connected

    def _analyze_all_events(self) -> PyDict:
        """Analyse tous les événements pour obtenir taille et centroïde"""
        print("Analyzing all events...")

        unique_ids = np.unique(self.id_connected[self.id_connected > 0])
        self.stats['original_events'] = len(unique_ids)

        event_info = {}

        for event_id in unique_ids:
            # Utilisation de numpy pour l'efficacité
            mask = (self.id_connected == event_id)
            voxel_coords = np.column_stack(np.where(mask))

            size = len(voxel_coords)
            centroid = np.mean(voxel_coords, axis=0)

            event_info[event_id] = {
                'size': size,
                'centroid': centroid,
                'voxels': voxel_coords
            }

        print(f"Found {len(unique_ids)} events")
        return event_info

    def _classify_events(self, event_info: PyDict) -> Tuple[PyDict, PyDict]:
        """Sépare les petits et grands groupes"""
        small_groups = {}
        large_groups = {}

        for event_id, info in event_info.items():
            if info['size'] < self.threshold_keep:
                small_groups[event_id] = info
            else:
                large_groups[event_id] = info

        self.stats['small_groups_found'] = len(small_groups)
        print(f"Small groups: {len(small_groups)}, Large groups: {len(large_groups)}")

        return small_groups, large_groups

    def _merge_neighboring_small_groups(self, small_groups: PyDict) -> List[PyDict]:
        """Fusionne les petits groupes voisins"""
        if not small_groups:
            return []

        print("Merging neighboring small groups...")

        # Créer une matrice de distances entre centroïdes
        small_ids = list(small_groups.keys())
        centroids = np.array([small_groups[sid]['centroid'] for sid in small_ids])

        # Seuil de voisinage adaptatif basé sur la taille moyenne des voxels
        T, Z, Y, X = self.id_connected.shape
        spatial_threshold = np.sqrt((Z * Y * X) / len(small_groups)) if small_groups else 10

        # Utiliser NearestNeighbors pour l'efficacité
        nn = NearestNeighbors(radius=spatial_threshold, metric='euclidean')
        nn.fit(centroids)

        # Graph de connectivité
        adjacency = nn.radius_neighbors_graph(centroids)

        # Composantes connexes pour grouper les voisins
        n_components, labels = self._connected_components(adjacency)

        # Créer les groupes fusionnés
        merged_groups = []
        groups_by_component = defaultdict(list)

        for i, label in enumerate(labels):
            groups_by_component[label].append(small_ids[i])

        for component_ids in groups_by_component.values():
            if len(component_ids) > 1:
                self.stats['groups_merged'] += len(component_ids) - 1

            # Fusionner physiquement les groupes
            merged_group = self._merge_physical_groups(component_ids, small_groups)
            merged_groups.append(merged_group)

        print(f"Created {len(merged_groups)} merged groups from {len(small_groups)} small groups")
        return merged_groups

    def _connected_components(self, adjacency):
        """Trouve les composantes connexes dans le graphe d'adjacence"""
        from scipy.sparse.csgraph import connected_components
        return connected_components(adjacency, directed=False)

    def _merge_physical_groups(self, group_ids: List[int], small_groups: PyDict) -> PyDict:
        """Fusionne physiquement plusieurs groupes en un seul"""
        if len(group_ids) == 1:
            return {
                'original_ids': group_ids,
                'size': small_groups[group_ids[0]]['size'],
                'centroid': small_groups[group_ids[0]]['centroid'],
                'voxels': small_groups[group_ids[0]]['voxels']
            }

        # Combiner tous les voxels
        all_voxels = []
        total_size = 0

        for gid in group_ids:
            all_voxels.append(small_groups[gid]['voxels'])
            total_size += small_groups[gid]['size']

        combined_voxels = np.vstack(all_voxels)
        combined_centroid = np.mean(combined_voxels, axis=0)

        return {
            'original_ids': group_ids,
            'size': total_size,
            'centroid': combined_centroid,
            'voxels': combined_voxels
        }

    def _assign_to_closest_large_groups(self, merged_small_groups: List[PyDict],
                                        large_groups: PyDict):
        """Assigne chaque groupe fusionné au grand groupe le plus proche"""
        if not large_groups or not merged_small_groups:
            return

        print("Assigning merged groups to closest large groups...")

        # Préparer les centroïdes des grands groupes
        large_ids = list(large_groups.keys())
        large_centroids = np.array([large_groups[lid]['centroid'] for lid in large_ids])

        # Utiliser NearestNeighbors pour l'efficacité
        nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn.fit(large_centroids)

        assignments_made = 0

        for merged_group in merged_small_groups:
            # Seuil de distance adaptatif
            max_distance = self._compute_adaptive_distance_threshold(merged_group, large_groups)

            # Trouver le grand groupe le plus proche
            distances, indices = nn.kneighbors([merged_group['centroid']], n_neighbors=1)
            closest_distance = distances[0][0]

            if closest_distance <= max_distance:
                # Assigner au groupe le plus proche
                target_id = large_ids[indices[0][0]]
                self._reassign_voxels(merged_group, target_id)
                assignments_made += 1
            # Sinon, le groupe reste isolé et sera traité plus tard

        self.stats['groups_assigned'] = assignments_made
        print(f"Assigned {assignments_made} merged groups to large groups")

    def _compute_adaptive_distance_threshold(self, merged_group: PyDict,
                                             large_groups: PyDict) -> float:
        """Calcule un seuil de distance adaptatif basé sur la taille des groupes"""
        # Seuil basé sur la racine cubique du volume pour tenir compte de la 3D
        base_threshold = np.power(merged_group['size'], 1 / 3) * 2

        # Ajuster selon la densité des grands groupes
        if large_groups:
            avg_large_size = np.mean([g['size'] for g in large_groups.values()])
            scale_factor = np.power(avg_large_size / self.threshold_keep, 1 / 3)
            base_threshold *= scale_factor

        return max(base_threshold, 5.0)  # Minimum de 5 unités

    def _reassign_voxels(self, merged_group: PyDict, target_id: int):
        """Réassigne tous les voxels d'un groupe fusionné à un nouvel ID"""
        # Marquer les anciens IDs pour suppression
        for old_id in merged_group['original_ids']:
            self.id_connected[self.id_connected == old_id] = target_id

    def _remove_isolated_small_groups(self):
        """Supprime les groupes isolés trop petits"""
        print("Removing isolated small groups...")

        # Réanalyser après les fusions/assignations
        current_events = {}
        unique_ids = np.unique(self.id_connected[self.id_connected > 0])

        for event_id in unique_ids:
            size = np.sum(self.id_connected == event_id)
            current_events[event_id] = size

        removed_count = 0

        for event_id, size in current_events.items():
            if size < self.threshold_remove:
                self.id_connected[self.id_connected == event_id] = 0
                removed_count += 1

        self.stats['groups_removed'] = removed_count
        self.stats['final_events'] = len(current_events) - removed_count

        print(f"Removed {removed_count} isolated small groups")

    def _print_statistics(self):
        """Affiche les statistiques du traitement"""
        print("\n=== PROCESSING STATISTICS ===")
        print(f"Original events: {self.stats['original_events']}")
        print(f"Small groups found: {self.stats['small_groups_found']}")
        print(f"Groups merged: {self.stats['groups_merged']}")
        print(f"Groups assigned to large groups: {self.stats['groups_assigned']}")
        print(f"Groups removed (too small): {self.stats['groups_removed']}")
        print(f"Final events: {self.stats['final_events']}")
        print(f"Reduction: {self.stats['original_events'] - self.stats['final_events']} events")


def process_small_groups_optimized(id_connected: np.ndarray,
                                   threshold_keep: int = 400,
                                   threshold_remove: int = 20) -> np.ndarray:
    """
    Fonction principale pour traiter les petits groupes selon la logique spécifiée

    Args:
        id_connected: Array 4D (T,Z,Y,X) avec les IDs des événements
        threshold_keep: Seuil au-dessus duquel un groupe est gardé (400 voxels)
        threshold_remove: Seuil en-dessous duquel un groupe isolé est supprimé (20 voxels)

    Returns:
        Array 4D traité avec les groupes fusionnés/supprimés
    """
    print("=== STARTING OPTIMIZED SMALL GROUPS PROCESSING ===")
    merger = EventMergerOptimized(id_connected, threshold_keep, threshold_remove)
 
    return merger.process_small_groups()


# Test de performance
def test_merger_performance():
    """Test de performance avec des données synthétiques"""
    print("=== TESTING MERGER PERFORMANCE ===")

    # Créer des données de test
    shape = (70, 32, 502, 320)  # Taille réaliste
    id_connected = np.zeros(shape, dtype=np.int32)

    # Générer des événements synthétiques de différentes tailles
    np.random.seed(42)
    event_id = 1

    # Quelques grands événements
    for _ in range(5):
        t_start = np.random.randint(0, shape[0] - 10)
        z_start = np.random.randint(0, shape[1] - 5)
        y_start = np.random.randint(0, shape[2] - 50)
        x_start = np.random.randint(0, shape[3] - 50)

        # Grand événement (> 400 voxels)
        id_connected[t_start:t_start + 8, z_start:z_start + 3,
        y_start:y_start + 20, x_start:x_start + 20] = event_id
        event_id += 1

    # Beaucoup de petits événements
    for _ in range(50):
        t = np.random.randint(0, shape[0])
        z = np.random.randint(0, shape[1])
        y_start = np.random.randint(0, shape[2] - 10)
        x_start = np.random.randint(0, shape[3] - 10)

        # Petit événement (< 400 voxels)
        size = np.random.randint(5, 100)
        coords = []
        for _ in range(size):
            dy = np.random.randint(0, 10)
            dx = np.random.randint(0, 10)
            if (y_start + dy < shape[2] and x_start + dx < shape[3]):
                coords.append((t, z, y_start + dy, x_start + dx))

        for coord in coords:
            id_connected[coord] = event_id
        event_id += 1

    print(f"Created test data with {np.max(id_connected)} events")
    print(f"Total voxels assigned: {np.sum(id_connected > 0):,}")

    # Test du merger
    start_time = time.time()
    result = process_small_groups_optimized(id_connected)
    processing_time = time.time() - start_time

    print(f"Processing completed in {processing_time:.2f}s")
    print(f"Speed: {id_connected.size / processing_time:.0f} voxels/second")

    return result


if __name__ == "__main__":
    test_merger_performance()