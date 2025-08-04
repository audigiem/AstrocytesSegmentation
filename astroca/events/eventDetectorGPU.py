"""
@file eventDetectorGPU.py
@brief GPU-optimized event detector for calcium events in 4D data using PyTorch.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
import time
import os
from astroca.tools.exportData import export_data
from tqdm import tqdm
from collections import defaultdict


class EventDetectorGPU:
    """
    @class EventDetectorGPU
    @brief GPU-optimized event detector for calcium events in 4D data using PyTorch.
    """

    def __init__(self, av_data: torch.Tensor, threshold_size_3d: int = 10,
                 threshold_size_3d_removed: int = 5, threshold_corr: float = 0.5,
                 device: str = None, plot: bool = False):
        """
        @fn __init__
        @brief Initialize the GPU EventDetector.
        @param av_data 4D numpy array of input data
        @param threshold_size_3d Minimum size for event retention
        @param threshold_size_3d_removed Minimum size for not removing small events
        @param threshold_corr Correlation threshold
        @param device GPU device ('cuda', 'cuda:0', etc.) or None for auto-detection
        @param plot Enable plotting (default: False)
        """
        print("=== GPU Event Detector for 4D Data ===")

        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        if self.device.type == 'cuda':
            print(f"GPU Memory: {torch.cuda.get_device_properties(self.device).total_memory / 1e9:.1f} GB")

        print(f"Input data range: [{av_data.min():.3f}, {av_data.max():.3f}]")
        print(f"Non-zero voxels density: {np.count_nonzero(av_data) / av_data.size:.5f}")

        # Convert to PyTorch tensor on GPU
        self.av_ = av_data
        self.time_length_, self.depth_, self.height_, self.width_ = av_data.shape

        self.threshold_size_3d_ = threshold_size_3d
        self.threshold_size_3d_removed_ = threshold_size_3d_removed
        self.threshold_corr_ = threshold_corr
        self.plot_ = plot

        # Masque pré-calculé pour les voxels non-zéro sur GPU
        self.nonzero_mask_ = self.av_ != 0

        # ID tensor sur GPU
        max_events_estimate = torch.count_nonzero(self.av_).item() // threshold_size_3d + 1000
        dtype = torch.int16 if max_events_estimate < 32000 else torch.int32
        self.id_connected_voxel_ = torch.zeros_like(self.av_, dtype=dtype, device=self.device)

        self.final_id_events_ = []

        # Cache optimisé
        self.pattern_cache_size_limit_ = 10000
        self.pattern_: Dict[Tuple[int, int, int, int], torch.Tensor] = {}

        # Offsets pré-calculés sur GPU
        self._neighbor_offsets = self._generate_neighbor_offsets()
        self._neighbor_offsets_4d = self._generate_neighbor_offsets_4d()

        # Statistiques
        self.stats_ = {
            "patterns_computed": 0,
            "correlations_computed": 0,
            "events_retained": 0,
            "events_merged": 0,
            "events_removed": 0,
            "gpu_memory_used": 0
        }

        self.small_av_group_set_ = set()

        # Structures optimisées pour le traitement en batch
        self._batch_size = 2000  # Plus grand batch pour GPU

    def _generate_neighbor_offsets(self) -> torch.Tensor:
        """Generate 26-neighbor offsets for 3D spatial connectivity on GPU."""
        offsets = []
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == dy == dz == 0:
                        continue
                    offsets.append([dz, dy, dx])
        return torch.tensor(offsets, dtype=torch.int32, device=self.device)

    def _generate_neighbor_offsets_4d(self) -> torch.Tensor:
        """Generate neighbor offsets for 4D connectivity on GPU."""
        offsets = []
        for dt in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == dy == dz == dt == 0:
                            continue
                        offsets.append([dt, dz, dy, dx])
        return torch.tensor(offsets, dtype=torch.int32, device=self.device)

    def find_events(self) -> None:
        """Main GPU-optimized event finding method."""
        print(
            f"Thresholds -> size: {self.threshold_size_3d_}, removed: {self.threshold_size_3d_removed_}, corr: {self.threshold_corr_}")
        start_time = time.time()

        # Vérification préliminaire
        if not torch.any(self.nonzero_mask_):
            print("No non-zero voxels found!")
            return

        event_id = 1
        waiting_for_processing = []
        small_AV_groups = []
        id_small_AV_groups = []

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
                else:
                    self.final_id_events_.append(event_id)
                    self.stats_["events_retained"] += 1

                event_id += 1

                # Gestion de la mémoire du cache
                if len(self.pattern_) > self.pattern_cache_size_limit_:
                    self._cleanup_pattern_cache()

        print(
            f" - Found {event_id - 1} events with {len(self.final_id_events_)} retained and {len(small_AV_groups)} small groups.")

        # Traitement des petits groupes
        if small_AV_groups:
            self._process_small_groups(small_AV_groups, id_small_AV_groups)

        self._compute_final_id_events()

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
        """Clean up pattern cache to prevent memory overflow."""
        items = list(self.pattern_.items())
        keep_count = len(items) // 2
        self.pattern_ = dict(items[-keep_count:])

    def _grow_region_optimized(self, waiting_for_processing: List, event_id: int) -> None:
        """GPU-optimized region growing with batch processing."""
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

    def _find_connected_AV_optimized(self, seed: List[int], pattern: torch.Tensor,
                                     event_id: int, waiting_list: List[List[int]]) -> None:
        """GPU-optimized search for connected neighbors."""
        t, z, y, x = seed

        # Extraire les frames sur GPU
        av_frame = self.av_[t]
        id_frame = self.id_connected_voxel_[t]

        # Obtenir les coordonnées des voisins valides
        valid_coords = self._get_valid_neighbors_gpu(z, y, x)

        if len(valid_coords) == 0:
            return

        # GPU batch check des conditions
        valid_mask = self._batch_check_conditions_gpu(av_frame, id_frame, valid_coords)
        valid_neighbors = valid_coords[valid_mask]

        if len(valid_neighbors) == 0:
            return

        # Traitement GPU des patterns et corrélations
        self._process_neighbors_gpu(valid_neighbors, pattern, event_id, waiting_list, t)

    def _get_valid_neighbors_gpu(self, z: int, y: int, x: int) -> torch.Tensor:
        """Get valid neighbor coordinates using GPU operations."""
        # Créer les coordonnées des voisins
        coords = torch.tensor([z, y, x], device=self.device).unsqueeze(0) + self._neighbor_offsets

        # Filtrer les coordonnées valides
        valid_mask = (
                (coords[:, 0] >= 0) & (coords[:, 0] < self.depth_) &
                (coords[:, 1] >= 0) & (coords[:, 1] < self.height_) &
                (coords[:, 2] >= 0) & (coords[:, 2] < self.width_)
        )

        return coords[valid_mask]

    def _batch_check_conditions_gpu(self, av_frame: torch.Tensor, id_frame: torch.Tensor,
                                    coords: torch.Tensor) -> torch.Tensor:
        """GPU batch check of conditions for multiple coordinates."""
        if len(coords) == 0:
            return torch.tensor([], dtype=torch.bool, device=self.device)

        # Extraction vectorisée des valeurs
        z_coords, y_coords, x_coords = coords[:, 0], coords[:, 1], coords[:, 2]
        av_values = av_frame[z_coords, y_coords, x_coords]
        id_values = id_frame[z_coords, y_coords, x_coords]

        return (av_values != 0) & (id_values == 0)

    def _process_neighbors_gpu(self, valid_neighbors: torch.Tensor, pattern: torch.Tensor,
                               event_id: int, waiting_list: List[List[int]], t: int) -> None:
        """Process neighbors using GPU batch operations."""
        if len(valid_neighbors) == 0:
            return

        # Extraire tous les profils d'intensité en une seule opération
        z_coords, y_coords, x_coords = valid_neighbors[:, 0], valid_neighbors[:, 1], valid_neighbors[:, 2]
        intensity_profiles = self.av_[:, z_coords, y_coords, x_coords]  # [T, N]

        # Traitement batch des patterns et corrélations
        for i in range(len(valid_neighbors)):
            nz, ny, nx = valid_neighbors[i].cpu().numpy()
            intensity_profile = intensity_profiles[:, i]

            neighbor_pattern = self._detect_pattern_optimized(intensity_profile, t)
            if neighbor_pattern is None:
                continue

            # Calcul de corrélation GPU
            max_corr = self._compute_max_ncc_gpu(pattern, neighbor_pattern)
            self.stats_["correlations_computed"] += 1

            if max_corr > self.threshold_corr_:
                self.id_connected_voxel_[t, nz, ny, nx] = event_id
                pattern_key = (t, nz, ny, nx)
                self.pattern_[pattern_key] = neighbor_pattern
                waiting_list.append([t, nz, ny, nx])

                # Ajout des points temporels
                start_t = t
                while start_t > 0 and intensity_profile[start_t - 1] != 0:
                    start_t -= 1

                for p in range(len(neighbor_pattern)):
                    tp = start_t + p
                    if (tp < self.time_length_ and
                            self.id_connected_voxel_[tp, nz, ny, nx] == 0):
                        self.id_connected_voxel_[tp, nz, ny, nx] = event_id
                        pattern_key = (tp, nz, ny, nx)
                        self.pattern_[pattern_key] = neighbor_pattern
                        waiting_list.append([tp, nz, ny, nx])

    def _compute_max_ncc_gpu(self, pattern1: torch.Tensor, pattern2: torch.Tensor) -> float:
        """GPU-optimized computation of maximum normalized cross-correlation."""
        if len(pattern1) == 0 or len(pattern2) == 0:
            return 0.0

        # Calcul des normes
        norm1 = torch.sqrt(torch.dot(pattern1, pattern1))
        norm2 = torch.sqrt(torch.dot(pattern2, pattern2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Corrélation croisée complète
        correlation = F.conv1d(
            pattern2.unsqueeze(0).unsqueeze(0),
            pattern1.flip(0).unsqueeze(0).unsqueeze(0),
            padding=len(pattern1) - 1
        ).squeeze()

        # Normalisation et maximum
        normalized_corr = correlation / (norm1 * norm2)
        return torch.max(normalized_corr).item()

    def _find_seed_point_fast(self, t0: int) -> Optional[Tuple[int, int, int]]:
        """GPU-optimized seed search."""
        av_frame = self.av_[t0]
        id_frame = self.id_connected_voxel_[t0]

        # Masque des candidats valides
        valid_mask = (av_frame != 0) & (id_frame == 0)

        if not torch.any(valid_mask):
            return None

        # Trouver le maximum parmi les candidats valides
        masked_values = torch.where(valid_mask, av_frame, torch.tensor(-1.0, device=self.device))
        flat_idx = torch.argmax(masked_values)

        # Convertir l'index plat en coordonnées 3D
        z = flat_idx // (self.height_ * self.width_)
        remainder = flat_idx % (self.height_ * self.width_)
        y = remainder // self.width_
        x = remainder % self.width_

        return (int(x), int(y), int(z))

    def _detect_pattern_optimized(self, intensity_profile: torch.Tensor, t: int) -> Optional[torch.Tensor]:
        """GPU-optimized pattern detection."""
        if intensity_profile[t] == 0:
            return None

        # Trouver les limites du pattern
        start, end = self._find_nonzero_pattern_bounds_gpu(intensity_profile, t)
        pattern = intensity_profile[start:end].clone()

        self.stats_["patterns_computed"] += 1
        return pattern

    def _find_nonzero_pattern_bounds_gpu(self, intensity_profile: torch.Tensor, t: int) -> Tuple[int, int]:
        """GPU-optimized pattern bounds detection."""
        start = t
        while start > 0 and intensity_profile[start - 1] != 0:
            start -= 1

        end = t
        while end < len(intensity_profile) and intensity_profile[end] != 0:
            end += 1

        return start, end

    def _process_small_groups(self, small_av_groups: List, id_small_av_groups: List) -> None:
        """Process small groups with GPU optimization where possible."""
        self.small_av_groups_set_ = set(id_small_av_groups)

        self._group_small_neighborhood_regions(small_av_groups, id_small_av_groups)

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
                    pbar.total -= 1
                else:
                    if len(group) >= self.threshold_size_3d_removed_:
                        self.final_id_events_.append(group_id)
                    else:
                        coords = torch.tensor(group, device=self.device)
                        self.id_connected_voxel_[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]] = 0
                        self.stats_["events_removed"] += 1

                    del small_av_groups[i]
                    del id_small_av_groups[i]
                    self.small_av_groups_set_.discard(group_id)
                    pbar.total -= 1

                pbar.update(1)

    def _group_small_neighborhood_regions(self, small_av_groups: List, list_ids_small_av_group: List) -> None:
        """GPU-optimized grouping of small neighborhood regions."""
        bounds = torch.tensor([self.time_length_, self.depth_, self.height_, self.width_], device=self.device)

        with tqdm(total=len(small_av_groups), desc="Grouping neighborhoods", unit="group") as pbar:
            id_ = 0
            while id_ < len(small_av_groups):
                list_av = small_av_groups[id_]
                group_id = list_ids_small_av_group[id_]

                neighbor_id_counts = defaultdict(int)

                # Conversion en tensor GPU pour traitement vectorisé
                coords_array = torch.tensor(list_av, device=self.device)

                # Calcul vectorisé des voisins
                coords_expanded = coords_array.unsqueeze(1) + self._neighbor_offsets_4d.unsqueeze(0)
                coords_flat = coords_expanded.view(-1, 4)

                # Vérification vectorisée des limites
                valid_mask = torch.all(
                    (coords_flat >= 0) & (coords_flat < bounds),
                    dim=1
                )

                if torch.any(valid_mask):
                    valid_coords = coords_flat[valid_mask]

                    # Accès vectorisé aux neighbor_ids
                    neighbor_ids = self.id_connected_voxel_[
                        valid_coords[:, 0],
                        valid_coords[:, 1],
                        valid_coords[:, 2],
                        valid_coords[:, 3]
                    ]

                    # Filtrage vectorisé
                    valid_neighbors_mask = (
                            (neighbor_ids != 0) &
                            (neighbor_ids != group_id)
                    )

                    # Filtrage supplémentaire pour small_av_groups_set_
                    if torch.any(valid_neighbors_mask):
                        valid_neighbors = neighbor_ids[valid_neighbors_mask]

                        # Conversion CPU pour vérification du set
                        valid_neighbors_cpu = valid_neighbors.cpu().numpy()
                        final_mask = np.array([nid in self.small_av_groups_set_ for nid in valid_neighbors_cpu])

                        if np.any(final_mask):
                            final_neighbors = valid_neighbors_cpu[final_mask]
                            unique_ids, counts = np.unique(final_neighbors, return_counts=True)

                            for nid, count in zip(unique_ids, counts):
                                neighbor_id_counts[nid] += count

                if neighbor_id_counts:
                    max_count = max(neighbor_id_counts.values())
                    candidates = [(nid, count) for nid, count in neighbor_id_counts.items()
                                  if count == max_count]
                    new_id = min(candidates)[0]

                    new_id_index = list_ids_small_av_group.index(new_id)
                    max_neighbor_count = neighbor_id_counts[new_id]

                    if max_neighbor_count >= len(list_av):
                        self.id_connected_voxel_[coords_array[:, 0], coords_array[:, 1],
                        coords_array[:, 2], coords_array[:, 3]] = new_id
                        small_av_groups[new_id_index].extend(list_av)

                        self.small_av_groups_set_.discard(group_id)
                        del small_av_groups[id_]
                        del list_ids_small_av_group[id_]
                        pbar.total -= 1
                        pbar.update(1)
                        continue

                id_ += 1
                pbar.update(1)

    def _change_id_small_regions(self, list_av: List, list_ids_small_av_group: List) -> bool:
        """GPU-optimized ID change for small regions."""
        small_av_set = set(list_ids_small_av_group)
        bounds = torch.tensor([self.time_length_, self.depth_, self.height_, self.width_], device=self.device)

        # Conversion en tensor GPU
        coords_array = torch.tensor(list_av, device=self.device)

        # Calcul vectorisé des voisins
        coords_expanded = coords_array.unsqueeze(1) + self._neighbor_offsets_4d.unsqueeze(0)
        coords_flat = coords_expanded.view(-1, 4)

        # Vérification vectorisée des limites
        valid_mask = torch.all(
            (coords_flat >= 0) & (coords_flat < bounds),
            dim=1
        )

        if not torch.any(valid_mask):
            return False

        valid_coords = coords_flat[valid_mask]

        # Accès vectorisé aux neighbor_ids
        neighbor_ids = self.id_connected_voxel_[
            valid_coords[:, 0],
            valid_coords[:, 1],
            valid_coords[:, 2],
            valid_coords[:, 3]
        ]

        # Conversion CPU pour vérification du set
        neighbor_ids_cpu = neighbor_ids.cpu().numpy()
        valid_neighbors_mask = (
                (neighbor_ids_cpu != 0) &
                np.array([nid not in small_av_set for nid in neighbor_ids_cpu])
        )

        if not np.any(valid_neighbors_mask):
            return False

        valid_neighbors = neighbor_ids_cpu[valid_neighbors_mask]
        unique_ids, counts = np.unique(valid_neighbors, return_counts=True)

        # Calcul du maximum
        max_count = np.max(counts)
        max_indices = counts == max_count
        candidates = unique_ids[max_indices]
        new_id = np.min(candidates)

        # Assignation GPU
        self.id_connected_voxel_[coords_array[:, 0], coords_array[:, 1],
        coords_array[:, 2], coords_array[:, 3]] = new_id
        return True

    def _compute_final_id_events(self) -> None:
        """GPU-optimized computation of final IDs."""
        if not self.final_id_events_:
            return

        max_id = self.id_connected_voxel_.max().item()
        if max_id == 0:
            return

        # Remappage vectorisé sur GPU
        id_map = torch.zeros(max_id + 1, dtype=self.id_connected_voxel_.dtype, device=self.device)

        final_ids = sorted(self.final_id_events_)
        for new_id, old_id in enumerate(final_ids, start=1):
            id_map[old_id] = new_id

        self.id_connected_voxel_ = id_map[self.id_connected_voxel_]

    def get_results(self) -> Tuple[torch.Tensor, List[int]]:
        """Return the final results, converting back to CPU numpy arrays."""
        return self.id_connected_voxel_, self.final_id_events_

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about detected events."""
        stats = {
            'nb_events': len(self.final_id_events_),
            'event_sizes': [],
            'total_event_voxels': 0,
            'gpu_memory_used': 0
        }

        if self.device.type == 'cuda':
            stats['gpu_memory_used'] = torch.cuda.memory_allocated(self.device) / 1e6  # MB

        stats.update(self.stats_)
        return stats


def detect_calcium_events_gpu(av_data: torch.Tensor, params_values: dict = None, device: str = None) -> Tuple[
    torch.Tensor, int]:
    """
    GPU-optimized function to detect calcium events in 4D data using PyTorch.

    @param av_data 4D numpy array of input data
    @param params_values Dictionary containing detection parameters
    @param device GPU device specification
    @return Tuple of (event_connections_array, number_of_events)
    """
    required_keys = {'events_extraction', 'save', 'paths'}
    if not required_keys.issubset(params_values.keys()):
        raise ValueError(f"Missing required parameters: {required_keys - params_values.keys()}")

    threshold_size_3d = int(params_values['events_extraction']['threshold_size_3d'])
    threshold_size_3d_removed = int(params_values['events_extraction']['threshold_size_3d_removed'])
    threshold_corr = float(params_values['events_extraction']['threshold_corr'])
    save_results = int(params_values['save']['save_events']) == 1
    output_directory = params_values['paths']['output_dir']

    # Créer le détecteur GPU
    detector = EventDetectorGPU(av_data, threshold_size_3d,
                                threshold_size_3d_removed, threshold_corr, device)

    # Exécuter la détection
    detector.find_events()
    id_connections, id_events = detector.get_results()

    # Sauvegarder si nécessaire
    if save_results:
        if output_directory is None:
            raise ValueError("Output directory must be specified if save_results is True.")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        id_connections_export = id_connections.cpu().numpy()  # Convertir en numpy pour l'export
        export_data(id_connections_export, output_directory, export_as_single_tif=True, file_name="ID_calciumEvents_GPU")

    print(60 * "=")
    print()
    return id_connections, len(id_events)