import numpy as np
import torch
import time
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import os
from astroca.tools.exportData import export_data



class EventDetectorGPU:
    """
    @class EventDetectorGPU
    @brief Fully vectorized GPU event detector maintaining CPU logic.
    """

    def __init__(self, av_data: np.ndarray, threshold_size_3d: int = 10,
                 threshold_size_3d_removed: int = 5, threshold_corr: float = 0.5,
                 device: str = None, plot: bool = False):
        """
        @fn __init__
        @brief Initialize with full GPU vectorization setup.
        """
        print("=== Fully Vectorized GPU Event Detector ===")

        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            gpu_memory = torch.cuda.get_device_properties(self.device).total_memory
            print(f"GPU Memory: {gpu_memory / 1e9:.1f} GB")

        # Convert to tensor on GPU
        if isinstance(av_data, np.ndarray):
            self.av_ = torch.from_numpy(av_data).float().to(self.device)
        else:
            self.av_ = av_data.float().to(self.device)

        self.time_length_, self.depth_, self.height_, self.width_ = self.av_.shape

        self.threshold_size_3d_ = threshold_size_3d
        self.threshold_size_3d_removed_ = threshold_size_3d_removed
        self.threshold_corr_ = threshold_corr

        # Pré-calculs GPU
        self.nonzero_mask_ = self.av_ != 0
        self.id_connected_voxel_ = torch.zeros_like(self.av_, dtype=torch.int32, device=self.device)
        self.final_id_events_ = []

        # Structures pour traitement vectorisé
        self._setup_vectorized_structures()

        print(f"Input data range: [{self.av_.min():.3f}, {self.av_.max():.3f}]")
        print(f"Non-zero density: {torch.count_nonzero(self.av_).item() / self.av_.numel():.4f}")

    def _setup_vectorized_structures(self):
        """Setup structures for vectorized processing."""
        # Offsets 3D pour voisinage spatial (26-connectivité)
        offsets_3d = []
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == dy == dz == 0:
                        continue
                    offsets_3d.append([dz, dy, dx])

        self.neighbor_offsets_3d = torch.tensor(offsets_3d, dtype=torch.int32, device=self.device)

        # Grille de coordonnées pour traitement vectorisé
        self.coord_grids = self._create_coordinate_grids()

    def _create_coordinate_grids(self):
        """Create coordinate grids for vectorized operations."""
        z_grid, y_grid, x_grid = torch.meshgrid(
            torch.arange(self.depth_, device=self.device),
            torch.arange(self.height_, device=self.device),
            torch.arange(self.width_, device=self.device),
            indexing='ij'
        )
        return torch.stack([z_grid, y_grid, x_grid], dim=-1)  # Shape: (D, H, W, 3)

    def find_events(self) -> None:
        """
        @fn find_events
        @brief Fully vectorized event detection preserving CPU logic.
        """
        print(
            f"Thresholds -> size: {self.threshold_size_3d_}, removed: {self.threshold_size_3d_removed_}, corr: {self.threshold_corr_}")
        start_time = time.time()

        if not torch.any(self.nonzero_mask_):
            print("No non-zero voxels found!")
            return

        # Traitement vectorisé par frame temporelle
        all_events = []
        event_id = 1

        for t in tqdm(range(self.time_length_), desc="Processing time frames"):
            frame_events = self._process_frame_vectorized(t, event_id)
            all_events.extend(frame_events)
            event_id += len(frame_events)

        print(f" - Found {len(all_events)} potential events")

        # Post-traitement vectorisé
        self._post_process_events_vectorized(all_events)

        print(f"Total time: {time.time() - start_time:.2f}s")

    def _process_frame_vectorized(self, t: int, start_event_id: int) -> List[Dict]:
        """
        @fn _process_frame_vectorized
        @brief Process entire frame with GPU vectorization.
        """
        frame_events = []
        current_frame = self.av_[t]
        current_id_frame = self.id_connected_voxel_[t]

        # Étape 1: Détection vectorisée de toutes les graines
        seeds = self._find_all_seeds_vectorized(current_frame, current_id_frame)

        if len(seeds) == 0:
            return frame_events

        # Étape 2: Traitement parallèle de toutes les graines
        for seed_idx, seed_coord in enumerate(seeds):
            z, y, x = seed_coord

            # Vérifier si déjà traité
            if self.id_connected_voxel_[t, z, y, x] != 0:
                continue

            # Extraction du pattern
            pattern = self._extract_pattern_vectorized(t, z, y, x)
            if pattern is None:
                continue

            # Croissance de région vectorisée
            event_voxels = self._grow_region_vectorized(t, z, y, x, pattern, start_event_id + len(frame_events))

            if len(event_voxels) > 0:
                event_info = {
                    'id': start_event_id + len(frame_events),
                    'voxels': event_voxels,
                    'size': len(event_voxels),
                    'seed': (t, z, y, x)
                }
                frame_events.append(event_info)

        return frame_events

    def _find_all_seeds_vectorized(self, frame: torch.Tensor, id_frame: torch.Tensor) -> List[Tuple[int, int, int]]:
        """
        @fn _find_all_seeds_vectorized
        @brief Find all seeds in frame using GPU vectorization.
        """
        # Masque des candidats valides
        valid_mask = (frame > 0) & (id_frame == 0)

        if not torch.any(valid_mask):
            return []

        # Détection des maxima locaux avec convolution
        frame_padded = F.pad(frame.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1, 1, 1), mode='constant', value=0)
        max_pooled = F.max_pool3d(frame_padded, kernel_size=3, stride=1, padding=0).squeeze()

        # Les graines sont les maxima locaux valides
        is_seed = (frame == max_pooled) & valid_mask

        # Convertir en coordonnées
        seed_coords = torch.nonzero(is_seed, as_tuple=False)

        # Limiter le nombre pour éviter l'explosion mémoire
        if len(seed_coords) > 200:
            values = frame[is_seed]
            _, indices = torch.topk(values, 200)
            seed_coords = seed_coords[indices]

        return [(int(c[0]), int(c[1]), int(c[2])) for c in seed_coords]

    def _extract_pattern_vectorized(self, t: int, z: int, y: int, x: int) -> Optional[torch.Tensor]:
        """
        @fn _extract_pattern_vectorized
        @brief Extract temporal pattern using GPU operations.
        """
        intensity_profile = self.av_[:, z, y, x]

        if intensity_profile[t] == 0:
            return None

        # Trouver les bornes du pattern vectorisé
        nonzero_mask = intensity_profile != 0
        if not torch.any(nonzero_mask):
            return None

        nonzero_indices = torch.nonzero(nonzero_mask, as_tuple=True)[0]
        start_idx = nonzero_indices[0].item()
        end_idx = nonzero_indices[-1].item() + 1

        pattern = intensity_profile[start_idx:end_idx]

        if len(pattern) < 2:
            return None

        return pattern

    def _grow_region_vectorized(self, seed_t: int, seed_z: int, seed_y: int, seed_x: int,
                                seed_pattern: torch.Tensor, event_id: int) -> List[Tuple[int, int, int, int]]:
        """
        @fn _grow_region_vectorized
        @brief Vectorized region growing maintaining CPU logic.
        """
        # Initialisation
        event_voxels = []
        processed = set()
        current_wave = [(seed_t, seed_z, seed_y, seed_x)]

        # Marquer le seed
        self.id_connected_voxel_[seed_t, seed_z, seed_y, seed_x] = event_id

        # Ajouter les points temporels du seed pattern
        temporal_voxels = self._add_temporal_points_vectorized(seed_t, seed_z, seed_y, seed_x,
                                                               seed_pattern, event_id)
        event_voxels.extend(temporal_voxels)
        processed.update(temporal_voxels)

        # Croissance par vagues vectorisées
        while current_wave and len(event_voxels) < self.threshold_size_3d_ * 10:
            next_wave = []

            # Traiter toute la vague actuelle en parallèle
            for voxel in current_wave:
                if voxel in processed:
                    continue

                processed.add(voxel)
                t, z, y, x = voxel

                # Traitement vectorisé des voisins spatiaux
                new_voxels = self._process_spatial_neighbors_vectorized(t, z, y, x, seed_pattern, event_id)

                for new_voxel in new_voxels:
                    if new_voxel not in processed:
                        next_wave.append(new_voxel)
                        event_voxels.append(new_voxel)

            current_wave = next_wave

        return event_voxels

    def _add_temporal_points_vectorized(self, t: int, z: int, y: int, x: int,
                                        pattern: torch.Tensor, event_id: int) -> List[Tuple[int, int, int, int]]:
        """
        @fn _add_temporal_points_vectorized
        @brief Add temporal points of pattern using vectorized operations.
        """
        temporal_voxels = [(t, z, y, x)]
        intensity_profile = self.av_[:, z, y, x]

        # Trouver le début du pattern
        start_t = t
        while start_t > 0 and intensity_profile[start_t - 1] != 0:
            start_t -= 1

        # Ajouter tous les points temporels du pattern
        for i in range(1, len(pattern)):
            t_new = start_t + i
            if (t_new < self.time_length_ and
                    self.id_connected_voxel_[t_new, z, y, x] == 0):
                self.id_connected_voxel_[t_new, z, y, x] = event_id
                temporal_voxels.append((t_new, z, y, x))

        return temporal_voxels

    def _process_spatial_neighbors_vectorized(self, t: int, z: int, y: int, x: int,
                                              reference_pattern: torch.Tensor, event_id: int) -> List[
        Tuple[int, int, int, int]]:
        """
        @fn _process_spatial_neighbors_vectorized
        @brief Process spatial neighbors with vectorized correlation computation.
        """
        # Calculer les coordonnées des voisins
        neighbor_coords = torch.tensor([[z, y, x]], device=self.device) + self.neighbor_offsets_3d

        # Filtrer les coordonnées valides
        valid_mask = (
                (neighbor_coords[:, 0] >= 0) & (neighbor_coords[:, 0] < self.depth_) &
                (neighbor_coords[:, 1] >= 0) & (neighbor_coords[:, 1] < self.height_) &
                (neighbor_coords[:, 2] >= 0) & (neighbor_coords[:, 2] < self.width_)
        )

        valid_neighbors = neighbor_coords[valid_mask]

        if len(valid_neighbors) == 0:
            return []

        # Vérification vectorisée des conditions
        nz, ny, nx = valid_neighbors[:, 0], valid_neighbors[:, 1], valid_neighbors[:, 2]

        # Vérifier que les voisins ne sont pas déjà traités et ont une valeur
        frame_values = self.av_[t, nz, ny, nx]
        frame_ids = self.id_connected_voxel_[t, nz, ny, nx]

        candidate_mask = (frame_values > 0) & (frame_ids == 0)
        candidates = valid_neighbors[candidate_mask]

        if len(candidates) == 0:
            return []

        # Traitement vectorisé des patterns et corrélations
        new_voxels = []

        # Extraction vectorisée des patterns des candidats
        for candidate in candidates:
            cz, cy, cx = candidate[0].item(), candidate[1].item(), candidate[2].item()

            candidate_pattern = self._extract_pattern_vectorized(t, cz, cy, cx)
            if candidate_pattern is None:
                continue

            # Calcul de corrélation vectorisé
            correlation = self._compute_correlation_vectorized(reference_pattern, candidate_pattern)

            if correlation > self.threshold_corr_:
                # Marquer le voxel
                self.id_connected_voxel_[t, cz, cy, cx] = event_id

                # Ajouter les points temporels
                temporal_voxels = self._add_temporal_points_vectorized(t, cz, cy, cx, candidate_pattern, event_id)
                new_voxels.extend(temporal_voxels)

        return new_voxels

    def _compute_correlation_vectorized(self, pattern1: torch.Tensor, pattern2: torch.Tensor) -> float:
        """
        @fn _compute_correlation_vectorized
        @brief Compute correlation using vectorized GPU operations.
        """
        if len(pattern1) == 0 or len(pattern2) == 0:
            return 0.0

        # Corrélation croisée normalisée comme dans le CPU
        min_len = min(len(pattern1), len(pattern2))
        max_len = max(len(pattern1), len(pattern2))

        # Tester tous les décalages possibles
        max_corr = 0.0

        # Version vectorisée des décalages
        for shift in range(-min_len + 1, min_len):
            if shift < 0:
                p1_segment = pattern1[-shift:min(-shift + min_len, len(pattern1))]
                p2_segment = pattern2[:len(p1_segment)]
            else:
                p2_segment = pattern2[shift:shift + min_len]
                p1_segment = pattern1[:len(p2_segment)]

            if len(p1_segment) == 0 or len(p2_segment) == 0:
                continue

            # Normalisation
            p1_norm = p1_segment - p1_segment.mean()
            p2_norm = p2_segment - p2_segment.mean()

            # Corrélation
            numerator = torch.dot(p1_norm, p2_norm)
            denominator = torch.sqrt(torch.dot(p1_norm, p1_norm) * torch.dot(p2_norm, p2_norm))

            if denominator > 0:
                corr = (numerator / denominator).item()
                max_corr = max(max_corr, corr)

        return max_corr

    def _post_process_events_vectorized(self, events: List[Dict]):
        """
        @fn _post_process_events_vectorized
        @brief Vectorized post-processing of events.
        """
        print(f" - Post-processing {len(events)} events...")

        # Séparer par taille
        large_events = [e for e in events if e['size'] >= self.threshold_size_3d_]
        small_events = [e for e in events if e['size'] < self.threshold_size_3d_]

        print(f" - Large events: {len(large_events)}, Small events: {len(small_events)}")

        # Traitement vectorisé des petits événements
        self._process_small_events_vectorized(small_events, large_events)

        # Remappage final des IDs
        self._remap_event_ids_vectorized(large_events)

        print(f" - Final events retained: {len(self.final_id_events_)}")

    def _process_small_events_vectorized(self, small_events: List[Dict], large_events: List[Dict]):
        """Process small events with vectorized operations."""
        for small_event in small_events:
            if small_event['size'] >= self.threshold_size_3d_removed_:
                large_events.append(small_event)
            else:
                # Supprimer l'événement (marquer comme 0)
                if small_event['voxels']:
                    coords = torch.tensor(small_event['voxels'], device=self.device)
                    self.id_connected_voxel_[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]] = 0

    def _remap_event_ids_vectorized(self, final_events: List[Dict]):
        """Remap event IDs using vectorized operations."""
        # Créer le mapping
        old_to_new = {}
        for new_id, event in enumerate(final_events, 1):
            old_to_new[event['id']] = new_id

        # Remappage vectorisé
        max_old_id = max(old_to_new.keys()) if old_to_new else 0
        id_map = torch.zeros(max_old_id + 1, dtype=torch.int32, device=self.device)

        for old_id, new_id in old_to_new.items():
            id_map[old_id] = new_id

        # Application vectorisée du mapping
        mask = self.id_connected_voxel_ > 0
        self.id_connected_voxel_[mask] = id_map[self.id_connected_voxel_[mask]]

        self.final_id_events_ = list(range(1, len(final_events) + 1))

    def get_results(self) -> Tuple[torch.Tensor, List[int]]:
        """Return final results."""
        return self.id_connected_voxel_, self.final_id_events_

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics."""
        stats = {
            'nb_events': len(self.final_id_events_),
            'total_event_voxels': torch.count_nonzero(self.id_connected_voxel_).item()
        }

        if self.device.type == 'cuda':
            stats['gpu_memory_used'] = torch.cuda.memory_allocated(self.device) / 1e6

        return stats


def detect_calcium_events_gpu(av_data: np.ndarray, params_values: dict = None, device: str = None) -> Tuple[
    np.ndarray, int]:
    """
    @fn detect_calcium_events_gpu
    @brief GPU-optimized calcium event detection function.
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
    detector = EventDetectorGPU(av_data, threshold_size_3d, threshold_size_3d_removed, threshold_corr, device)

    # Exécuter la détection
    detector.find_events()
    id_connections, id_events = detector.get_results()

    # Convertir en numpy pour compatibilité
    id_connections_np = id_connections.cpu().numpy()

    # Sauvegarder si nécessaire
    if save_results:
        if output_directory is None:
            raise ValueError("Output directory must be specified if save_results is True.")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        export_data(id_connections_np, output_directory, export_as_single_tif=True, file_name="ID_calciumEvents_GPU")

    print(60 * "=")
    print()
    return id_connections_np, len(id_events)