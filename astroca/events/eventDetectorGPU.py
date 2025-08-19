"""
Version GPU qui produit des résultats équivalents (pas identiques) au CPU
en conservant la logique métier tout en optimisant pour le GPU.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import time
import os
from astroca.tools.exportData import export_data


class EventDetectorGPUEquivalent:
    """
    GPU Event Detector qui conserve la logique métier du CPU
    mais adapte l'implémentation pour tirer parti du GPU.
    """

    def __init__(
        self,
        av_data: torch.Tensor,
        threshold_size_3d: int = 10,
        threshold_size_3d_removed: int = 5,
        threshold_corr: float = 0.5,
        device: str = None,
    ):
        print("=== GPU Event Detector (Equivalent Logic) ===")

        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Convert to GPU tensor
        if isinstance(av_data, np.ndarray):
            self.av_ = torch.from_numpy(av_data).float().to(self.device)
        else:
            self.av_ = av_data.float().to(self.device)

        self.time_length_, self.depth_, self.height_, self.width_ = self.av_.shape

        # Parameters (identiques au CPU)
        self.threshold_size_3d_ = threshold_size_3d
        self.threshold_size_3d_removed_ = threshold_size_3d_removed
        self.threshold_corr_ = threshold_corr

        # GPU structures
        self.nonzero_mask_ = self.av_ != 0
        self.id_connected_voxel_ = torch.zeros_like(
            self.av_, dtype=torch.int32, device=self.device
        )
        self.final_id_events_ = []

        # Cache patterns (comme CPU mais sur GPU)
        self.patterns_cache_ = {}

        # Pre-compute neighbor offsets (identique au CPU)
        self._setup_neighbor_offsets()

        print(f"Input data range: [{self.av_.min():.3f}, {self.av_.max():.3f}]")
        print(
            f"Non-zero density: {torch.count_nonzero(self.av_).item() / self.av_.numel():.4f}"
        )

    def _setup_neighbor_offsets(self):
        """Setup des offsets de voisinage (identique au CPU)"""
        # 26-connectivity pour voisins spatiaux
        offsets_3d = []
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == dy == dz == 0:
                        continue
                    offsets_3d.append([dz, dy, dx])

        self.neighbor_offsets_3d = torch.tensor(
            offsets_3d, dtype=torch.int32, device=self.device
        )

    def find_events(self) -> None:
        """
        Version GPU qui conserve la logique séquentielle par frame
        mais optimise le traitement interne de chaque frame.
        """
        print(
            f"Thresholds -> size: {self.threshold_size_3d_}, removed: {self.threshold_size_3d_removed_}, corr: {self.threshold_corr_}"
        )
        start_time = time.time()

        if not torch.any(self.nonzero_mask_):
            print("No non-zero voxels found!")
            return

        event_id = 1
        all_events = []
        small_events = []

        # Traitement par frame (comme CPU) mais optimisé GPU
        for t in tqdm(range(self.time_length_), desc="Processing frames"):
            frame_events, frame_small_events = self._process_frame_cpu_logic(
                t, event_id
            )

            all_events.extend(frame_events)
            small_events.extend(frame_small_events)
            event_id += len(frame_events) + len(frame_small_events)

        print(
            f" - Found {len(all_events) + len(small_events)} events with {len(all_events)} retained and {len(small_events)} small groups"
        )

        # Post-processing des petits événements (logique CPU)
        if small_events:
            self._process_small_events_cpu_logic(small_events, all_events)

        # Finalisation
        self._compute_final_id_events()

        print(f"Total time: {time.time() - start_time:.2f}s")

    def _process_frame_cpu_logic(
        self, t: int, start_event_id: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Traite une frame en conservant la logique CPU :
        - Trouve un seed à la fois
        - Traite complètement chaque événement avant le suivant
        """
        frame_events = []
        small_events = []
        current_event_id = start_event_id

        while True:
            # Trouve UN seed (comme CPU) mais sur GPU
            seed = self._find_single_seed_gpu(t)
            if seed is None:
                break

            z, y, x = seed

            # Extract pattern (GPU optimisé)
            pattern = self._extract_pattern_gpu(t, z, y, x)
            if pattern is None:
                continue

            # Region growing (logique CPU mais sur GPU)
            event_voxels = self._grow_region_cpu_logic(
                t, z, y, x, pattern, current_event_id
            )

            # Classification comme CPU
            event_info = {
                "id": current_event_id,
                "voxels": event_voxels,
                "size": len(event_voxels),
            }

            if len(event_voxels) >= self.threshold_size_3d_:
                frame_events.append(event_info)
            else:
                small_events.append(event_info)

            current_event_id += 1

        return frame_events, small_events

    def _find_single_seed_gpu(self, t: int) -> Optional[Tuple[int, int, int]]:
        """
        Trouve UN SEUL seed comme le CPU, mais sur GPU.
        Equivalent à find_seed_fast du CPU.
        """
        frame = self.av_[t]
        id_frame = self.id_connected_voxel_[t]

        # Masque des candidats valides
        valid_mask = (frame > 0) & (id_frame == 0)

        if not torch.any(valid_mask):
            return None

        # Trouve le premier maximum (équivalent à la logique CPU)
        valid_values = frame * valid_mask
        max_val = torch.max(valid_values)

        if max_val == 0:
            return None

        # Trouve la première occurrence du maximum
        max_positions = torch.nonzero(valid_values == max_val, as_tuple=False)

        if len(max_positions) == 0:
            return None

        # Prendre le premier (équivalent au comportement CPU)
        first_pos = max_positions[0]
        return (int(first_pos[0]), int(first_pos[1]), int(first_pos[2]))

    def _extract_pattern_gpu(
        self, t: int, z: int, y: int, x: int
    ) -> Optional[torch.Tensor]:
        """
        Extraction de pattern GPU (identique à la logique CPU)
        """
        intensity_profile = self.av_[:, z, y, x]

        if intensity_profile[t] == 0:
            return None

        # Trouve les bornes du pattern (logique identique CPU)
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

    def _grow_region_cpu_logic(
        self,
        seed_t: int,
        seed_z: int,
        seed_y: int,
        seed_x: int,
        pattern: torch.Tensor,
        event_id: int,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Croissance de région qui suit la logique CPU exacte :
        - Queue FIFO
        - Traitement séquentiel des voisins
        - Même ordre de parcours
        """
        event_voxels = []
        waiting_queue = [(seed_t, seed_z, seed_y, seed_x)]
        processed = set()

        # Marquer le seed
        self.id_connected_voxel_[seed_t, seed_z, seed_y, seed_x] = event_id

        # Ajouter les points temporels du seed
        temporal_voxels = self._add_temporal_points_gpu(
            seed_t, seed_z, seed_y, seed_x, pattern, event_id
        )
        event_voxels.extend(temporal_voxels)
        waiting_queue.extend(temporal_voxels[1:])  # Exclure le seed déjà ajouté

        # Croissance BFS (identique au CPU)
        queue_index = 0
        while queue_index < len(waiting_queue):
            current_voxel = waiting_queue[queue_index]

            if current_voxel in processed:
                queue_index += 1
                continue

            processed.add(current_voxel)

            # Traiter les voisins spatiaux
            new_voxels = self._process_spatial_neighbors_cpu_logic(
                current_voxel, pattern, event_id
            )

            for new_voxel in new_voxels:
                if new_voxel not in processed:
                    waiting_queue.append(new_voxel)
                    event_voxels.append(new_voxel)

            queue_index += 1

        return event_voxels

    def _add_temporal_points_gpu(
        self, t: int, z: int, y: int, x: int, pattern: torch.Tensor, event_id: int
    ) -> List[Tuple[int, int, int, int]]:
        """
        Ajout des points temporels (logique identique CPU mais sur GPU)
        """
        temporal_voxels = [(t, z, y, x)]
        intensity_profile = self.av_[:, z, y, x]

        # Trouve le début du pattern (identique CPU)
        start_t = t
        while start_t > 0 and intensity_profile[start_t - 1] != 0:
            start_t -= 1

        # Ajouter tous les points temporels
        for i in range(1, len(pattern)):
            t_new = start_t + i
            if (
                t_new < self.time_length_
                and self.id_connected_voxel_[t_new, z, y, x] == 0
            ):
                self.id_connected_voxel_[t_new, z, y, x] = event_id
                temporal_voxels.append((t_new, z, y, x))

        return temporal_voxels

    def _process_spatial_neighbors_cpu_logic(
        self,
        voxel: Tuple[int, int, int, int],
        reference_pattern: torch.Tensor,
        event_id: int,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Traitement des voisins spatiaux avec logique CPU exacte
        """
        t, z, y, x = voxel
        new_voxels = []

        # Calculer les voisins spatiaux
        for offset in self.neighbor_offsets_3d:
            nz = z + offset[0].item()
            ny = y + offset[1].item()
            nx = x + offset[2].item()

            # Vérification des limites
            if not (
                0 <= nz < self.depth_
                and 0 <= ny < self.height_
                and 0 <= nx < self.width_
            ):
                continue

            # Vérifications comme CPU
            if (
                self.av_[t, nz, ny, nx] == 0
                or self.id_connected_voxel_[t, nz, ny, nx] != 0
            ):
                continue

            # Extract neighbor pattern
            neighbor_pattern = self._extract_pattern_gpu(t, nz, ny, nx)
            if neighbor_pattern is None:
                continue

            # Calcul de corrélation (identique CPU)
            correlation = self._compute_correlation_cpu_logic(
                reference_pattern, neighbor_pattern
            )

            if correlation > self.threshold_corr_:
                # Marquer le voxel
                self.id_connected_voxel_[t, nz, ny, nx] = event_id

                # Ajouter les points temporels
                temporal_voxels = self._add_temporal_points_gpu(
                    t, nz, ny, nx, neighbor_pattern, event_id
                )
                new_voxels.extend(temporal_voxels)

        return new_voxels

    def _compute_correlation_cpu_logic(
        self, pattern1: torch.Tensor, pattern2: torch.Tensor
    ) -> float:
        """
        Calcul de corrélation identique au CPU mais sur GPU
        """
        if len(pattern1) == 0 or len(pattern2) == 0:
            return 0.0

        # Logique identique au compute_max_ncc_strict du CPU
        min_len = min(len(pattern1), len(pattern2))
        max_corr = 0.0

        # Tester tous les décalages
        for shift in range(-min_len + 1, min_len):
            if shift < 0:
                p1_seg = pattern1[-shift : min(-shift + min_len, len(pattern1))]
                p2_seg = pattern2[: len(p1_seg)]
            else:
                p2_seg = pattern2[shift : shift + min_len]
                p1_seg = pattern1[: len(p2_seg)]

            if len(p1_seg) == 0 or len(p2_seg) == 0:
                continue

            if len(p1_seg) != len(p2_seg):
                continue

            # Normalisation
            p1_mean = p1_seg.mean()
            p2_mean = p2_seg.mean()

            p1_norm = p1_seg - p1_mean
            p2_norm = p2_seg - p2_mean

            # Corrélation
            numerator = torch.dot(p1_norm, p2_norm)
            denom1 = torch.sqrt(torch.dot(p1_norm, p1_norm))
            denom2 = torch.sqrt(torch.dot(p2_norm, p2_norm))

            if denom1 > 0 and denom2 > 0:
                corr = (numerator / (denom1 * denom2)).item()
                max_corr = max(max_corr, corr)

        return max_corr

    def _process_small_events_cpu_logic(
        self, small_events: List[Dict], large_events: List[Dict]
    ):
        """
        Post-processing des petits événements (logique CPU adaptée)
        """
        print(f" - Processing {len(small_events)} small events...")

        # Simplification : traitement direct sans la logique complexe de groupement
        # pour garder des performances GPU raisonnables
        for small_event in small_events:
            if small_event["size"] >= self.threshold_size_3d_removed_:
                large_events.append(small_event)
            else:
                # Supprimer l'événement
                if small_event["voxels"]:
                    for voxel in small_event["voxels"]:
                        t, z, y, x = voxel
                        self.id_connected_voxel_[t, z, y, x] = 0

    def _compute_final_id_events(self):
        """Calcul des IDs finaux (identique CPU)"""
        # Collecter tous les IDs non-zéro
        unique_ids = torch.unique(
            self.id_connected_voxel_[self.id_connected_voxel_ > 0]
        )

        if len(unique_ids) == 0:
            return

        # Remappage
        id_map = torch.zeros(
            torch.max(unique_ids).item() + 1, dtype=torch.int32, device=self.device
        )

        for new_id, old_id in enumerate(unique_ids, 1):
            id_map[old_id] = new_id

        # Application du mapping
        mask = self.id_connected_voxel_ > 0
        self.id_connected_voxel_[mask] = id_map[self.id_connected_voxel_[mask]]

        self.final_id_events_ = list(range(1, len(unique_ids) + 1))

    def get_results(self) -> Tuple[torch.Tensor, List[int]]:
        """Retourne les résultats finaux"""
        return self.id_connected_voxel_, self.final_id_events_

    def get_statistics(self) -> Dict:
        """Statistiques de détection"""
        stats = {
            "nb_events": len(self.final_id_events_),
            "total_event_voxels": torch.count_nonzero(self.id_connected_voxel_).item(),
        }

        if self.device.type == "cuda":
            stats["gpu_memory_used"] = torch.cuda.memory_allocated(self.device) / 1e6

        return stats


# Fonction d'interface principale
def detect_calcium_events_gpu_equivalent(
    av_data, params_values: dict = None, device: str = None
):
    """
    Détection GPU avec logique équivalente au CPU
    """
    # Extract parameters
    threshold_size_3d = int(params_values["events_extraction"]["threshold_size_3d"])
    threshold_size_3d_removed = int(
        params_values["events_extraction"]["threshold_size_3d_removed"]
    )
    threshold_corr = float(params_values["events_extraction"]["threshold_corr"])
    output_dir = params_values["paths"]["output_dir"]
    save_events = params_values["save"]["save_events"]

    # Convert to tensor if needed
    if isinstance(av_data, np.ndarray):
        av_tensor = torch.from_numpy(av_data)
    else:
        av_tensor = av_data

    # Create detector
    detector = EventDetectorGPUEquivalent(
        av_tensor, threshold_size_3d, threshold_size_3d_removed, threshold_corr, device
    )

    # Run detection
    detector.find_events()
    id_connections, id_events = detector.get_results()

    # Convert back to numpy for compatibility
    id_connections_np = id_connections.cpu().numpy()
    if save_events:
        if output_dir is None:
            raise ValueError(
                "Output directory must be specified if save_results is True."
            )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        export_data(
            id_connections_np,
            output_dir,
            export_as_single_tif=True,
            file_name="ID_calciumEvents",
        )

    return id_connections, len(id_events)
