import numpy as np
import numba as nb
from numba import njit, prange
from typing import List, Tuple, Optional, Any, Dict
import time
from scipy import ndimage
from skimage.measure import label
import matplotlib.pyplot as plt
import os
from astroca.tools.exportData import export_data
from tqdm import tqdm
from collections import deque

# Fonctions numba pour l'optimisation
@njit
def _find_nonzero_pattern_bounds(intensity_profile: np.ndarray, t: int) -> Tuple[int, int]:
    """Find start and end indices of non-zero pattern around time t."""
    start = t
    while start > 0 and intensity_profile[start - 1] != 0:
        start -= 1
    
    end = t
    while end < len(intensity_profile) and intensity_profile[end] != 0:
        end += 1
    
    return start, end

@njit
def _compute_ncc_fast(pattern1: np.ndarray, pattern2: np.ndarray) -> np.ndarray:
    """Optimized normalized cross-correlation computation."""
    # Cross-correlation with zero boundary conditions
    vout = np.correlate(pattern1, pattern2, 'full')
    
    # Auto-correlation for normalization
    auto_corr_v1 = np.dot(pattern1, pattern1)
    auto_corr_v2 = np.dot(pattern2, pattern2)
    
    # Normalization
    den = np.sqrt(auto_corr_v1 * auto_corr_v2)
    if den == 0:
        return np.zeros_like(vout)
    
    return vout / den

@njit
def _find_seed_fast(frame_data: np.ndarray, id_mask: np.ndarray) -> Tuple[int, int, int, float]:
    """Fast seed finding using numba."""
    max_val = 0.0
    best_x, best_y, best_z = -1, -1, -1
    
    depth, height, width = frame_data.shape
    
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                val = frame_data[z, y, x]
                if val > max_val and id_mask[z, y, x] == 0:
                    max_val = val
                    best_x, best_y, best_z = x, y, z
    
    return best_x, best_y, best_z, max_val

class EventDetectorOptimized:
    """
    Event Detector for calcium events in 4D data (time, depth, height, width).
    Optimized version with improved performance while preserving exact behavior.
    """

    def __init__(self, av_data: np.ndarray, threshold_size_3d: int = 10,
                 threshold_size_3d_removed: int = 5, threshold_corr: float = 0.5,
                 plot: bool = False):
        """
        Initialize the EventDetector with the provided active voxel data and thresholds.
        """
        print("=== Finding events in 4D data ===")
        print(f"Input data range: [{av_data.min():.3f}, {av_data.max():.3f}]")
        print(f"Non-zero voxels: {np.count_nonzero(av_data)}/{av_data.size}")

        self.av_ = av_data.astype(np.float32)
        self.time_length_, self.depth_, self.height_, self.width_ = av_data.shape

        self.threshold_size_3d_ = threshold_size_3d
        self.threshold_size_3d_removed_ = threshold_size_3d_removed
        self.threshold_corr_ = threshold_corr
        self.plot_ = plot

        # Pre-compute non-zero mask for faster access
        self.nonzero_mask_ = self.av_ != 0
        self.nonzero_cords_ = np.where(self.nonzero_mask_)

        self.id_connected_voxel_ = np.zeros_like(self.av_, dtype=np.int32)
        self.final_id_events_ = []

        # Cache optimization: use tuple keys for better performance
        self.pattern_cache_: Dict[Tuple[int, int, int, int], np.ndarray] = {}
        self.pattern_: Dict[Tuple[int, int, int, int], np.ndarray] = {}

        # Pre-allocated arrays for neighbor search
        self._neighbor_offsets = self._generate_neighbor_offsets()
        self._neighbor_offsets_4d = self._generate_neighbor_offsets_4d()

        self.stats_ = {
            "patterns_computed": 0,
            "regions_grown": 0,
            "correlations_computed": 0,
            "events_retained": 0,
            "events_merged": 0,
            "events_removed": 0,
        }

    def _generate_neighbor_offsets(self) -> np.ndarray:
        """Pre-generate 26-neighbor offsets for 3D spatial connectivity."""
        offsets = []
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == dy == dz == 0:
                        continue
                    offsets.append([dz, dy, dx])
        return np.array(offsets, dtype=np.int32)

    def _generate_neighbor_offsets_4d(self) -> np.ndarray:
        """Pre-generate neighbor offsets for 4D connectivity (space + time)."""
        offsets = []
        for dt in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == dy == dz == dt == 0:
                            continue
                        offsets.append([dt, dz, dy, dx])
        return np.array(offsets, dtype=np.int32)

    def find_events(self) -> None:
        """
        Main method to find events in the active voxel data.
        Optimized version with pre-allocated structures and vectorized operations.
        """
        print(f"Thresholds -> size: {self.threshold_size_3d_}, removed: {self.threshold_size_3d_removed_}, corr: {self.threshold_corr_}")
        start_time = time.time()
        
        if len(self.nonzero_cords_[0]) == 0:
            print("No non-zero voxels found!")
            return

        event_id = 1
        # Use deque for better performance on append/pop operations
        waiting_for_processing = deque()
        small_AV_groups = []
        id_small_AV_groups = []

        for t in tqdm(range(self.time_length_), desc="Processing time frames", unit="frame"):
        # for t in range(self.time_length_):
            # print(f"\nProcessing time frame {t}, searching seed ...")
            frame_time = time.time()
            
            seed = self._find_seed_point_fast(t)
            while seed is not None:
                x, y, z = seed
                # print(f"Found seed at ({x}, {y}, {z}) in frame {t}, identifier {event_id}")

                intensity_profile = self.av_[:, z, y, x]
                pattern = self._detect_pattern_optimized(intensity_profile, t)
                if pattern is None:
                    # print(f"No valid pattern found for seed ({x}, {y}, {z})")
                    break

                # Add the seed to the waiting list
                pattern_key = (t, z, y, x)
                self.pattern_[pattern_key] = pattern
                self.id_connected_voxel_[t, z, y, x] = event_id
                waiting_for_processing.append([t, z, y, x])

                # Add all points of the current pattern
                for i in range(1, len(pattern)):
                    t0 = t + i
                    if t0 < self.time_length_ and self.id_connected_voxel_[t0, z, y, x] == 0:
                        self.id_connected_voxel_[t0, z, y, x] = event_id
                        pattern_key = (t0, z, y, x)
                        self.pattern_[pattern_key] = pattern
                        waiting_for_processing.append([t0, z, y, x])

                # Process neighbors using optimized region growing
                self._grow_region_optimized(waiting_for_processing, event_id)
                
                group_size = len(waiting_for_processing)
                # print(f"    Size of group ID={event_id} : {group_size}")

                # Check if the group is below threshold
                if group_size < self.threshold_size_3d_:
                    # print(f"    Group ID={event_id} is too small ({group_size} voxels), adding to small groups")
                    small_AV_groups.append(list(waiting_for_processing))
                    id_small_AV_groups.append(event_id)
                else:
                    self.final_id_events_.append(event_id)
                    self.stats_["events_retained"] += 1

                # Update for next iteration
                event_id += 1
                waiting_for_processing.clear()
                seed = self._find_seed_point_fast(t)
            
            # print(f"Time for frame {t}: {time.time() - frame_time:.2f} seconds")

        # Process small groups
        if small_AV_groups:
            # print(f"\nFound {len(small_AV_groups)} small groups, trying to merge them with larger groups...")
            self._process_small_groups(small_AV_groups, id_small_AV_groups)

        print(f"\nTotal events found: {len(self.final_id_events_)}")
        print(f"Size of each final event:")
        for event_id in self.final_id_events_:
            size = np.sum(self.id_connected_voxel_ == event_id)
            print(f"    Event ID={event_id}: {size} voxels")

        self._compute_final_id_events()
                 
        print(f"Total time taken: {time.time() - start_time:.2f} seconds")
        print()

    def _grow_region_optimized(self, waiting_for_processing: deque, event_id: int) -> None:
        """
        Optimized region growing using pre-computed neighbor offsets.
        """
        index_waiting = 0
        waiting_list = list(waiting_for_processing)
        
        while index_waiting < len(waiting_list):
            seed = waiting_list[index_waiting]
            pattern_key = tuple(seed)
            pattern = self.pattern_[pattern_key]
            if pattern is None:
                raise ValueError(f"Invalid pattern for voxel {seed}")
            
            self._find_connected_AV_optimized(seed, pattern, event_id, waiting_list)
            index_waiting += 1
        
        waiting_for_processing.clear()
        waiting_for_processing.extend(waiting_list)

    def _find_connected_AV_optimized(self, seed: List[int], pattern: np.ndarray, 
                                   event_id: int, waiting_list: List[List[int]]) -> None:
        """
        Optimized neighbor search using pre-computed offsets.
        """
        t, z, y, x = seed
        
        # Use vectorized neighbor checking
        for offset in self._neighbor_offsets:
            dz, dy, dx = offset
            nz, ny, nx = z + dz, y + dy, x + dx
            
            # Bounds checking
            if (nz < 0 or nz >= self.depth_ or 
                ny < 0 or ny >= self.height_ or 
                nx < 0 or nx >= self.width_):
                continue

            if (self.av_[t, nz, ny, nx] != 0 and 
                self.id_connected_voxel_[t, nz, ny, nx] == 0):
                
                # Extract intensity profile and detect pattern
                intensity_profile = self.av_[:, nz, ny, nx]
                neighbor_pattern = self._detect_pattern_optimized(intensity_profile, t)
                if neighbor_pattern is None:
                    continue

                # Compute correlation using optimized function
                correlation = _compute_ncc_fast(pattern, neighbor_pattern)
                max_corr = np.max(correlation)
                
                if max_corr > self.threshold_corr_:
                    self.id_connected_voxel_[t, nz, ny, nx] = event_id
                    pattern_key = (t, nz, ny, nx)
                    self.pattern_[pattern_key] = neighbor_pattern
                    waiting_list.append([t, nz, ny, nx])

                    # Find pattern start time efficiently
                    start_t = t
                    while start_t > 0 and intensity_profile[start_t - 1] != 0:
                        start_t -= 1

                    # Add temporal pattern points
                    for p in range(len(neighbor_pattern)):
                        tp = start_t + p
                        if (tp < self.time_length_ and 
                            self.id_connected_voxel_[tp, nz, ny, nx] == 0):
                            self.id_connected_voxel_[tp, nz, ny, nx] = event_id
                            pattern_key = (tp, nz, ny, nx)
                            self.pattern_[pattern_key] = neighbor_pattern
                            waiting_list.append([tp, nz, ny, nx])

    def _find_seed_point_fast(self, t0: int) -> Optional[Tuple[int, int, int]]:
        """
        Optimized seed point finding using numba.
        """
        x, y, z, max_val = _find_seed_fast(
            self.av_[t0], 
            self.id_connected_voxel_[t0]
        )
        
        if x == -1:
            return None
        return (x, y, z)

    def _detect_pattern_optimized(self, intensity_profile: np.ndarray, t: int) -> Optional[np.ndarray]:
        """
        Optimized pattern detection using numba for bounds finding.
        """
        if intensity_profile[t] == 0:
            return None
        
        # Use numba-optimized bounds finding
        start, end = _find_nonzero_pattern_bounds(intensity_profile, t)
        pattern = intensity_profile[start:end].copy()  # Copy to avoid reference issues
        
        if self.plot_:
            print(f"Detected pattern from {start} to {end} for time frame {t}, length: {len(pattern)}")
            plt.figure(figsize=(10, 4))
            plt.plot(np.arange(start, end), pattern, label=f"Pattern at t={t}")
            plt.title(f"Detected Pattern at t={t}")
            plt.show()
        
        self.stats_["patterns_computed"] += 1
        return pattern

    def _compute_normalized_cross_correlation(self, pattern1: np.ndarray, pattern2: np.ndarray) -> np.ndarray:
        """
        Wrapper for the optimized correlation computation.
        """
        correlation = _compute_ncc_fast(pattern1, pattern2)
        self.stats_["correlations_computed"] += 1

        if self.plot_:
            zero_position_vout = len(pattern2) - 1
            plt.plot(np.arange(-zero_position_vout, len(pattern1)), correlation)
            plt.title("Normalized cross correlation")
            plt.xlabel("time")
            plt.ylabel("value")
            plt.show()

        return correlation

    def _process_small_groups(self, small_av_groups: List, id_small_av_groups: List) -> None:
        """
        Optimized processing of small groups with better indexing.
        """
        self._group_small_neighborhood_regions(small_av_groups, id_small_av_groups)
        
        # Process groups in reverse order to handle removals safely
        for i in range(len(small_av_groups) - 1, -1, -1):
            group = small_av_groups[i]
            group_id = id_small_av_groups[i]
            
            change_id = self._change_id_small_regions(group, id_small_av_groups)
            if change_id:
                # print((f"Group {group_id} merged into larger group")
                self.stats_["events_merged"] += 1
            else:
                if len(group) >= self.threshold_size_3d_removed_:
                    # print((f"Group {group_id} is large enough ({len(group)} voxels), adding to final events")
                    self.final_id_events_.append(group_id)
                else:
                    # print((f"Group {group_id} is isolated and too small ({len(group)} voxels), removing it")
                    # Vectorized removal
                    for t, z, y, x in group:
                        self.id_connected_voxel_[t, z, y, x] = 0
                    self.stats_["events_removed"] += 1
            
            # Remove processed group
            del small_av_groups[i]
            del id_small_av_groups[i]

    def _group_small_neighborhood_regions(self, small_av_groups: List, list_ids_small_av_group: List) -> None:
        """
        Optimized grouping using vectorized operations where possible.
        """
        id_ = 0
        while id_ < len(small_av_groups):
            list_av = small_av_groups[id_]
            group_id = list_ids_small_av_group[id_]

            neighbor_id_counts = {}
            
            # Vectorized neighbor checking using pre-computed offsets
            for t, z, y, x in list_av:
                for offset in self._neighbor_offsets_4d:
                    dt, dz, dy, dx = offset
                    nt, nz, ny, nx = t + dt, z + dz, y + dy, x + dx
                    
                    if (nt < 0 or nt >= self.time_length_ or
                        nz < 0 or nz >= self.depth_ or
                        ny < 0 or ny >= self.height_ or
                        nx < 0 or nx >= self.width_):
                        continue
                        
                    neighbor_id = self.id_connected_voxel_[nt, nz, ny, nx]
                    if (neighbor_id != 0 and
                        neighbor_id in list_ids_small_av_group and
                        neighbor_id != group_id):
                        neighbor_id_counts[neighbor_id] = neighbor_id_counts.get(neighbor_id, 0) + 1

            if neighbor_id_counts:
                new_id = max(neighbor_id_counts, key=neighbor_id_counts.get)
                new_id_index = list_ids_small_av_group.index(new_id)
                
                if len(small_av_groups[new_id_index]) >= len(list_av):
                    # Vectorized ID assignment
                    for t, z, y, x in list_av:
                        self.id_connected_voxel_[t, z, y, x] = new_id
                    small_av_groups[new_id_index].extend(list_av)
                    
                    del small_av_groups[id_]
                    del list_ids_small_av_group[id_]
                    continue
            id_ += 1

    def _change_id_small_regions(self, list_av: List, list_ids_small_av_group: List) -> bool:
        """
        Optimized ID changing with better neighbor counting.
        """
        neighbor_counts = {}
        
        for t, z, y, x in list_av:
            for offset in self._neighbor_offsets_4d:
                dt, dz, dy, dx = offset[0], offset[1], offset[2], offset[3]
                nt, nz, ny, nx = t + dt, z + dz, y + dy, x + dx
                
                if (nt < 0 or nt >= self.time_length_ or
                    nz < 0 or nz >= self.depth_ or
                    ny < 0 or ny >= self.height_ or
                    nx < 0 or nx >= self.width_):
                    continue
                    
                neighbor_id = self.id_connected_voxel_[nt, nz, ny, nx]
                if neighbor_id != 0 and neighbor_id not in list_ids_small_av_group:
                    neighbor_counts[neighbor_id] = neighbor_counts.get(neighbor_id, 0) + 1

        if neighbor_counts:
            new_id = max(neighbor_counts, key=neighbor_counts.get)
            # Vectorized assignment
            for t, z, y, x in list_av:
                self.id_connected_voxel_[t, z, y, x] = new_id
            return True
        
        return False

    def _compute_final_id_events(self) -> None:
        """
        Optimized ID remapping using vectorized operations.
        """
        if not self.final_id_events_:
            return

        max_id = self.id_connected_voxel_.max()
        if max_id == 0:
            return
            
        id_map = np.zeros(max_id + 1, dtype=self.id_connected_voxel_.dtype)
        
        final_ids = sorted(self.final_id_events_)
        for new_id, old_id in enumerate(final_ids, start=1):
            id_map[old_id] = new_id

        self.id_connected_voxel_ = id_map[self.id_connected_voxel_]

    def get_results(self) -> Tuple[np.ndarray, List[int]]:
        """Return the final results."""
        return self.id_connected_voxel_, self.final_id_events_

    def get_statistics(self) -> Dict:
        """
        Get comprehensive statistics about detected events.
        """
        stats = {
            'nb_events': len(self.final_id_events_),
            'event_sizes': [],
            'total_event_voxels': 0
        }

        # Vectorized size computation
        unique_ids, counts = np.unique(self.id_connected_voxel_[self.id_connected_voxel_ > 0], 
                                      return_counts=True)
        
        for event_id in self.final_id_events_:
            if event_id in unique_ids:
                size = counts[unique_ids == event_id][0]
                stats['event_sizes'].append(size)
                stats['total_event_voxels'] += size

        if stats['event_sizes']:
            event_sizes = np.array(stats['event_sizes'])
            stats['mean_event_size'] = np.mean(event_sizes)
            stats['median_event_size'] = np.median(event_sizes)
            stats['max_event_size'] = np.max(event_sizes)
            stats['min_event_size'] = np.min(event_sizes)

        # Add processing stats
        stats.update(self.stats_)
        
        return stats


def detect_calcium_events_opti(av_data: np.ndarray, params_values: dict = None,
                         save_results: bool = False,
                         output_directory: str = None) -> Tuple[np.ndarray, List[int]]:
    """
    Optimized function to detect calcium events in 4D active voxel data.
    """
    threshold_size_3d = int(params_values['events_extraction']['threshold_size_3d'])
    threshold_size_3d_removed = int(params_values['events_extraction']['threshold_size_3d_removed'])
    threshold_corr = float(params_values['events_extraction']['threshold_corr'])
    save_results = int(params_values['files']['save_results']) == 1
    output_directory = params_values['paths']['output_dir'] if output_directory is None else output_directory

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
    
    print(60*"=")
    print()
    return id_connections, id_events


def test_with_synthetic_data():
    """Test function with synthetic data generation."""
    print("=== TESTING WITH SYNTHETIC DATA ===")

    shape = (8, 32, 512, 320)
    av_data = np.zeros(shape, dtype=np.float32)

    av_data[2:5, 10:15, 100:120, 50:70] = np.random.rand(3, 5, 20, 20) * 0.5 + 0.5
    av_data[1:4, 20:25, 200:230, 100:130] = np.random.rand(3, 5, 30, 30) * 0.3 + 0.7
    av_data[5:7, 5:8, 400:405, 200:205] = np.random.rand(2, 3, 5, 5) * 0.8 + 0.2

    print(f"Created synthetic data with shape {shape}")
    print(f"Non-zero voxels: {np.count_nonzero(av_data)}")

    # Create minimal params for testing
    params_values = {
        'events_extraction': {
            'threshold_size_3d': 10,
            'threshold_size_3d_removed': 5,
            'threshold_corr': 0.5
        },
        'files': {'save_results': 0},
        'paths': {'output_dir': './output'}
    }

    results = detect_calcium_events_opti(av_data, params_values)
    return results


if __name__ == "__main__":
    test_results = test_with_synthetic_data()