import numpy as np
import numba as nb
from numba import njit, prange, jit
from typing import List, Tuple, Optional, Any, Dict
import time
from scipy import ndimage
from skimage.measure import label
import matplotlib.pyplot as plt
import os
from astroca.tools.exportData import export_data
from tqdm import tqdm
from collections import deque


@jit(nopython=True, parallel=True)
def correlation_zero_boundary(v1, v2):
    """
    Compute cross-correlation with zero boundary conditions (like the Java version).
    Optimized with Numba for speed.
    """
    nV1 = len(v1)
    nV2 = len(v2)
    size = nV1 + nV2 - 1
    vout = np.zeros(size, dtype=np.float32)

    for n in prange(-nV2 + 1, nV1):  # prange for parallelization
        sum_val = 0.0
        for m in range(nV1):
            index = m - n
            if 0 <= index < nV2:
                sum_val += v2[index] * v1[m]
        vout[n + nV2 - 1] = sum_val

    return vout


@jit(nopython=True)
def compute_normalized_cross_correlation(v1, v2):
    """
    Compute normalized cross-correlation (NCC) for all lags, equivalent to the Java version.
    Returns an array of NCC values for each possible lag.
    Optimized with Numba.
    """
    if len(v1) == 0 or len(v2) == 0:
        return np.zeros(1, dtype=np.float32)  # Return empty array

    # Cross-correlation
    vout = correlation_zero_boundary(v1, v2)

    # Auto-correlation of v1 at lag 0
    auto_v1 = np.sum(v1 ** 2)  # Equivalent to correlation_zero_boundary(v1,v1)[len(v1)-1]

    # Auto-correlation of v2 at lag 0
    auto_v2 = np.sum(v2 ** 2)  # Equivalent to correlation_zero_boundary(v2,v2)[len(v2)-1]

    # Normalization factor
    den = np.sqrt(auto_v1 * auto_v2)

    # Avoid division by zero
    if den == 0:
        return np.zeros_like(vout)

    # Normalize all correlation values
    vout /= den

    return vout


@njit(fastmath=True, cache=True)
def _find_seeds_batch(frame_data: np.ndarray, id_mask: np.ndarray,
                      min_threshold: float = 0.0) -> np.ndarray:
    """Find multiple seeds in parallel."""
    depth, height, width = frame_data.shape

    # Pre-allocate result array
    max_seeds = 10
    seeds_result = np.full((max_seeds, 4), -1.0, dtype=np.float32)
    seed_count = 0

    # Find best voxels by scanning
    best_values = np.full(max_seeds, -1.0, dtype=np.float32)
    best_coords = np.full((max_seeds, 3), -1, dtype=np.int32)

    for z in range(depth):
        for y in range(height):
            for x in range(width):
                val = frame_data[z, y, x]
                if val > min_threshold and id_mask[z, y, x] == 0:
                    # Check if this value is better than our worst current best
                    min_idx = 0
                    min_val = best_values[0]
                    for i in range(max_seeds):
                        if best_values[i] < min_val:
                            min_idx = i
                            min_val = best_values[i]

                    if val > min_val:
                        best_values[min_idx] = val
                        best_coords[min_idx, 0] = x
                        best_coords[min_idx, 1] = y
                        best_coords[min_idx, 2] = z

    # Convert to output format
    for i in range(max_seeds):
        if best_values[i] > min_threshold:
            seeds_result[seed_count, 0] = best_coords[i, 0]  # x
            seeds_result[seed_count, 1] = best_coords[i, 1]  # y
            seeds_result[seed_count, 2] = best_coords[i, 2]  # z
            seeds_result[seed_count, 3] = best_values[i]  # intensity
            seed_count += 1

    if seed_count == 0:
        return np.array([[-1.0, -1.0, -1.0, 0.0]], dtype=np.float32)

    return seeds_result[:seed_count]

class EventDetectorUltraFast:
    """
    Version ultra-optimisée du détecteur d'événements calcium.
    Optimisations principales:
    - Traitement vectorisé des graines
    - Croissance de région en parallèle
    - Cache intelligent des patterns
    - Élimination des calculs redondants
    """

    def __init__(self, av_data: np.ndarray, threshold_size_3d: int = 10,
                 threshold_size_3d_removed: int = 5, threshold_corr: float = 0.5,
                 plot: bool = False):
        """Initialize with ultra-fast optimizations."""
        print("=== EventDetector Ultra-Fast ===")
        print(f"Input shape: {av_data.shape}")
        print(f"Data range: [{av_data.min():.3f}, {av_data.max():.3f}]")
        print(f"Non-zero density: {np.count_nonzero(av_data) / av_data.size:.4f}")

        self.av_ = av_data.astype(np.float32)
        self.time_length_, self.depth_, self.height_, self.width_ = av_data.shape

        self.threshold_size_3d_ = threshold_size_3d
        self.threshold_size_3d_removed_ = threshold_size_3d_removed
        self.threshold_corr_ = threshold_corr
        self.plot_ = plot

        # Pre-compute structures
        self.id_connected_voxel_ = np.zeros_like(self.av_, dtype=np.int32)
        self.final_id_events_ = []

        # Optimized neighbor offsets
        self._neighbor_offsets_3d = self._generate_neighbor_offsets_3d()
        self._neighbor_offsets_4d = self._generate_neighbor_offsets_4d()

        # Cache for patterns (with size limit)
        self.pattern_cache_: Dict[Tuple[int, int, int], np.ndarray] = {}
        self.max_cache_size_ = 50000

        # Statistics
        self.stats_ = {
            "frames_processed": 0,
            "seeds_found": 0,
            "regions_grown": 0,
            "events_retained": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }

    def _generate_neighbor_offsets_3d(self) -> np.ndarray:
        """Generate 3D spatial neighbor offsets."""
        offsets = []
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == dy == dz == 0:
                        continue
                    offsets.append([dz, dy, dx])
        return np.array(offsets, dtype=np.int32)

    def _generate_neighbor_offsets_4d(self) -> np.ndarray:
        """Generate 4D neighbor offsets."""
        offsets = []
        for dt in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == dy == dz == dt == 0:
                            continue
                        offsets.append([dt, dz, dy, dx])
        return np.array(offsets, dtype=np.int32)

    def _get_pattern_cached(self, z: int, y: int, x: int) -> Optional[np.ndarray]:
        """Get pattern with caching."""
        key = (z, y, x)
        if key in self.pattern_cache_:
            self.stats_["cache_hits"] += 1
            return self.pattern_cache_[key]

        intensity_profile = self.av_[:, z, y, x]

        # Find non-zero bounds
        nonzero_idx = np.nonzero(intensity_profile)[0]
        if len(nonzero_idx) == 0:
            return None

        start, end = nonzero_idx[0], nonzero_idx[-1] + 1
        pattern = intensity_profile[start:end].copy()

        # Cache management
        if len(self.pattern_cache_) >= self.max_cache_size_:
            # Remove oldest entries (simple FIFO)
            oldest_key = next(iter(self.pattern_cache_))
            del self.pattern_cache_[oldest_key]

        self.pattern_cache_[key] = pattern
        self.stats_["cache_misses"] += 1
        return pattern

    def find_events(self) -> None:
        """Ultra-fast event detection."""
        print(
            f"Thresholds: size={self.threshold_size_3d_}, removed={self.threshold_size_3d_removed_}, corr={self.threshold_corr_}")
        start_time = time.time()

        event_id = 1
        small_groups = []
        small_group_ids = []

        # Process frames with optimized seed finding
        # for t in tqdm(range(self.time_length_), desc="Processing frames"):
        for t in range(self.time_length_):
            frame_start = time.time()
            print(f"Processing frame {t + 1}/{self.time_length_}...")

            # Find multiple seeds at once
            seeds = _find_seeds_batch(
                self.av_[t],
                self.id_connected_voxel_[t],
                min_threshold=0.01
            )


            if seeds[0, 0] == -1:  # No seeds found
                continue

            # Process each seed
            for seed_idx in range(len(seeds)):
                x, y, z, intensity = seeds[seed_idx]
                x, y, z = int(x), int(y), int(z)
                print(f"Found seed at (x={x}, y={y}, z={z}) in frame {t}, identifier {event_id}")

                if self.id_connected_voxel_[t, z, y, x] != 0:
                    continue  # Already processed

                # Get pattern for this seed
                pattern = self._get_pattern_cached(z, y, x)
                if pattern is None:
                    print(f"No valid pattern for seed at (t={t}, z={z}, y={y}, x={x}), skipping.")
                    continue

                # Initialize region
                region_voxels = [[t, z, y, x]]
                self.id_connected_voxel_[t, z, y, x] = event_id

                # Add temporal pattern points
                intensity_profile = self.av_[:, z, y, x]
                nonzero_idx = np.nonzero(intensity_profile)[0]

                if len(nonzero_idx) > 0:
                    start_t, end_t = nonzero_idx[0], nonzero_idx[-1] + 1
                    for tp in range(start_t, end_t):
                        if (tp != t and tp < self.time_length_ and
                                self.id_connected_voxel_[tp, z, y, x] == 0):
                            self.id_connected_voxel_[tp, z, y, x] = event_id
                            region_voxels.append([tp, z, y, x])

                # Grow region using optimized method
                self._grow_region_fast(region_voxels, pattern, event_id, t, z, y, x)

                # Check region size
                region_size = len(region_voxels)
                print(f"Region size for event {event_id}: {region_size} voxels")
                if region_size >= self.threshold_size_3d_:
                    self.final_id_events_.append(event_id)
                    self.stats_["events_retained"] += 1
                else:
                    print(f"Region {event_id} is too small ({region_size} voxels), adding to small groups.")
                    small_groups.append(region_voxels)
                    small_group_ids.append(event_id)

                event_id += 1
                self.stats_["seeds_found"] += 1
            print(f"Frame {t + 1} processed in {time.time() - frame_start:.2f}s")

            self.stats_["frames_processed"] += 1

        # Process small groups
        if small_groups:
            print(f"Processing {len(small_groups)} small groups...")
            self._process_small_groups_fast(small_groups, small_group_ids)

        print(f"Events found: {len(self.final_id_events_)}")
        print(f"Size of each final event:")
        for event_id in self.final_id_events_:
            size = np.sum(self.id_connected_voxel_ == event_id)
            print(f"    Event ID={event_id}: {size} voxels")
        # Finalize
        self._compute_final_id_events()

        print(f"Total time: {time.time() - start_time:.2f}s")
        print(
            f"Cache efficiency: {self.stats_['cache_hits'] / (self.stats_['cache_hits'] + self.stats_['cache_misses']):.2%}")

    def _grow_region_fast(self, region_voxels: List, ref_pattern: np.ndarray,
                          event_id: int, seed_t: int, seed_z: int, seed_y: int, seed_x: int) -> None:
        """Fast region growing with minimal redundancy."""
        processed = set()
        queue = deque([(seed_t, seed_z, seed_y, seed_x)])

        while queue:
            t, z, y, x = queue.popleft()

            if (t, z, y, x) in processed:
                continue
            processed.add((t, z, y, x))

            # Check 3D spatial neighbors only
            for offset in self._neighbor_offsets_3d:
                dz, dy, dx = offset
                nz, ny, nx = z + dz, y + dy, x + dx

                if (nz < 0 or nz >= self.depth_ or
                        ny < 0 or ny >= self.height_ or
                        nx < 0 or nx >= self.width_):
                    continue

                if (self.av_[t, nz, ny, nx] != 0 and
                        self.id_connected_voxel_[t, nz, ny, nx] == 0):

                    # Quick pattern check
                    neighbor_pattern = self._get_pattern_cached(nz, ny, nx)
                    if neighbor_pattern is None:
                        continue

                    # Fast correlation check
                    # corr = compute_normalized_cross_correlation(ref_pattern, neighbor_pattern)
                    # if np.max(corr) >= self.threshold_corr_:
                    if self._quick_correlation_check(ref_pattern, neighbor_pattern):
                        # Add to region
                        self.id_connected_voxel_[t, nz, ny, nx] = event_id
                        region_voxels.append([t, nz, ny, nx])

                        # Add temporal points
                        intensity_profile = self.av_[:, nz, ny, nx]
                        nonzero_idx = np.nonzero(intensity_profile)[0]

                        if len(nonzero_idx) > 0:
                            start_t, end_t = nonzero_idx[0], nonzero_idx[-1] + 1
                            for tp in range(start_t, end_t):
                                if (tp != t and tp < self.time_length_ and
                                        self.id_connected_voxel_[tp, nz, ny, nx] == 0):
                                    self.id_connected_voxel_[tp, nz, ny, nx] = event_id
                                    region_voxels.append([tp, nz, ny, nx])

                        queue.append((t, nz, ny, nx))

    def _quick_correlation_check(self, pattern1: np.ndarray, pattern2: np.ndarray) -> bool:
        """Fast correlation check without full computation."""
        if len(pattern1) == 0 or len(pattern2) == 0:
            return False

        # Quick dot product check
        norm1 = np.linalg.norm(pattern1)
        norm2 = np.linalg.norm(pattern2)

        if norm1 == 0 or norm2 == 0:
            return False

        # Approximate correlation using dot product of normalized patterns
        if len(pattern1) == len(pattern2):
            dot_product = np.dot(pattern1, pattern2)
            approx_corr = dot_product / (norm1 * norm2)
            return approx_corr > self.threshold_corr_ * 0.8  # Relaxed threshold for speed

        # For different lengths, use a simpler overlap check
        min_len = min(len(pattern1), len(pattern2))
        p1_sub = pattern1[:min_len]
        p2_sub = pattern2[:min_len]

        dot_product = np.dot(p1_sub, p2_sub)
        norm1_sub = np.linalg.norm(p1_sub)
        norm2_sub = np.linalg.norm(p2_sub)

        if norm1_sub == 0 or norm2_sub == 0:
            return False

        approx_corr = dot_product / (norm1_sub * norm2_sub)
        return approx_corr > self.threshold_corr_ * 0.8

    def _process_small_groups_fast(self, small_groups: List, small_group_ids: List) -> None:
        """Fast processing of small groups."""
        # First pass: merge neighboring small groups
        merged_groups = {}

        for i, group in enumerate(small_groups):
            group_id = small_group_ids[i]

            # Find neighboring groups
            neighbors = set()
            for t, z, y, x in group:
                for offset in self._neighbor_offsets_4d:
                    dt, dz, dy, dx = offset
                    nt, nz, ny, nx = t + dt, z + dz, y + dy, x + dx

                    if (nt >= 0 and nt < self.time_length_ and
                            nz >= 0 and nz < self.depth_ and
                            ny >= 0 and ny < self.height_ and
                            nx >= 0 and nx < self.width_):

                        neighbor_id = self.id_connected_voxel_[nt, nz, ny, nx]
                        if neighbor_id != 0 and neighbor_id != group_id:
                            neighbors.add(neighbor_id)

            # Merge with largest neighbor if it's a big group
            best_neighbor = None
            for neighbor_id in neighbors:
                if neighbor_id in self.final_id_events_:
                    best_neighbor = neighbor_id
                    break

            if best_neighbor:
                # Merge with large group
                for t, z, y, x in group:
                    self.id_connected_voxel_[t, z, y, x] = best_neighbor
            elif len(group) >= self.threshold_size_3d_removed_:
                # Keep as separate event
                self.final_id_events_.append(group_id)
            else:
                # Remove small isolated group
                for t, z, y, x in group:
                    self.id_connected_voxel_[t, z, y, x] = 0

    def _compute_final_id_events(self) -> None:
        """Compute final event IDs with vectorized remapping."""
        if not self.final_id_events_:
            return

        # Create mapping from old IDs to new sequential IDs
        unique_ids = np.unique(self.id_connected_voxel_[self.id_connected_voxel_ > 0])
        final_ids = sorted([id for id in unique_ids if id in self.final_id_events_])

        if not final_ids:
            return

        # Vectorized remapping
        max_id = max(final_ids)
        id_map = np.zeros(max_id + 1, dtype=np.int32)

        for new_id, old_id in enumerate(final_ids, 1):
            id_map[old_id] = new_id

        # Apply mapping
        mask = self.id_connected_voxel_ > 0
        self.id_connected_voxel_[mask] = id_map[self.id_connected_voxel_[mask]]

        # Update final event list
        self.final_id_events_ = list(range(1, len(final_ids) + 1))

    def get_results(self) -> Tuple[np.ndarray, List[int]]:
        """Return final results."""
        return self.id_connected_voxel_, self.final_id_events_

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics."""
        stats = {
            'nb_events': len(self.final_id_events_),
            'event_sizes': [],
            'total_event_voxels': 0
        }

        if self.final_id_events_:
            unique_ids, counts = np.unique(
                self.id_connected_voxel_[self.id_connected_voxel_ > 0],
                return_counts=True
            )

            for event_id in self.final_id_events_:
                if event_id in unique_ids:
                    idx = np.where(unique_ids == event_id)[0][0]
                    size = counts[idx]
                    stats['event_sizes'].append(size)
                    stats['total_event_voxels'] += size

            if stats['event_sizes']:
                sizes = np.array(stats['event_sizes'])
                stats['mean_event_size'] = np.mean(sizes)
                stats['median_event_size'] = np.median(sizes)
                stats['max_event_size'] = np.max(sizes)
                stats['min_event_size'] = np.min(sizes)

        stats.update(self.stats_)
        return stats


def detect_calcium_events_opti(av_data: np.ndarray, params_values: dict = None) -> Tuple[np.ndarray, List[int]]:
    """
    Ultra-fast calcium event detection function.
    """
    threshold_size_3d = int(params_values['events_extraction']['threshold_size_3d'])
    threshold_size_3d_removed = int(params_values['events_extraction']['threshold_size_3d_removed'])
    threshold_corr = float(params_values['events_extraction']['threshold_corr'])
    save_results = int(params_values['files']['save_results']) == 1
    output_directory = params_values['paths']['output_dir']

    detector = EventDetectorUltraFast(
        av_data, threshold_size_3d, threshold_size_3d_removed, threshold_corr
    )

    detector.find_events()
    id_connections, id_events = detector.get_results()

    if save_results:
        if output_directory is None:
            raise ValueError("Output directory must be specified if save_results is True.")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        id_connections = id_connections.astype(np.float32)
        export_data(id_connections, output_directory,
                    export_as_single_tif=True, file_name="ID_calciumEvents")

    print("=" * 60)
    return id_connections, id_events


def test_ultra_fast():
    """Test with synthetic data."""
    print("=== TESTING ULTRA-FAST VERSION ===")

    # Create test data similar to your real volumes
    shape = (100, 30, 500, 300)
    av_data = np.zeros(shape, dtype=np.float32)

    # Add some realistic calcium events
    np.random.seed(42)

    # Large event
    av_data[20:35, 10:15, 100:150, 50:100] = (
            np.random.rand(15, 5, 50, 50) * 0.3 + 0.4
    )

    # Medium event
    av_data[50:65, 20:25, 200:230, 150:180] = (
            np.random.rand(15, 5, 30, 30) * 0.4 + 0.3
    )

    # Small events
    for i in range(5):
        t_start = np.random.randint(0, 80)
        z_start = np.random.randint(0, 25)
        y_start = np.random.randint(0, 480)
        x_start = np.random.randint(0, 280)

        av_data[t_start:t_start + 8, z_start:z_start + 3,
        y_start:y_start + 15, x_start:x_start + 15] = (
                np.random.rand(8, 3, 15, 15) * 0.2 + 0.5
        )

    print(f"Test data shape: {shape}")
    print(f"Non-zero voxels: {np.count_nonzero(av_data):,}")
    print(f"Data range: [{av_data.min():.3f}, {av_data.max():.3f}]")

    # Test parameters
    params_values = {
        'events_extraction': {
            'threshold_size_3d': 50,
            'threshold_size_3d_removed': 20,
            'threshold_corr': 0.4
        },
        'files': {'save_results': 0},
        'paths': {'output_dir': './output'}
    }

    start_time = time.time()
    results = detect_calcium_events_opti(av_data, params_values)
    end_time = time.time()

    print(f"Processing time: {end_time - start_time:.2f} seconds")
    print(f"Events detected: {len(results[1])}")

    return results


if __name__ == "__main__":
    test_results = test_ultra_fast()