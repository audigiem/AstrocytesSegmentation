import numpy as np
import numba as nb
from numba import njit, prange
from typing import List, Tuple, Optional, Any
import time
from scipy import ndimage
from skimage.measure import label
import matplotlib.pyplot as plt
import os
from astroca.tools.exportData import export_data
from tqdm import tqdm
from collections import deque

class EventDetectorOptimized:
    """
    Event Detector for calcium events in 4D data (time, depth, height, width).
    """

    def __init__(self, av_data: np.ndarray, threshold_size_3d: int = 10,
                 threshold_size_3d_removed: int = 5, threshold_corr: float = 0.5,
                 plot: bool = False):
        """
        Initialize the EventDetector with the provided active voxel data and thresholds.
        @param av_data: 4D numpy array of shape (T, Z, Y, X) representing the active voxel data.
        @param threshold_size_3d: Minimum size of the 3D region to be considered an event.
        @param threshold_size_3d_removed: Minimum size of the 3D region to be retained after merging small events.
        @param threshold_corr: Threshold for normalized cross-correlation to link voxels.
        @param plot: Boolean flag to enable plotting of detected patterns.
        """
        print("=== Finding events in 4D data ===")
        # print(f"Input data shape: {av_data.shape}")
        print(f"Input data range: [{av_data.min():.3f}, {av_data.max():.3f}]")
        print(f"Non-zero voxels: {np.count_nonzero(av_data)}/{av_data.size}")

        self.av_ = av_data.astype(np.float32)
        self.time_length_, self.depth_, self.height_, self.width_ = av_data.shape

        self.threshold_size_3d_ = threshold_size_3d
        self.threshold_size_3d_removed_ = threshold_size_3d_removed
        self.threshold_corr_ = threshold_corr
        self.plot_ = plot

        self.nonzero_mask_ = self.av_ != 0
        self.nonzero_cords_ = np.where(self.nonzero_mask_)

        self.id_connected_voxel_ = np.zeros_like(self.av_, dtype=np.int32)
        self.final_id_events_ = []

        self.pattern_cache_ = {}
        self.pattern_ = {}

        self.stats_ = {
            "patterns_computed": 0,
            "regions_grown": 0,
            "correlations_computed": 0,
            "events_retained": 0,
            "events_merged": 0,
            "events_removed": 0,
        }



    def find_events(self) -> None:
        """
        Main method to find events in the active voxel data.
        """
        print(f"Thresholds -> size: {self.threshold_size_3d_}, removed: {self.threshold_size_3d_removed_}, corr: {self.threshold_corr_}")
        start_time = time.time()
        if len(self.nonzero_cords_[0]) == 0:
            print("No non-zero voxels found!")
            return

        event_id = 1
        # Active voxels belonging to the current event whose neighbors have not been processed yet
        waiting_for_processing = []
        index_waiting_for_processing = 0
        small_AV_groups = []
        id_small_AV_groups = []

        for t in tqdm(range(self.time_length_), desc="Processing time frames", unit="frame"):
            # print(f"\nProcessing time frame {t}, searching seed ...")
            # frame_time = time.time()
            seed = self._find_seed_point(t)
            while seed is not None:
                x, y, z = seed
                # print(f"Found seed at ({x}, {y}, {z}) in frame {t}, identifier {event_id}")

                intensity_profile = self.av_[:, z, y, x]
                pattern = self._detect_pattern_optimized(intensity_profile, t)
                if pattern is None:
                    print(f"No valid pattern found for seed ({x}, {y}, {z})")
                    break

                # add the seed to the waiting list
                self.pattern_[t, z, y, x] = pattern
                self.id_connected_voxel_[t, z, y, x] = event_id
                waiting_for_processing.append([t, z, y, x])

                # all the points of the current pattern {t,z,y,x}_t = {(t0,z,y,x) | t0 in [t_start, t_end]}
                # belong to the same event
                for i in range(1, len(pattern)):
                    t0 = t + i
                    if t0 < self.time_length_ and self.id_connected_voxel_[t0, z, y, x] == 0:
                        self.id_connected_voxel_[t0, z, y, x] = event_id
                        self.pattern_[t0, z, y, x] = pattern
                        waiting_for_processing.append([t0, z, y, x])

                # Check the neighbors of the seed point
                while index_waiting_for_processing < len(waiting_for_processing):
                    index = waiting_for_processing[index_waiting_for_processing]
                    pattern = self.pattern_[index[0], index[1], index[2], index[3]]
                    if pattern is None:
                        raise ValueError(f"Invalid pattern for voxel ({index[3]}, {index[2]}, {index[1]}, {index[0]})")
                    self._find_connected_AV(index, pattern, event_id, waiting_for_processing)
                    index_waiting_for_processing += 1
                # print(f"    Size of group ID={event_id} : {len(waiting_for_processing)}")

                # check if the number of voxels in the group is below the threshold
                if len(waiting_for_processing) < self.threshold_size_3d_:
                    # print(f"    Group ID={event_id} is too small ({len(waiting_for_processing)} voxels), adding to small groups")
                    small_AV_groups.append(waiting_for_processing.copy())
                    id_small_AV_groups.append(event_id)
                else:
                    self.final_id_events_.append(event_id)
                    self.stats_["events_retained"] += 1  # TRACE
                # update
                event_id += 1
                waiting_for_processing.clear()
                index_waiting_for_processing = 0
                seed = self._find_seed_point(t)
            # print(f"Time for frame {t}: {time.time() - frame_time:.2f} seconds")

        # if there are small groups, try to merge them with larger groups
        if len(small_AV_groups) > 0:
            # print(f"\nFound {len(small_AV_groups)} small groups, trying to merge them with larger groups...")
            self._group_small_neighborhood_regions(small_AV_groups, id_small_AV_groups)

            for i, group in enumerate(small_AV_groups):
                change_id = self._change_id_small_regions(group, id_small_AV_groups)
                if change_id:
                    # print(f"Group {id_small_AV_groups[i]} merged into larger group")
                    self.stats_["events_merged"] += 1
                    small_AV_groups.pop(i)
                    id_small_AV_groups.pop(i)
                    i -= 1  # Adjust index after removal
                else:
                    # if the small group is not isolated, check its size and add/remove it
                    if len(group) >= self.threshold_size_3d_removed_:
                        # print(f"Group {id_small_AV_groups[i]} is large enough ({len(group)} voxels), adding to final events")
                        self.final_id_events_.append(id_small_AV_groups[i])
                    else:
                        # print(f"Group {id_small_AV_groups[i]} is isolated and too small ({len(group)} voxels), removing it")
                        # remove the small group from the id_connected_voxel_
                        for index_point in group:
                            t, z, y, x = index_point
                            self.id_connected_voxel_[t, z, y, x] = 0
                        self.stats_["events_removed"] += 1  # TRACE
                    small_AV_groups.pop(i)
                    id_small_AV_groups.pop(i)
                    i -= 1  # Adjust index after removal
            small_AV_groups.clear()
            id_small_AV_groups.clear()
        print(f"\nTotal events found: {len(self.final_id_events_)}")

        self._compute_final_id_events()

        print(f"Total time taken: {time.time() - start_time:.2f} seconds")
        print()


    def _find_connected_AV(self, seed: List[int], pattern: np.ndarray, event_id: int, waiting_for_processing: deque[List[int]]):
        """
        Look for the 26-connected neighbors of the seed voxel. For each neighbor:
        1. Extract the intensity profile and its temporal pattern.
        2. Compute the normalized cross-validation (NCC) between the seed pattern and the neighbor pattern.
        3. If the max of NCC is above the threshold, add the neighbor to the event and continue processing.
        @param seed: List of coordinates [t, z, y, x] of the seed voxel.
        @param pattern: 1D numpy array of the detected pattern for the seed voxel.
        @param event_id: Identifier of the current event.
        @param waiting_for_processing: List of voxels waiting for processing, where each voxel is represented as [t, z, y, x].
        """
        t, z, y, x = seed
        for dz in [-1, 0, 1]:
            nz = z + dz
            if nz < 0 or nz >= self.depth_:
                continue
            for dy in [-1, 0, 1]:
                ny = y + dy
                if ny < 0 or ny >= self.height_:
                    continue
                for dx in [-1, 0, 1]:
                    nx = x + dx
                    if dx == dy == dz == 0:
                        continue
                    if nx < 0 or nx >= self.width_:
                        continue

                    # neighbor = [nx, ny, nz, t]
                    # index = nz + (nx + ny * self.width_) * self.depth_

                    if self.av_[t, nz, ny, nx] != 0 and self.id_connected_voxel_[t, nz, ny, nx] == 0:
                        # 1. Extract the intensity profile of the neighbor voxel
                        intensity_profile = self.av_[:, nz, ny, nx]
                        neighbor_pattern = self._detect_pattern_optimized(intensity_profile, t)
                        if neighbor_pattern is None:
                            continue

                        # 2. Compute normalized cross-correlation (NCC) between the seed pattern and the neighbor pattern
                        correlation = self._compute_normalized_cross_correlation(pattern, neighbor_pattern)
                        max_corr = np.max(correlation)
                        if max_corr > self.threshold_corr_:
                            self.id_connected_voxel_[t, nz, ny, nx] = event_id
                            self.pattern_[(t, nz, ny, nx)] = neighbor_pattern
                            waiting_for_processing.append([t, nz, ny, nx])

                            # Determine the start time of the pattern
                            start_t = t
                            while start_t > 0 and intensity_profile[start_t - 1] != 0:
                                start_t -= 1

                            # Add all the points of the current pattern {t,z,y,x}_t = {(t0,z,y,x) | t0 in [start_t, t_end]}
                            for p in range(len(neighbor_pattern)):
                                tp = start_t + p
                                if tp < self.time_length_ and self.id_connected_voxel_[tp, nz, ny, nx] == 0:
                                    self.id_connected_voxel_[tp, nz, ny, nx] = event_id
                                    self.pattern_[(tp, nz, ny, nx)] = neighbor_pattern
                                    waiting_for_processing.append([tp, nz, ny, nx])



    def _find_seed_point(self, t0: int) -> None | tuple[int, int, int] | tuple[Any, Any, Any]:
        """
        Find a seed point on the frame t0 (not necessarily the first frame) such that
        x_t0 = {x,y,z,t0} = max(av[t0, z, y, x]) over (x,y,z) in [0, width-1] x [0, height-1] x [0, depth-1].
        @param t0: Time frame to search for the seed point.
        @return: x_t0 = (x, y, z) coordinates of the seed point or None if no seed found.
        """
        max_val = 0.0
        best_seed = (-1, -1, -1)
        possible_cords = np.where(self.av_[t0] > 0)

        for z, y, x in zip(*possible_cords):
            if self.av_[t0, z, y, x] > max_val and self.id_connected_voxel_[t0, z, y, x] == 0:
                max_val = self.av_[t0, z, y, x]
                best_seed = (x, y, z)

        if best_seed == (-1, -1, -1):
            return None
        return best_seed


    def _detect_pattern_optimized(self, intensity_profile: np.ndarray, t: int) -> Optional[np.ndarray]:
        """
        Detect the temporal pattern of the intensity profile that starts or belongs to the time frame t.
        @param intensity_profile: temporal profile of the voxel at (x,y,z).
        @param t: Time frame to start the pattern detection.
        @return: pattern: 1D numpy array of the detected pattern or None if no valid pattern found.
        """
        if intensity_profile[t] == 0:
            return None
        # check if the pattern is already cached
        # if t in self.pattern_cache_:
        #     return self.pattern_cache_[t]
        # Find the start and end of the pattern
        while t > 0 and intensity_profile[t - 1] != 0:
            t -= 1
        start = t
        while t < len(intensity_profile) and intensity_profile[t] != 0:
            t += 1
        end = t
        pattern = intensity_profile[start:end]
        if self.plot_:
            print(f"Detected pattern from {start} to {end} for time frame {t}, length: {len(pattern)}")
            plt.figure(figsize=(10, 4))
            plt.plot(np.arange(start, end), pattern, label=f"Pattern at t={t}")
            plt.title(f"Detected Pattern at t={t}")
            plt.show()
        self.stats_["patterns_computed"] += 1  # TRACE
        # self.pattern_cache_[t] = pattern
        return pattern


    def _compute_normalized_cross_correlation(self, pattern1: np.ndarray, pattern2: np.ndarray) -> np.ndarray:
        """
        Compute the normalized cross-correlation between two patterns.
        The signals are supposed to be causal, meaning that pattern1 starts at t=0 and pattern2 starts at t=0.
        and compactly supported.
        @param pattern1: 1D numpy array representing the first pattern.
        @param pattern2: 1D numpy array representing the second pattern.
        @return: 1D numpy array of the normalized cross-correlation values.
        """
        # Cross-correlation with zero boundary conditions
        vout = np.correlate(pattern1, pattern2, mode='full')

        # Auto-correlation for normalization
        output1 = np.correlate(pattern1, pattern1, mode='full')
        auto_corr_v1 = output1[len(pattern1) - 1]

        output2 = np.correlate(pattern2, pattern2, mode='full')
        auto_corr_v2 = output2[len(pattern2) - 1]

        # Normalization
        den = np.sqrt(auto_corr_v1 * auto_corr_v2)
        if den == 0:
            return np.zeros_like(vout)
        vout = vout / den

        self.stats_["correlations_computed"] += 1  # TRACE

        if self.plot_:
            zero_position_vout = len(pattern2) - 1
            plt.plot(np.arange(-zero_position_vout, len(pattern1)), vout)
            plt.title("Normalized cross correlation")
            plt.xlabel("time")
            plt.ylabel("value")
            plt.show()

        return vout

    def _group_small_neighborhood_regions(self, small_av_groups: list, list_ids_small_av_group: list):
        """
        For each small AV group, find the largest neighboring small AV group.
        If such a neighbor exists and is at least as large, merge the current group into it.

        Args:
            small_av_groups: List of small AV groups (each group is a list of [t, z, y, x]).
            list_ids_small_av_group: List of IDs for each small AV group.
        """
        id_ = 0
        while id_ < len(small_av_groups):
            list_av = small_av_groups[id_]
            group_id = list_ids_small_av_group[id_]

            neighbor_id_counts = {}
            # For each voxel in the group, check all 26 neighbors in space and +/-1 in time
            for index_point in list_av:
                t, z, y, x = index_point
                for dt in [-1, 0, 1]:
                    nt = t + dt
                    if nt < 0 or nt >= self.time_length_:
                        continue
                    for dz in [-1, 0, 1]:
                        nz = z + dz
                        if nz < 0 or nz >= self.depth_:
                            continue
                        for dy in [-1, 0, 1]:
                            ny = y + dy
                            if ny < 0 or ny >= self.height_:
                                continue
                            for dx in [-1, 0, 1]:
                                nx = x + dx
                                if dx == dy == dz == dt == 0:
                                    continue
                                if nx < 0 or nx >= self.width_:
                                    continue
                                neighbor_id = self.id_connected_voxel_[nt, nz, ny, nx]
                                if (neighbor_id != 0 and
                                        neighbor_id in list_ids_small_av_group and
                                        neighbor_id != group_id):
                                    neighbor_id_counts[neighbor_id] = neighbor_id_counts.get(neighbor_id, 0) + 1

            if neighbor_id_counts:
                # Find the neighbor group with the most contacts
                new_id = max(neighbor_id_counts, key=neighbor_id_counts.get)
                new_id_index = list_ids_small_av_group.index(new_id)
                # Only merge if the neighbor group is at least as large
                if len(small_av_groups[new_id_index]) >= len(list_av):
                    for index_point in list_av:
                        t, z, y, x = index_point
                        self.id_connected_voxel_[t, z, y, x] = new_id
                        small_av_groups[new_id_index].append(index_point)
                    # Remove merged group
                    del small_av_groups[id_]
                    del list_ids_small_av_group[id_]
                    # Do not increment id_ to check the new group at this index
                    continue
            id_ += 1

    def _change_id_small_regions(self, list_av: list, list_ids_small_av_group: list) -> bool:
        """
        For a small AV group, find the largest neighboring AV group (not in small groups).
        If found, change the group's ID to the largest neighbor's ID.

        Args:
            list_av: List of voxels [t, z, y, x] in the small AV group.
            list_ids_small_av_group: List of IDs of all small AV groups.

        Returns:
            True if the group ID was changed, False otherwise.
        """
        list_ids_neighborhood = []
        list_count = []

        for index_point in list_av:
            t, z, y, x = index_point
            for dt in [-1, 0, 1]:
                nt = t + dt
                if nt < 0 or nt >= self.time_length_:
                    continue
                for dz in [-1, 0, 1]:
                    nz = z + dz
                    if nz < 0 or nz >= self.depth_:
                        continue
                    for dy in [-1, 0, 1]:
                        ny = y + dy
                        if ny < 0 or ny >= self.height_:
                            continue
                        for dx in [-1, 0, 1]:
                            nx = x + dx
                            if dx == dy == dz == 0:
                                continue
                            if nx < 0 or nx >= self.width_:
                                continue
                            neighbor_id = self.id_connected_voxel_[nt, nz, ny, nx]
                            if neighbor_id != 0 and neighbor_id not in list_ids_small_av_group:
                                if neighbor_id in list_ids_neighborhood:
                                    idx = list_ids_neighborhood.index(neighbor_id)
                                    list_count[idx] += 1
                                else:
                                    list_ids_neighborhood.append(neighbor_id)
                                    list_count.append(1)

        if list_ids_neighborhood:
            index_new_id = int(np.argmax(list_count))
            new_id = list_ids_neighborhood[index_new_id]
            for index_point in list_av:
                t, z, y, x = index_point
                self.id_connected_voxel_[t, z, y, x] = new_id
            return True
        else:
            return False

    def _compute_final_id_events(self):
        """
        Remap event IDs in id_connected_voxel_ to consecutive values (1, 2, 3, ...)
        using final_id_events_ as reference.
        """
        if not self.final_id_events_:
            return  # Rien à faire

        # Étape 1 : créer un mapping numpy de taille max_id + 1
        max_id = self.id_connected_voxel_.max()
        id_map = np.zeros(max_id + 1, dtype=self.id_connected_voxel_.dtype)

        # Étape 2 : construire le mapping {old_id → new_id}
        final_ids = sorted(self.final_id_events_)
        for new_id, old_id in enumerate(final_ids, start=1):
            id_map[old_id] = new_id

        # Étape 3 : appliquer le mapping vectorisé
        self.id_connected_voxel_ = id_map[self.id_connected_voxel_]


    def get_results(self) -> Tuple[np.ndarray, List[int]]:
        return self.id_connected_voxel_, self.final_id_events_

    def get_statistics(self) -> dict:
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


def detect_calcium_events(av_data: np.ndarray, params_values: dict = None,
                                   save_results: bool = False,
                                   output_directory: str = None) -> Tuple[np.ndarray, List[int]]:
    """
    Function to detect calcium events in 4D active voxel data using the optimized EventDetector.
    @param av_data: 4D numpy array of shape (T, Z, Y, X) representing the active voxel data.
    @param params_values: Dictionary containing parameters for event detection.
    @param save_results: Boolean flag to save results to disk.
    @param output_directory: Directory to save results if save_results is True.
    @return: Tuple containing: list of detected events, their IDs, and statistics.
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
        id_connections = id_connections.astype(np.float32)  # Ensure the data is in float32 format
        export_data(id_connections, output_directory, export_as_single_tif=True, file_name="ID_calciumEvents")
    print(60*"=")
    print()
    return id_connections, id_events


def test_with_synthetic_data():
    print("=== TESTING WITH SYNTHETIC DATA ===")

    shape = (8, 32, 512, 320)
    av_data = np.zeros(shape, dtype=np.float32)

    av_data[2:5, 10:15, 100:120, 50:70] = np.random.rand(3, 5, 20, 20) * 0.5 + 0.5

    av_data[1:4, 20:25, 200:230, 100:130] = np.random.rand(3, 5, 30, 30) * 0.3 + 0.7

    av_data[5:7, 5:8, 400:405, 200:205] = np.random.rand(2, 3, 5, 5) * 0.8 + 0.2

    print(f"Created synthetic data with shape {shape}")
    print(f"Non-zero voxels: {np.count_nonzero(av_data)}")

    results = detect_calcium_events(av_data)

    return results

if __name__ == "__main__":
    test_results = test_with_synthetic_data()