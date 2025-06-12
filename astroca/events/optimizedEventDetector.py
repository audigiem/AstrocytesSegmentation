"""
Optimized Event Detector for Calcium Imaging Data
"""

import numpy as np
import numba as nb
from numba import njit, prange
import time

# Global arrays to avoid repeated allocations
_temp_pattern = np.empty(1000, dtype=np.float32)
_temp_correlation = np.empty(2000, dtype=np.float32)
_temp_neighbors = np.empty((26, 4), dtype=np.int32)


@njit
def _get_linear_idx(x, y, z, width, depth):
    """Convert 3D coordinates to linear index"""
    return z + (x + y * width) * depth


@njit
def _compute_profile(av_data, x, y, z):
    """Extract intensity profile for given voxel"""
    return av_data[:, z, y, x]


@njit
def _detect_pattern_core(profile, t):
    """Core pattern detection - returns start_time and length"""
    if profile[t] == 0.0:
        return -1, 0

    # Find pattern start
    start_t = t
    while start_t > 0 and profile[start_t - 1] != 0.0:
        start_t -= 1

    # Find pattern length
    length = 0
    for i in range(start_t, len(profile)):
        if profile[i] != 0.0:
            length += 1
        else:
            break

    return start_t, length


@njit
def _extract_pattern(profile, start_t, length, pattern_buffer):
    """Extract pattern values into buffer"""
    for i in range(length):
        pattern_buffer[i] = profile[start_t + i]


@njit
def _normalized_cross_correlation(p1, len1, p2, len2, result_buffer):
    """Optimized normalized cross-correlation"""
    result_len = len1 + len2 - 1

    # Compute means
    mean1 = 0.0
    mean2 = 0.0
    for i in range(len1):
        mean1 += p1[i]
    for i in range(len2):
        mean2 += p2[i]
    mean1 /= len1
    mean2 /= len2

    # Compute standard deviations
    std1 = 0.0
    std2 = 0.0
    for i in range(len1):
        diff = p1[i] - mean1
        std1 += diff * diff
    for i in range(len2):
        diff = p2[i] - mean2
        std2 += diff * diff

    std1 = np.sqrt(std1 / len1)
    std2 = np.sqrt(std2 / len2)

    if std1 == 0.0 or std2 == 0.0:
        for i in range(result_len):
            result_buffer[i] = 0.0
        return result_len

    # Compute normalized cross-correlation
    for lag in range(result_len):
        sum_corr = 0.0
        count = 0
        for i in range(len1):
            j = lag - i
            if 0 <= j < len2:
                sum_corr += ((p1[i] - mean1) / std1) * ((p2[j] - mean2) / std2)
                count += 1
        result_buffer[lag] = sum_corr / count if count > 0 else 0.0

    return result_len


@njit
def _find_max_value(arr, length):
    """Find maximum value and its index"""
    max_val = arr[0]
    max_idx = 0
    for i in range(1, length):
        if arr[i] > max_val:
            max_val = arr[i]
            max_idx = i
    return max_val, max_idx


@njit
def _get_26_neighbors(x, y, z, t, width, height, depth, time_length, neighbors_buffer):
    """Get all valid 26-connected neighbors"""
    count = 0
    for i in range(-1, 2):
        nz = z + i
        if nz < 0 or nz >= depth:
            continue
        for k in range(-1, 2):
            ny = y + k
            if ny < 0 or ny >= height:
                continue
            for j in range(-1, 2):
                if i == 0 and j == 0 and k == 0:
                    continue
                nx = x + j
                if nx < 0 or nx >= width:
                    continue
                neighbors_buffer[count, 0] = nx
                neighbors_buffer[count, 1] = ny
                neighbors_buffer[count, 2] = nz
                neighbors_buffer[count, 3] = t
                count += 1
    return count


@njit
def _find_seed_point(av_data, id_connected_voxel, t, width, height, depth,
                     index_x_min, index_x_max):
    """Find next unexplored seed point at time t"""
    for z in range(depth):
        for y in range(height):
            for x in range(index_x_min[z], index_x_max[z] + 1):
                if (av_data[t, z, y, x] != 0.0 and
                        id_connected_voxel[t, _get_linear_idx(x, y, z, width, depth)] == 0):
                    return x, y, z, t
    return -1, -1, -1, -1


@njit
def _process_connected_neighbors(av_data, id_connected_voxel, pattern_storage,
                                 x, y, z, t, pattern_vals, pattern_len, event_id,
                                 threshold_corr, width, height, depth, time_length,
                                 waiting_list, waiting_count):
    """Process all 26-connected neighbors"""
    neighbors_count = _get_26_neighbors(x, y, z, t, width, height, depth,
                                        time_length, _temp_neighbors)

    added_count = 0
    for n_idx in range(neighbors_count):
        nx, ny, nz, nt = _temp_neighbors[n_idx, 0], _temp_neighbors[n_idx, 1], \
            _temp_neighbors[n_idx, 2], _temp_neighbors[n_idx, 3]

        linear_idx = _get_linear_idx(nx, ny, nz, width, depth)

        if (av_data[nt, nz, ny, nx] != 0.0 and
                id_connected_voxel[nt, linear_idx] == 0):

            # Extract neighbor profile and detect pattern
            neighbor_profile = _compute_profile(av_data, nx, ny, nz)
            n_start, n_len = _detect_pattern_core(neighbor_profile, nt)

            if n_len == 0:
                continue

            # Extract neighbor pattern
            _extract_pattern(neighbor_profile, n_start, n_len, _temp_pattern)

            # Compute cross-correlation
            corr_len = _normalized_cross_correlation(pattern_vals, pattern_len,
                                                     _temp_pattern, n_len,
                                                     _temp_correlation)
            max_corr, _ = _find_max_value(_temp_correlation, corr_len)

            if max_corr > threshold_corr:
                # Add neighbor to group
                id_connected_voxel[nt, linear_idx] = event_id

                # Store pattern
                for p_idx in range(n_len):
                    pattern_storage[nt, linear_idx, p_idx] = _temp_pattern[p_idx]
                pattern_storage[nt, linear_idx, 1000] = n_len  # Store length at end

                # Add to waiting list
                waiting_list[waiting_count, 0] = nx
                waiting_list[waiting_count, 1] = ny
                waiting_list[waiting_count, 2] = nz
                waiting_list[waiting_count, 3] = nt
                waiting_count += 1
                added_count += 1

                # Add all voxels in the pattern
                for p in range(n_len):
                    pt = n_start + p
                    if (pt != nt and pt < time_length and
                            id_connected_voxel[pt, linear_idx] == 0):
                        id_connected_voxel[pt, linear_idx] = event_id

                        # Store pattern
                        for p_idx in range(n_len):
                            pattern_storage[pt, linear_idx, p_idx] = _temp_pattern[p_idx]
                        pattern_storage[pt, linear_idx, 1000] = n_len

                        # Add to waiting list
                        waiting_list[waiting_count, 0] = nx
                        waiting_list[waiting_count, 1] = ny
                        waiting_list[waiting_count, 2] = nz
                        waiting_list[waiting_count, 3] = pt
                        waiting_count += 1
                        added_count += 1

    return waiting_count


@njit
def _change_small_region_ids(small_group, group_size, id_connected_voxel,
                             small_group_ids, width, height, depth, time_length):
    """Change IDs of small regions by merging with largest neighbor"""
    neighbor_counts = nb.typed.Dict.empty(nb.int32, nb.int32)

    # Count neighbor IDs
    for i in range(group_size):
        x, y, z, t = small_group[i, 0], small_group[i, 1], small_group[i, 2], small_group[i, 3]

        # Check 4D neighborhood
        for dt in range(-1, 2):
            nt = t + dt
            if nt < 0 or nt >= time_length:
                continue

            neighbors_count = _get_26_neighbors(x, y, z, nt, width, height, depth,
                                                time_length, _temp_neighbors)

            for n_idx in range(neighbors_count):
                nx, ny, nz = _temp_neighbors[n_idx, 0], _temp_neighbors[n_idx, 1], _temp_neighbors[n_idx, 2]
                linear_idx = _get_linear_idx(nx, ny, nz, width, depth)
                neighbor_id = id_connected_voxel[nt, linear_idx]

                if neighbor_id != 0:
                    # Check if it's not a small group ID
                    is_small = False
                    for j in range(len(small_group_ids)):
                        if neighbor_id == small_group_ids[j]:
                            is_small = True
                            break

                    if not is_small:
                        if neighbor_id in neighbor_counts:
                            neighbor_counts[neighbor_id] += 1
                        else:
                            neighbor_counts[neighbor_id] = 1

    if len(neighbor_counts) > 0:
        # Find most frequent neighbor ID
        max_count = 0
        new_id = 0
        for nid in neighbor_counts:
            if neighbor_counts[nid] > max_count:
                max_count = neighbor_counts[nid]
                new_id = nid

        # Update all voxels in small group
        for i in range(group_size):
            x, y, z, t = small_group[i, 0], small_group[i, 1], small_group[i, 2], small_group[i, 3]
            linear_idx = _get_linear_idx(x, y, z, width, depth)
            id_connected_voxel[t, linear_idx] = new_id

        return True

    return False


@njit
def _compute_final_consecutive_ids(id_connected_voxel, final_event_ids,
                                   width, height, depth, time_length,
                                   index_x_min, index_x_max):
    """Update IDs to be consecutive"""
    # Sort final IDs
    final_ids_sorted = np.sort(final_event_ids)

    for t in range(time_length):
        for z in range(depth):
            for y in range(height):
                for x in range(index_x_min[z], index_x_max[z] + 1):
                    linear_idx = _get_linear_idx(x, y, z, width, depth)
                    current_id = id_connected_voxel[t, linear_idx]

                    if current_id != 0:
                        # Find position in sorted array
                        for i in range(len(final_ids_sorted)):
                            if final_ids_sorted[i] == current_id:
                                desired_id = i + 1
                                if current_id != desired_id:
                                    id_connected_voxel[t, linear_idx] = desired_id
                                break


def find_events(av_data, threshold_size_3d=400, threshold_size_3d_removed=20,
                threshold_corr=0.6):
    """
    Main function to find calcium events

    Args:
        av_data: 4D numpy array (time, depth, height, width)
        threshold_size_3d: minimum group size threshold
        threshold_size_3d_removed: threshold below which groups are removed
        threshold_corr: cross-correlation threshold

    Returns:
        id_connected_voxel: array of connected voxel IDs
        final_event_ids: list of final event IDs
    """
    start_time = time.time()
    print("\nFinding events")

    # Get dimensions
    time_length, depth, height, width = av_data.shape
    total_voxels = depth * height * width

    # Initialize arrays
    id_connected_voxel = np.zeros((time_length, total_voxels), dtype=np.int32)
    pattern_storage = np.zeros((time_length, total_voxels, 1001), dtype=np.float32)  # 1000 for pattern + 1 for length

    # Index bounds (all voxels by default)
    index_x_min = np.zeros(depth, dtype=np.int32)
    index_x_max = np.full(depth, width - 1, dtype=np.int32)

    # Working arrays
    waiting_list = np.zeros((100000, 4), dtype=np.int32)  # Large buffer for waiting list
    small_groups = []
    small_group_ids = []
    final_event_ids = []

    event_id = 1

    # Main loop over time
    for t in range(time_length):
        # Find seed points
        while True:
            x, y, z, seed_t = _find_seed_point(av_data, id_connected_voxel, t,
                                               width, height, depth,
                                               index_x_min, index_x_max)

            if x == -1:  # No more seeds
                break

            # Extract and detect pattern for seed
            profile = _compute_profile(av_data, x, y, z)
            pattern_start, pattern_len = _detect_pattern_core(profile, seed_t)

            if pattern_len == 0:
                continue

            # Extract pattern values
            _extract_pattern(profile, pattern_start, pattern_len, _temp_pattern)

            # Initialize seed
            linear_idx = _get_linear_idx(x, y, z, width, depth)
            id_connected_voxel[seed_t, linear_idx] = event_id

            # Store pattern
            for p_idx in range(pattern_len):
                pattern_storage[seed_t, linear_idx, p_idx] = _temp_pattern[p_idx]
            pattern_storage[seed_t, linear_idx, 1000] = pattern_len

            # Initialize waiting list
            waiting_list[0, 0] = x
            waiting_list[0, 1] = y
            waiting_list[0, 2] = z
            waiting_list[0, 3] = seed_t
            waiting_count = 1

            # Add all voxels in seed pattern
            for p in range(1, pattern_len):
                pt = pattern_start + p
                if (pt < time_length and
                        id_connected_voxel[pt, linear_idx] == 0):
                    id_connected_voxel[pt, linear_idx] = event_id

                    # Store pattern
                    for p_idx in range(pattern_len):
                        pattern_storage[pt, linear_idx, p_idx] = _temp_pattern[p_idx]
                    pattern_storage[pt, linear_idx, 1000] = pattern_len

                    waiting_list[waiting_count, 0] = x
                    waiting_list[waiting_count, 1] = y
                    waiting_list[waiting_count, 2] = z
                    waiting_list[waiting_count, 3] = pt
                    waiting_count += 1

            # Process neighborhood
            waiting_idx = 0
            while waiting_idx < waiting_count:
                wx, wy, wz, wt = (waiting_list[waiting_idx, 0], waiting_list[waiting_idx, 1],
                                  waiting_list[waiting_idx, 2], waiting_list[waiting_idx, 3])

                # Get stored pattern
                w_linear_idx = _get_linear_idx(wx, wy, wz, width, depth)
                w_pattern_len = int(pattern_storage[wt, w_linear_idx, 1000])

                # Extract pattern values
                for p_idx in range(w_pattern_len):
                    _temp_pattern[p_idx] = pattern_storage[wt, w_linear_idx, p_idx]

                # Process neighbors
                waiting_count = _process_connected_neighbors(
                    av_data, id_connected_voxel, pattern_storage,
                    wx, wy, wz, wt, _temp_pattern, w_pattern_len, event_id,
                    threshold_corr, width, height, depth, time_length,
                    waiting_list, waiting_count
                )

                waiting_idx += 1

            # Check group size
            if waiting_count < threshold_size_3d:
                # Store as small group
                small_group = np.zeros((waiting_count, 4), dtype=np.int32)
                for i in range(waiting_count):
                    small_group[i] = waiting_list[i]
                small_groups.append((small_group, waiting_count))
                small_group_ids.append(event_id)
            else:
                final_event_ids.append(event_id)

            event_id += 1

    # Process small groups
    if small_groups:
        small_group_ids_array = np.array(small_group_ids, dtype=np.int32)

        i = 0
        while i < len(small_groups):
            small_group, group_size = small_groups[i]

            changed = _change_small_region_ids(
                small_group, group_size, id_connected_voxel,
                small_group_ids_array, width, height, depth, time_length
            )

            if changed:
                small_groups.pop(i)
                small_group_ids.pop(i)
                small_group_ids_array = np.array(small_group_ids, dtype=np.int32)
            else:
                if group_size < threshold_size_3d_removed:
                    # Remove group
                    for j in range(group_size):
                        x, y, z, t = small_group[j, 0], small_group[j, 1], small_group[j, 2], small_group[j, 3]
                        linear_idx = _get_linear_idx(x, y, z, width, depth)
                        id_connected_voxel[t, linear_idx] = 0
                else:
                    final_event_ids.append(small_group_ids[i])

                small_groups.pop(i)
                small_group_ids.pop(i)
                small_group_ids_array = np.array(small_group_ids, dtype=np.int32)
                continue
            i += 1

    nb_events = len(final_event_ids)
    print(f"There are {nb_events} calcium events in the image!\n")

    # Compute final consecutive IDs
    if final_event_ids:
        final_event_ids_array = np.array(final_event_ids, dtype=np.int32)
        _compute_final_consecutive_ids(id_connected_voxel, final_event_ids_array,
                                       width, height, depth, time_length,
                                       index_x_min, index_x_max)

    end_time = time.time()
    duration = (end_time - start_time) / 60
    print(f"Duration find_events() = {duration:.2f}min")

    return id_connected_voxel, final_event_ids


def detect_pattern(profile, t):
    """Standalone pattern detection function"""
    if profile[t] == 0:
        return None

    pattern = []
    while t > 0 and profile[t - 1] != 0:
        t -= 1

    for i in range(t, len(profile)):
        if profile[i] != 0:
            pattern.append(profile[i])
        else:
            break

    return np.array(pattern, dtype=np.float32) if pattern else None