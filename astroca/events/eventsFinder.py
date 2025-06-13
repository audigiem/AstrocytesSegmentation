import numpy as np
from numba import njit, prange, types
from numba.typed import Dict, List

import numpy as np
from numba import njit, prange, types
from numba.typed import Dict, List


# @njit(parallel=True)
def find_events(av: np.ndarray, threshold_size3D: int,
                threshold_size3D_removed: int, threshold_corr: float):
    T, Z, Y, X = av.shape
    id_connected = np.zeros_like(av, dtype=np.int32)
    group_id = np.int32(1)  # Explicitement en int32
    final_ids = List()  # Liste vide typée dynamiquement

    # Conversion explicite des paramètres
    threshold_size3D = np.int32(threshold_size3D)
    threshold_size3D_removed = np.int32(threshold_size3D_removed)
    threshold_corr = np.float32(threshold_corr)

    # Pré-calcul des profils avec vérification des dimensions
    all_profiles = np.empty((X, Y, Z, T), dtype=np.float32)
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                for t in range(T):
                    all_profiles[x, y, z, t] = av[t, z, y, x]

    for t in range(T):
        while True:
            seed = _find_seed_optimized(av, id_connected, t, Z, Y, X)
            if seed[0] == -1:
                break

            x, y, z, t0 = seed
            t0 = np.int32(t0)  # Conversion explicite
            profile = all_profiles[x, y, z]
            pattern_range = _detect_pattern_range(profile, t0)
            if pattern_range[0] == -1:
                continue

            t_start, t_end = pattern_range
            t_start = np.int32(t_start)
            t_end = np.int32(t_end)

            # Vérification des bornes temporelles
            t_curr_max = min(t_end, T)
            for dt in range(t_end - t_start):
                t_curr = t_start + dt
                if t_curr < T:
                    id_connected[t_curr, z, y, x] = group_id

            size = _find_connected_voxels_optimized(
                av, id_connected, all_profiles,
                (np.int32(x), np.int32(y), np.int32(z), t0),
                profile[t_start:t_end].copy(),
                group_id,
                threshold_corr,
                np.int32(T), np.int32(Z), np.int32(Y), np.int32(X),
                t_start, t_end
            )

            if size >= threshold_size3D:
                final_ids.append(np.int32(group_id))  # Conversion explicite
            group_id += np.int32(1)
    _process_groups_optimized(id_connected, final_ids, threshold_size3D_removed, T, Z, Y, X)

    print("Returning id_voxels shape:", id_connected.shape)
    print("Returning events_ids shape:", len(final_ids))
    return id_connected, np.int32(len(final_ids))  # Conversion explicite


# @njit
def _find_seed_optimized(av, id_connected, t0, Z, Y, X):
    max_val = 0.0
    best = (np.int32(-1), np.int32(-1), np.int32(-1), np.int32(-1))
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                val = av[t0, z, y, x]
                if val > max_val and id_connected[t0, z, y, x] == 0:
                    max_val = val
                    best = (np.int32(x), np.int32(y), np.int32(z), np.int32(t0))
    return best


# @njit
def _detect_pattern_range(profile, t):
    t = np.int32(t)
    if profile[t] == 0:
        return (np.int32(-1), np.int32(-1))

    start = t
    while start > 0 and profile[start - 1] != 0:
        start -= 1

    end = t
    while end < len(profile) and profile[end] != 0:
        end += 1

    return (np.int32(start), np.int32(end))


# @njit
def _ncc_optimized(v1, v2):
    mean1 = np.mean(v1)
    mean2 = np.mean(v2)
    v1 = v1 - mean1
    v2 = v2 - mean2
    norm1 = np.sqrt(np.sum(v1 ** 2))
    norm2 = np.sqrt(np.sum(v2 ** 2))
    if norm1 == 0.0 or norm2 == 0.0:
        return np.float32(0.0)
    return np.float32(np.sum(v1 * v2) / (norm1 * norm2))


# @njit
def _find_connected_voxels_optimized(av, id_connected, all_profiles,
                                     seed, pattern, group_id,
                                     threshold_corr, T, Z, Y, X,
                                     t_start, t_end):
    # Utilisation d'une liste avec type explicite
    queue = List()
    queue.append((
        np.int32(seed[0]), np.int32(seed[1]),
        np.int32(seed[2]), np.int32(seed[3])
    ))

    head = np.int32(0)
    count = np.int32(0)

    while head < len(queue):
        x, y, z, t = queue[head]
        head += 1
        count += 1

        for dz in (-1, 0, 1):
            nz = z + dz
            if nz < 0 or nz >= Z:
                continue
            for dy in (-1, 0, 1):
                ny = y + dy
                if ny < 0 or ny >= Y:
                    continue
                for dx in (-1, 0, 1):
                    nx = x + dx
                    if (dx == 0 and dy == 0 and dz == 0) or nx < 0 or nx >= X:
                        continue

                    if av[t, nz, ny, nx] > 0 and id_connected[t, nz, ny, nx] == 0:
                        profile = all_profiles[nx, ny, nz]
                        if profile[t] == 0:
                            continue
                        start, end = _detect_pattern_range(profile, t)
                        if start == -1:
                            continue
                        sub = profile[start:end].copy()
                        if len(sub) != len(pattern):
                            min_len = min(len(sub), len(pattern))
                            corr = _ncc_optimized(pattern[:min_len], sub[:min_len])
                        else:
                            corr = _ncc_optimized(pattern, sub)
                        if corr > threshold_corr:
                            for dt in range(end - start):
                                t_curr = start + dt
                                if t_curr < T:
                                    id_connected[t_curr, nz, ny, nx] = group_id
                            queue.append((np.int32(nx), np.int32(ny), np.int32(nz), np.int32(t)))
    return count


# @njit
def _process_groups_optimized(id_connected, final_ids, threshold_size3D_removed, T, Z, Y, X):
    group_counts = Dict.empty(key_type=types.int32, value_type=types.int32)

    for t in range(T):
        for z in range(Z):
            for y in range(Y):
                for x in range(X):
                    gid = id_connected[t, z, y, x]
                    if gid > 0:
                        group_counts[gid] = group_counts.get(gid, 0) + 1

    small_groups = List.empty_list(types.int32)
    for gid in group_counts:
        if group_counts[gid] < threshold_size3D_removed:
            small_groups.append(gid)

    for i in range(len(small_groups)):
        gid = small_groups[i]
        voxels = List.empty_list(types.int32[:])
        for t in range(T):
            for z in range(Z):
                for y in range(Y):
                    for x in range(X):
                        if id_connected[t, z, y, x] == gid:
                            voxels.append(np.array([x, y, z, t], dtype=np.int32))

        neighbor_counts = Dict.empty(key_type=types.int32, value_type=types.int32)
        for idx in range(len(voxels)):
            x, y, z, t = voxels[idx]
            for dz in (-1, 0, 1):
                nz = z + dz
                if nz < 0 or nz >= Z:
                    continue
                for dy in (-1, 0, 1):
                    ny = y + dy
                    if ny < 0 or ny >= Y:
                        continue
                    for dx in (-1, 0, 1):
                        nx = x + dx
                        if (dx == 0 and dy == 0 and dz == 0) or nx < 0 or nx >= X:
                            continue
                        neighbor_id = id_connected[t, nz, ny, nx]
                        if neighbor_id > 0 and neighbor_id != gid:
                            neighbor_counts[neighbor_id] = neighbor_counts.get(neighbor_id, 0) + 1

        best_neighbor = -1
        max_count = -1
        for neighbor_id in neighbor_counts:
            if neighbor_counts[neighbor_id] > max_count:
                max_count = neighbor_counts[neighbor_id]
                best_neighbor = neighbor_id

        for idx in range(len(voxels)):
            x, y, z, t = voxels[idx]
            if best_neighbor != -1:
                id_connected[t, z, y, x] = best_neighbor
            else:
                id_connected[t, z, y, x] = 0

    # Mise à jour finale des IDs conservés
    valid_final_ids = List.empty_list(types.int32)
    for i in range(len(final_ids)):
        gid = final_ids[i]
        if gid in group_counts and group_counts[gid] >= threshold_size3D_removed:
            valid_final_ids.append(gid)

    final_ids.clear()
    for i in range(len(valid_final_ids)):
        final_ids.append(valid_final_ids[i])
