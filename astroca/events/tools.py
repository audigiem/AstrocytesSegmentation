"""
@file tools.py
@brief This module provides utility functions for comparing and visualizing 3D volumes, particularly in the context of event detection in astrocyte imaging data.
"""

import numpy as np
import numba as nb
from numba import njit, prange
from typing import List, Tuple, Optional, Any, Dict
import time
import os
from astroca.tools.exportData import export_data
from tqdm import tqdm
from collections import deque


@njit
def find_nonzero_pattern_bounds(
    intensity_profile: np.ndarray, t: int
) -> Tuple[int, int]:
    """
    @fn _find_nonzero_pattern_bounds
    @brief Find start and end indices of non-zero pattern around time t.
    @param intensity_profile 1D numpy array of intensity values
    @param t Time index
    @return Tuple (start, end) indices of the non-zero pattern
    """
    start = t
    while start > 0 and intensity_profile[start - 1] != 0:
        start -= 1

    end = t
    while end < len(intensity_profile) and intensity_profile[end] != 0:
        end += 1

    return start, end


@njit
def compute_ncc_fast(pattern1: np.ndarray, pattern2: np.ndarray) -> np.ndarray:
    """
    @fn _compute_ncc_fast
    @brief Optimized normalized cross-correlation computation.
    @param pattern1 First 1D numpy array
    @param pattern2 Second 1D numpy array
    @return 1D numpy array of normalized cross-correlation values
    """
    vout = np.correlate(pattern1, pattern2, "full")

    auto_corr_v1 = np.dot(pattern1, pattern1)
    auto_corr_v2 = np.dot(pattern2, pattern2)

    den = np.sqrt(auto_corr_v1 * auto_corr_v2)
    if den == 0:
        return np.zeros_like(vout)

    return vout / den


@njit
def compute_max_ncc_fast(pattern1: np.ndarray, pattern2: np.ndarray) -> float:
    """
    @fn _compute_max_ncc_fast
    @brief Optimized computation of ONLY the maximum correlation.
    @param pattern1 First 1D numpy array
    @param pattern2 Second 1D numpy array
    @return Maximum normalized cross-correlation value (float)
    """
    if len(pattern1) == 0 or len(pattern2) == 0:
        return 0.0

    # Pre-compute norms
    norm1 = np.sqrt(np.dot(pattern1, pattern1))
    norm2 = np.sqrt(np.dot(pattern2, pattern2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    # Only compute correlations that could be maximum
    max_corr = 0.0
    min_len = min(len(pattern1), len(pattern2))

    # Check correlation at key positions only
    for offset in range(min_len):
        if offset < len(pattern2):
            # Forward correlation
            end_idx = min(len(pattern1), len(pattern2) - offset)
            if end_idx > 0:
                corr = np.dot(pattern1[:end_idx], pattern2[offset : offset + end_idx])
                corr = corr / (norm1 * norm2)
                if corr > max_corr:
                    max_corr = corr

        if offset > 0 and offset < len(pattern1):
            # Backward correlation
            end_idx = min(len(pattern1) - offset, len(pattern2))
            if end_idx > 0:
                corr = np.dot(pattern1[offset : offset + end_idx], pattern2[:end_idx])
                corr = corr / (norm1 * norm2)
                if corr > max_corr:
                    max_corr = corr

    return max_corr


@njit
def correlation_zero_boundary_conditions(v1, v2):
    nV1 = len(v1)
    nV2 = len(v2)
    size = nV1 + nV2 - 1
    vout = np.zeros(size, dtype=np.float32)

    for n in range(-nV2 + 1, nV1):
        sum_val = 0.0
        for m in range(nV1):
            idx = m - n
            if 0 <= idx < nV2:
                sum_val += v2[idx] * v1[m]
        vout[n + nV2 - 1] = sum_val

    return vout


@njit
def compute_max_ncc_strict(v1, v2):
    # Cross-correlation
    vout = correlation_zero_boundary_conditions(v1, v2)

    # Auto-correlations at t = 0
    auto1 = correlation_zero_boundary_conditions(v1, v1)[len(v1) - 1]
    auto2 = correlation_zero_boundary_conditions(v2, v2)[len(v2) - 1]

    if auto1 == 0.0 or auto2 == 0.0:
        return 0.0

    den = np.sqrt(auto1 * auto2)

    max_corr = 0.0
    for i in range(len(vout)):
        norm_corr = vout[i] / den
        if norm_corr > max_corr:
            max_corr = norm_corr

    return max_corr


@njit
def batch_check_conditions(
    av_frame: np.ndarray, id_frame: np.ndarray, coords: np.ndarray
) -> np.ndarray:
    """
    @fn _batch_check_conditions
    @brief Batch check of conditions for multiple coordinates.
    @param av_frame 3D numpy array of data
    @param id_frame 3D numpy array of event IDs
    @param coords Array of coordinates to check (N, 3)
    @return Boolean mask array indicating valid coordinates
    """
    valid_mask = np.zeros(len(coords), dtype=nb.boolean)

    for i in range(len(coords)):
        z, y, x = coords[i]
        if av_frame[z, y, x] != 0 and id_frame[z, y, x] == 0:
            valid_mask[i] = True

    return valid_mask


@njit
def find_seed_fast(
    frame_data: np.ndarray, id_mask: np.ndarray
) -> Tuple[int, int, int, float]:
    """
    @fn _find_seed_fast
    @brief Fast seed finding using numba.
    @param frame_data 3D numpy array of data
    @param id_mask 3D numpy array of event IDs
    @return Tuple (x, y, z, max_val) of the best seed found
    """
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


@njit
def process_neighbors_batch(
    av_data: np.ndarray,
    id_mask: np.ndarray,
    neighbor_coords: np.ndarray,
    t: int,
    pattern: np.ndarray,
    threshold_corr: float,
) -> np.ndarray:
    """
    @fn _process_neighbors_batch
    @brief Process multiple neighbors in batch for better performance.
    @param av_data 4D numpy array of data
    @param id_mask 4D numpy array of event IDs
    @param neighbor_coords Array of neighbor coordinates (N, 3)
    @param t Time index
    @param pattern 1D numpy array, reference pattern
    @param threshold_corr Correlation threshold
    @return Array of valid neighbors (z, y, x, start, end)
    """
    valid_neighbors = []

    for i in range(neighbor_coords.shape[0]):
        z, y, x = neighbor_coords[i]

        if av_data[t, z, y, x] != 0 and id_mask[t, z, y, x] == 0:
            # Extract intensity profile
            intensity_profile = av_data[:, z, y, x]

            # Find pattern bounds
            start, end = find_nonzero_pattern_bounds(intensity_profile, t)
            if start < end:
                neighbor_pattern = intensity_profile[start:end]

                # Compute correlation
                correlation = compute_ncc_fast(pattern, neighbor_pattern)
                max_corr = np.max(correlation)

                if max_corr > threshold_corr:
                    valid_neighbors.append((z, y, x, start, end))

    return np.array(valid_neighbors)


@njit
def get_valid_neighbors(
    z: int,
    y: int,
    x: int,
    depth: int,
    height: int,
    width: int,
    neighbor_offsets: np.ndarray,
) -> np.ndarray:
    """
    @fn _get_valid_neighbors
    @brief Get valid neighbor coordinates using pre-computed offsets.
    @param z Z coordinate
    @param y Y coordinate
    @param x X coordinate
    @param depth Depth of the volume
    @param height Height of the volume
    @param width Width of the volume
    @param neighbor_offsets Array of neighbor offsets (N, 3)
    @return Array of valid neighbor coordinates (N, 3)
    """
    valid_coords = []

    for i in range(neighbor_offsets.shape[0]):
        dz, dy, dx = neighbor_offsets[i]
        nz, ny, nx = z + dz, y + dy, x + dx

        if 0 <= nz < depth and 0 <= ny < height and 0 <= nx < width:
            valid_coords.append((nz, ny, nx))

    return np.array(valid_coords)
