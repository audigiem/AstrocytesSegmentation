"""
@file cropper.py
@brief This module provides functionality to crop boundaries of 3D image sequences with time dimension (if needed).
"""

from astroca.tools.exportData import export_data
import os
import numpy as np
from typing import Tuple


def detect_null_band_X_dir(data: np.ndarray) -> Tuple[int, int]:
    """
    @fn detect_null_band_X_dir
    @brief Detect the first and last non null band in the X direction of a 4D image sequence.
    @param file_path Path to the 4D image sequence file
    @return Tuple containing the first and last non null band indices in the X direction
    """
    print(" - Detecting null bands in X direction...")
    if len(data.shape) != 4:
        raise ValueError(
            f"Input data must be a 4D numpy array with shape (T, Z, Y, X) but got shape {data.shape}."
        )

    T, Z, Y, X = data.shape

    # Optimized approach: compute sum along T, Z, Y axes for each X slice
    # This creates a 1D array where each element is the sum of all values in that X slice
    x_sums = np.sum(data, axis=(0, 1, 2))

    # Find non-zero indices (bands with data)
    non_zero_indices = np.nonzero(x_sums)[0]

    if len(non_zero_indices) == 0:
        raise ValueError("No non-null bands found in the X direction.")

    first_non_null_band = int(non_zero_indices[0])
    last_non_null_band = int(non_zero_indices[-1])

    print(
        f"    First non-null band: {first_non_null_band}, Last non-null band: {last_non_null_band}"
    )

    return first_non_null_band, last_non_null_band


def crop_boundaries(data: np.ndarray, params: dict) -> np.ndarray:
    """
    @brief Crop the boundaries of a 3D image sequence with time dimension.

    @param data: 4D numpy array of shape (T, Z, Y, X) representing the image sequence.
    @param params: Dictionary containing the cropping parameters:
        - pixel_cropped: Number of pixels to crop from the height dimension.
        - x_min: Minimum x-coordinate for cropping.
        - x_max: Maximum x-coordinate for cropping.
        - save_results: Boolean indicating whether to save the cropped data.
        - output_directory: Directory to save the cropped data if save_results is True.
    @return 4D numpy array of shape (T, Z, Y', X') representing the cropped image sequence,
    where Y' = Y - pixel_cropped and X' = x_max - x_min
    """
    print("=== Cropping boundaries and compute boundaries ===")
    print(" - Cropping the boundaries of the image sequence...")

    # extract necessary parameters
    required_keys = {"preprocessing", "save", "paths"}
    if not required_keys.issubset(params.keys()):
        raise ValueError(
            f"Missing required parameters: {required_keys - params.keys()}"
        )

    try:
        x_min, x_max = detect_null_band_X_dir(data)
    except ValueError as e:
        raise ValueError(f"Error detecting null bands in X direction: {e}")
    pixel_cropped = int(params["preprocessing"]["pixel_cropped"])
    save_results = (
        int(params["save"]["save_cropp_boundaries"]) == 1
    )  # Convert to boolean
    output_directory = params["paths"]["output_dir"]

    if len(data.shape) != 4 and len(data.shape) != 3:
        raise ValueError(
            f"Input data must be a 4D (or 3D) numpy array with shape (T, Z, Y, X) or (Z, Y, X) but got shape {data.shape}."
        )

    T, Z, Y, X = data.shape

    start_depth, end_depth = (0, Z)  # No cropping in depth
    start_height, end_height = (
        pixel_cropped,
        Y,
    )  # Crop pixel_cropped pixels from the top
    start_width, end_width = (x_min, x_max + 1)  # Crop from x_min to x_max (inclusive)

    # for all the frames in the time dimension, perform the cropping
    cropped_data = data[
        :, start_depth:end_depth, start_height:end_height, start_width:end_width
    ]

    print(f"    Cropped data shape: {cropped_data.shape}")

    if save_results:
        if output_directory is None:
            raise ValueError(
                "output_directory must be specified if save_results is True."
            )
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        export_data(
            cropped_data,
            output_directory,
            export_as_single_tif=True,
            file_name="cropped_image_sequence",
        )
    print()

    return cropped_data
