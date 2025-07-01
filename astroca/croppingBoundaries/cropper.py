"""
@file cropper.py
@brief This module provides functionality to crop boundaries of 3D image sequences with time dimension (if needed).
"""

from astroca.tools.exportData import export_data
import os
import numpy as np

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
    required_keys = {'preprocessing', 'files', 'paths'}
    if not required_keys.issubset(params.keys()):
        raise ValueError(f"Missing required parameters: {required_keys - params.keys()}")
    x_min = int(params['preprocessing']['x_min'])
    x_max = int(params['preprocessing']['x_max'])
    pixel_cropped = int(params['preprocessing']['pixel_cropped'])
    save_results = int(params['files']['save_results']) == 1  # Convert to boolean
    output_directory = params['paths']['output_dir']
    
    
    if len(data.shape) != 4 and len(data.shape) != 3:
        raise ValueError(f"Input data must be a 4D (or 3D) numpy array with shape (T, Z, Y, X) or (Z, Y, X) but got shape {data.shape}.")
    
    T, Z, Y, X = data.shape    

    start_depth, end_depth = (0, Z)  # No cropping in depth
    start_height, end_height = (pixel_cropped, Y)  # Crop pixel_cropped pixels from the top
    start_width, end_width = (x_min, x_max + 1)  # Crop from x_min to x_max (inclusive)

    # for all the frames in the time dimension, perform the cropping
    cropped_data = data[:, start_depth:end_depth, start_height:end_height, start_width:end_width]
    
    print(f"    Cropped data shape: {cropped_data.shape}")

    if save_results:
        if output_directory is None:
            raise ValueError("output_directory must be specified if save_results is True.")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        export_data(cropped_data, output_directory, export_as_single_tif=True, file_name="cropped_image_sequence")
    print()
    
    return cropped_data


