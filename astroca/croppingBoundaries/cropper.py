"""
@file cropper.py
@brief This module provides functionality to crop boundaries of 3D image sequences with time dimension (if needed).
"""

from astroca.tools.scene import ImageSequence3DPlusTime
from astroca.tools.exportData import export_data
import os

def crop_boundaries(image_sequence: ImageSequence3DPlusTime, coordinate_range_per_dimension: list, save_results: bool = False, output_directory: str = None) -> None:
    """
    @brief Crop the boundaries of a 3D image sequence with time dimension in place.

    @param image_sequence: An instance of ImageSequence3DPlusTime containing the image data.
    @param coordinate_range_per_dimension: A list containing tuples for each dimension (start, end).
                                          The length of this list should match the number of spatial dimensions in the data.
    @param save_results: If True, the cropped data will be saved to the image_sequence object.
    @param output_directory: Directory where the cropped data will be saved if save_results is True.
    @raises ValueError: If coordinate_range_per_dimension does not contain exactly 3 tuples.
    """
    print("Cropping boundaries...")
    
    if len(coordinate_range_per_dimension) != 3:
        raise ValueError("coordinate_range_per_dimension must contain exactly 3 tuples for (depth, height, width).")

    start_depth, end_depth = coordinate_range_per_dimension[0]
    start_height, end_height = coordinate_range_per_dimension[1]
    start_width, end_width = coordinate_range_per_dimension[2]

    # for all the frames in the time dimension, perform the cropping
    cropped_data = image_sequence.get_data()[:, start_depth:end_depth, start_height:end_height, start_width:end_width]
    cropped_time_length, cropped_depth, cropped_height, cropped_width = cropped_data.shape

    # modify the image sequence in place
    image_sequence.set_data(cropped_data)
    image_sequence.set_dimensions(cropped_time_length, cropped_depth, cropped_height, cropped_width)
    print(f"Cropped data shape: {cropped_data.shape}")
    print()

    if save_results:
        if output_directory is None:
            raise ValueError("output_directory must be specified if save_results is True.")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        export_data(cropped_data, output_directory, export_as_single_tif=True, file_name="cropped_image_sequence")
        print(f"Cropped data saved to {output_directory}/cropped_image_sequence.tif")



