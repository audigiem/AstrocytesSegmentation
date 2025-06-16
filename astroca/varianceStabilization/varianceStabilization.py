"""
@file varianceStabilization.py
@brief This module performs variance stabilization using the Anscombe transform.
@detail It converts Poisson noise to approximately Gaussian noise with stabilized variance (mean = 0, variance = 1).
The transform is applied only within the meaningful X-boundaries [index_xmin[z], index_xmax[z]] for each Z slice.
"""

import numpy as np
from astroca.tools.scene import ImageSequence3DPlusTime
import os
from astroca.tools.exportData import export_data

def compute_variance_stabilization(image_sequence: ImageSequence3DPlusTime,
                                    index_xmin: np.ndarray,
                                    index_xmax: np.ndarray,
                                    save_results: bool = False,
                                    output_directory: str = None
                                   ):
    """
    @brief Applies the Anscombe variance stabilization transform in-place to the image sequence.
    @param image_sequence: ImageSequence3DPlusTime instance
    @param index_xmin: 1D array of shape (depth,) with left cropping bounds per z
    @param index_xmax: 1D array of shape (depth,) with right cropping bounds per z
    @param save_results: If True, saves the transformed data to the specified output directory
    @param output_directory: Directory to save the transformed data if save_results is True
    """
    print("Applying variance stabilization using Anscombe transform...")
    data = image_sequence.get_data()  # shape (T, Z, Y, X)
    T, Z, Y, X = data.shape

    for z in range(Z):
        x_min = index_xmin[z]
        x_max = index_xmax[z] + 1  # Python exclusive slicing

        if x_min >= x_max:
            continue  # avoid invalid slices

        # Create a slice view for the (T, Y, X) region to be transformed
        # shape: (T, Y, x_max - x_min)
        sub_volume = data[:, z, :, x_min:x_max]

        # Apply Anscombe transform: 2 * sqrt(3/8 + x)
        np.sqrt(sub_volume + 3.0 / 8.0, out=sub_volume)
        sub_volume *= 2.0  # in-place multiplication

    image_sequence.set_data(data)  # Update the data in place
    print("Variance stabilization completed.")

    if save_results:
        if output_directory is None:
            raise ValueError("Output directory must be specified when save_results is True.")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        export_data(data, output_directory, export_as_single_tif=True, file_name="variance_stabilized_sequence")

    print()

def check_variance(image_sequence: ImageSequence3DPlusTime,
                   index_xmin: np.ndarray,
                   index_xmax: np.ndarray) -> bool:
    """
    @brief Checks if the variance of the transformed data is approximately 1.
    @param image_sequence: ImageSequence3DPlusTime instance
    @param index_xmin: 1D array of shape (depth,) with left cropping bounds per z
    @param index_xmax: 1D array of shape (depth,) with right cropping bounds per z
    @return: True if variance is approximately 1, False otherwise
    """
    data = image_sequence.get_data()  # shape (T, Z, Y, X)
    T, Z, Y, X = data.shape

    for z in range(Z):
        x_min = index_xmin[z]
        x_max = index_xmax[z] + 1  # Python exclusive slicing

        if x_min >= x_max:
            continue  # avoid invalid slices

        sub_volume = data[:, z, :, x_min:x_max]
        variance = np.var(sub_volume)

        if not np.isclose(variance, 1.0):
            return False

    return True


def anscombe_inverse(data: np.ndarray, index_xmin: np.ndarray, index_xmax: np.ndarray, save_results: bool = False, output_directory: str = None) -> np.ndarray:
    """
    Compute inverse of Anscombe transform to compute the amplitude of the image
    @param data: 4D numpy array of shape (T, Z, Y, X)
    @param index_xmin: 1D array of shape (depth,) with left cropping bounds per z
    @param index_xmax:1D array of shape (depth,) with right cropping bounds per z
    @param save_results: If True, saves the transformed data to the specified output directory
    @param output_directory: Directory to save the transformed data if save_results is True
    @return:
    """
    print("Applying inverse Anscombe transform...")
    T, Z, Y, X = data.shape

    for z in range(Z):
        x_min = index_xmin[z]
        x_max = index_xmax[z] + 1  # Python exclusive slicing

        if x_min >= x_max:
            continue  # avoid invalid slices

        # Create a slice view for the (T, Y, X) region to be transformed
        sub_volume = data[:, z, :, x_min:x_max]

        # Apply inverse Anscombe transform: (x / 2)^2 - 3/8
        sub_volume /= 2.0
        np.square(sub_volume, out=sub_volume)
        sub_volume -= 3.0 / 8.0

    if save_results:
        if output_directory is None:
            raise ValueError("Output directory must be specified when save_results is True.")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        export_data(data, output_directory, export_as_single_tif=True, file_name="inverse_anscombe_transformed_sequence")
    return data