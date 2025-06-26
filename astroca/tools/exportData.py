"""
@file exportData.py
@brief This module provides functionality to export 3D image sequences with time dimension to various formats.
@detail It includes methods to save the data either as one .tif file that contains all time frames or as multiple .tif files, one for each time frame in a directory.
"""

import os
import numpy as np
from astroca.tools.scene import ImageSequence3DPlusTime
from tifffile import imwrite

def export_data(data: np.ndarray,
                output_path: str,
                export_as_single_tif: bool = True,
                file_name: str = "exported_sequence",
                directory_name: str = "exported_data"):
    """
    Export 3D image sequence data to .tif files, readable as T-Z-X-Y in FIJI.

    @param data: 4D numpy array of shape (T, Z, Y, X) representing the image sequence.
    @param output_path: Path where the exported files will be saved.
    @param export_as_single_tif: If True, saves all time frames in one .tif file; otherwise saves each time frame as a separate .tif file.
    @param file_name: Name of the file to save if exporting as a single .tif file.
    @param directory_name: Name of the directory to save the exported files if exporting multiple .tif files.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if export_as_single_tif:
        T, Z, Y, X = data.shape

        # Reshape into a 5D array for ImageJ: (T, Z, C=1, Y, X)
        data_5d = data[:, :, np.newaxis, :, :]  # (T, Z, 1, Y, X)

        imwrite(
            os.path.join(output_path, file_name + ".tif"),
            data_5d,
            imagej=True,
            metadata={
                'axes': 'TZCYX',
                'Frames': T,
                'Slices': Z,
                'Channels': 1
            }
        )
        print(f"Exported all time frames to {os.path.join(output_path, file_name + '.tif')}")
    else:
        # Export each time frame separately (no ImageJ metadata needed)
        directory_path = os.path.join(output_path, directory_name)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        for t in range(data.shape[0]):
            imwrite(os.path.join(directory_path, f'time_frame_{t}.tif'), data[t])
            print(f"Exported time frame {t} to {os.path.join(directory_path, f'time_frame_{t}.tif')}")


def save_numpy_tab(data: np.ndarray, output_path: str, file_name: str = "exported_data.npy"):
    """
    Save a numpy array to a .npy file.

    @param data: Numpy array to save.
    @param output_path: Path where the .npy file will be saved.
    @param file_name: Name of the .npy file.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    np.save(os.path.join(output_path, file_name), data)
    print(f"Saved numpy array to {os.path.join(output_path, file_name)}")