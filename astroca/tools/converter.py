"""
@file converter.py
@brief This module provides functionality to convert (3D + time) image sequences fom a series of .tif files to a single .tif file.
"""

import os
import numpy as np
from tifffile import imread, imwrite
from astroca.tools.loadData import load_data
from astroca.tools.exportData import export_data

def convert_tif_series_to_single_tif(input_folder: str, output_path: str, output_file_name: str) -> None:
    """
    @brief Convert a series of .tif files in a folder to a single .tif file.
    @param input_folder: Path to the folder containing the .tif files.
    @param output_path: Path to the directory where the output .tif file will be saved.
    @param output_file_name: Path to the output .tif file.
    """
    image_seq = load_data(input_folder)
    export_data(image_seq, output_path, export_as_single_tif=True, file_name=output_file_name)


def convert_single_tif_to_series(input_file: str, output_folder: str) -> None:
    """
    @brief Convert a single .tif file to a series of .tif files.
    @param input_file: Path to the input .tif file.
    @param output_folder: Path to the folder where the output .tif files will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    data = load_data(input_file)
    export_data(data, output_folder, export_as_single_tif=False)


if __name__ == "__main__":
    input_folder = "/home/matteo/Bureau/INRIA/assets/20steps"
    output_path = "/home/matteo/Bureau/INRIA/assets/"
    output_file_name = "20stepsTimeScene"

    convert_tif_series_to_single_tif(input_folder, output_path, output_file_name)