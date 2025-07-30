#! /usr/bin/env python3
"""
@file compareFiles.py
@brief This script compares my output files with the expected output files.
@detail It checks if the files are identical and prints somme statistics about the differences.
"""

import os
import numpy as np
from astroca.tools.loadData import load_data
from tifffile import imwrite
from tests.comparingTools.compareCSVFiles import compare_csv_files


def compute_frame_accuracy(expected_frame: np.ndarray, differences: np.ndarray, percentage_accuracy: float) -> float:
    """
    Compute the accuracy of the output frame compared to the expected frame.

    @param expected_frame: The expected frame data.
    @param differences: The differences between the expected frame and the output frame.
    @param percentage_accuracy: The acceptable percentage difference for the comparison.
    @return: The accuracy of the output frame as a percentage.
    """
    mask = np.abs(expected_frame) > 0
    if np.any(mask):
        relative_diff = differences[mask] / np.abs(expected_frame[mask])
        correct = relative_diff <= percentage_accuracy
        frame_accuracy = np.mean(correct) * 100
    else:
        frame_accuracy = 0.0
    return frame_accuracy

def compare_files(expected_file_path: str, output_file_path: str, percentage_accuracy: float = 0.01, save_diff: bool = False):
    """
    Compare two files and print statistics about their differences.
    
    @param expected_file_path: Path to the expected output file.
    @param output_file_path: Path to the output file to compare.
    @param percentage_accuracy: The acceptable percentage difference for the comparison.
    @param save_diff: If True, saves the differences to a new file.
    @raises FileNotFoundError: If either the expected file or the output file does not exist.
    @raises ValueError: If the shapes of the expected and output files do not match.
    """
    if not os.path.exists(expected_file_path):
        raise FileNotFoundError(f"Expected file {expected_file_path} does not exist.")
    
    if not os.path.exists(output_file_path):
        raise FileNotFoundError(f"Output file {output_file_path} does not exist.")
    
    expected_data = load_data(expected_file_path)
    output_data = load_data(output_file_path)
    
    if output_data.ndim == 3:
        # If the output data is 3D, we need to add a time dimension
        output_data = np.expand_dims(output_data, axis=0)
        
    if expected_data.ndim == 3:
        # If the expected data is 3D, we need to add a time dimension
        expected_data = np.expand_dims(expected_data, axis=0)
    
    if expected_data.shape != output_data.shape:
        raise ValueError(f"Shape mismatch: expected {expected_data.shape}, got {output_data.shape}")
    
    # Calculate differences
    differences = np.abs(expected_data - output_data)
    tag_differences = False
    
    if np.all(differences == 0):
        print("Files are identical.")
    elif np.max(differences) < 1e-5:
        print("Files are similar within a very small margin (less than 1e-5).")
    else:
        tag_differences = True
        print("Files differ.")
        max_diff = np.max(differences)
        mean_diff = np.mean(differences)
        number_of_significant_differences = np.sum(differences > percentage_accuracy * np.abs(expected_data))
        print(f"Max difference: {max_diff}")
        print(f"Mean difference: {mean_diff}")
        print(f"Number of significant differences: {number_of_significant_differences}")

    if save_diff and tag_differences:
        # Save the differences to a new file
        diff_file_name = os.path.basename(output_file_path).replace('.tif', '_diff.tif')
        diff_file_path = os.path.join(os.path.dirname(output_file_path), diff_file_name)
        imwrite(diff_file_path, differences.astype(np.float32))
        print(f"Differences saved to {diff_file_path}")

def compare_sequence(expected_sequence_path: str, output_sequence_path: str, percentage_accuracy: float = 0.01, save_diff: bool = False):
    """
    Compare two sequences of files and print statistics about their differences.

    @param expected_sequence_path: Path to either the directory containing expected output files or the .tif file containing the expected output.
    @param output_sequence_path: Path to either the directory containing output files or the .tif file containing the output.
    @param percentage_accuracy: The acceptable percentage difference for the comparison for each frame.
    @param save_diff: If True, saves the differences to a new .tif file containing all the differences frames.
    @raise FileNotFoundError: If either the expected sequence path or the output sequence path does not exist.
    @raise ValueError: If the shapes of the expected and output sequences do not match.
    """
    if not os.path.exists(expected_sequence_path):
        raise FileNotFoundError(f"Expected sequence path {expected_sequence_path} does not exist.")
    if not os.path.exists(output_sequence_path):
        raise FileNotFoundError(f"Output sequence path {output_sequence_path} does not exist.")

    # load the expected and output sequences from either directories or single .tif files
    expected_data = load_data(expected_sequence_path)
    output_data = load_data(output_sequence_path)

    if expected_data.shape != output_data.shape:
        raise ValueError(f"Shape mismatch: expected {expected_data.shape}, got {output_data.shape}")

    T, Z, Y, X = expected_data.shape
    # for each time frame, compare the expected and output data
    differences = np.zeros_like(expected_data, dtype=np.float32)
    differences_exist = False
    list_max_diff = []
    list_mean_diff = []
    list_accuracy = []
    list_nb_significant_differences = []
    list_diff_frames = []
    list_acceptable_frames = []
    for t in range(T):
        expected_frame = expected_data[t]
        output_frame = output_data[t]

        # Calculate differences
        frame_differences = np.abs(expected_frame - output_frame)
        differences[t] = frame_differences

        if np.all(frame_differences == 0):
            # print(f"Frame {t}: Files are identical.")
            continue
        elif np.max(frame_differences) < 1e-5:
            # print(f"Frame {t}: Files are similar within a very small margin (less than 1e-5).")
            list_acceptable_frames.append(t)
            continue
        else:
            differences_exist = True
            nb_significant_differences = np.sum(frame_differences > percentage_accuracy * np.abs(expected_frame))
            # print(f"Frame {t}: Files differ.")
            list_diff_frames.append(t)
            max_diff = np.max(frame_differences)
            mean_diff = np.mean(frame_differences)
            list_max_diff.append(max_diff)
            list_mean_diff.append(mean_diff)
            list_accuracy.append(compute_frame_accuracy(expected_frame, frame_differences, percentage_accuracy))
            list_nb_significant_differences.append(nb_significant_differences)

    if differences_exist:
        # display global statistics for the differences
        print(f"Time frames where differences were found: {list_diff_frames}")
        print("\nGlobal statistics for differences across all frames:")
        # max_diff = np.max(list_max_diff)
        # mean_max_diff = np.mean(list_max_diff)
        # mean_mean_diff = np.mean(list_mean_diff)
        # mean_percentage_accuracy = np.mean(list_accuracy)
        print(f"Max differences across all frames: {[int(x) for x in list_max_diff]}")
        # print(f"Mean differences across all frames: {[float(x) for x in list_mean_diff]}")
        # print(f"Percentages accuracy across all frames: {[float(x) for x in list_accuracy]}")
        print(f"Number of significant differences across all frames: {[int(x) for x in list_nb_significant_differences]}")
        print(f"Total frames compared: {T}")

        if save_diff:
            # Save the differences to a new file
            diff_file_name = os.path.basename(output_sequence_path).replace('.tif', '_diff.tif')
            diff_file_path = os.path.join(os.path.dirname(output_sequence_path), diff_file_name)
            imwrite(diff_file_path, differences.astype(np.float32))
            print(f"Differences saved to {diff_file_path}")
    else:
        if len(list_acceptable_frames) > 0:
            print(f"All frames are either identical or within the acceptable margin of 1e-5.")
        else:
            print("All frames are identical.")
    return differences

def show_offset_voxels_diff(differences: np.ndarray, expected_sequence_path: str = None, output_sequence_path: str = None):
    """
    Display the coordinates of the voxels that differ between the expected and output data.

    @param differences: A 4D numpy array containing the differences between the expected and output data.
    """
    if differences.ndim != 4:
        raise ValueError("Differences array must be 4D (T, Z, Y, X).")

    expected_vol = load_data(expected_sequence_path)

    output_vol = load_data(output_sequence_path)

    T, Z, Y, X = differences.shape
    diff_coords = np.argwhere(differences > 0.0001)
    
    if diff_coords.size == 0:
        print("No differences found.")
    else:
        print(f"Found {len(diff_coords)} differing voxels:")
        for coord in diff_coords:
            t, z, y, x = coord
            print(f"Time: {t}, Z: {z}, Y: {y}, X: {x}")
            print(f"Expected value: {expected_vol[t, z, y, x]}, Output value: {output_vol[t, z, y, x]}")



def main():
    # Define paths to expected and output files
    EXPECTED_DIR_PATH = "/home/matteo/Bureau/INRIA/codeJava/outputdirTest/"
    OUTPUT_DIR_PATH = "/home/matteo/Bureau/INRIA/codePython/outputdir/testDir/"

    expected_f0_path = EXPECTED_DIR_PATH + "F0.tif"
    output_f0_path = OUTPUT_DIR_PATH + "F0_estimated_numba.tif"

    expected_cropped_path = EXPECTED_DIR_PATH + "data_cropped.tif"
    output_cropped_path = OUTPUT_DIR_PATH + "cropped_image_sequence.tif"

    expected_boundaries_path = EXPECTED_DIR_PATH + "data_boundaries.tif"
    output_boundaries_path = OUTPUT_DIR_PATH + "bounded_image_sequence.tif"

    expected_anscombe_path = EXPECTED_DIR_PATH + "anscombeTransform.tif"
    output_anscombe_path = OUTPUT_DIR_PATH + "variance_stabilized_sequence.tif"

    expected_dF_path = EXPECTED_DIR_PATH + "dF.tif"
    output_dF_path = OUTPUT_DIR_PATH + "dynamic_image_dF.tif"

    expected_Zscore_path = EXPECTED_DIR_PATH + "Zscore.tif"
    output_Zscore_path = OUTPUT_DIR_PATH + "zScore.tif"

    expected_closing_path = EXPECTED_DIR_PATH + "Closing_in_space.tif"
    output_closing_path = OUTPUT_DIR_PATH + "filledSpaceMorphology.tif"

    expected_median_path = EXPECTED_DIR_PATH + "Median.tif"
    output_median_path = OUTPUT_DIR_PATH + "medianFiltered_2.tif"

    expected_active_voxels_path = EXPECTED_DIR_PATH + "AV.tif"
    output_active_voxels_path = OUTPUT_DIR_PATH + "activeVoxels.tif"

    expected_ID_calcium_events_path = EXPECTED_DIR_PATH + "ID_calciumEvents.tif"
    output_ID_calcium_events_path = OUTPUT_DIR_PATH + "ID_calciumEvents.tif"

    expected_anscombe_inverse_path = EXPECTED_DIR_PATH + "F0_inv.tif"
    output_anscombe_inverse_path = OUTPUT_DIR_PATH + "inverse_anscombe_transformed_volume.tif"
    
    expected_amplitude_image_path = EXPECTED_DIR_PATH + "amplitude.tif"
    output_amplitude_image_path = OUTPUT_DIR_PATH + "image_amplitude.tif"
    
    expected_csv_path = EXPECTED_DIR_PATH + "Features.csv"
    output_csv_path = OUTPUT_DIR_PATH + "Features.csv"

    save_results = False
    features_float_precision = 3

    print("Comparing files after each step...")
    print("Step 1: Comparing files after crop and boundaries computations...")
    
    # compare_sequence(expected_cropped_path, output_cropped_path, save_diff=save_results, percentage_accuracy=1e-6)
    compare_sequence(expected_boundaries_path, output_boundaries_path, save_diff=save_results, percentage_accuracy=1e-6)
    print()
    
    # print("Step 2: Comparing files after Anscombe transform...")
    # compare_sequence(expected_anscombe_path, output_anscombe_path, save_diff=save_results, percentage_accuracy=1e-6)
    # print()
    #
    print("Step 3: Comparing files after F0 estimation...")
    compare_files(expected_f0_path, output_f0_path, save_diff=save_results, percentage_accuracy=1e-6)
    print()
    
    # print("Step 4: Comparing files after dF computation...")
    # compare_sequence(expected_dF_path, output_dF_path, save_diff=save_results, percentage_accuracy=1e-6)
    # print()
    #
    # print("Step 5: Comparing files after Z-score computation...")
    # compare_sequence(expected_Zscore_path, output_Zscore_path, save_diff=save_results, percentage_accuracy=1e-6)
    # print()
    #
    # print("Step 6: Comparing files after closing in space...")
    # compare_sequence(expected_closing_path, output_closing_path, save_diff=save_results, percentage_accuracy=1e-6)
    # print()
    #
    # print("Step 7: Comparing files after median filtering...")
    # compare_sequence(expected_median_path, output_median_path, save_diff=save_results, percentage_accuracy=1e-6)
    # print()
    #
    # print("Step 8: Comparing files after active voxels detection...")
    # compare_sequence(expected_active_voxels_path, output_active_voxels_path, save_diff=save_results, percentage_accuracy=1e-6)
    # print()
    
    print("Step 9: Comparing files after calcium events detection...")
    compare_sequence(expected_ID_calcium_events_path, output_ID_calcium_events_path, save_diff=save_results, percentage_accuracy=1e-6)
    print()

    # print("Step 10: Comparing files after Anscombe inverse transform...")
    # compare_files(expected_anscombe_inverse_path, output_anscombe_inverse_path, save_diff=save_results, percentage_accuracy=1e-6)
    # print()
    
    print("Step 11: Comparing files after amplitude computation...")
    compare_sequence(expected_amplitude_image_path, output_amplitude_image_path, save_diff=save_results, percentage_accuracy=1e-6)
    print()
    
    print(f"Step 12: Comparing CSV files for features with a float precision of {10 ** -features_float_precision}...")
    compare_csv_files(expected_csv_path, output_csv_path, float_precision=features_float_precision)
    print()
    
    print("All comparisons completed.")

def compare_worklow():
    """
    Compare the output of the workflow with the expected output.
    This function assumes that the expected output files are in a specific directory.
    """
    expected_dir = "/home/matteo/Bureau/INRIA/codePython/outputdir/checkDirectoryFewerTime/"
    output_dir = "/home/matteo/Bureau/INRIA/workflows/astrocaWorkflow/Data/"
    
    expected_files = [
        "bounded_image_sequence.tif",
        "variance_stabilized_sequence.tif",
        "F0_estimated.tif",
        "dynamic_image_dF.tif",
        "zScore.tif",
        "filledSpaceMorphology.tif",
        "medianFiltered_2.tif",
        "activeVoxels.tif",      
    ]
    
    output_files = [
        "Boundaries computation/",
        "Anscombe/",
        "Baseline fluorescence estimation/",
        "Dynamic  image/",
        "Zscore/",
        "Space closing/",
        "Median  filter/",
        "Av finder/",
    ]
    
    for expected_file, output_file in zip(expected_files, output_files):
        expected_path = os.path.join(expected_dir, expected_file)
        output_path = os.path.join(output_dir, output_file)
        
        if not os.path.exists(expected_path):
            print(f"Expected file {expected_path} does not exist.")
            continue
        
        if not os.path.exists(output_path):
            print(f"Output file {output_path} does not exist.")
            continue
        if expected_file == "F0_estimated.tif":
            print(f"Comparing {expected_file} with {output_file}...")
            compare_files(expected_path, output_path, save_diff=False, percentage_accuracy=1e-6)
            print()
        else:
            try:
                print(f"Comparing {expected_file} with {output_file}...")
                compare_sequence(expected_path, output_path, save_diff=False, percentage_accuracy=1e-6)
                print()
            except Exception as e:
                print(f"Error comparing {expected_file} with {output_file}: {e}")
    print("All comparisons completed.")

if __name__ == "__main__":
    main()