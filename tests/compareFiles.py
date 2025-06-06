"""
@file compareFiles.py
@brief This script compares my output files with the expected output files.
@detail It checks if the files are identical and prints somme statistics about the differences.
"""

import os
import numpy as np
from astroca.tools.loadData import load_data
from tifffile import imwrite

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
    
    if expected_data.shape != output_data.shape:
        raise ValueError(f"Shape mismatch: expected {expected_data.shape}, got {output_data.shape}")
    
    # Calculate differences
    differences = np.abs(expected_data - output_data)
    tag_differences = False
    
    if np.all(differences == 0):
        print("Files are identical.")
    elif np.all(differences < percentage_accuracy * np.abs(expected_data)):
        print(f"Files are similar within the acceptable percentage ({percentage_accuracy * 100:.6f}%).")
    elif np.max(differences) < 1e-5:
        print("Files are similar within a very small margin (less than 1e-5).")
    else:
        tag_differences = True
        print("Files differ.")
        max_diff = np.max(differences)
        mean_diff = np.mean(differences)

        # compute the number of voxels that differ by more than the acceptable percentage
        threshold = percentage_accuracy * np.abs(expected_data)
        significant_differences = np.sum(differences > threshold)
        total_voxels = expected_data.size
        percentage_differences = (significant_differences / total_voxels) * 100
        print(f"Percentage of voxels differing by more than {percentage_accuracy * 100:.6f}%: {percentage_differences:.6f}%")
        print(f"Max difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")

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
    for t in range(T):
        expected_frame = expected_data[t]
        output_frame = output_data[t]

        # Calculate differences
        frame_differences = np.abs(expected_frame - output_frame)
        differences[t] = frame_differences

        if np.all(frame_differences == 0):
            print(f"Frame {t}: Files are identical.")

        elif np.all(frame_differences < percentage_accuracy * np.abs(expected_frame)):
            print(f"Frame {t}: Files are similar within the acceptable percentage ({percentage_accuracy * 100:.6f}%).")
        elif np.max(frame_differences) < 1e-5:
            print(f"Frame {t}: Files are similar within a very small margin (less than 1e-5).")
        else:
            differences_exist = True
            print(f"Frame {t}: Files differ.")
            max_diff = np.max(frame_differences)
            mean_diff = np.mean(frame_differences)

            # compute the number of voxels that differ by more than the acceptable percentage
            threshold = percentage_accuracy * np.abs(expected_frame)
            significant_differences = np.sum(frame_differences > threshold)
            total_voxels = expected_frame.size
            percentage_differences = (significant_differences / total_voxels) * 100
            print(f"Frame {t}: Percentage of voxels differing by more than {percentage_accuracy * 100:.6f}%: {percentage_differences:.6f}%")
            print(f"Frame {t}: Max difference: {max_diff:.6f}")
            print(f"Frame {t}: Mean difference: {mean_diff:.6f}")
    if save_diff and differences_exist:
        # Save the differences to a new file
        diff_file_name = os.path.basename(output_sequence_path).replace('.tif', '_diff.tif')
        diff_file_path = os.path.join(os.path.dirname(output_sequence_path), diff_file_name)
        imwrite(diff_file_path, differences.astype(np.float32))
        print(f"Differences saved to {diff_file_path}")
    elif not differences_exist:
        print("No differences found in any frame.")






if __name__ == "__main__":
    # Define paths to expected and output files
    expected_f0_path = "/home/matteo/Bureau/INRIA/codeJava/outputdir/F0.tif"
    output_f0_path = "/home/matteo/Bureau/INRIA/codePython/outputdir/checkDirectory/F0_single_block.tif"

    expected_cropped_path = "/home/matteo/Bureau/INRIA/codeJava/outputdir/data_cropped.tif"
    output_cropped_path = "/home/matteo/Bureau/INRIA/codePython/outputdir/checkDirectory/cropped_image_sequence.tif"

    expected_boundaries_path = "/home/matteo/Bureau/INRIA/codeJava/outputdir/data_boundaries.tif"
    output_boundaries_path = "/home/matteo/Bureau/INRIA/codePython/outputdir/checkDirectory/cropped_image_sequence_with_boundaries.tif"

    expected_anscombe_path = "/home/matteo/Bureau/INRIA/codeJava/outputdir/anscombeTransform.tif"
    output_anscombe_path = "/home/matteo/Bureau/INRIA/codePython/outputdir/checkDirectory/anscombe_transform.tif"
    
    expected_dF_path = "/home/matteo/Bureau/INRIA/codeJava/outputdir/dF.tif"
    output_dF_path = "/home/matteo/Bureau/INRIA/codePython/outputdir/checkDirectory/dF.tif"

    print("Comparing files after each step...")
    print("Step 1: Comparing files after crop and boundaries computations...")

    compare_sequence(expected_cropped_path, output_cropped_path, save_diff=False, percentage_accuracy=1e-6)
    compare_sequence(expected_boundaries_path, output_boundaries_path, save_diff=False, percentage_accuracy=1e-6)
    print()

    print("Step 2: Comparing files after Anscomb transform...")
    compare_sequence(expected_anscombe_path, output_anscombe_path, save_diff=False, percentage_accuracy=1e-6)
    print()

    print("Step 3: Comparing files after F0 estimation...")
    compare_files(expected_f0_path, output_f0_path, save_diff=False, percentage_accuracy=1e-6)
    print()
    
    print("Step 4: Comparing files after dF computation...")
    compare_files(expected_dF_path, output_dF_path, save_diff=False, percentage_accuracy=1e-6)
    print()
    
    print("All comparisons completed.")