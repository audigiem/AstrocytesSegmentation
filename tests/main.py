#! /usr/bin/env python3
"""
@file main.py
@brief Entry point, pipeline of astrocytes cells segmentation
"""



from astroca.tools.scene import ImageSequence3DPlusTime
from astroca.tools.loadData import load_data
from astroca.croppingBoundaries.cropper import crop_boundaries
from astroca.croppingBoundaries.computeBoundaries import compute_boundaries
from astroca.varianceStabilization.varianceStabilization import compute_variance_stabilization
from astroca.dynamicImage.dynamicImage import compute_dynamic_image, background_estimation_numba, \
    background_estimation_numpy, background_estimation_single_block, background_estimation_single_block_ultra_optimized
from astroca.parametersNoise.parametersNoise import estimate_std_over_time
from astroca.tools.exportData import export_data
from astroca.activeVoxels.activeVoxelsFinder import find_active_voxels

import os

def main():
    # === Paramètres ===
    input_folder = "/home/matteo/Bureau/INRIA/assets/fewTimeStepScene/"
    pixel_cropped = 10
    moving_window = 7
    x_min = 0
    x_max = 319
    method = "percentile"  # or "min"
    method2 = "Med"       # or "Med"
    percentile = 10.0
    save_results = True
    output_dir = "/home/matteo/Bureau/INRIA/codePython/outputdir/checkDirectory"
    threshold_Zscore = 2.8

    # === Loading ===
    data = load_data(input_folder)  # shape (T, Z, Y, X)
    T, Z, Y, X = data.shape
    print(f"Loaded data of shape: {data.shape}")
    #
    # === Initialization ===
    image_seq = ImageSequence3DPlusTime(data, time_length=T, width=X, height=Y, depth=Z)

    # === Crop + boundaries ===
    crop_boundaries(image_seq, [(0, Z), (pixel_cropped, Y), (x_min, x_max+1)], save_results=save_results, output_directory=output_dir)
    index_xmin, index_xmax, _, = compute_boundaries(image_seq, pixel_cropped=pixel_cropped, save_results=save_results, output_directory=output_dir)

    # === Variance Stabilization ===
    compute_variance_stabilization(image_seq, index_xmin, index_xmax, save_results=save_results, output_directory=output_dir)

    # === F0 estimation ===
    # F0_time_window = background_estimation_numpy(image_seq, index_xmin, index_xmax, moving_window, time_window, method, method2, percentile)
    F0 = background_estimation_single_block(image_seq, index_xmin, index_xmax, moving_window, method, method2, percentile, save_results=save_results, output_directory=output_dir)
    # F0_optimized = background_estimation_single_block_ultra_optimized(image_seq, index_xmin, index_xmax, moving_window, method, method2, percentile)
    # F0_numba = background_estimation_numba(image_seq, index_xmin, index_xmax, moving_window, T, method, method2, percentile)

    print(f"F0 shape: {F0.shape}")

    # === Compute dF and background noise estimation ===
    # dF_time_window, mean_noise_time_window = compute_dynamic_image(image_seq, F0_time_window, index_xmin, index_xmax, time_window)
    dF, mean_noise = compute_dynamic_image(image_seq, F0, index_xmin, index_xmax, T, save_results=save_results, output_directory=output_dir)
    std_noise = estimate_std_over_time(dF, index_xmin, index_xmax)



    print(f"Computation of dF and noise estimation completed. dF shape: {dF.shape}, std_noise: {std_noise:.6f}")

    active_voxels = find_active_voxels(dF, std_noise, mean_noise, threshold_Zscore, index_xmin, index_xmax, radius = (1,1,1), save_results=save_results, output_directory=output_dir)







    

if __name__ == "__main__":
    main()
