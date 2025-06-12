#! /usr/bin/env python3
"""
@file main.py
@brief Entry point, pipeline of astrocytes cells segmentation
"""
from skimage.filters.rank import threshold

from astroca.tools.scene import ImageSequence3DPlusTime
from astroca.tools.loadData import load_data, read_config
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

    # loading parameters from config file
    params = read_config()

    input_folder = params['paths']['input_folder']
    output_folder = params['paths']['output_dir']
    pixel_cropped = int(params['preprocessing']['pixel_cropped'])
    x_min = int(params['preprocessing']['x_min'])
    x_max = int(params['preprocessing']['x_max'])
    moving_window = int(params['background_estimation']['moving_window'])
    method = params['background_estimation']['method']
    method2 = params['background_estimation']['method2']
    percentile = float(params['background_estimation']['percentile'])
    threshold_Zscore = float(params['processing']['threshold_zscore'])
    save_results = bool(params['processing']['save_results'])

    print("Parameters loaded successfully")


    # === Loading ===
    data = load_data(input_folder)  # shape (T, Z, Y, X)
    T, Z, Y, X = data.shape
    print(f"Loaded data of shape: {data.shape}")
    print()

    # === Initialization ===
    image_seq = ImageSequence3DPlusTime(data, time_length=T, width=X, height=Y, depth=Z)

    # === Crop + boundaries ===
    crop_boundaries(image_seq, [(0, Z), (pixel_cropped, Y), (x_min, x_max+1)], save_results=save_results, output_directory=output_folder)
    index_xmin, index_xmax, _, = compute_boundaries(image_seq, pixel_cropped=pixel_cropped, save_results=save_results, output_directory=output_folder)

    # === Variance Stabilization ===
    compute_variance_stabilization(image_seq, index_xmin, index_xmax, save_results=save_results, output_directory=output_folder)

    # === F0 estimation ===
    # F0_time_window = background_estimation_numpy(image_seq, index_xmin, index_xmax, moving_window, time_window, method, method2, percentile)
    F0 = background_estimation_single_block(image_seq, index_xmin, index_xmax, moving_window, method, method2, percentile, save_results=save_results, output_directory=output_folder)
    # F0_optimized = background_estimation_single_block_ultra_optimized(image_seq, index_xmin, index_xmax, moving_window, method, method2, percentile)
    # F0_numba = background_estimation_numba(image_seq, index_xmin, index_xmax, moving_window, T, method, method2, percentile)

    # === Compute dF and background noise estimation ===
    # dF_time_window, mean_noise_time_window = compute_dynamic_image(image_seq, F0_time_window, index_xmin, index_xmax, time_window)
    dF, mean_noise = compute_dynamic_image(image_seq, F0, index_xmin, index_xmax, T, save_results=save_results, output_directory=output_folder)
    std_noise = estimate_std_over_time(dF, index_xmin, index_xmax)


    active_voxels = find_active_voxels(dF, std_noise, mean_noise, threshold_Zscore, index_xmin, index_xmax, radius = (1,1,1), save_results=save_results, output_directory=output_folder)







    

if __name__ == "__main__":
    main()
