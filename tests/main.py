"""
@file main.py
@brief Entry point, pipeline of astrocytes cells segmentation
"""


import numpy as np
from astroca.tools.scene import ImageSequence3DPlusTime
from astroca.tools.loadData import load_data
from astroca.croppingBoundaries.cropper import crop_boundaries
from astroca.croppingBoundaries.computeBoundaries import compute_boundaries
from astroca.varianceStabilization.varianceStabilization import compute_variance_stabilization
from astroca.dynamicImage.dynamicImage import compute_dynamic_image, background_estimation_numba, \
    background_estimation_numpy, background_estimation_single_block, background_estimation_single_block_ultra_optimized
from astroca.parametersNoise.parametersNoise import estimate_std_over_time
from astroca.tools.exportData import export_data

import os

def main():
    # === Paramètres ===
    input_folder = "/home/matteo/Bureau/INRIA/assets/fewTimeStepScene/"
    pixel_cropped = 10
    moving_window = 7
    time_window = 10
    x_min = 0
    x_max = 319
    method = "percentile"  # or "min"
    method2 = "Med"       # or "Med"
    percentile = 10.0
    save_results = False
    output_dir = "/home/matteo/Bureau/INRIA/codePython/outputdir/checkDirectory/"

    # === 1. Chargement ===
    data = load_data(input_folder)  # shape (T, Z, Y, X)
    T, Z, Y, X = data.shape
    print(f"Loaded data of shape: {data.shape}")

    # === 2. Initialisation ===
    image_seq = ImageSequence3DPlusTime(data, time_length=T, width=X, height=Y, depth=Z)
    
    # === 3. Détection des bandes vides (cropping) ===
    crop_boundaries(image_seq, [(0, Z), (pixel_cropped, Y), (x_min, x_max+1)])
    if save_results:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        export_data(image_seq.get_data(), output_dir, file_name="cropped_image_sequence", export_as_single_tif=True)
    index_xmin, index_xmax, _, = compute_boundaries(image_seq, pixel_cropped=pixel_cropped)
    if save_results:
        export_data(image_seq.get_data(), output_dir, file_name="cropped_image_sequence_with_boundaries", export_as_single_tif=True)

    
    
    # # === 4. Variance Stabilization ===
    compute_variance_stabilization(image_seq, index_xmin, index_xmax)
    if save_results:
        export_data(image_seq.get_data(), output_dir, file_name="anscombe_transform", export_as_single_tif=True)

    # === 5. Estimation du fond F0 ===

    # F0_time_window = background_estimation_numpy(image_seq, index_xmin, index_xmax, moving_window, time_window, method, method2, percentile)
    F0 = background_estimation_single_block(image_seq, index_xmin, index_xmax, moving_window, method, method2, percentile)
    # F0_optimized = background_estimation_single_block_ultra_optimized(image_seq, index_xmin, index_xmax, moving_window, method, method2, percentile)
    # F0_numba = background_estimation_numba(image_seq, index_xmin, index_xmax, moving_window, T, method, method2, percentile)

    print(f"F0 shape: {F0.shape}")
    if save_results:

        # export_data(F0_time_window, output_dir, file_name="F0_time_window", export_as_single_tif=True)
        export_data(F0, output_dir, file_name="F0_single_block", export_as_single_tif=True)
        # export_data(F0_optimized, output_dir, file_name="F0_single_block_optimized", export_as_single_tif=True)

    # === 6. Calcul dF = data - F0 et estimation bruit moyen ===
    # dF_time_window, mean_noise_time_window = compute_dynamic_image(image_seq, F0_time_window, index_xmin, index_xmax, time_window)
    dF, mean_noise = compute_dynamic_image(image_seq, F0, index_xmin, index_xmax, T)
    if save_results:
        export_data(dF, output_dir, file_name="dF", export_as_single_tif=True)
    #
    # === 7. Estimation de l'écart-type du bruit spatial ===
    std_noise = estimate_std_over_time(dF, index_xmin, index_xmax)



    
    # === Résumé ===
    print(f"\n--- Summary ---")
    print(f"Result with time window = {T}:")
    print(f"mean_noise = {mean_noise:.6f}")
    print(f"std_noise  = {std_noise:.6f}")
    print()
    # print(f"Result with time window = {time_window}:")
    # print(f"mean_noise_time_window = {mean_noise_time_window:.6f}")
    # print(f"std_noise_time_window  = {std_noise_time_window:.6f}")
    # print(f"Processed data shape: {image_seq.get_data().shape}")

if __name__ == "__main__":
    main()
