"""
@file main.py
@brief Entry point, pipeline of astrocytes cells segmentation
"""


import numpy as np
from astroca.init.scene import ImageSequence3DPlusTime
from astroca.init.loadData import load_data_from_file
from astroca.croppingBoundaries.cropper import crop_boundaries
from astroca.croppingBoundaries.computeBoundaries import compute_boundaries
from astroca.varianceStabilization.varianceStabilization import compute_variance_stabilization
from astroca.dynamicImage.dynamicImage import compute_dynamic_image, background_estimation_numba, background_estimation_numpy
from astroca.parametersNoise.parametersNoise import estimate_std_over_time

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
    output_dir = "/home/matteo/Bureau/INRIA/codePython/outputdir/"

    # === 1. Chargement ===
    data = load_data_from_file(input_folder)  # shape (T, Z, Y, X)
    T, Z, Y, X = data.shape
    print(f"Loaded data of shape: {data.shape}")

    # === 2. Initialisation ===
    image_seq = ImageSequence3DPlusTime(data, time_length=T, width=X, height=Y, depth=Z)
    
    # === 3. Détection des bandes vides (cropping) ===
    crop_boundaries(image_seq, [(0, Z), (0, Y), (x_min, x_max+1)])
    index_xmin, index_xmax, _, = compute_boundaries(image_seq, pixel_cropped=pixel_cropped)
    
    
    # === 4. Variance Stabilization ===
    compute_variance_stabilization(image_seq, index_xmin, index_xmax)
    
    # === 5. Estimation du fond F0 ===

    F0 = background_estimation_numpy(image_seq, index_xmin, index_xmax, moving_window, T, method, method2, percentile)
    F0_numba = background_estimation_numba(image_seq, index_xmin, index_xmax, moving_window, T, method, method2, percentile)

    # === 6. Calcul dF = data - F0 et estimation bruit moyen ===
    dF, mean_noise = compute_dynamic_image(image_seq, F0, index_xmin, index_xmax, T)
    # image_seq.set_data(dF)
    
    # === 7. Estimation de l'écart-type du bruit spatial ===
    std_noise = estimate_std_over_time(dF, index_xmin, index_xmax)
    
    # === Résumé ===
    print(f"\n--- Summary ---")
    print(f"mean_noise = {mean_noise:.6f}")
    print(f"std_noise  = {std_noise:.6f}")
    print(f"Processed data shape: {image_seq.get_data().shape}")

if __name__ == "__main__":
    main()
