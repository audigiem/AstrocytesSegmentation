#! /usr/bin/env python3
"""
@file main.py
@brief Entry point, pipeline of astrocytes cells segmentation
"""
from networkx.algorithms.components import is_connected
from skimage.filters.rank import threshold
import os
# os.environ['NUMBA_THREADING_LAYER'] = 'tbb'

from astroca.tools.scene import ImageSequence3DPlusTime
from astroca.tools.loadData import load_data, read_config
from astroca.croppingBoundaries.cropper import crop_boundaries
from astroca.croppingBoundaries.computeBoundaries import compute_boundaries
from astroca.varianceStabilization.varianceStabilization import compute_variance_stabilization
from astroca.dynamicImage.dynamicImage import compute_dynamic_image, background_estimation_single_block, compute_image_amplitude
from astroca.parametersNoise.parametersNoise import estimate_std_over_time
from astroca.activeVoxels.activeVoxelsFinder import find_active_voxels

from astroca.events.eventDetector import detect_calcium_events
from astroca.events.eventDetectorPreCompute import detect_calcium_events_ultra_optimized
from astroca.events.eventDetectorPreComputeSafe import detect_calcium_events_safe
from astroca.events.eventDetectorScipy import detect_events, show_results

from astroca.features.featuresComputation import save_features_from_events



def main():

    # loading parameters from config file
    params = read_config()

    print("Parameters loaded successfully")


    # === Loading ===
    data = load_data(params['paths']['input_folder'])  # shape (T, Z, Y, X)
    T, Z, Y, X = data.shape
    print(f"Loaded data of shape: {data.shape}")
    print()

    # === Crop + boundaries ===
    cropped_data = crop_boundaries(data, params)
    index_xmin, index_xmax, _, data = compute_boundaries(cropped_data, params)

    # === Variance Stabilization ===
    data = compute_variance_stabilization(data, index_xmin, index_xmax, params)

    # === F0 estimation ===
    F0 = background_estimation_single_block(data, index_xmin, index_xmax, params)

    # === Compute dF and background noise estimation ===
    dF, mean_noise = compute_dynamic_image(data, F0, index_xmin, index_xmax, T, params)
    std_noise = estimate_std_over_time(dF, index_xmin, index_xmax)

    # === Compute Z-score, closing morphology, median filter ===
    active_voxels = find_active_voxels(dF, std_noise, mean_noise, index_xmin, index_xmax, params)



    # active_voxels = load_data("/home/matteo/Bureau/INRIA/codeJava/outputdirFewerTime/AV.tif")
    # === Detect calcium events ===
    # id_connected_voxels, events_ids = detect_calcium_events(active_voxels, params_values=params)
    # id_connected_voxels, events_ids = detect_calcium_events_ultra_optimized(active_voxels, params_values=params)
    # id_connected_voxels, events_ids = detect_calcium_events_safe(active_voxels, params_values=params)
    # id_connected_voxels = detect_events(active_voxels, params_values=params)
    # events_ids = list(set(id_connected_voxels.flatten()) - {0})  # Exclude background label (0)
    # show_results(id_connected_voxels)
    
    # # === Compute image amplitude ===
    # image_amplitude = compute_image_amplitude(cropped_data, index_xmin, index_xmax, save_results=save_results, output_directory=output_folder)

    # # === Compute features ===
    # save_features_from_events(id_connected_voxels, events_ids, image_amplitude, params_values=params['features_extraction'], save_result=save_results, output_directory=output_folder)

    

if __name__ == "__main__":
    main()
