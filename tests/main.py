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
from astroca.events.eventDetectorOptimized import detect_calcium_events_optimized
from astroca.events.eventDetectorPreCompute import detect_calcium_events_ultra_optimized
from astroca.events.eventDetectorPreComputeSafe import detect_calcium_events_safe
from astroca.features.featuresComputation import save_features_from_events



def main():

    # loading parameters from config file
    params = read_config()

    # input_folder = params['paths']['input_folder']
    # output_folder = params['paths']['output_dir']
    #
    # save_results = int(params['files']['save_results'])
    # if save_results == 0:
    #     save_results = False
    # else :
    #     save_results = True
    #
    # pixel_cropped = int(params['preprocessing']['pixel_cropped'])
    # x_min = int(params['preprocessing']['x_min'])
    # x_max = int(params['preprocessing']['x_max'])
    #
    # print("Parameters loaded successfully")
    #
    #
    # # === Loading ===
    # data = load_data(input_folder)  # shape (T, Z, Y, X)
    # T, Z, Y, X = data.shape
    # print(f"Loaded data of shape: {data.shape}")
    # print()
    #
    # # === Initialization ===
    # image_seq = ImageSequence3DPlusTime(data, time_length=T, width=X, height=Y, depth=Z)
    #
    # # === Crop + boundaries ===
    # crop_boundaries(image_seq, [(0, Z), (pixel_cropped, Y), (x_min, x_max+1)], save_results=save_results, output_directory=output_folder)
    # data_cropped = image_seq.get_data()  # shape (T, Z, Y, X)
    # index_xmin, index_xmax, _, = compute_boundaries(image_seq, pixel_cropped=pixel_cropped, save_results=save_results, output_directory=output_folder)
    #
    # # === Variance Stabilization ===
    # compute_variance_stabilization(image_seq, index_xmin, index_xmax, save_results=save_results, output_directory=output_folder)
    #
    # # === F0 estimation ===
    # F0 = background_estimation_single_block(image_seq, index_xmin, index_xmax, params_values=params['background_estimation'], save_results=save_results, output_directory=output_folder)
    #
    # # === Compute dF and background noise estimation ===
    # dF, mean_noise = compute_dynamic_image(image_seq, F0, index_xmin, index_xmax, T, save_results=save_results, output_directory=output_folder)
    # std_noise = estimate_std_over_time(dF, index_xmin, index_xmax)
    #
    # # === Compute Z-score, closing morphology, median filter ===
    # active_voxels = find_active_voxels(dF, std_noise, mean_noise, index_xmin, index_xmax, params_values=params['active_voxels'], save_results=save_results, output_directory=output_folder)
    #


    active_voxels = load_data("/home/matteo/Bureau/INRIA/codeJava/outputdirFewerTime/AV.tif")
    # === Detect calcium events ===
    # id_connected_voxels, events_ids = detect_calcium_events(active_voxels, params_values=params)
    # id_connected_voxels, events_ids = detect_calcium_events_optimized(active_voxels, params_values=params)
    # id_connected_voxels, events_ids = detect_calcium_events_ultra_optimized(active_voxels, params_values=params)
    id_connected_voxels, events_ids = detect_calcium_events_safe(active_voxels, params_values=params)
    # labels, stats = detect_events_4d(active_voxels, params=params)

    # # === Compute image amplitude ===
    # image_amplitude = compute_image_amplitude(data_cropped, index_xmin, index_xmax, save_results=save_results, output_directory=output_folder)
    #
    # # === Compute features ===
    # save_features_from_events(id_connected_voxels, events_ids, image_amplitude, params_values=params['features_extraction'], save_result=save_results, output_directory=output_folder)

    

if __name__ == "__main__":
    main()
