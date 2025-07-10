#! /usr/bin/env python3
"""
@file main.py
@brief Entry point, pipeline of astrocytes cells segmentation
"""
import os

from astroca.tools.loadData import load_data, read_config
from astroca.croppingBoundaries.cropper import crop_boundaries
from astroca.croppingBoundaries.computeBoundaries import compute_boundaries
from astroca.varianceStabilization.varianceStabilization import compute_variance_stabilization
from astroca.dynamicImage.dynamicImage import compute_dynamic_image, compute_image_amplitude
from astroca.dynamicImage.backgroundEstimator import background_estimation_single_block
from astroca.parametersNoise.parametersNoise import estimate_std_over_time
from astroca.activeVoxels.activeVoxelsFinder import find_active_voxels

from astroca.events.eventDetector import detect_calcium_events
from astroca.events.eventDetectorScipy import detect_events, show_results
from astroca.events.eventDetectorAccurate import VoxelGroupingAlgorithm
from astroca.events.eventDetectorCorrected import detect_calcium_events_opti
from astroca.features.featuresComputation import save_features_from_events
import time
import tracemalloc
from typing import List, Dict, Tuple, Any
import sys


def run_pipeline_with_statistics(enable_memory_profiling: bool = False) -> None:
    """
    @brief Run the pipeline with memory and time statistics.
    @param enable_memory_profiling: If True, enables memory profiling using tracemalloc.
    """
    def run_step(name, func, *args, **kwargs) -> Any:
        """
        @brief Run a single step of the pipeline and collect statistics.
        @param name: Name of the step for logging
        @param func: Function to execute for the step
        @param args: Positional arguments for the function
        @param kwargs: Keyword arguments for the function
        @return: Result of the function execution
        """
        if enable_memory_profiling:
            tracemalloc.start()
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        if enable_memory_profiling:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            time_stats[name] = elapsed_time
            memory_stats[name] = peak / 10**6
        else:
            time_stats[name] = elapsed_time
            memory_stats[name] = None

        return result

    time_stats = {}
    memory_stats = {}

    print("=== Starting pipeline with statistics ===")

    # === Configuration ===
    params = run_step("read_config", read_config)
    bool_save_results = int(params['files']['save_results']) == 1

    # === Loading ===
    data = run_step("load_data", load_data, params['paths']['input_folder'])
    T, Z, Y, X = data.shape
    print(f"Loaded data of shape: {data.shape}\n")

    # === Crop + boundaries ===
    cropped_data = run_step("crop_boundaries", crop_boundaries, data, params)
    index_xmin, index_xmax, _, raw_data = run_step("compute_boundaries", compute_boundaries, cropped_data, params)

    # === Variance Stabilization ===
    data = run_step("variance_stabilization", compute_variance_stabilization, raw_data, index_xmin, index_xmax, params)

    # === Background estimation (F0) ===
    F0 = run_step("background_estimation", background_estimation_single_block, data, index_xmin, index_xmax, params)

    # === Compute dF and noise ===
    dF, mean_noise = run_step("compute_dynamic_image", compute_dynamic_image, data, F0, index_xmin, index_xmax, T,
                              params)
    std_noise = run_step("estimate_std_noise", estimate_std_over_time, dF, index_xmin, index_xmax)

    # === Active voxels ===
    active_voxels = run_step("find_active_voxels", find_active_voxels, dF, std_noise, mean_noise, index_xmin,
                             index_xmax, params)

    # === Detect events ===
    id_connections, ids_events = run_step("detect_calcium_events", detect_calcium_events_opti, active_voxels,
                                          params_values=params)

    # === Amplitude ===
    image_amplitude = run_step("compute_image_amplitude", compute_image_amplitude, raw_data, F0, index_xmin,
                               index_xmax, params)

    # === Features ===
    run_step("save_features", save_features_from_events, id_connections, ids_events, image_amplitude,
             params)

    print("\n=== Pipeline completed ===")
    total_time = sum(time_stats.values())
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Save results: {bool_save_results}")
    if enable_memory_profiling:
        for step in time_stats:
            print(f"{step}: {time_stats[step]:.2f} seconds | Peak Memory: {memory_stats[step]:.2f} MB")
    else:
        for step in time_stats:
            print(f"{step}: {time_stats[step]:.2f} seconds")

def run_pipeline():
    # loading parameters from config file
    time_start = time.time()
    params = read_config()

    print("Parameters loaded successfully")

    # === Loading ===
    data = load_data(params['paths']['input_folder'])  # shape (T, Z, Y, X)
    T, Z, Y, X = data.shape
    print(f"Loaded data of shape: {data.shape}")
    # print()

    # === Crop + boundaries ===
    cropped_data = crop_boundaries(data, params)
    index_xmin, index_xmax, _, raw_data = compute_boundaries(cropped_data, params)

    # === Variance Stabilization ===
    data = compute_variance_stabilization(raw_data, index_xmin, index_xmax, params)

    # === F0 estimation ===
    F0 = background_estimation_single_block(data, index_xmin, index_xmax, params)

    # === Compute dF and background noise estimation ===
    dF, mean_noise = compute_dynamic_image(data, F0, index_xmin, index_xmax, T, params)
    std_noise = estimate_std_over_time(dF, index_xmin, index_xmax)

    # === Compute Z-score, closing morphology, median filter ===
    active_voxels = find_active_voxels(dF, std_noise, mean_noise, index_xmin, index_xmax, params)

    # === Detect calcium events ===
    id_connections, ids_events = detect_calcium_events_opti(active_voxels, params_values=params)
    
    # === Compute image amplitude ===
    image_amplitude = compute_image_amplitude(raw_data, F0, index_xmin, index_xmax, params)
    
    # === Compute features ===
    save_features_from_events(id_connections, ids_events, image_amplitude, params_values=params)
    end_time = time.time() - time_start
    bool_save_results = int(params['files']['save_results']) == 1
    print(f"Pipeline completed in {end_time:.2f} {"while saving results" if bool_save_results else "without saving results"}.")


def main():
    """
    @brief Main function to run the pipeline.
    """
    profile_memory = False
    profile_time = False

    if len(sys.argv) > 2:
        raise ValueError(f"Too many arguments. Usage: {sys.argv[0]} [--stats | --memstats]")

    if len(sys.argv) == 2:
        arg = sys.argv[1]
        if arg == "--stats":
            profile_time = True
        elif arg == "--memstats":
            print("WARNING: Memory profiling is enabled, this may slow down the pipeline execution.")
            profile_time = True
            profile_memory = True
        elif arg == "--quiet":
            profile_time = False
            profile_memory = False
            print("Running pipeline in quiet mode, no statistics nor execution trace will be printed.")
            with open(os.devnull, 'w') as devnull:
                sys.stdout = devnull 
                run_pipeline()
                return
            
        elif arg == "--help":
            print(f"Usage: {sys.argv[0]} [--stats | --memstats | --quiet | --help]")
            print("  --stats: Run pipeline with time statistics")
            print("  --memstats: Run pipeline with memory and time statistics")
            print("  --quiet: Run pipeline without statistics and without execution trace")
            print("  --help: Show this help message")
            return       
        else:
            raise ValueError(f"Invalid argument '{arg}'. Usage: {sys.argv[0]} [--stats | --memstats]")

    if profile_time:
        print(f"Running pipeline with {'memory and time' if profile_memory else 'time'} statistics...")
        run_pipeline_with_statistics(enable_memory_profiling=profile_memory)
    else:
        print("Running pipeline without statistics...")
        run_pipeline()

if __name__ == "__main__":
    main()
