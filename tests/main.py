#! /usr/bin/env python3
"""
@file main.py
@brief Entry point, pipeline of astrocytes cells segmentation
"""
import os

from astroca.tools.loadData import load_data, read_config
from astroca.croppingBoundaries.cropper import crop_boundaries
from astroca.croppingBoundaries.computeBoundaries import compute_boundaries
from astroca.varianceStabilization.varianceStabilization import (
    compute_variance_stabilization,
)
from astroca.dynamicImage.dynamicImage import (
    compute_dynamic_image,
    compute_image_amplitude,
)
from astroca.dynamicImage.backgroundEstimator import (
    background_estimation_single_block,
    background_estimation_single_block_numba,
)
from astroca.parametersNoise.parametersNoise import (
    estimate_std_over_time,
    estimate_std_over_time_optimized,
)
from astroca.activeVoxels.activeVoxelsFinder import find_active_voxels
from astroca.events.eventDetector import detect_calcium_events_opti
from astroca.features.featuresComputation import save_features_from_events
from astroca.tools.runLogger import RunLogger
import time
import tracemalloc
from typing import List, Dict, Tuple, Any
import sys
import torch


def run_pipeline_with_statistics(enable_memory_profiling: bool = False) -> None:
    """
    @brief Run the pipeline with memory and time statistics.
    @param enable_memory_profiling: If True, enables memory profiling using tracemalloc.
    """

    logger = RunLogger(config_path="config.ini", base_dir="runs")

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

    GPU_AVAILABLE = torch.cuda.is_available()

    print(
        f"=== Starting pipeline with statistics, using {'GPU' if GPU_AVAILABLE else 'CPU'} ===\n"
    )

    # === Configuration ===
    params = run_step("read_config", read_config)
    params["GPU_AVAILABLE"] = 1 if GPU_AVAILABLE else 0

    # === Loading ===
    data = run_step(
        "load_data", load_data, params["paths"]["input_folder"], GPU_AVAILABLE
    )
    T, Z, Y, X = data.shape
    print(f"Loaded data of shape: {data.shape}\n")

    # === Crop + boundaries ===
    cropped_data = run_step("crop_boundaries", crop_boundaries, data, params)
    index_xmin, index_xmax, _, raw_data = run_step(
        "compute_boundaries", compute_boundaries, cropped_data, params
    )

    # === Variance Stabilization ===
    data = run_step(
        "variance_stabilization",
        compute_variance_stabilization,
        raw_data,
        index_xmin,
        index_xmax,
        params,
    )

    # === Background estimation (F0) ===
    F0 = run_step(
        "background_estimation",
        background_estimation_single_block,
        data,
        index_xmin,
        index_xmax,
        params,
    )

    # === Compute dF and noise ===
    dF, mean_noise = run_step(
        "compute_dynamic_image",
        compute_dynamic_image,
        data,
        F0,
        index_xmin,
        index_xmax,
        T,
        params,
    )
    std_noise = run_step(
        "estimate_std_noise",
        estimate_std_over_time,
        dF,
        index_xmin,
        index_xmax,
        GPU_AVAILABLE,
    )

    std_noise = 1.1693237

    # === Active voxels ===
    active_voxels = run_step(
        "find_active_voxels",
        find_active_voxels,
        dF,
        std_noise,
        mean_noise,
        index_xmin,
        index_xmax,
        params,
    )

    # === Detect events ===
    if GPU_AVAILABLE:
        print("GPU is available, forcing event detection on CPU for compatibility.")
        active_voxels = active_voxels.cpu().numpy()
    id_connections, ids_events = run_step(
        "detect_calcium_events",
        detect_calcium_events_opti,
        active_voxels,
        params_values=params,
    )

    # === Amplitude ===
    image_amplitude = run_step(
        "compute_image_amplitude",
        compute_image_amplitude,
        raw_data,
        F0,
        index_xmin,
        index_xmax,
        params,
    )

    # === Features ===
    run_step(
        "save_features",
        save_features_from_events,
        id_connections,
        ids_events,
        image_amplitude,
        params,
    )

    print("\n=== Pipeline completed ===")
    total_time = sum(time_stats.values())
    print(f"Total time: {total_time:.2f} seconds")

    summary = {
        "original shape": f"{T}x{Z}x{Y}x{X}",
        "indexes xmin": index_xmin.tolist(),
        "indexes xmax": index_xmax.tolist(),
        "mean_noise": mean_noise,
        "std_noise": std_noise,
        "number_of_events": ids_events,
        "total_time_sec": round(total_time, 2),
        "steps": {},
        "memory_profiling": enable_memory_profiling,
    }

    for step in time_stats:
        step_time = time_stats[step]
        percent = step_time / total_time * 100
        summary["steps"][step] = {
            "time_seconds": round(step_time, 2),
            "percent": round(percent, 2),
        }
        if enable_memory_profiling:
            mem = memory_stats[step]
            summary["steps"][step]["peak_memory_MB"] = round(mem, 2) if mem else None
            print(f"{step}: {step_time:.2f}s ({percent:.2f}%) | Peak: {mem:.2f} MB")
        else:
            print(f"{step}: {step_time:.2f}s ({percent:.2f}%)")

    print("\n")
    logger.save_summary(summary)


def run_pipeline():
    """
    @fn run_pipeline
    @brief Run the main pipeline with logging support.
    @return None
    """
    logger = RunLogger(config_path="config.ini", base_dir="runs")

    time_start = time.time()
    params = read_config()
    GPU_AVAILABLE = torch.cuda.is_available()
    params["GPU_AVAILABLE"] = 1 if GPU_AVAILABLE else 0

    print("Parameters loaded successfully")

    # === Loading ===
    data = load_data(params["paths"]["input_folder"])  # shape (T, Z, Y, X)
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
    std_noise = estimate_std_over_time(dF, index_xmin, index_xmax, GPU_AVAILABLE)

    # mean_noise = 1.187468
    # std_noise = 1.1693237

    # === Compute Z-score, closing morphology, median filter ===
    active_voxels = find_active_voxels(
        dF, std_noise, mean_noise, index_xmin, index_xmax, params
    )

    # === Detect calcium events (force this step on CPU) ===
    if GPU_AVAILABLE:
        print("GPU is available, forcing event detection on CPU for compatibility.")
        torch_device = torch.device("cpu")
        active_voxels = active_voxels.to(torch_device)
    id_connections, ids_events = detect_calcium_events_opti(
        active_voxels, params_values=params
    )

    # === Compute image amplitude ===
    image_amplitude = compute_image_amplitude(
        raw_data, F0, index_xmin, index_xmax, params
    )

    # === Compute features ===
    save_features_from_events(
        id_connections, ids_events, image_amplitude, params_values=params
    )
    end_time = time.time() - time_start
    print(f"Pipeline completed in {end_time:.2f} seconds.")

    # Save summary with logger
    summary = {
        "original shape": f"{T}x{Z}x{Y}x{X}",
        "indexes xmin": index_xmin.tolist(),
        "indexes xmax": index_xmax.tolist(),
        "mean_noise": mean_noise,
        "std_noise": std_noise,
        "number_of_events": ids_events,
        "total_time_sec": round(end_time, 2),
        "memory_profiling": False,
    }

    logger.save_summary(summary)


def main():
    """
    @brief Main function to run the pipeline.
    """
    profile_memory = False
    profile_time = False

    if len(sys.argv) > 2:
        raise ValueError(
            f"Too many arguments. Usage: {sys.argv[0]} [--stats | --memstats | --quiet | --help]"
        )

    if len(sys.argv) == 2:
        arg = sys.argv[1]
        if arg == "--stats":
            profile_time = True
        elif arg == "--memstats":
            print(
                "WARNING: Memory profiling is enabled, this may slow down the pipeline execution."
            )
            profile_time = True
            profile_memory = True
        elif arg == "--quiet":
            print(
                "Running pipeline in quiet mode, no statistics nor execution trace will be printed."
            )
            with open(os.devnull, "w") as devnull:
                sys.stdout = devnull
                run_pipeline()
                return

        elif arg == "--help":
            print(f"Usage: {sys.argv[0]} [--stats | --memstats | --quiet | --help]")
            print("  --stats: Run pipeline with time statistics")
            print("  --memstats: Run pipeline with memory and time statistics")
            print(
                "  --quiet: Run pipeline without statistics and without execution trace"
            )
            print("  --help: Show this help message")
            return
        else:
            raise ValueError(
                f"Invalid argument '{arg}'. Usage: {sys.argv[0]} [--stats | --memstats]"
            )

    if profile_time:
        print(
            f"Running pipeline with {'memory and time' if profile_memory else 'time'} statistics..."
        )
        run_pipeline_with_statistics(enable_memory_profiling=profile_memory)
    else:
        print("Running pipeline without statistics...")
        run_pipeline()


if __name__ == "__main__":
    main()
