#!/usr/bin/env python3
"""
@file main.py
@brief Entry point, pipeline of astrocytes cells segmentation
"""
import os
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional, Callable
from contextlib import contextmanager

import torch

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
from astroca.dynamicImage.backgroundEstimator import background_estimation_single_block
from astroca.parametersNoise.parametersNoise import estimate_std_over_time
from astroca.activeVoxels.activeVoxelsFinder import find_active_voxels
from astroca.events.eventDetector import detect_calcium_events_opti
from astroca.features.featuresComputation import save_features_from_events
from astroca.tools.runLogger import RunLogger


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution"""

    enable_memory_profiling: bool = False
    enable_time_profiling: bool = False
    quiet_mode: bool = False
    config_path: str = "config.ini"
    base_dir: str = "runs"


@dataclass
class PipelineStats:
    """Statistics collector for pipeline execution"""

    time_stats: Dict[str, float] = field(default_factory=dict)
    memory_stats: Dict[str, Optional[float]] = field(default_factory=dict)
    export_threads: List[Any] = field(default_factory=list)

    def add_step_stats(
        self, name: str, elapsed_time: float, peak_memory: Optional[float] = None
    ):
        """Add statistics for a pipeline step"""
        self.time_stats[name] = elapsed_time
        self.memory_stats[name] = peak_memory

    def get_total_time(self) -> float:
        """Get total execution time"""
        return sum(self.time_stats.values())

    def print_summary(self, enable_memory_profiling: bool = False):
        """Print execution summary"""
        total_time = self.get_total_time()
        print(f"\nTotal time: {total_time:.2f} seconds")

        for step, step_time in self.time_stats.items():
            percent = step_time / total_time * 100 if total_time > 0 else 0
            if enable_memory_profiling and self.memory_stats[step] is not None:
                mem = self.memory_stats[step]
                print(f"{step}: {step_time:.2f}s ({percent:.2f}%) | Peak: {mem:.2f} MB")
            else:
                print(f"{step}: {step_time:.2f}s ({percent:.2f}%)")

    def to_dict(
        self,
        original_shape: Tuple[int, ...],
        index_xmin,
        index_xmax,
        mean_noise,
        std_noise,
        ids_events,
        enable_memory_profiling: bool,
    ) -> Dict[str, Any]:
        """Convert stats to dictionary for logging"""
        total_time = self.get_total_time()
        summary = {
            "original shape": f"{original_shape[0]}x{original_shape[1]}x{original_shape[2]}x{original_shape[3]}",
            "indexes xmin": index_xmin.tolist(),
            "indexes xmax": index_xmax.tolist(),
            "mean_noise": mean_noise,
            "std_noise": std_noise,
            "number_of_events": ids_events,
            "total_time_sec": round(total_time, 2),
            "steps": {},
            "memory_profiling": enable_memory_profiling,
        }

        for step, step_time in self.time_stats.items():
            percent = step_time / total_time * 100 if total_time > 0 else 0
            step_data = {
                "time_seconds": round(step_time, 2),
                "percent": round(percent, 2),
            }
            if enable_memory_profiling and self.memory_stats[step] is not None:
                step_data["peak_memory_MB"] = round(self.memory_stats[step], 2)
            summary["steps"][step] = step_data

        return summary


class PipelineExecutor:
    """Main pipeline executor class"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = RunLogger(
            config_path=config.config_path, base_dir=config.base_dir
        )
        self.stats = PipelineStats()
        self.gpu_available = torch.cuda.is_available()

    @contextmanager
    def _step_profiler(self, step_name: str):
        """Context manager for profiling individual steps"""
        if self.config.enable_memory_profiling:
            tracemalloc.start()

        start_time = time.time()

        try:
            yield
        finally:
            elapsed_time = time.time() - start_time
            peak_memory = None

            if self.config.enable_memory_profiling:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                peak_memory = peak / 10**6  # Convert to MB

            self.stats.add_step_stats(step_name, elapsed_time, peak_memory)

    def _run_step(self, name: str, func: Callable, *args, **kwargs) -> Any:
        """Run a single pipeline step with profiling"""
        with self._step_profiler(name):
            return func(*args, **kwargs)

    def _track_export_thread(self, obj: Any, step_name: str):
        """Track export thread if object has one"""
        if hasattr(obj, "_export_thread") and obj._export_thread:
            self.stats.export_threads.append(obj._export_thread)
            if not self.config.quiet_mode:
                print(f"Added {step_name} export thread to tracking list")

    def _setup_gpu_config(self, params: Dict[str, Any]) -> bool:
        """Configure GPU availability based on parameters"""
        if params["general"]["execution_mode"] == "cpu":
            if not self.config.quiet_mode:
                print("Forcing CPU execution mode.")
            return False
        return self.gpu_available

    def _wait_for_export_threads(self):
        """Wait for all export threads to complete"""
        if not self.stats.export_threads:
            if not self.config.quiet_mode:
                print("No async export threads to wait for")
            return

        if not self.config.quiet_mode:
            print("=== Waiting for all exports to complete ===")

        for i, thread in enumerate(self.stats.export_threads):
            if thread and thread.is_alive():
                if not self.config.quiet_mode:
                    print(
                        f"  Waiting for export thread {i + 1}/{len(self.stats.export_threads)}..."
                    )
                thread.join(timeout=300)  # 5 minutes timeout

                if thread.is_alive():
                    if not self.config.quiet_mode:
                        print(
                            f"  ⚠️ Warning: Thread {i + 1} still running after timeout"
                        )
                else:
                    if not self.config.quiet_mode:
                        print(f"  ✓ Thread {i + 1} completed")

        if not self.config.quiet_mode:
            print("✅ All exports completed successfully")

    def _cleanup_threads_on_error(self):
        """Cleanup export threads on error"""
        for thread in self.stats.export_threads:
            if thread and thread.is_alive():
                if not self.config.quiet_mode:
                    print("  Waiting for cleanup...")
                thread.join(timeout=10)

    def run_pipeline(self) -> Dict[str, Any]:
        """Execute the complete pipeline"""
        start_time = time.time()

        if not self.config.quiet_mode:
            mode = "GPU" if self.gpu_available else "CPU"
            print(f"=== Starting pipeline, using {mode} ===\n")

        # Load configuration
        params = self._run_step("read_config", read_config)
        gpu_available = self._setup_gpu_config(params)
        params["GPU_AVAILABLE"] = 1 if gpu_available else 0

        # Load data
        data = self._run_step(
            "load_data", load_data, params["paths"]["input_folder"], gpu_available
        )
        T, Z, Y, X = data.shape

        if not self.config.quiet_mode:
            print(f"Loaded data of shape: {data.shape}\n")

        try:
            # Execute pipeline steps
            results = self._execute_pipeline_steps(data, params, gpu_available, T)

            # Wait for exports
            self._wait_for_export_threads()

            # Create and save summary
            summary = self._create_summary(data.shape, results)
            self.logger.save_summary(summary)

            if not self.config.quiet_mode:
                print("\n=== Pipeline completed ===")
                if self.config.enable_time_profiling:
                    self.stats.print_summary(self.config.enable_memory_profiling)
                else:
                    total_time = time.time() - start_time
                    print(f"Pipeline completed in {total_time:.2f} seconds.")

            return summary

        except Exception as e:
            if not self.config.quiet_mode:
                print(f"⚠ Pipeline error: {e}")
                import traceback

                traceback.print_exc()

            self._cleanup_threads_on_error()
            raise

    def _execute_pipeline_steps(self, data, params, gpu_available, T) -> Dict[str, Any]:
        """Execute all pipeline processing steps"""
        # Crop + boundaries
        cropped_data = self._run_step("crop_boundaries", crop_boundaries, data, params)
        index_xmin, index_xmax, _, raw_data = self._run_step(
            "compute_boundaries", compute_boundaries, cropped_data, params
        )
        self._track_export_thread(raw_data, "raw data")

        # Variance Stabilization
        data = self._run_step(
            "variance_stabilization",
            compute_variance_stabilization,
            raw_data,
            index_xmin,
            index_xmax,
            params,
        )
        self._track_export_thread(data, "variance stabilization")

        # Background estimation
        F0 = self._run_step(
            "background_estimation",
            background_estimation_single_block,
            data,
            index_xmin,
            index_xmax,
            params,
        )
        self._track_export_thread(F0, "F0")

        # Compute dF and noise
        dF, mean_noise = self._run_step(
            "compute_dynamic_image",
            compute_dynamic_image,
            data,
            F0,
            index_xmin,
            index_xmax,
            T,
            params,
        )
        self._track_export_thread(dF, "dF")

        std_noise = self._run_step(
            "estimate_std_noise",
            estimate_std_over_time,
            dF,
            index_xmin,
            index_xmax,
            gpu_available,
        )
        self._track_export_thread(std_noise, "noise estimation")

        # Active voxels
        active_voxels = self._run_step(
            "find_active_voxels",
            find_active_voxels,
            dF,
            std_noise,
            mean_noise,
            index_xmin,
            index_xmax,
            params,
        )
        self._track_export_thread(active_voxels, "active voxels")

        # Event detection (force CPU)
        if gpu_available:
            if not self.config.quiet_mode:
                print(
                    "GPU is available, forcing event detection on CPU for compatibility."
                )
            active_voxels = active_voxels.cpu().numpy()

        id_connections, ids_events = self._run_step(
            "detect_calcium_events",
            detect_calcium_events_opti,
            active_voxels,
            params_values=params,
        )
        self._track_export_thread(id_connections, "event detection")

        # Amplitude computation
        image_amplitude = self._run_step(
            "compute_image_amplitude",
            compute_image_amplitude,
            raw_data,
            F0,
            index_xmin,
            index_xmax,
            params,
        )
        self._track_export_thread(image_amplitude, "amplitude")

        # Features computation
        self._run_step(
            "save_features",
            save_features_from_events,
            id_connections,
            ids_events,
            image_amplitude,
            params,
        )

        return {
            "index_xmin": index_xmin,
            "index_xmax": index_xmax,
            "mean_noise": mean_noise,
            "std_noise": std_noise,
            "ids_events": ids_events,
        }

    def _create_summary(
        self, original_shape: Tuple[int, ...], results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create pipeline execution summary"""
        if self.config.enable_time_profiling:
            return self.stats.to_dict(
                original_shape,
                results["index_xmin"],
                results["index_xmax"],
                results["mean_noise"],
                results["std_noise"],
                results["ids_events"],
                self.config.enable_memory_profiling,
            )
        else:
            return {
                "original shape": f"{original_shape[0]}x{original_shape[1]}x{original_shape[2]}x{original_shape[3]}",
                "indexes xmin": results["index_xmin"].tolist(),
                "indexes xmax": results["index_xmax"].tolist(),
                "mean_noise": results["mean_noise"],
                "std_noise": results["std_noise"],
                "number_of_events": results["ids_events"],
                "memory_profiling": False,
            }


def parse_arguments() -> PipelineConfig:
    """Parse command line arguments and return configuration"""
    config = PipelineConfig()

    if len(sys.argv) > 2:
        raise ValueError(
            f"Too many arguments. Usage: {sys.argv[0]} [--stats | --memstats | --quiet | --help]"
        )

    if len(sys.argv) == 2:
        arg = sys.argv[1]

        if arg == "--stats":
            config.enable_time_profiling = True
        elif arg == "--memstats":
            print(
                "WARNING: Memory profiling is enabled, this may slow down the pipeline execution."
            )
            config.enable_time_profiling = True
            config.enable_memory_profiling = True
        elif arg == "--quiet":
            config.quiet_mode = True
        elif arg == "--help":
            print(f"Usage: {sys.argv[0]} [--stats | --memstats | --quiet | --help]")
            print("  --stats: Run pipeline with time statistics")
            print("  --memstats: Run pipeline with memory and time statistics")
            print(
                "  --quiet: Run pipeline without statistics and without execution trace"
            )
            print("  --help: Show this help message")
            sys.exit(0)
        else:
            raise ValueError(
                f"Invalid argument '{arg}'. Usage: {sys.argv[0]} [--stats | --memstats | --quiet | --help]"
            )

    return config


def main():
    """Main function to run the pipeline"""
    try:
        config = parse_arguments()

        if config.quiet_mode:
            print("Running pipeline in quiet mode...")
            with open(os.devnull, "w") as devnull:
                original_stdout = sys.stdout
                sys.stdout = devnull
                try:
                    executor = PipelineExecutor(config)
                    executor.run_pipeline()
                finally:
                    sys.stdout = original_stdout
        else:
            if config.enable_time_profiling:
                mode = "memory and time" if config.enable_memory_profiling else "time"
                print(f"Running pipeline with {mode} statistics...")
            else:
                print("Running pipeline without statistics...")

            executor = PipelineExecutor(config)
            executor.run_pipeline()

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
