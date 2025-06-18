#! /usr/bin/env python3
import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import cpu_count
from astroca.activeVoxels.spaceMorphology import (median_3d_for_4d_stack, apply_median_filter_spherical_numba,
                                                  median_filter_3d, apply_median_filter_3d_per_time)
from astroca.tools.loadData import load_data
from astroca.tools.exportData import export_data

class Median3DTestBench:
    def __init__(self):
        self.results = {}
        self.timings = {}
        self.metrics = {}

    def generate_test_data(self, shape=(5, 20, 128, 128), noise_level=0.2):
        """Generate synthetic 4D test data with moving spheres + noise"""
        print("Generating synthetic 4D test data...")
        t, z, y, x = shape
        data = np.zeros(shape, dtype=np.float32)

        # Create moving spheres with different intensities
        for frame in range(t):
            center = (
                int(y / 2 + y / 4 * np.sin(frame / t * 2 * np.pi)),
                int(x / 2 + x / 4 * np.cos(frame / t * 2 * np.pi)),
                int(z / 2)
            )
            radius = min(y, x, z) // 4

            zz, yy, xx = np.ogrid[:z, :y, :x]
            distance = np.sqrt((xx - center[1]) ** 2 + (yy - center[0]) ** 2 + (zz - center[2]) ** 2)
            data[frame, distance <= radius] = 0.8 + 0.2 * np.random.rand()  # Variable intensity

        # Add different noise types
        data += noise_level * np.random.randn(*shape)  # Gaussian noise
        data = np.clip(data, 0, 1)

        # Add some hot pixels
        hot_pixels = np.random.rand(*shape) > 0.995
        data[hot_pixels] = 1.0

        return data.astype(np.float32)

    def export_synthetic_data(self, data, output_path="/home/matteo/Bureau/INRIA/assets/syntheticData/", export_as_single_tif=True, file_name="synthetic_data"):
        """Export synthetic data to a TIFF file"""
        export_data(data, output_path, export_as_single_tif=export_as_single_tif, file_name=file_name)


    def run_benchmark(self, data, radius=1.5, n_workers=None):
        """Run benchmark on all implementations"""
        if n_workers is None:
            n_workers = min(4, cpu_count())

        implementations = {
            "median_3d_for_4d_stack": lambda d: median_3d_for_4d_stack(d, radius=2, n_workers=8),
            "apply_median_filter_spherical_numba": lambda d: apply_median_filter_spherical_numba(d, radius),
            "median_filter_3d": lambda d: median_filter_3d(d, radius),
            "apply_median_filter_3d_per_time": lambda d: apply_median_filter_3d_per_time(d, radius)
        }

        # Use Java implementation as reference
        reference = load_data("/home/matteo/Bureau/INRIA/codeJava/outputdirFewerTime/Median.tif")

        # Test all implementations
        for name, func in implementations.items():
            print(f"\nTesting {name}...")
            try:
                start = time.time()
                result = func(data.copy())  # Use copy to avoid modifying input
                elapsed = time.time() - start

                # Compute advanced metrics
                diff = np.abs(reference - result)
                metrics = {
                    'time': elapsed,
                    'max_diff': np.max(diff),
                    'mean_diff': np.mean(diff),
                    'psnr': psnr(reference, result, data_range=1.0),
                    'mse': mean_squared_error(reference, result),
                    'correlation': np.corrcoef(reference.ravel(), result.ravel())[0, 1]
                }

                self.results[name] = result
                self.timings[name] = elapsed
                self.metrics[name] = metrics

                print(f"Time: {elapsed:.2f}s")
                print(f"Max diff: {metrics['max_diff']:.2e}, Mean diff: {metrics['mean_diff']:.2e}")
                print(f"PSNR: {metrics['psnr']:.2f}dB, MSE: {metrics['mse']:.2e}")
                print(f"Correlation: {metrics['correlation']:.4f}")
                self.analyze_temporal_errors(reference)

            except Exception as e:
                print(f"Error in {name}: {str(e)}")
                self.results[name] = None
                self.timings[name] = None
                self.metrics[name] = None

    def analyze_temporal_errors(self, reference):
        """ Analyze temporal errors for each implementation against a reference volume."""
        if not self.results:
            print("No results to analyze.")
            return

        for name, result in self.results.items():
            if result is None:
                continue
            print(f"\n=== Temporal analysis for {name} ===")
            t = reference.shape[0]
            frame_mse = []
            for i in range(t):
                mse = mean_squared_error(reference[i], result[i])
                frame_mse.append((i, mse))
            frame_mse.sort(key=lambda x: x[1], reverse=True)

            print("Top 5 worst frames (highest MSE):")
            for idx, mse_val in frame_mse[:5]:
                print(f"  Frame {idx}: MSE = {mse_val:.3e}")

            self.metrics[name]['worst_frames'] = frame_mse[:5]

    def visualize_results(self, data, frame=0, slice=0):
        """Visual comparison of results"""
        plt.figure(figsize=(18, 10))

        # Original and reference
        plt.subplot(2, 4, 1)
        plt.imshow(data[frame, slice], cmap='gray', vmin=0, vmax=1)
        plt.title("Original Data")
        plt.axis('off')



        # Show all implementations
        for i, (name, result) in enumerate(self.results.items(), 3):
            plt.subplot(2, 4, i)
            if result is not None:
                plt.imshow(result[frame, slice], cmap='gray', vmin=0, vmax=1)
                time_str = f"{self.timings[name]:.2f}s" if self.timings[name] else "Error"
                metrics = self.metrics.get(name, {})
                psnr_str = f"\nPSNR: {metrics.get('psnr', 'N/A'):.1f}dB" if metrics else ""
                plt.title(f"{name}\n{time_str}{psnr_str}")
            else:
                plt.text(0.5, 0.5, "Failed", ha='center', va='center')
                plt.title(f"{name}\nError")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def visualize_worst_frames(self, data, max_frames=3, slice=0):
        for name, result in self.results.items():
            worst = self.metrics.get(name, {}).get('worst_frames')
            if not worst:
                continue

            print(f"\nVisualizing worst frames for {name}:")
            n = min(len(worst), max_frames)
            plt.figure(figsize=(5 * n, 6))

            for i, (frame_idx, mse_val) in enumerate(worst[:n]):
                plt.subplot(2, n, i + 1)
                plt.imshow(data[frame_idx, slice], cmap='gray', vmin=0, vmax=1)
                plt.title(f"Original Frame {frame_idx}")
                plt.axis('off')

                plt.subplot(2, n, i + 1 + n)
                plt.imshow(result[frame_idx, slice], cmap='gray', vmin=0, vmax=1)
                plt.title(f"{name} - Frame {frame_idx}\nMSE={mse_val:.2e}")
                plt.axis('off')

            plt.tight_layout()
            plt.show()

    def plot_metrics(self):
        """Plot comparative metrics"""
        if not self.metrics:
            return

        names = [n for n in self.metrics if self.metrics[n] is not None]
        if not names:
            return

        metrics = ['time', 'psnr', 'mse', 'correlation']
        titles = ['Execution Time (s)', 'PSNR (dB)', 'MSE (lower is better)', 'Correlation']

        plt.figure(figsize=(15, 10))
        for i, (metric, title) in enumerate(zip(metrics, titles), 1):
            plt.subplot(2, 2, i)
            values = [self.metrics[n][metric] for n in names]

            if metric == 'time':
                plt.bar(names, values, color='skyblue')
                plt.ylabel('Seconds')
            elif metric == 'psnr':
                plt.bar(names, values, color='lightgreen')
                plt.ylabel('dB')
            elif metric == 'mse':
                plt.bar(names, values, color='salmon')
                plt.yscale('log')
            else:
                plt.bar(names, values, color='gold')

            plt.title(title)
            plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    tester = Median3DTestBench()

    # # 1. Test with synthetic data
    # print("=== Testing with Synthetic Data ===")
    # synthetic_data = tester.generate_test_data(shape=(10, 30, 256, 256))
    # tester.export_synthetic_data(synthetic_data)
    # # pick the Java results to compare with
    #
    # # tester.run_benchmark(synthetic_data, radius=1.5)
    # #
    # # # Visualization
    # # tester.visualize_results(synthetic_data, frame=1, slice=15)
    # # tester.plot_metrics()
    #
    # 2. Test with real data (if available)
    try:
        print("\n=== Testing with Real Data ===")
        real_data = load_data("/home/matteo/Bureau/INRIA/codeJava/outputdirFewerTime/Median.tif")
        tester.run_benchmark(real_data, radius=1.5)
        # tester.visualize_worst_frames(real_data)
        tester.plot_metrics()
    except Exception as e:
        print(f"Could not load real data: {str(e)}")