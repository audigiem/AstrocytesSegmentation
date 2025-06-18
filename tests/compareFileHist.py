#! /usr/bin/env python3
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt


from astroca.tools.loadData import load_data

def compare_volumes(vol1, vol2, threshold=1e-4):
    assert vol1.shape == vol2.shape, "Volumes must have the same shape"

    diff = np.abs(vol1.astype(np.float32) - vol2.astype(np.float32))

    n_voxels = diff.size
    vox_diff_count = np.sum(diff > threshold)
    vox_diff_percent = vox_diff_count / n_voxels * 100

    print(f"Total voxels: {n_voxels}")
    print(f"Voxels differing by more than {threshold}: {vox_diff_count} ({vox_diff_percent:.2f}%)")
    print(f"Max difference: {diff.max()}")
    print(f"Mean difference: {diff.mean()}")

    return diff, vox_diff_percent


def plot_histograms(vol1, vol2, diff):
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.hist(vol1.ravel(), bins=100, color='blue', alpha=0.6, label='Volume 1')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.hist(vol2.ravel(), bins=100, color='green', alpha=0.6, label='Volume 2')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.hist(diff.ravel(), bins=100, color='red', alpha=0.6, label='Difference')
    plt.legend()

    plt.suptitle("Histograms of Volumes and Differences")
    plt.show()


def plot_slice_comparison(vol1, vol2, diff, frame=0, z_slice=0):
    # Extract 2D slices (Y, X) at given frame and z
    slice1 = vol1[frame, z_slice, :, :]
    slice2 = vol2[frame, z_slice, :, :]
    diff_slice = diff[frame, z_slice, :, :]

    vmin = min(slice1.min(), slice2.min())
    vmax = max(slice1.max(), slice2.max())

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(slice1, cmap='gray', vmin=vmin, vmax=vmax)
    plt.title(f"Volume 1 - Frame {frame} Slice Z={z_slice}")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(slice2, cmap='gray', vmin=vmin, vmax=vmax)
    plt.title(f"Volume 2 - Frame {frame} Slice Z={z_slice}")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(diff_slice, cmap='hot')
    plt.title("Difference")
    plt.colorbar()

    plt.show()


if __name__ == "__main__":
    # Remplace par tes fichiers .tif
    expected_closing_path = "/home/matteo/Bureau/INRIA/codeJava/outputdirFewerTime/Median.tif"
    output_closing_path = "/home/matteo/Bureau/INRIA/codePython/outputdir/checkDirectoryFewerTime/medianFiltered_2.tif"

    vol1 = load_data(expected_closing_path)
    vol2 = load_data(output_closing_path)
    if vol1.ndim != 4 or vol2.ndim != 4:
        raise ValueError(f"Both volumes must be 4D arrays (T, Z, Y, X), got shapes {vol1.shape} and {vol2.shape}")
    elif vol1.shape != vol2.shape:
        raise ValueError(f"Volumes must have the same shape, got {vol1.shape} and {vol2.shape}")
    diff, percent_diff = compare_volumes(vol1, vol2, threshold=1e-4)
    plot_histograms(vol1, vol2, diff)

    # Change frame et slice Z pour inspecter d’autres slices
    plot_slice_comparison(vol1, vol2, diff, frame=0, z_slice=0)
