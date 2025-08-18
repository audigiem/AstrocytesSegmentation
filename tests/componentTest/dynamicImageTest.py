#! /usr/bin/env python3
"""
@file medianFilterTest.py
@brief This module provides a test case for dynamic image processing functionality in the astroca package.
"""
from astroca.tools.loadData import load_data
import numpy as np
import torch
from astroca.dynamicImage.dynamicImage import compute_dynamic_image_GPU

def test_dynamic_image_processing():
    """
    @brief Test the dynamic image processing functionality.
    """
    print("Testing dynamic image processing functionality...")

    # Load the data
    data = load_data("/home/maudigie/data/outputData/testGPU/variance_stabilized_sequence.tif", True)
    F0 = load_data("/home/maudigie/data/outputData/testGPU/F0.tif", True)
    index_xmin = np.load("/home/maudigie/data/outputData/testGPU/index_xmin.npy")
    index_xmax = np.load("/home/maudigie/data/outputData/testGPU/index_xmax.npy")
    # convert to torch tensor if necessary
    xmin = torch.tensor(index_xmin, dtype=torch.int32)
    xmax = torch.tensor(index_xmax, dtype=torch.int32)

    T, Z, Y, X = data.shape
    params = {
        'save' : {
            'save_df' : 0
        },
        'paths' : {
            'output_dir' : None
        }
    }
    dF, mean_noise = compute_dynamic_image_GPU(
        data,
        F0,
        xmin,
        xmax,
        T,
    )


if __name__ == "__main__":
    test_dynamic_image_processing()
    print("Dynamic image processing test completed successfully.")
