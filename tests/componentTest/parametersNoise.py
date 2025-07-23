#! /usr/bin/env python3
"""
@file featuresTest.py
@brief This module provides component tests for the features computation functionality in the astroca package.
"""
import unittest
from astroca.tools.loadData import load_data
import numpy as np
import torch
from astroca.parametersNoise.parametersNoise import estimate_std_over_time_CPU
from astroca.parametersNoise.parametersNoiseGPU import estimate_std_over_time_GPU
from tests.comparingTools.compareFiles import compare_sequence, compare_files


class FeaturesTest(unittest.TestCase):
    """
    @brief Test case for features computation functionality.
    """

    def setUp(self):
        """
        @brief Set up the test case with synthetic data.
        """
        self.base_data = load_data("/home/maudigie/data/outputData/testCPU/dynamic_image_dF.tif")
        self.index_xmin = np.load("home/maudigie/data/outputData/testCPU/index_Xmin.npy")
        self.index_xmax = np.load("home/maudigie/data/outputData/testCPU/index_Xmax.npy")
        self.stdMapGPU = load_data("home/maudigie/data/outputData/debug/stdMapGPU.tif")
        self.stdMapCPU = load_data("home/maudigie/data/outputData/debug/stdMapCPU.tif")
        self.residualGPU = load_data("home/maudigie/data/outputData/debug/residualsGPU.tif")
        self.residualCPU = load_data("home/maudigie/data/outputData/debug/residualsCPU.tif")

    def test_event_detection(self):
        """
        @brief Test the event detection functionality.
        """
        std_noise_CPU = estimate_std_over_time_CPU(self.base_data, self.index_xmin, self.index_xmax)
        # std_noise_GPU = estimate_std_over_time_GPU(torch.tensor(self.base_data, dtype=torch.float32),
        #                                             torch.tensor(self.index_xmin, dtype=torch.int32),
        #                                             torch.tensor(self.index_xmax, dtype=torch.int32))
        # Check if the CPU and GPU estimates are close enough
        # self.assertAlmostEqual(std_noise_CPU, std_noise_GPU.item(), places=5,
        #                        msg="CPU and GPU standard deviation estimates do not match.")

    def test_compare_data(self):
        """
        @brief Test the comparison of data files.
        """
        # Compare the standard deviation maps
        self.assertTrue(compare_files(self.stdMapCPU, self.stdMapGPU, percentage_accuracy=1e-5),
                        msg="Standard deviation maps do not match within the threshold.")

        # Compare the residuals
        self.assertTrue(compare_files(self.residualCPU, self.residualGPU, percentage_accuracy=1e-5),
                        msg="Residuals do not match within the threshold.")


    def tearDown(self):
        """
        @brief Clean up after the test case.
        """
        # No specific cleanup needed for this test case
        pass


if __name__ == "__main__":
    unittest.main()
# To run the test, use the command: python -m unittest eventDetectionTest
