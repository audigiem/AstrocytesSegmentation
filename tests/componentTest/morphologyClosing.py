#! /usr/bin/env python3
"""
@file medianFilterTest.py
@brief This module provides a test case for the median filter functionality in the astroca package.
"""
import unittest
from astroca.tools.loadData import load_data
from astroca.tools.exportData import export_data
from astroca.activeVoxels.spaceMorphology import (
    closing_morphology_in_space_CPU,
    closing_morphology_in_space_GPU,
)
import numpy as np
import torch

from tests.comparingTools.compareFiles import compare_sequence


class EventDetectionTest(unittest.TestCase):
    """
    @brief Test case for event detection functionality.
    """

    def setUp(self):
        """
        @brief Set up the test case with synthetic data.
        """
        self.target_dir = "/home/maudigie/data/outputData/testGPU/"
        self.src_dir = "/home/maudigie/data/outputData/testCPU/"
        self.gpu_available = True
        self.save_results = True
        self.dataGPU = load_data(self.src_dir + "zScore.tif", self.gpu_available)
        self.dataCPU = load_data(self.src_dir + "zScore.tif", GPU_AVAILABLE=False)
        self.border_condition = ["ignore"]
        self.radius = 1

    def test_median_filter(self):
        """
        @brief Test the median filter functionality.
        """
        print("Testing closing morphology in space functionality...")
        for border_condition in self.border_condition:
            print(f"Testing border condition: {border_condition}")
            closed_data_CPU = closing_morphology_in_space_CPU(
                self.dataCPU, self.radius, border_condition
            )
            closed_data_GPU = closing_morphology_in_space_GPU(
                self.dataGPU, self.radius, border_condition
            )
            if self.save_results:
                export_data(
                    closed_data_CPU,
                    self.src_dir,
                    export_as_single_tif=True,
                    file_name=f"closing_in_space_CPU_{border_condition}",
                )
                export_data(
                    closed_data_GPU.cpu().numpy(),
                    self.target_dir,
                    export_as_single_tif=True,
                    file_name=f"closing_in_space_GPU_{border_condition}",
                )
            compare_sequence(
                self.src_dir + f"closing_in_space_CPU_{border_condition}.tif",
                self.target_dir + f"closing_in_space_CPU_{border_condition}.tif",
            )

    def tearDown(self):
        """
        @brief Clean up after the test case.
        """
        pass


if __name__ == "__main__":
    unittest.main()
# To run the test, use the command: python -m unittest eventDetectionTest
