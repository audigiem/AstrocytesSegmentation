#! /usr/bin/env python3
""" 
@file medianFilterTest.py
@brief This module provides a test case for the median filter functionality in the astroca package.
"""
import unittest
from astroca.tools.loadData import load_data
from astroca.tools.exportData import export_data
from astroca.activeVoxels.medianFilter import (
    unified_median_filter_3d_cpu,
    unified_median_filter_3d_gpu,
)
from tests.comparingTools.compareFiles import compare_sequence
import numpy as np
import torch


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
        self.data_GPU = load_data(
            self.src_dir + "closing_in_space_CPU_reflect.tif", self.gpu_available
        )
        self.data_CPU = load_data(
            self.src_dir + "closing_in_space_CPU_reflect.tif", GPU_AVAILABLE=False
        )
        self.border_condition = "reflect"
        self.radius = 1.5

    def test_median_filter(self):
        """
        @brief Test the median filter functionality.
        """
        print("Testing median filter functionality...")
        data_CPU = unified_median_filter_3d_cpu(
            self.data_CPU, self.radius, self.border_condition
        )
        data_GPU = unified_median_filter_3d_gpu(
            self.data_GPU, self.radius, self.border_condition
        )
        if self.save_results:
            export_data(
                data_CPU,
                self.src_dir,
                export_as_single_tif=True,
                file_name="medianFiltered_CPU",
            )
            export_data(
                data_GPU.cpu().numpy(),
                self.target_dir,
                export_as_single_tif=True,
                file_name="medianFiltered_GPU",
            )
        compare_sequence(
            self.src_dir + "medianFiltered_CPU.tif",
            self.target_dir + "medianFiltered_GPU.tif",
        )

    def tearDown(self):
        """
        @brief Clean up after the test case.
        """
        pass


if __name__ == "__main__":
    unittest.main()
# To run the test, use the command: python -m unittest eventDetectionTest
