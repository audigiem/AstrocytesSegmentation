#! /usr/bin/env python3
""" 
@file medianFilterTest.py
@brief This module provides a test case for the median filter functionality in the astroca package.
"""
import unittest
from astroca.tools.loadData import load_data
from astroca.tools.exportData import export_data
from astroca.activeVoxels.medianFilter import unified_median_filter_3d
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
        self.gpu_available = True
        self.save_results = True
        self.data = load_data(self.target_dir + "filledSpaceMorphology.tif", self.gpu_available)
        self.border_condition = 'ignore'
        self.radius = 1.5

    def test_median_filter(self):
        """ 
        @brief Test the median filter functionality.
        """
        print("Testing median filter functionality...")
        data = unified_median_filter_3d(self.data, self.radius, self.border_condition, use_gpu=self.gpu_available)
        if self.save_results:
            if isinstance(data, torch.Tensor):
                data_to_export = data.cpu().numpy()
            else:
                data_to_export = data
            export_data(data_to_export, self.target_dir, export_as_single_tif=True, file_name="medianFiltered_2")

    def tearDown(self):
        """ 
        @brief Clean up after the test case.
        """
        pass


if __name__ == "__main__":
    unittest.main()
# To run the test, use the command: python -m unittest eventDetectionTest
