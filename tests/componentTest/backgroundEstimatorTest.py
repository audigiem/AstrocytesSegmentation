#! /usr/bin/env python3
""" 
@file eventDetectionTest.py
@brief This module provides component tests for the event detection functionality in the astroca package.
"""
import unittest
from astroca.tools.loadData import load_data
from astroca.dynamicImage.backgroundEstimator import background_estimation_single_block, background_estimation_single_block_numba
from tests.comparingTools.compareFiles import compare_files
import numpy as np


class EventDetectionTest(unittest.TestCase):
    """ 
    @brief Test case for event detection functionality.
    """
    
    def setUp(self):
        """ 
        @brief Set up the test case with synthetic data.
        """
        self.target_dir = "/home/matteo/Bureau/INRIA/codePython/outputdir/checkDir20/"
        self.data = load_data(self.target_dir + "variance_stabilized_sequence.tif")
        self.xmins = np.load(self.target_dir + "index_Xmin.npy")
        self.xmaxs = np.load(self.target_dir + "index_Xmax.npy")

        self.params_values = {
            'background_estimation': {
                'moving_window': 9,
                'method': 'percentile',
                'method2': 'Med',
                'percentile': 10.0
            },
            'paths': {
                'output_dir': self.target_dir
            },
            'save': {
                'save_background_estimation': 1
            }
        }

    def test_background_estimation(self):
        """ 
        @brief Test the background estimation functionality.
        """
        print("Starting background estimation test...")
        F0 = background_estimation_single_block(self.data, self.xmins, self.xmaxs, params_values=self.params_values)
        F0_numba = background_estimation_single_block_numba(self.data, self.xmins, self.xmaxs, params_values=self.params_values)
        compare_files(self.target_dir + "F0_estimated.tif", self.target_dir + "F0_estimated_numba.tif")

    def tearDown(self):
        """ 
        @brief Clean up after the test case.
        """
        pass


if __name__ == "__main__":
    unittest.main()
# To run the test, use the command: python -m unittest eventDetectionTest
