#! /usr/bin/env python3
""" 
@file eventDetectionTest.py
@brief This module provides component tests for the event detection functionality in the astroca package.
"""
import unittest
from astroca.tools.loadData import load_data
from astroca.parametersNoise.parametersNoise import estimate_std_over_time, estimate_std_over_time_optimized
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
        self.data = load_data("/home/matteo/Bureau/INRIA/codeJava/outputdir/dF.tif")
        self.xmins = np.load(self.target_dir + "index_Xmin.npy")
        self.xmaxs = np.load(self.target_dir + "index_Xmax.npy")


    def test_noise_estimation(self):
        """ 
        @brief Test the noise estimation functionality.
        """
        print("Starting noise estimation test...")
        std = estimate_std_over_time(self.data, self.xmins, self.xmaxs)
        std_optimized = estimate_std_over_time_optimized(self.data, self.xmins, self.xmaxs)
        np.testing.assert_array_equal(std, std_optimized)

    def tearDown(self):
        """ 
        @brief Clean up after the test case.
        """
        pass


if __name__ == "__main__":
    unittest.main()
# To run the test, use the command: python -m unittest eventDetectionTest
