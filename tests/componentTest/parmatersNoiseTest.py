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
        self.test_dirs = ["/home/matteo/Bureau/INRIA/codePython/outputdir/testDir/",
                          "/home/matteo/Bureau/INRIA/codePython/outputdir/checkDir20/"]
        self.data = [load_data(dir_path + "dynamic_image_dF.tif") for dir_path in self.test_dirs]
        self.xmins = np.load(self.target_dir + "index_Xmin.npy")
        self.xmaxs = np.load(self.target_dir + "index_Xmax.npy")


    def test_noise_estimation(self):
        """ 
        @brief Test the noise estimation functionality.
        """
        print("Starting noise estimation test...")
        for data in self.data:
            std = estimate_std_over_time(data, self.xmins, self.xmaxs)
            std_optimized = estimate_std_over_time_optimized(data, self.xmins, self.xmaxs)
            # assert that the standard deviations are close enough
            self.assertTrue(np.allclose(std, std_optimized, rtol=1e-5, atol=1e-8),
                            "Standard deviations from both methods do not match.")

    def tearDown(self):
        """ 
        @brief Clean up after the test case.
        """
        pass


if __name__ == "__main__":
    unittest.main()
# To run the test, use the command: python -m unittest eventDetectionTest
