#! /usr/bin/env python3
""" 
@file featuresTest.py
@brief This module provides component tests for the features computation functionality in the astroca package.
"""
import unittest
from astroca.tools.loadData import load_data
from astroca.features.featuresComputation import save_features_from_events
from tests.comparingTools.compareCSVFiles import compare_csv_files

class FeaturesTest(unittest.TestCase):
    """ 
    @brief Test case for features computation functionality.
    """
    
    def setUp(self):
        """ 
        @brief Set up the test case with synthetic data.
        """
        self.amplitude = load_data("/home/matteo/Bureau/INRIA/codeJava/outputdir20/amplitude.tif")
        self.calcium_events = load_data("/home/matteo/Bureau/INRIA/codeJava/outputdir20/ID_calciumEvents.tif")
        
        self.params_values = {
            'features_extraction': {
                'voxel_size_x': 0.1025,
                'voxel_size_y': 0.1025,
                'voxel_size_z': 0.1344,
                'threshold_median_localized': 0.5,
                'volume_localized': 0.0434
            },
            'paths': {
                'output_dir': "/home/matteo/Bureau/INRIA/codePython/outputdir/checkDir20/"
            },
            'save': {
                'save_features': 1
            }
        }

    def test_event_detection(self):
        """ 
        @brief Test the event detection functionality.
        """
        print("Starting features computation test...")
        save_features_from_events(self.calcium_events, 73, self.amplitude, params_values=self.params_values)
        # Here we would typically check if the features were saved correctly,
        compare_csv_files(
            "/home/matteo/Bureau/INRIA/codeJava/outputdir20/Features.csv",
            "/home/matteo/Bureau/INRIA/codeJava/outputdir20/Features.csv"
        )
        
        
    def tearDown(self):
        """ 
        @brief Clean up after the test case.
        """
        # No specific cleanup needed for this test case
        pass
    
        
if __name__ == "__main__":
    unittest.main()
# To run the test, use the command: python -m unittest eventDetectionTest
    