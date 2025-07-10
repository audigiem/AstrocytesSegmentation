#! /usr/bin/env python3
""" 
@file eventDetectionTest.py
@brief This module provides component tests for the event detection functionality in the astroca package.
"""
import unittest
from astroca.tools.loadData import load_data
from astroca.tools.exportData import export_data

class EventDetectionTest(unittest.TestCase):
    """ 
    @brief Test case for event detection functionality.
    """
    
    def setUp(self):
        """ 
        @brief Set up the test case with synthetic data.
        """
        self.data = load_data("/home/matteo/Bureau/INRIA/codeJava/outputdirFewerTime/AV.tif")
        self.params_values = {
            'events_extraction' : {
                'threshold_size_3d' : 400,
                'threshold_size_3d_removed' : 20,
                'threshold_corr' : 0.6
            }
        }
    
    def test_event_detection(self):
        """ 
        @brief Test the event detection functionality.
        """
        print("Starting event detection test...")
        # active_voxels = detect_events(self.data, params_values=self.params_values)
        # events_ids = list(set(active_voxels.flatten()) - {0})  # Exclude background label (0)
        
        # self.assertGreater(len(events_ids), 0, "No events detected.")
        
        # show_results(active_voxels)
        # active_voxels = active_voxels.astype('uint16')
        # # Optionally save results
        # export_data(active_voxels, "/home/matteo/Bureau/INRIA/codePython/AstrocytesSegmentation/tests/assets/eventDetectionResults/", export_as_single_tif=True, file_name="detected_events")
        
    def tearDown(self):
        """ 
        @brief Clean up after the test case.
        """
        # No specific cleanup needed for this test case
        pass
    
        
if __name__ == "__main__":
    unittest.main()
# To run the test, use the command: python -m unittest eventDetectionTest