#! /usr/bin/env python3
""" 
@file eventDetectionTest.py
@brief This module provides component tests for the event detection functionality in the astroca package.
"""
import unittest
from astroca.tools.loadData import load_data
from astroca.tools.exportData import export_data
from astroca.events.eventDetector import detect_calcium_events_opti

class EventDetectionTest(unittest.TestCase):
    """ 
    @brief Test case for event detection functionality.
    """
    
    def setUp(self):
        """ 
        @brief Set up the test case with synthetic data.
        """
        self.dataPython = load_data("/home/matteo/Bureau/INRIA/codePython/outputdir/checkDirLatestTest/activeVoxels.tif")
        self.targetResults = load_data("/home/matteo/Bureau/INRIA/codeJava/outputdirLatestTest/ID_calciumEvents.tif")

        self.params_values = {
            'events_extraction' : {
                'threshold_size_3d' : 400,
                'threshold_size_3d_removed' : 20,
                'threshold_corr' : 0.6
            },
            'paths': {
                'output_dir': "/home/matteo/Bureau/INRIA/codePython/outputdir/checkDirLatestTest/"
            },
            'save': {
                'save_events': 1
            }
        }

    def test_event_detection(self):
        """ 
        @brief Test the event detection functionality.
        """
        print("Starting event detection test...")
        id_connections, nb_events = detect_calcium_events_opti(self.dataPython, params_values=self.params_values)
        
        
        
    def tearDown(self):
        """ 
        @brief Clean up after the test case.
        """
        # No specific cleanup needed for this test case
        pass
    
        
if __name__ == "__main__":
    unittest.main()
# To run the test, use the command: python -m unittest eventDetectionTest
    # data = load_data("/home/matteo/Bureau/INRIA/codeJava/outputdir20/AV.tif")
    # param = {
    #     'events_extraction' : {
    #         'threshold_size_3d' : 400,
    #         'threshold_size_3d_removed' : 20,
    #         'threshold_corr' : 0.6
    #     },
    #     'paths': {
    #         'output_dir': None
    #     },
    #     'save': {
    #         'save_events': 0
    #     }
    # }
    # id_connections, nb_events = detect_calcium_events_opti(data, params_values=param)
