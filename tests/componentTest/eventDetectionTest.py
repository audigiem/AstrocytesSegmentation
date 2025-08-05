#! /usr/bin/env python3
""" 
@file eventDetectionTest.py
@brief This module provides component tests for the event detection functionality in the astroca package.
"""
import unittest

import torch

from astroca.tools.loadData import load_data
from astroca.tools.exportData import export_data
from astroca.events.eventDetectorCorrected import detect_calcium_events_opti
from astroca.events.eventDetectorGPU import detect_calcium_events_gpu

class EventDetectionTest(unittest.TestCase):
    """ 
    @brief Test case for event detection functionality.
    """
    
    def setUp(self):
        """ 
        @brief Set up the test case with synthetic data.
        """
        self.dataPython = load_data("/home/maudigie/data/outputData/testCPU/activeVoxels.tif")
        self.targetResults = load_data("/home/maudigie/data/outputData/testCPU/ID_calciumEvents.tif")

        self.params_values = {
            'events_extraction' : {
                'threshold_size_3d' : 400,
                'threshold_size_3d_removed' : 20,
                'threshold_corr' : 0.6
            },
            'paths': {
                'output_dir': "/home/maudigie/data/outputData/testGPU/"
            },
            'save': {
                'save_events': 0
            }
        }

    # def test_event_detection(self):
    #     """
    #     @brief Test the event detection functionality.
    #     """
    #     print("Starting event detection test...")
    #     id_connections, nb_events = detect_calcium_events_opti(self.dataPython, params_values=self.params_values)
        
    def test_event_detection_GPU(self):
        """
        @brief Test the event detection functionality with GPU support.
        """
        print("Starting event detection test with GPU...")
        # Ensure the data is in the correct format for GPU processing
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your GPU setup.")
        self.dataPython = torch.tensor(self.dataPython, dtype=torch.float32).cuda()
        id_connections, nb_events = detect_calcium_events_gpu(self.dataPython, params_values=self.params_values)
        # Compare the results with the target results
        self.assertTrue(torch.allclose(id_connections, self.targetResults))
        
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
