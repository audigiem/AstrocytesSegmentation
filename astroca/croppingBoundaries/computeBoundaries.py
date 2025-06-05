"""
@file computeBoundaries.py
@brief This module provides functionality to compute cropping boundaries for 3D image sequences with time dimension.
@detail Due to the acquisition process, the data is supposed to have empty band(s) for each z-slice.
To speed up the calculations over the images, we spot those empty bands whose value is equal to default_value
(0.0 or 50.0) and compute the cropping boundaries for each z-slice. The cropping boundaries are computed as
the first and last non-empty band in each z-slice.
"""

import numpy as np
from astroca.init.scene import ImageSequence3DPlusTime

def compute_boundaries(image_sequence: ImageSequence3DPlusTime, pixel_cropped: int = 2) -> tuple:
    """
    @brief Compute cropping boundaries (empty bands) for a 3D image sequence with time dimension in place.
    @param image_sequence: An instance of ImageSequence3DPlusTime containing the image data.
    @param pixel_cropped: Number of pixels to remove at both ends of the valid region (in X) for each Z slice.
    @return: Tuple (index_xmin, index_xmax, default_value) updated in-place in the image_sequence object.
    """
    
    print("Computing cropping boundaries...")
    data = image_sequence.get_data()  # shape: (T, Z, Y, X)
    T, Z, Y, X = data.shape
    t = 0  # Use the first time frame to compute cropping bounds


    # Sample 10% of Y values
    nb_y_tested = max(1, int(0.1 * Y))
    y_array = np.random.choice(Y, size=nb_y_tested, replace=False)

    # Define default value from a known empty pixel
    default_value = float(data[t, 0, 0, X - 1])
    image_sequence.set_default_value(default_value)

    index_xmin = np.full(Z, -1, dtype=int)
    index_xmax = np.full(Z, X - 1, dtype=int)

    for z in range(Z):
        for x in range(X):
            values = data[t, z, y_array, x]
            if not np.all(values == default_value):
                if index_xmin[z] == -1:
                    index_xmin[z] = x
            elif index_xmin[z] != -1:
                index_xmax[z] = x - 1
                break

    # Replace values on borders with default_value
    for t in range(T):
        for z in range(Z):
            for y in range(Y):
                data[t, z, y, index_xmin[z]:index_xmin[z] + pixel_cropped] = default_value
                data[t, z, y, index_xmax[z] - pixel_cropped + 1:index_xmax[z] + 1] = default_value

    index_xmin += pixel_cropped
    index_xmax -= pixel_cropped

    image_sequence.set_data(data)  # In-place update
    # image_sequence.index_xmin = index_xmin
    # image_sequence.index_xmax = index_xmax
    # image_sequence.default_value = default_value
    
    print(f"Cropping boundaries computed: \n    index_xmin={index_xmin}, \n    index_xmax={index_xmax}, \n    default_value={default_value}")
    print()

    return index_xmin, index_xmax, default_value
