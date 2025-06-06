import numpy as np


class ImageSequence3DPlusTime:
    """
    A class to represent a 3D image sequence with time dimension.
    We will load the data from a file and then instantiate the class with the data.
    """

    def __init__(self, data: np.ndarray, time_length: int,
                 width: int, height: int, depth: int):
        """
        @brief Initialize the ImageSequence3DPlusTime object from a loaded data array.

        @param param data: 4D numpy array with shape (time_length, depth, height, width)
        @param param time_length: Length of the time dimension
        @param param width: Length of the x dimension
        @param param height: Length of the y dimension
        @pram param depth: Length of the z dimension
        """
        self.data = data
        self.time_length = time_length
        self.width = width
        self.height = height
        self.depth = depth
        self.default_value = 0.0

    def get_data(self) -> np.ndarray:
        """
        @brief Get the data of the image sequence.

        @return: 4D numpy array with shape (time_length, depth, height, width)
        """
        return self.data

    def get_time_length(self) -> int:
        """
        @brief Get the length of the time dimension.

        @return: Length of the time dimension
        """
        return self.time_length

    def get_width(self) -> int:
        """
        @brief Get the width of the image sequence.

        @return: Width of the image sequence
        """
        return self.width

    def get_height(self) -> int:
        """
        @brief Get the height of the image sequence.

        @return: Height of the image sequence
        """
        return self.height

    def get_depth(self) -> int:
        """
        @brief Get the depth of the image sequence.

        @return: Depth of the image sequence
        """
        return self.depth

    def get_default_value(self) -> float:
        """
        @brief Get the default value used for empty bands in the image sequence.

        @return: Default value used for empty bands in the image sequence.
        """
        return self.default_value



    def save_scene(self, filename: str):
        """
        @brief Save the image sequence to a file.

        @param filename: The name of the file to save the image sequence to.
        """
        np.savez_compressed(filename, data=self.data, time_length=self.time_length,
                            width=self.width, height=self.height, depth=self.depth)


    def set_data(self, data: np.ndarray):
        """
        @brief Set the data of the image sequence.

        @param data: 4D numpy array with shape (time_length, depth, height, width)
        """
        self.data = data
        self.time_length, self.depth, self.height, self.width = data.shape

    def set_time_length(self,  time_length: int):
        """
        @brief Set the length of the time dimension.

        @param time_length: Length of the time dimension
        """
        self.time_length = time_length

    def set_width(self, width: int):
        """
        @brief Set the width of the image sequence.

        @param width: Width of the image sequence
        """
        self.width = width

    def set_height(self, height: int):
        """
        @brief Set the height of the image sequence.

        @param height: Height of the image sequence
        """
        self.height = height

    def set_depth(self, depth: int):
        """
        @brief Set the depth of the image sequence.

        @param depth: Depth of the image sequence
        """
        self.depth = depth

    def set_dimensions(self, time_length: int, depth: int, height: int, width: int):
        """
        @brief Set the dimensions of the image sequence.

        @param time_length: Length of the time dimension
        @param depth: Length of the z dimension
        @param height: Length of the y dimension
        @param width: Length of the x dimension
        """
        self.time_length = time_length
        self.depth = depth
        self.height = height
        self.width = width

    def set_default_value(self, default_value: float):
        """
        @brief Set the default value used for empty bands in the image sequence.

        @param default_value: Default value used for empty bands in the image sequence.
        """
        self.default_value = default_value


    def __repr__(self):
        """
        @brief String representation of the ImageSequence3DPlusTime object.

        @return: String representation of the object.
        """
        return (f"ImageSequence3DPlusTime(time_length={self.time_length}, "
                f"depth={self.depth}, height={self.height}, width={self.width})")

    def __str__(self):
        """
        @brief String representation of the ImageSequence3DPlusTime object.

        @return: String representation of the object.
        """
        return (f"ImageSequence3DPlusTime with time_length={self.time_length}, "
                f"depth={self.depth}, height={self.height}, width={self.width}")