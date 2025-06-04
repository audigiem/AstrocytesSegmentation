import numpy as np


class Frame:
    """A class representing a frame at a specific time, containing attributes and their values."""

    def __init__(self, time: float, attributes: dict):
        """
        Initializes a Frame instance.

        @param :param time: The time at which the frame is recorded.
        :param attributes: A dictionary of attributes and their values.
        """
        self.time = time
        self.attributes = attributes