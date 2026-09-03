# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from abc import ABC, abstractmethod
from core.python.postprocess.interfaces import PostprocessingOutput

import numpy as np


class VisualizerAbstract(ABC):
    """This is an abstract class for visualizer. All task specific classes should inherit from it"""

    def __init__(
        self,
    ):
        pass

    @abstractmethod
    def visualizer(
        self, image: np.ndarray, results: PostprocessingOutput
    ) -> np.ndarray | None:
        """This method would display the post processed output onto original image
        based on the task e.g. detection, face detection etc."""
        pass
