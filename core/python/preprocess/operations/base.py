# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from abc import ABC, abstractmethod
from numpy import ndarray


class Base(ABC):
    """This is the abstract base class. All of the operations need to inherit from this class"""

    def __init__(self):
        pass

    def __call__(self, image: ndarray) -> ndarray:
        return self.run(image)

    @abstractmethod
    def run(self, image: ndarray) -> ndarray:
        pass
