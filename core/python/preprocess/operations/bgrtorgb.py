# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import numpy
import cv2

from core.python.preprocess.operations.base import Base


class BgrToRgb(Base):
    """This class implements changing color space from BGR to RGB"""

    def __init__(self):
        super().__init__()

    def run(self, image: numpy.ndarray) -> numpy.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
