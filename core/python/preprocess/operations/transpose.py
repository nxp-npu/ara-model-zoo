# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import numpy

from core.python.preprocess.operations.base import Base


class ChannelFirstTranspose(Base):
    """This class transposes the channels"""

    def __init__(self):
        super().__init__()

    def run(self, image: numpy.ndarray) -> numpy.ndarray:
        # Below code should work for for both 3 and 1 channels. But it has not been tested for 1 channel
        image = image.transpose(2, 0, 1)
        return image
