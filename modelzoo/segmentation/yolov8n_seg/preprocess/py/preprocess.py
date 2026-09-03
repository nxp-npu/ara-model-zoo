# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import numpy
from core.python.config import Config
from core.python.preprocess.preprocessing_builder import PreprocessingBuilder
from core.python.preprocess.interfaces import PreprocessOutput


def preprocess(image: numpy.ndarray, config: Config) -> PreprocessOutput:
    return PreprocessingBuilder(config).execute_pipeline(image)
