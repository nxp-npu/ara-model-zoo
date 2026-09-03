# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from copy import deepcopy

import numpy as np

from core.python.config import Config
from core.python.preprocess.preprocessing_builder import PreprocessingBuilder
from core.python.preprocess.interfaces import PreprocessOutput
from core.python.preprocess.operations import BgrToRgb, CenterCrop


def preprocess(image: np.ndarray, config: Config) -> PreprocessOutput:
    updated_config: Config = deepcopy(config)

    # BGR -> RGB conversion
    bgr_to_rgb = BgrToRgb()
    image = bgr_to_rgb(image)
    updated_config.preprocess.bgr_to_rgb = False

    if config.preprocess.centercrop is None:
        raise ValueError("centercrop is not initialized")

    # MobileNet Center Crop
    fractional_central_crop = CenterCrop(config.preprocess.centercrop)
    image = fractional_central_crop(image)
    updated_config.preprocess.centercrop = None

    return PreprocessingBuilder(updated_config).execute_pipeline(image)
