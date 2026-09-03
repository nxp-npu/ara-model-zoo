# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import copy

import numpy as np

from core.python.config import Config
from core.python.preprocess.interfaces import PreprocessOutput
from core.python.preprocess.operations.bgrtorgb import BgrToRgb
from core.python.preprocess.preprocessing_builder import PreprocessingBuilder


def preprocess(image: np.ndarray, config: Config) -> PreprocessOutput:
    """Preprocess a single image for EfficientNet-Lite4 inference.

    The image goes through: colour conversion, centre crop at the original
    resolution, resize to the model input size, normalization and a
    channel-order transpose to produce a tensor ready for the hardware.
    """
    updated_config = copy.deepcopy(config)
    updated_config.preprocess.bgr_to_rgb = False
    updated_config.preprocess.centercrop = None

    assert config.preprocess.centercrop is not None

    image = BgrToRgb()(image)
    image_h, image_w = image.shape[:2]
    side = int(min(image_h, image_w) * config.preprocess.centercrop.fraction)
    top = (image_h - side) // 2
    left = (image_w - side) // 2
    image = image[top : top + side, left : left + side]
    return PreprocessingBuilder(updated_config).execute_pipeline(image)
