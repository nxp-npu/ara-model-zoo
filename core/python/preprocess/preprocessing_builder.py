# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import enum
import numpy as np

from core.python.preprocess.operations import Base
from core.python.preprocess.operations.bgrtorgb import BgrToRgb
from core.python.preprocess.operations.meansubtraction import MeanSubtraction
from core.python.preprocess.operations.resize import Resize
from core.python.preprocess.operations.scale import Scale
from core.python.preprocess.operations.tofloat import ToFloat
from core.python.preprocess.operations.transpose import ChannelFirstTranspose
from core.python.preprocess.operations.centercrop import CenterCrop
from core.python.preprocess.interfaces import PreprocessOutput
from core.python.config.config import Config


class PreprocessingBuilder:
    """This class creates a preprocessing pipeline based on the configs present in run.yaml"""

    class VALUES(str, enum.Enum):
        MEAN = "mean"
        SCALE = "scale"
        RESIZE = "resize"
        TO_FLOAT = "to_float"
        BGR_TO_RGB = "bgr_to_rgb"
        CHANNEL_FIRST_TRANSPOSE = "hwc_to_chw"
        CENTER_CROP = "centercrop"

    def __init__(self, config: Config):
        self.config = config
        values = self.VALUES
        self.module_precedence = {
            0: [values.BGR_TO_RGB, values.RESIZE, values.CENTER_CROP],
            1: [values.TO_FLOAT],
            2: [values.MEAN, values.SCALE],
            3: [values.CHANNEL_FIRST_TRANSPOSE],
        }

        self.pipeline = [
            transform
            for _, level in self.module_precedence.items()
            for module_name in level
            if (transform := self._get_module(module_name))
        ]

    def _get_module(self, key: VALUES) -> Base | None:
        values = self.VALUES

        if key.value == values.TO_FLOAT.value:
            if self.config.preprocess.to_float:
                return ToFloat()

        elif key.value == values.BGR_TO_RGB.value:
            if self.config.preprocess.bgr_to_rgb:
                return BgrToRgb()

        elif key.value == values.RESIZE.value:
            if self.config.preprocess.resize is not None:
                model_input_shape = self.config.preprocess.input_shape
                padding_value = self.config.preprocess.resize.padding

                output_shape = (model_input_shape.width, model_input_shape.height)
                interpolation = self.config.preprocess.resize.interpolation
                maintain_aspect_ratio = self.config.preprocess.resize.aspect_ratio
                scale = self.config.preprocess.resize.scale

                return Resize(
                    output_shape,
                    interpolation=interpolation,
                    maintain_aspect_ratio=maintain_aspect_ratio,
                    padding=True
                    if self.config.preprocess.resize.padding is not None
                    else False,
                    padding_value=padding_value,
                    scale=scale,
                    resize=self.config.preprocess.resize.resize_size,
                )

        elif key.value == values.CENTER_CROP.value:
            if self.config.preprocess.centercrop is not None:
                return CenterCrop(self.config.preprocess.centercrop)

        elif key.value == values.MEAN.value:
            mean_list = [
                self.config.preprocess.mean.r,
                self.config.preprocess.mean.g,
                self.config.preprocess.mean.b,
            ]
            return MeanSubtraction(
                mean_list, self.config.preprocess.input_shape.channels
            )

        elif key.value == values.SCALE.value:
            scale_list = [
                self.config.preprocess.scale.r,
                self.config.preprocess.scale.g,
                self.config.preprocess.scale.b,
            ]
            return Scale(scale_list, self.config.preprocess.input_shape.channels)

        elif key.value == values.CHANNEL_FIRST_TRANSPOSE.value:
            if self.config.preprocess.hwc_to_chw:
                return ChannelFirstTranspose()

        else:
            raise NotImplementedError(f"No implementation for `{key.value}`")

    def execute_pipeline(
        self,
        input_image: np.ndarray,
    ) -> PreprocessOutput:
        original_image = input_image.copy()

        for transform in self.pipeline:
            input_image = transform(input_image)

        return PreprocessOutput(
            original_image=original_image, processed_image=input_image
        )
