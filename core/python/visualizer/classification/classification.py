# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import numpy as np

from typing import cast
from core.python.visualizer.base import VisualizerAbstract
from core.python.visualizer.utils import draw_classification_output
from core.python.postprocess.interfaces import (
    PostprocessingOutput,
    ClassificationOutput,
)
from core.python.logger import logger


class Classification(VisualizerAbstract):
    def __init__(
        self,
        thickness: int = 2,
        font_scale: float = 0.4,
        score_format: str = "{:.1f}%",
    ):
        self.thickness = thickness
        self.font_scale = font_scale
        self.score_format = score_format

    def visualizer(
        self,
        image: np.ndarray,
        results: PostprocessingOutput,
    ) -> np.ndarray | None:

        results = cast(ClassificationOutput, results)

        image = draw_classification_output(results.top_n, image)
        logger.info("Labels have been drawn on the image")

        return image
