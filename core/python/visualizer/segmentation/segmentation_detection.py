# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from typing import cast

import numpy as np

from core.python.logger import logger
from core.python.postprocess.detection.coco_classes_labels import COCO_CLASSES, COLORS
from core.python.postprocess.interfaces import SegDetectionOutput, PostprocessingOutput
from core.python.visualizer.base import VisualizerAbstract
from core.python.visualizer.utils import draw_segmentations


class SegmentationDetection(VisualizerAbstract):
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
        results = cast(SegDetectionOutput, results)
        boxes = np.asarray(results.boxes)
        scores = np.asarray(results.scores)
        masks = np.asarray(results.masks)
        class_ids = (
            np.asarray(results.classes)
            if np.asarray(results.classes).size > 0
            else np.zeros(len(boxes), dtype=int)
        )

        if not boxes.size or not scores.size:
            logger.warning("No drawable object detections found.")
            return None

        return draw_segmentations(
            image,
            boxes,
            scores,
            masks,
            class_ids,
            colors=COLORS,
            class_labels=list(results.class_labels or COCO_CLASSES),
            score_format=self.score_format,
            box_thickness=self.thickness,
            font_scale=self.font_scale,
        )
