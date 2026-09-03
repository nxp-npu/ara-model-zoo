# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import numpy as np

from typing import cast
from core.python.visualizer.base import VisualizerAbstract
from core.python.postprocess.interfaces import PostprocessingOutput, FaceDetectionOutput
from core.python.visualizer.utils import draw_detections, draw_keypoints
from core.python.logger import logger


class FaceDetection(VisualizerAbstract):
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

        results = cast(FaceDetectionOutput, results)
        boxes = np.asarray(results.boxes)
        scores = np.asarray(results.scores)
        class_ids = (
            np.asarray(results.classes)
            if np.asarray(results.classes).size > 0
            else np.zeros(len(boxes), dtype=int)
        )
        keypoints = np.asarray(results.keypoints)

        has_drawn = False
        if boxes.size and scores.size:
            logger.info("Going to draw detections")
            img = draw_detections(
                image,
                boxes,
                scores,
                class_ids,
                class_labels=["face"],
            )
            has_drawn = True

        if keypoints.size:
            logger.info("Going to draw keypoints")
            img = draw_keypoints(img, keypoints)
            has_drawn = True

        if not has_drawn:
            logger.warning("No drawable results found.")
            return None

        return img
