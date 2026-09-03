# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from typing import cast

import numpy as np

from core.python.logger import logger
from core.python.postprocess.interfaces import (
    PoseEstimationOutput,
    PostprocessingOutput,
)
from core.python.visualizer.base import VisualizerAbstract
from core.python.visualizer.utils import draw_detections, draw_pose_keypoints


class PoseEstimation(VisualizerAbstract):
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
        results = cast(PoseEstimationOutput, results)
        boxes = np.asarray(results.boxes)
        scores = np.asarray(results.scores)
        class_ids = (
            np.asarray(results.classes)
            if np.asarray(results.classes).size > 0
            else np.zeros(len(boxes), dtype=int)
        )
        keypoints = np.asarray(results.keypoints)

        img = image.copy()
        has_drawn = False

        if boxes.size and scores.size:
            logger.info("Going to draw pose detections")
            img = draw_detections(
                img,
                boxes,
                scores,
                class_ids,
                class_labels=list(results.class_labels or ("person",)),
                score_format=self.score_format,
                box_thickness=self.thickness,
                font_scale=self.font_scale,
            )
            has_drawn = True

        if keypoints.size:
            logger.info("Going to draw pose keypoints")
            img = draw_pose_keypoints(
                img,
                keypoints,
                line_thickness=self.thickness,
            )
            has_drawn = True

        if not has_drawn:
            logger.warning("No drawable pose estimation results found.")
            return None

        return img
