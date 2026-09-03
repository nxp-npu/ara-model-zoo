# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from core.python.logger import logger
from core.python.postprocess.interfaces import (
    FaceDetectionOutput,
    ObjectDetectionOutput,
    PoseEstimationOutput,
    PostprocessingOutput,
    ClassificationOutput,
    SegDetectionOutput,
)
from core.python.visualizer.face_detection.facedetection import FaceDetection
from core.python.visualizer.object_detection.objectdetection import ObjectDetection
from core.python.visualizer.pose_estimation.poseestimation import PoseEstimation
from core.python.visualizer.classification.classification import Classification
from core.python.visualizer.segmentation.segmentation_detection import (
    SegmentationDetection,
)
from core.python.visualizer.utils import save

import numpy as np


PostProcessingOutputClasses = [
    (FaceDetectionOutput, FaceDetection),
    (ObjectDetectionOutput, ObjectDetection),
    (PoseEstimationOutput, PoseEstimation),
    (ClassificationOutput, Classification),
    (SegDetectionOutput, SegmentationDetection),
]


class Visualizer:
    def __init__(
        self,
        output_dir: str,
        thickness: int = 2,
        font_scale: float = 0.4,
        score_format: str = "{:.1f}%",
    ):
        self.output_dir = output_dir
        self.thickness = thickness
        self.font_scale = font_scale
        self.score_format = score_format

    def visualize(
        self,
        image: np.ndarray,
        results: PostprocessingOutput,
        output_filename: str = "frame.png",
    ) -> str | None:
        visualizer = None
        for postprocessedoutput, task in PostProcessingOutputClasses:
            if isinstance(results, postprocessedoutput):
                visualizer = task(
                    thickness=self.thickness,
                    font_scale=self.font_scale,
                    score_format=self.score_format,
                )

                break

        if visualizer is None:
            logger.error("No visualizer registered for result type: %s", type(results))
            return None

        image: np.ndarray | None = visualizer.visualizer(
            image=image,
            results=results,
        )

        if image is not None:
            return save(image, output_filename, self.output_dir)

        return None
