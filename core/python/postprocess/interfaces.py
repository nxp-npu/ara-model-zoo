# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .detection.coco_classes_labels import COCO_CLASSES
from .detection.visualization import object_detection_console_output
from ..output_display import (
    classification_console_output,
    face_estimation_console_output,
    pose_estimation_console_output,
)


@dataclass
class PostprocessingOutput(ABC):
    @abstractmethod
    def print_to_console(self) -> None:
        raise NotImplementedError


@dataclass
class SegDetectionOutput(PostprocessingOutput):
    """
    Structured representation of object detection predictions for a single image.
    """

    image_name: str
    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray
    masks: np.ndarray
    class_labels: tuple[str, ...] | list[str] | None = None

    def print_to_console(self) -> None:
        object_detection_console_output(
            image_path=self.image_name,
            model_input_shape=[0, 0, 0],
            boxes=self.boxes,
            scores=self.scores,
            cls_ids=self.classes,
            class_labels=list(self.class_labels or COCO_CLASSES),
        )


@dataclass
class ObjectDetectionOutput(PostprocessingOutput):
    """
    Structured representation of object detection predictions for a single image.
    """

    image_name: str
    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray
    class_labels: tuple[str, ...] | list[str] | None = None

    def print_to_console(self) -> None:
        object_detection_console_output(
            image_path=self.image_name,
            model_input_shape=[0, 0, 0],
            boxes=self.boxes,
            scores=self.scores,
            cls_ids=self.classes,
            class_labels=list(self.class_labels or COCO_CLASSES),
        )


@dataclass
class FaceDetectionOutput(PostprocessingOutput):
    """
    Structured representation of face detection predictions for a single image.

    Attributes
    ----------
    image_name : str
        Name of the image associated with these predictions.

    boxes : np.ndarray
        Bounding box predictions of shape (N, 4) in [x1, y1, x2, y2] format.
        Empty array with shape (0, 4) if no detections.

    scores : np.ndarray
        Confidence scores of shape (N,).
        Empty array with shape (0,) if no detections.

    classes : np.ndarray
        Class IDs of shape (N,).
        Empty array with shape (0,) if no detections.

    keypoints : np.ndarray
        Keypoint coordinates of shape (N, 5, 3), where each keypoint is
        (x, y, confidence). Empty array with shape (0, 5, 3) if no detections.
    """

    image_name: str
    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray
    keypoints: np.ndarray

    def print_to_console(self) -> None:
        face_estimation_console_output(
            image_name=self.image_name,
            kpts=self.keypoints,
            boxes=self.boxes,
            classes=self.classes,
            scores=self.scores,
        )


@dataclass
class ClassificationOutput(PostprocessingOutput):
    """
    This class represents the output for classification postprocessor

    Attributes
    ----------
    image_name : str
        Name of the image associated with these predictions.

    top_n: list[dict[str, np.int64 | np.float32]]
        top probabilities returned  by the postprocessor

    """

    top_n: list[dict[str, np.int64 | np.float32]]
    image_name: str

    def print_to_console(self) -> None:
        classification_console_output(
            top_k=self.top_n,
            image_path=self.image_name,
        )


@dataclass
class PoseEstimationOutput(PostprocessingOutput):
    """
    Structured representation of pose estimation predictions for a single image.
    """

    image_name: str
    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray
    keypoints: np.ndarray
    class_labels: tuple[str, ...] | list[str] | None = None

    def print_to_console(self) -> None:
        pose_estimation_console_output(
            image_name=self.image_name,
            kpts=self.keypoints,
            boxes=self.boxes,
            classes=self.classes,
            scores=self.scores,
        )
