# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from core.python.config import Config
from core.python.inference import ModelOutput
from core.python.postprocess.detection.coco_classes_labels import COCO_CLASSES
from core.python.postprocess.detection_utils import scale_boxes, nms_yolov8
from core.python.postprocess.interfaces import ObjectDetectionOutput


def postprocess(
    model_output: ModelOutput,
    config: Config,
    original_image: npt.NDArray,
    image_name: str,
) -> ObjectDetectionOutput | None:

    # Extract prediction tensors
    # YOLOv7 output: (batch, num_boxes, x, y, w, h, obj_conf, cls_1, ..., cls_n)
    predictions = np.array(list(model_output.outputs.values())[0])

    # Fold objectness into class scores and reshape to nms_yolov8 format
    # Shape: (batch, 4 + num_classes, num_boxes)
    predictions[..., 5:] *= predictions[..., 4:5]
    predictions = predictions[..., [0, 1, 2, 3, *range(5, predictions.shape[2])]]
    predictions = predictions.transpose(0, -1, -2)

    assert config.postprocess.nms_score_threshold is not None
    assert config.postprocess.iou_threshold is not None
    assert config.postprocess.top_k is not None

    detections = nms_yolov8(
        pred=predictions,
        nms_score_threshold=config.postprocess.nms_score_threshold,
        iou_threshold=config.postprocess.iou_threshold,
        max_det=config.postprocess.top_k,
        is_agnostic_nms=False,
        multi_label=True,
    )[0]

    scaled_boxes = scale_boxes(
        (config.preprocess.input_shape.height, config.preprocess.input_shape.width),
        detections[:, :4].copy(),
        original_image.shape[:2],
    )

    return ObjectDetectionOutput(
        image_name=image_name,
        boxes=scaled_boxes,
        scores=detections[:, 4].astype(np.float32),
        classes=detections[:, 5].astype(np.int32),
        class_labels=COCO_CLASSES,
    )
