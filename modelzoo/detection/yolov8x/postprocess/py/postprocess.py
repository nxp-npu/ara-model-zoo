# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from core.python.config import Config
from core.python.inference import ModelOutput
from core.python.postprocess.detection.coco_classes_labels import COCO_CLASSES
from core.python.postprocess.detection_utils import nms_yolov8, scale_boxes
from core.python.postprocess.interfaces import ObjectDetectionOutput


def _normalize_output_tensor(tensor: np.ndarray) -> np.ndarray:
    if tensor.ndim != 3:
        raise ValueError(
            f"Expected YOLOv8 output tensor to be 3D, but received shape {tensor.shape}"
        )

    if tensor.shape[1] in {4, 80, 84}:
        return tensor.astype(np.float32, copy=False)
    if tensor.shape[2] in {4, 80, 84}:
        return tensor.transpose(0, 2, 1).astype(np.float32, copy=False)

    raise ValueError(f"Unsupported YOLOv8 output tensor shape: {tensor.shape}")


def _extract_prediction_tensor(model_output: ModelOutput) -> np.ndarray:
    tensors = [
        _normalize_output_tensor(np.asarray(output))
        for output in model_output.outputs.values()
    ]

    if len(tensors) == 1:
        tensor = tensors[0]
        if tensor.shape[1] != 84:
            raise ValueError(
                f"Expected final YOLOv8 tensor with 84 channels, got {tensor.shape}"
            )
        return tensor

    if len(tensors) == 2:
        box_tensor = next((tensor for tensor in tensors if tensor.shape[1] == 4), None)
        class_tensor = next(
            (tensor for tensor in tensors if tensor.shape[1] == 80), None
        )
        if box_tensor is None or class_tensor is None:
            raise ValueError(
                "Unable to identify YOLOv8 box/class tensors from model outputs"
            )
        return np.concatenate([box_tensor, class_tensor], axis=1)

    raise ValueError(f"Expected 1 or 2 YOLOv8 output tensors, received {len(tensors)}")


def postprocess(
    model_output: ModelOutput,
    config: Config,
    original_image: npt.NDArray,
    image_name: str,
) -> ObjectDetectionOutput | None:
    predictions = _extract_prediction_tensor(model_output)
    detections = nms_yolov8(
        pred=predictions,
        nms_score_threshold=config.postprocess.nms_score_threshold,
        iou_threshold=config.postprocess.iou_threshold,
        max_det=config.postprocess.top_k,
        multi_label=True,
    )[0]

    if detections.size == 0:
        return None

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
