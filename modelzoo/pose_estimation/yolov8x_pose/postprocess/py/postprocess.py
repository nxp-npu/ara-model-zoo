# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from __future__ import annotations

from typing import Mapping

import numpy as np
import numpy.typing as npt

from core.python.config import Config
from core.python.inference import ModelOutput
from core.python.postprocess.detection_utils import (
    nms_yolov8,
    scale_boxes,
    scale_pose_keypoints,
)
from core.python.postprocess.interfaces import PoseEstimationOutput


NUM_KEYPOINTS = 17
KEYPOINT_DIMS = 3

YOLOV8_POSE_CHANNELS = 4 + 1 + (17 * 3)


def _normalize_node_name(name: str) -> str:
    return name.replace(".", "_").replace("/", "_")


def _get_yolov8_pose_cutoff_tensor(
    outputs: Mapping[str, np.ndarray], marker: str
) -> np.ndarray:
    matches = [
        np.asarray(tensor)
        for name, tensor in outputs.items()
        if marker in _normalize_node_name(name)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one YOLOv8 pose cutoff tensor for '{marker}', "
            f"found {len(matches)} in {list(outputs)}"
        )
    return matches[0]


def merge_yolov8_pose_cutoff_outputs(
    outputs: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Merge ARA cutoff tensors into a single (1, 56, N) pose prediction tensor."""
    boxes = _get_yolov8_pose_cutoff_tensor(outputs, "Mul_2_output").reshape(1, 4, -1)
    boxes_sigmoid = _get_yolov8_pose_cutoff_tensor(outputs, "Sigmoid_output_0").reshape(
        1, 1, -1
    )

    pose_mul = _get_yolov8_pose_cutoff_tensor(outputs, "Mul_4_output").reshape(
        1, 17, 2, -1
    )
    pose_sig = _get_yolov8_pose_cutoff_tensor(outputs, "Sigmoid_1_output").reshape(
        1, 17, 1, -1
    )
    pose_out = np.concatenate((pose_mul, pose_sig), axis=2).reshape(1, 51, -1)
    return np.concatenate((boxes, boxes_sigmoid, pose_out), axis=1).astype(
        np.float32, copy=False
    )


def extract_yolov8_pose_prediction_tensor(
    outputs: Mapping[str, np.ndarray],
) -> np.ndarray:
    tensors = {name: np.asarray(value) for name, value in outputs.items()}

    if len(tensors) == 1:
        tensor = next(iter(tensors.values()))
    elif len(tensors) == 4:
        normalized_names = [_normalize_node_name(name) for name in tensors]
        required_markers = (
            "Mul_2_output",
            "Sigmoid_output_0",
            "Mul_4_output",
            "Sigmoid_1_output",
        )
        if all(
            any(marker in name for name in normalized_names)
            for marker in required_markers
        ):
            return merge_yolov8_pose_cutoff_outputs(tensors)
        raise ValueError(
            f"Expected a single YOLOv8 pose output tensor, received {len(tensors)}"
        )
    else:
        normalized = []
        for tensor in tensors.values():
            if tensor.ndim != 3:
                raise ValueError(
                    "Expected YOLOv8 pose output tensor to be 3D, "
                    f"but received shape {tensor.shape}"
                )
            if tensor.shape[1] == YOLOV8_POSE_CHANNELS:
                normalized.append(tensor.astype(np.float32, copy=False))
            elif tensor.shape[2] == YOLOV8_POSE_CHANNELS:
                normalized.append(
                    tensor.transpose(0, 2, 1).astype(np.float32, copy=False)
                )
            else:
                raise ValueError(
                    f"Unsupported YOLOv8 pose output tensor shape: {tensor.shape}"
                )

        if len(normalized) == 1:
            return normalized[0]

        pose_tensors = [
            tensor for tensor in normalized if tensor.shape[1] == YOLOV8_POSE_CHANNELS
        ]
        if len(pose_tensors) == 1:
            return pose_tensors[0]

        raise ValueError(
            f"Expected a single YOLOv8 pose output tensor, received {len(tensors)}"
        )

    if tensor.ndim != 3:
        raise ValueError(
            "Expected YOLOv8 pose output tensor to be 3D, "
            f"but received shape {tensor.shape}"
        )
    if tensor.shape[1] == YOLOV8_POSE_CHANNELS:
        return tensor.astype(np.float32, copy=False)
    if tensor.shape[2] == YOLOV8_POSE_CHANNELS:
        return tensor.transpose(0, 2, 1).astype(np.float32, copy=False)

    raise ValueError(f"Unsupported YOLOv8 pose output tensor shape: {tensor.shape}")


def postprocess(
    model_output: ModelOutput,
    config: Config,
    original_image: npt.NDArray,
    image_name: str,
) -> PoseEstimationOutput | None:
    predictions = extract_yolov8_pose_prediction_tensor(model_output.outputs)
    assert config.postprocess.nms_score_threshold is not None
    assert config.postprocess.iou_threshold is not None
    assert config.postprocess.top_k is not None
    detections = nms_yolov8(
        pred=predictions,
        nms_score_threshold=config.postprocess.nms_score_threshold,
        iou_threshold=config.postprocess.iou_threshold,
        max_det=config.postprocess.top_k,
        num_keypoints=NUM_KEYPOINTS,
        keypoint_dims=KEYPOINT_DIMS,
    )[0]

    if detections.size == 0:
        return None

    model_input_shape = (
        config.preprocess.input_shape.height,
        config.preprocess.input_shape.width,
    )
    image_shape = original_image.shape[:2]

    boxes = scale_boxes(
        model_input_shape,
        detections[:, :4].copy(),
        image_shape,
    ).astype(np.float32, copy=False)
    keypoints = scale_pose_keypoints(
        detections[:, 6:].copy(),
        model_input_shape,
        image_shape,
        num_keypoints=NUM_KEYPOINTS,
    )

    return PoseEstimationOutput(
        image_name=image_name,
        boxes=boxes,
        scores=detections[:, 4].astype(np.float32, copy=False),
        classes=detections[:, 5].astype(np.int32, copy=False),
        keypoints=keypoints,
        class_labels=("person",),
    )
