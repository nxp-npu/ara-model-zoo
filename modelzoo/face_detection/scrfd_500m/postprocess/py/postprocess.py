# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from __future__ import annotations

from typing import Iterable

import numpy as np
import numpy.typing as npt

from core.python.config import Config
from core.python.inference import ModelOutput
from core.python.postprocess.detection_utils import (
    numpy_nms,
    scale_boxes,
    scale_pose_keypoints,
)
from core.python.postprocess.interfaces import FaceDetectionOutput

_FEATURE_STRIDES = (8, 16, 32)
_KEYPOINTS_PER_FACE = 5
_KEYPOINT_VALUES_PER_FACE = _KEYPOINTS_PER_FACE * 2


def _decode_centers_to_xyxy(
    centers: npt.NDArray[np.float32],
    distances: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    x1 = centers[:, 0] - distances[:, 0]
    y1 = centers[:, 1] - distances[:, 1]
    x2 = centers[:, 0] + distances[:, 2]
    y2 = centers[:, 1] + distances[:, 3]
    return np.stack((x1, y1, x2, y2), axis=-1).astype(np.float32, copy=False)


def _decode_centers_to_keypoints(
    centers: npt.NDArray[np.float32],
    distances: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    points: list[npt.NDArray[np.float32]] = []
    for index in range(0, distances.shape[1], 2):
        points.append(centers[:, index % 2] + distances[:, index])
        points.append(centers[:, (index % 2) + 1] + distances[:, index + 1])
    return np.stack(points, axis=-1).astype(np.float32, copy=False)


def _normalize_to_nchw(
    tensor: npt.NDArray[np.float32],
    values_per_anchor: int,
) -> tuple[npt.NDArray[np.float32], int, int, int]:
    array = np.asarray(tensor, dtype=np.float32)
    if array.ndim == 3:
        array = np.expand_dims(array, axis=0)

    if array.ndim != 4:
        raise ValueError(
            f"Expected SCRFD head tensor to be 4D, received shape {array.shape}"
        )

    if array.shape[1] % values_per_anchor == 0:
        nchw = array
    elif array.shape[-1] % values_per_anchor == 0:
        nchw = array.transpose(0, 3, 1, 2)
    else:
        raise ValueError(
            f"Unable to interpret SCRFD head tensor shape {array.shape} for "
            f"{values_per_anchor} values per anchor"
        )

    batch, channels, height, width = nchw.shape
    if batch != 1:
        raise ValueError(
            f"SCRFD postprocess expects a single-image batch, received batch={batch}"
        )

    anchors_per_location = channels // values_per_anchor
    return nchw, anchors_per_location, height, width


def _flatten_head(
    tensor: npt.NDArray[np.float32],
    values_per_anchor: int,
) -> tuple[npt.NDArray[np.float32], int, int, int]:
    nchw, anchors_per_location, height, width = _normalize_to_nchw(
        tensor,
        values_per_anchor,
    )
    flattened = nchw.transpose(0, 2, 3, 1).reshape(-1, values_per_anchor)
    return flattened.astype(np.float32, copy=False), anchors_per_location, height, width


def _anchor_centers(
    height: int,
    width: int,
    stride: int,
    anchors_per_location: int,
) -> npt.NDArray[np.float32]:
    y_coords = np.arange(height, dtype=np.float32)
    x_coords = np.arange(width, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(y_coords, x_coords, indexing="ij")
    centers = np.empty((height * width, 2), dtype=np.float32)
    centers[:, 0] = grid_x.reshape(-1) * stride
    centers[:, 1] = grid_y.reshape(-1) * stride
    if anchors_per_location > 1:
        centers = np.repeat(centers, anchors_per_location, axis=0)
    return centers.astype(np.float32, copy=False)


def _decode_scale(
    tensors: Iterable[npt.NDArray[np.float32]],
    stride: int,
    score_threshold: float,
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    score_tensor, box_tensor, keypoint_tensor = tensors

    scores, score_anchors, height, width = _flatten_head(score_tensor, 1)
    boxes, box_anchors, box_height, box_width = _flatten_head(box_tensor, 4)
    keypoints, keypoint_anchors, keypoint_height, keypoint_width = _flatten_head(
        keypoint_tensor,
        _KEYPOINT_VALUES_PER_FACE,
    )

    if (
        score_anchors != box_anchors
        or score_anchors != keypoint_anchors
        or height != box_height
        or height != keypoint_height
        or width != box_width
        or width != keypoint_width
    ):
        raise ValueError("SCRFD head tensors for a scale do not have matching shapes")

    scores = scores.reshape(-1)
    keep = np.where(scores >= score_threshold)[0]
    if keep.size == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, _KEYPOINTS_PER_FACE, 2), dtype=np.float32),
        )

    centers = _anchor_centers(height, width, stride, score_anchors)
    decoded_boxes = _decode_centers_to_xyxy(centers, boxes * stride)[keep]
    decoded_scores = scores[keep]
    decoded_keypoints = _decode_centers_to_keypoints(centers, keypoints * stride)[keep]
    decoded_keypoints = decoded_keypoints.reshape(-1, _KEYPOINTS_PER_FACE, 2)

    valid_boxes = (decoded_boxes[:, 2] >= decoded_boxes[:, 0]) & (
        decoded_boxes[:, 3] >= decoded_boxes[:, 1]
    )

    return (
        decoded_boxes[valid_boxes].astype(np.float32, copy=False),
        decoded_scores[valid_boxes].astype(np.float32, copy=False),
        decoded_keypoints[valid_boxes].astype(np.float32, copy=False),
    )


def postprocess(
    model_output: ModelOutput,
    config: Config,
    original_image: npt.NDArray[np.uint8],
    image_name: str,
) -> FaceDetectionOutput | None:
    outputs = [
        np.asarray(output, dtype=np.float32) for output in model_output.outputs.values()
    ]
    if len(outputs) != len(_FEATURE_STRIDES) * 3:
        raise ValueError(
            f"Expected 9 SCRFD outputs, received {len(outputs)}: "
            f"{model_output.output_names}"
        )

    postprocess_config = config.postprocess
    if postprocess_config is None:
        raise ValueError("SCRFD postprocessing configuration is required")

    score_threshold = postprocess_config.nms_score_threshold
    iou_threshold = postprocess_config.iou_threshold
    top_k = postprocess_config.top_k
    assert score_threshold is not None
    assert iou_threshold is not None

    all_boxes: list[npt.NDArray[np.float32]] = []
    all_scores: list[npt.NDArray[np.float32]] = []
    all_keypoints: list[npt.NDArray[np.float32]] = []

    for scale_index, stride in enumerate(_FEATURE_STRIDES):
        start = scale_index * 3
        decoded_boxes, decoded_scores, decoded_keypoints = _decode_scale(
            outputs[start : start + 3],
            stride,
            score_threshold,
        )
        if decoded_scores.size == 0:
            continue
        all_boxes.append(decoded_boxes)
        all_scores.append(decoded_scores)
        all_keypoints.append(decoded_keypoints)

    if not all_scores:
        return None

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    keypoints = np.concatenate(all_keypoints, axis=0)

    keep = np.asarray(numpy_nms(boxes, scores, iou_threshold), dtype=np.int32)
    if top_k is not None and keep.size > top_k:
        keep = keep[:top_k]

    boxes = boxes[keep]
    scores = scores[keep]
    keypoints = keypoints[keep]

    if scores.size == 0:
        return None

    preprocess_config = config.preprocess
    if preprocess_config is None:
        raise ValueError("SCRFD preprocessing configuration is required")

    input_shape = preprocess_config.input_shape
    model_input_shape = (input_shape.height, input_shape.width)
    input_image_shape = original_image.shape[:2]

    boxes_scaled = scale_boxes(
        model_input_shape,
        boxes,
        input_image_shape,
    )

    keypoint_scores = np.repeat(scores[:, None, None], _KEYPOINTS_PER_FACE, axis=1)
    keypoints_with_scores = np.concatenate(
        [keypoints, keypoint_scores],
        axis=2,
    ).astype(np.float32, copy=False)
    keypoints_scaled = scale_pose_keypoints(
        keypoints_with_scores,
        model_input_shape,
        input_image_shape,
        num_keypoints=_KEYPOINTS_PER_FACE,
    )

    classes = np.zeros((scores.shape[0],), dtype=np.int32)

    return FaceDetectionOutput(
        image_name=image_name,
        boxes=boxes_scaled.astype(np.float32, copy=False),
        scores=scores.astype(np.float32, copy=False),
        classes=classes,
        keypoints=keypoints_scaled.astype(np.float32, copy=False),
    )
