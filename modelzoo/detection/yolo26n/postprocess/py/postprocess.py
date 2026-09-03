# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

"""
YOLO26n Postprocessing for Ara-model-zoo Pipeline.

Model outputs (varies by flow):

  Float eval, single combined tensor:
    output_0: (1, 300, 6)  top-300 detections [x1, y1, x2, y2, score, class_id]

  Device run, two split tensors:
    Concat_output:  (1, 4, 8400, 1)  raw boxes in grid units (before stride scaling)
    Sigmoid_output: (1, 80, 8400, 1) post-sigmoid class probabilities

  Anchor distribution across three FPN scales:
    80x80 grid = 6400 anchors at stride  8
    40x40 grid = 1600 anchors at stride 16
    20x20 grid =  400 anchors at stride 32
    Total:        8400 anchors

Pipeline:
    1. Squeeze trailing singleton dims from ARA outputs if present
    2. Identify output format: combined (1,300,6) or split (1,4,8400)+(1,80,8400)
    3a. if (1,300,6) then directly calculate scale boxes adn return it
    3b. For split tensors: multiply boxes by strides [8,16,32] to convert grid→pixel coords
    4. Concatenate to (1, 84, 8400)
    5. Two-pass top-k selection (replaces NMS):
       a. Per-anchor max class score → top-300 anchors
       b. Flatten 300x80 scores → top-300 global detections
    6. Filter by score threshold, finite values, and positive box area
    7. Scale boxes from model input space (640x640) → original image space
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from core.python.config import Config
from core.python.inference import ModelOutput
from core.python.postprocess.detection.coco_classes_labels import COCO_CLASSES
from core.python.postprocess.detection_utils import scale_boxes
from core.python.postprocess.interfaces import ObjectDetectionOutput

_NC = 80
_BOX_CHANNELS = 4

_STRIDES = np.concatenate([
    np.full(6400, 8.0),
    np.full(1600, 16.0),
    np.full(400, 32.0),
], dtype=np.float32).reshape(1, 1, 8400)


def _normalize_tensor(tensor: np.ndarray) -> np.ndarray:
    """Ensure tensor layout is (1, C, 8400).

    Args:
        tensor: 3D array, either (1, C, 8400) or (1, 8400, C).

    Returns:
        Array in (1, C, 8400) layout, float32.

    Raises:
        ValueError: If tensor is not 3D.
    """
    if tensor.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got shape {tensor.shape}")
    if tensor.shape[2] in {_BOX_CHANNELS, _NC, _BOX_CHANNELS + _NC}:
        tensor = tensor.transpose(0, 2, 1)
    return tensor.astype(np.float32, copy=False)


def _topk_detections(predictions: np.ndarray, max_det: int = 300) -> np.ndarray:
    """Two-pass top-k selection end2end postprocess.

    Replaces NMS for one-to-one head models. First pass selects top-k
    anchors by maximum class score. Second pass selects top-k globally
    from the k x 80 anchor-class score matrix.

    Args:
        predictions: Array of shape (1, 84, 8400) with xyxy boxes in pixel
            coordinates (channels 0-3) and sigmoid class probabilities
            (channels 4-83).
        max_det: Maximum number of detections to return.

    Returns:
        Array of shape (N, 6) where each row is
        [x1, y1, x2, y2, score, class_id] in 640x640 letterboxed pixel space.
        N <= max_det.
    """
    preds = predictions[0].transpose(1, 0)
    boxes = preds[:, :4]
    scores = preds[:, 4:]

    A, num_cls = scores.shape
    k = min(max_det, A)

    max_per_anchor = scores.max(axis=1)
    ori_index = np.argpartition(-max_per_anchor, k)[:k]
    ori_index = ori_index[np.argsort(-max_per_anchor[ori_index])]

    gathered = scores[ori_index]
    flat = gathered.ravel()
    flat_k = min(k, flat.size)
    flat_topk_idx = np.argpartition(-flat, flat_k)[:flat_k]
    flat_topk_idx = flat_topk_idx[np.argsort(-flat[flat_topk_idx])]

    flat_scores = flat[flat_topk_idx]
    anchor_in_gathered = flat_topk_idx // num_cls
    cls_ids = flat_topk_idx % num_cls

    sel_boxes = boxes[ori_index[anchor_in_gathered]]

    return np.column_stack([
        sel_boxes,
        flat_scores,
        cls_ids.astype(np.float32),
    ])


def _extract_detections(model_output: ModelOutput, max_det: int = 300) -> np.ndarray:
    """Route model outputs to the correct decoding path.

    Handles two output formats:
      - End2end (1, 300, 6): model with baked-in top-k selection.
      - Split (1, 4, 8400) + (1, 80, 8400): ARA compiled output with boxes
        in grid units requiring stride multiplication.

    DVM outputs may carry a trailing singleton dim (1, C, 8400, 1) which
    is removed before processing.

    Args:
        model_output: ModelOutput instance from the inference session.
            output.outputs is a dict mapping tensor names to numpy arrays.
        max_det: Maximum detections to return from top-k selection.

    Returns:
        Array of shape (N, 6): [x1, y1, x2, y2, score, class_id]
        in 640x640 letterboxed pixel coordinates.

    Raises:
        ValueError: If output tensors cannot be identified as any known format.
    """
    tensors = {
        name: np.asarray(t, dtype=np.float32)
        for name, t in model_output.outputs.items()
    }

    tensors = {
        name: t.reshape(t.shape[0], t.shape[1], -1) if t.ndim == 4 and t.shape[-1] == 1 else t
        for name, t in tensors.items()
    }

    for name, t in tensors.items():
        if t.ndim == 3 and t.shape[-1] == 6:
            return t.reshape(-1, 6)

    normed = {n: _normalize_tensor(t) for n, t in tensors.items() if t.ndim == 3}
    box = next((t for t in normed.values() if t.shape[1] == _BOX_CHANNELS), None)
    cls = next((t for t in normed.values() if t.shape[1] == _NC), None)

    if box is None or cls is None:
        raise ValueError(
            f"Cannot identify box/cls from: {[(n, t.shape) for n, t in tensors.items()]}"
        )

    box = box * _STRIDES
    combined = np.concatenate([box, cls], axis=1)
    return _topk_detections(combined, max_det)


def postprocess(
    model_output: ModelOutput,
    config: Config,
    original_image: npt.NDArray,
    image_name: str,
) -> ObjectDetectionOutput | None:
    """Decode model outputs into scaled bounding box detections.

    Entry point called by the Ara-model-zoo evaluation pipeline for each image.
    Extracts detections from raw model outputs, filters by score and validity,
    and scales coordinates from 640x640 letterbox space to original image space.

    Args:
        model_output: Raw inference outputs wrapped in a ModelOutput instance.
        config: Pipeline configuration with preprocess shape, score threshold,
            top_k, and other postprocess parameters.
        original_image: Original BGR image as read by cv2, shape (H, W, 3).
            Used only for its dimensions during coordinate scaling.
        image_name: Filename of the image, passed through to the output.

    Returns:
        ObjectDetectionOutput with boxes, scores, classes, and labels,
        or None if no valid detections remain after filtering.
    """
    max_det = config.postprocess.top_k or 300
    detections = _extract_detections(model_output, max_det)

    score_threshold = config.postprocess.nms_score_threshold or 0.001
    valid = detections[:, 4] > score_threshold
    valid &= np.isfinite(detections).all(axis=1)
    valid &= (detections[:, 2] > detections[:, 0]) & (detections[:, 3] > detections[:, 1])
    detections = detections[valid]

    if detections.shape[0] == 0:
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
