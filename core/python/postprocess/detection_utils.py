# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from typing import List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt


def scale_pose_keypoints(
    keypoints: npt.NDArray,
    model_input_shape: Tuple[int, int],
    image_shape: Tuple[int, int],
    num_keypoints: int = 17,
) -> npt.NDArray:
    """
    Scale pose keypoints from model-input coordinates to image coordinates.

    Assumed input shape:
        - (N, num_keypoints * 3)

    Also accepts:
        - (N, num_keypoints, 3)

    Args:
        keypoints:
            Pose keypoint tensor containing N detected poses.

        model_input_shape:
            Model input shape as (height, width).

        image_shape:
            Original image shape as (height, width).

        num_keypoints:
            Expected number of keypoints per pose.

            Note:
                num_keypoints is not explicitly validated.
                Negative values are interpreted by NumPy reshape as an inferred
                dimension rather than a literal keypoint count.

    Returns:
        Array of shape (N, num_keypoints, 3) when the reshape
        operation is compatible.

        Empty input returns an array of shape
        (0, num_keypoints, 3) with dtype float32.

    Raises:
        ValueError:
            If a pose cannot be reshaped to
            (num_keypoints, 3).
    """
    keypoint_list = []

    for kpt in keypoints:
        kpt = kpt.reshape(num_keypoints, 3)
        kpt = scale_coords(model_input_shape, kpt, image_shape)
        keypoint_list.append(kpt)

    return (
        np.stack(keypoint_list, axis=0).astype(np.float32, copy=False)
        if keypoint_list
        else np.zeros((0, num_keypoints, 3), dtype=np.float32)
    )


def scale_coords(
    model_input_shape: Tuple[int, int],
    coords: np.ndarray,
    img_shape: Tuple[int, int],
    ratio_pad: Tuple[Tuple[float], Tuple[float, float]] | None = None,
) -> np.ndarray:
    """
    Rescale coordinates from model input size to original image size.

    Args:
        model_input_shape (tuple): Height and width of model input.

        coords (np.ndarray): Coordinates to scale (shape [n, 2] or [n, 3]).
            Note:
                This function does not explicitly validate input
                dimensionality. Consequently, some higher-dimensional
                inputs may be accepted when the second dimension
                contains at least two elements, while others may
                raise indexing errors.

        img_shape (tuple): Original image height and width.
        ratio_pad (tuple, optional): Precomputed gain and padding. Defaults to None.

    Returns:
        np.ndarray: Scaled coordinates clipped to image boundaries.
    """
    if ratio_pad is None:
        gain = min(
            model_input_shape[0] / img_shape[0], model_input_shape[1] / img_shape[1]
        )
        pad = (
            (model_input_shape[1] - img_shape[1] * gain) / 2,
            (model_input_shape[0] - img_shape[0] * gain) / 2,
        )
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, 0] = np.clip((coords[:, 0] - pad[0]) / gain, 0, img_shape[1])
    coords[:, 1] = np.clip((coords[:, 1] - pad[1]) / gain, 0, img_shape[0])

    return coords


def scale_boxes(
    model_input_shape: Tuple[int, int],
    coords: np.ndarray,
    img_shape: Tuple[int, int],
    ratio_pad: Optional[Tuple[Tuple[float], Tuple[float, float]]] = None,
) -> np.ndarray:
    """
    Rescale bounding boxes from model input size to original image size.

    Args:
        model_input_shape (tuple): Height and width of model input.

        coords (np.ndarray): Bounding boxes to scale (shape [n, 4]).
            Note:
                This function does not explicitly validate input
                dimensionality. Consequently, some higher-dimensional
                inputs may be accepted when the second dimension
                contains at least two elements, while others may
                raise indexing errors.

        img_shape (tuple): Original image height and width.
        ratio_pad (tuple, optional): Precomputed gain and padding. Defaults to None.

    Returns:
        np.ndarray: Scaled bounding boxes clipped to image boundaries.
    """
    img_height, img_width = img_shape[:2]

    if ratio_pad is None:
        gain = min(
            model_input_shape[0] / img_height,
            model_input_shape[1] / img_width,
        )
        pad = (
            round((model_input_shape[1] - round(img_width * gain)) / 2 - 0.1),
            round((model_input_shape[0] - round(img_height * gain)) / 2 - 0.1),
        )
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] = np.clip((coords[:, [0, 2]] - pad[0]) / gain, 0, img_width)
    coords[:, [1, 3]] = np.clip((coords[:, [1, 3]] - pad[1]) / gain, 0, img_height)

    return coords


def scale_mask(mask: np.ndarray, shape: list) -> np.ndarray:
    """
    Scales the mask to match the dimensions of the original image
    and removes padding to align aspect ratio with original image

    Args:
        masks (np.ndarray): Detected binary mask of shape [h, w]
        shape (tuple): Shape of original image as [h, w]

    Returns:
        (np.ndarray): A scaled mask to the original image.
    """
    assert len(shape) == 2
    assert len(mask.shape) == 2

    if not ((mask == 0) | (mask == 1)).all():
        raise ValueError("Expected a binary mask containing only values 0 and 1.")

    mask_h, mask_w = mask.shape
    h, w = shape

    ratio = min(mask_h / h, mask_w / w)
    pad_y = int((mask_h - h * ratio) // 2)
    pad_x = int((mask_w - w * ratio) // 2)

    mask = mask[pad_y : mask_h - pad_y, pad_x : mask_w - pad_x]
    mask = mask.astype(np.uint8) * 255
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask = cv2.resize(mask, [w, h], interpolation=cv2.INTER_LINEAR)
    mask[mask > 127] = 255
    mask[mask <= 127] = 0

    return mask


def crop_mask(
    masks: np.ndarray,
    boxes: np.ndarray,
) -> np.ndarray:
    """
    Zero out mask pixels outside their corresponding bounding boxes.

    Each input mask is cropped by its corresponding bounding box.
    Pixels outside the box are set to zero while pixels inside the
    box are preserved.

    Args:
        masks:
            Binary or continuous masks of shape ``(N, H, W)``.

        boxes:
            Bounding boxes of shape ``(N, 4)`` in
            ``(x1, y1, x2, y2)`` format.

            Floating-point coordinates are supported and are
            interpreted using NumPy comparison semantics against
            integer pixel locations.

            Each bounding box must satisfy::

                x2 >= x1
                y2 >= y1

            All coordinates must be finite.

    Returns:
        np.ndarray:
            Cropped masks with the same shape and dtype as
            ``masks``.

    Raises:
        ValueError:
            If:
            - ``masks`` is not of shape ``(N, H, W)``.
            - ``boxes`` is not of shape ``(N, 4)``.
            - The number of masks and boxes differ.
            - Any bounding-box coordinate is non-finite.
            - Any bounding box is inverted
              (``x2 < x1`` or ``y2 < y1``).
    """

    if masks.ndim != 3:
        raise ValueError("Expected masks with shape (N, H, W).")

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("Expected boxes with shape (N, 4).")

    if masks.shape[0] != boxes.shape[0]:
        raise ValueError("Expected one bounding box per mask.")

    if np.any(boxes[:, 2:] < boxes[:, :2]):
        raise ValueError(
            "Expected bounding boxes in (x1, y1, x2, y2) "
            "format with x2 >= x1 and y2 >= y1."
        )

    _, h, w = masks.shape

    x1, y1, x2, y2 = np.split(
        boxes[:, :, None],
        4,
        axis=1,
    )
    # Each has shape (N, 1, 1)

    r = np.arange(w)[None, None, :]
    # Pixel x-coordinates, shape (1, 1, W)

    c = np.arange(h)[None, :, None]
    # Pixel y-coordinates, shape (1, H, 1)

    crop = (r >= x1) & (r < x2) & (c >= y1) & (c < y2)
    # Boolean crop mask, shape (N, H, W)

    return masks * crop


def process_mask(
    protos: np.ndarray,
    masks_in: np.ndarray,
    bboxes: np.ndarray,
    shape: tuple[int, int],
    upsample: bool = True,
) -> np.ndarray:
    """
    Apply prototype masks to bounding boxes.

    Args:
        protos:
            Prototype masks of shape ``(C, mask_h, mask_w)``.

        masks_in:
            Mask coefficients of shape ``(N, C)``,
            where ``N`` is the number of detections.

        bboxes:
            Bounding boxes of shape ``(N, 4)``
            in ``(x1, y1, x2, y2)`` format.

        shape:
            Output image shape ``(height, width)``.

        upsample:
            Currently unused and retained only for API compatibility.

    Returns:
        np.ndarray:
            Boolean masks of shape ``(N, height, width)``.

    Raises:
        ValueError:
            If:

            - ``shape`` is not ``(height, width)``.
            - Any target dimension is non-positive.
            - ``protos`` is not of shape ``(C, H, W)``.
            - ``masks_in`` is not of shape ``(N, C)``.
            - ``masks_in`` and ``protos`` have incompatible channel counts.
            - ``masks_in`` and ``bboxes`` have different numbers of detections.
            - ``protos`` or ``masks_in`` contain non-finite values.

            Validation of ``bboxes`` is delegated to
            ``crop_mask()``.
    """

    if len(shape) != 2:
        raise ValueError("Expected shape as (height, width).")

    height, width = shape

    if height <= 0 or width <= 0:
        raise ValueError("Expected positive output dimensions.")

    if protos.ndim != 3:
        raise ValueError("Expected protos with shape (C, H, W).")

    if masks_in.ndim != 2:
        raise ValueError("Expected masks_in with shape (N, C).")

    channels, mask_h, mask_w = protos.shape

    if channels == 0 or mask_h == 0 or mask_w == 0:
        raise ValueError("Expected non-empty prototype tensor.")

    if masks_in.shape[1] != channels:
        raise ValueError("Expected masks_in.shape[1] to equal protos.shape[0].")

    if masks_in.shape[0] != bboxes.shape[0]:
        raise ValueError("Expected one bounding box per mask.")

    num_masks = masks_in.shape[0]

    if num_masks == 0:
        return np.empty(
            (0, height, width),
            dtype=bool,
        )

    protos = protos.astype(
        np.float32,
        copy=False,
    )

    masks = (masks_in @ protos.reshape(channels, -1)).reshape(
        num_masks,
        mask_h,
        mask_w,
    )

    masks = np.stack(
        [
            cv2.resize(
                mask,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )
            for mask in masks
        ],
        axis=0,
    )

    masks = crop_mask(
        masks=masks,
        boxes=bboxes,
    )

    return masks > 0.0


def numpy_nms(
    dets: np.ndarray,
    scores: np.ndarray,
    thresh: float,
) -> list[int]:
    """
    Perform Non-Maximum Suppression (NMS) on detection boxes.

    Boxes are expected in ``(x1, y1, x2, y2)`` format, where
    ``(x1, y1)`` is the top-left corner and ``(x2, y2)`` is the
    bottom-right corner.

    Args:
        dets:
            Bounding boxes of shape ``(N, 4)`` in
            ``(x1, y1, x2, y2)`` format.

            Each bounding box must satisfy::

                x2 >= x1
                y2 >= y1

        scores:
            Confidence scores of shape ``(N,)``,
            one score per bounding box.

        thresh:
            IoU threshold for suppression.

            The threshold range is validated by the caller.

    Returns:
        list[int]:
            Indices of the boxes to keep, ordered by descending
            confidence score.

    Raises:
        ValueError:
            If:

            - ``dets`` is not of shape ``(N, 4)``.
            - ``scores`` is not of shape ``(N,)``.
            - The number of detections and scores differ.
            - Any bounding box is inverted
              (``x2 < x1`` or ``y2 < y1``).
    """

    if dets.ndim != 2 or dets.shape[1] != 4:
        raise ValueError("Expected dets with shape (N, 4).")

    if scores.ndim != 1:
        raise ValueError("Expected scores with shape (N,).")

    if dets.shape[0] != scores.shape[0]:
        raise ValueError("Expected one score per detection.")

    if np.any(dets[:, 2:] < dets[:, :2]):
        raise ValueError(
            "Expected bounding boxes in (x1, y1, x2, y2) "
            "format with x2 >= x1 and y2 >= y1."
        )

    x1, y1, x2, y2 = (
        dets[:, 0],
        dets[:, 1],
        dets[:, 2],
        dets[:, 3],
    )

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort(kind="mergesort")[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        order = order[np.where(ovr <= thresh)[0] + 1]

    return keep


def xyxy2xywh(
    x: np.ndarray,
) -> np.ndarray:
    """
    Convert bounding boxes from ``(x1, y1, x2, y2)`` to
    ``(x, y, width, height)``.

    The top-left corner ``(x, y)`` is preserved while the
    width and height are computed as::

        width = x2 - x1
        height = y2 - y1

    Args:
        x:
            Bounding boxes of shape ``(N, 4)`` in
            ``(x1, y1, x2, y2)`` format.

            Each bounding box must satisfy::

                x2 >= x1
                y2 >= y1

            All coordinates must be finite.

    Returns:
        np.ndarray:
            Bounding boxes of shape ``(N, 4)`` in
            ``(x, y, width, height)`` format.

            The returned array has the same dtype as the input
            and is a copy of the input array.

    Raises:
        ValueError:
            If:

            - ``x`` is not of shape ``(N, 4)``.
            - Any coordinate is non-finite.
            - Any bounding box is inverted
              (``x2 < x1`` or ``y2 < y1``).
    """

    if x.ndim != 2 or x.shape[1] != 4:
        raise ValueError("Expected boxes with shape (N, 4).")

    if np.any(x[:, 2:] < x[:, :2]):
        raise ValueError(
            "Expected bounding boxes in (x1, y1, x2, y2) format "
            "with x2 >= x1 and y2 >= y1."
        )

    y = np.copy(x)

    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height

    return y


def xywh2xyxy(
    x: np.ndarray,
) -> np.ndarray:
    """
    Convert bounding boxes from ``(center_x, center_y, width, height)``
    to ``(x1, y1, x2, y2)``.

    Args:
        x:
            Bounding boxes of shape ``(N, 4)`` in
            ``(center_x, center_y, width, height)`` format.
            All values must be finite, with ``width >= 0`` and
            ``height >= 0``.

    Returns:
        np.ndarray:
            A copy of ``x`` in ``(x1, y1, x2, y2)`` format with the
            same shape and dtype.

    Raises:
        ValueError:
            If ``x`` is not of shape ``(N, 4)``, contains non-finite
            values, or has negative widths or heights.
    """

    if x.ndim != 2 or x.shape[1] != 4:
        raise ValueError("Expected boxes with shape (N, 4).")

    if np.any(x[:, 2:] < 0):
        raise ValueError("Expected width >= 0 and height >= 0.")

    y = np.copy(x)

    half_width = x[:, 2] * 0.5
    half_height = x[:, 3] * 0.5

    y[:, 0] = x[:, 0] - half_width  # x1
    y[:, 1] = x[:, 1] - half_height  # y1
    y[:, 2] = x[:, 0] + half_width  # x2
    y[:, 3] = x[:, 1] + half_height  # y2

    return y


def nms_yolov8(
    pred: np.ndarray,
    nms_score_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_det: int = 300,
    is_agnostic_nms: bool = False,
    multi_label: bool = False,
    num_keypoints: int = 0,
    keypoint_dims: int = 3,
    nc: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Apply Non-Maximum Suppression (NMS) to YOLOv8 predictions.

    Supports detection, segmentation, and pose-estimation outputs.
    Set ``num_keypoints`` to the number of keypoints per instance for pose
    models (e.g. 17 for COCO); set ``nc`` explicitly for segmentation models
    that include mask coefficients after the class channels.

    Args:
        pred: Model predictions with shape [batch, channels, num_boxes].
            For detection: channels = 4 + num_classes.
            For segmentation: channels = 4 + num_classes + num_masks.
            For pose: channels = 4 + num_classes + (num_keypoints * keypoint_dims).
        nms_score_threshold: Confidence threshold for candidate filtering.
        iou_threshold: IoU threshold used for NMS suppression.
        max_det: Maximum number of detections per image.
        is_agnostic_nms: Whether to apply class-agnostic NMS.
        multi_label: Allow multiple labels per box.
        num_keypoints: Number of keypoints predicted per instance.
            0 for detection, e.g. 17 for COCO-format pose.
        keypoint_dims: Number of values per keypoint, typically 3 (x, y, visibility).
        nc: Number of classes. Required when the model output includes
            additional channels beyond the class scores (e.g. segmentation
            mask coefficients). If None, it is inferred as
            ``channels - 4 - (num_keypoints * keypoint_dims)``.
            When provided explicitly, it takes precedence over
            ``num_keypoints`` and ``keypoint_dims``; all remaining channels
            after the class scores are preserved as extra outputs.

    Returns:
        List[np.ndarray]: List of detections per image. Each detection has
        format [x1, y1, x2, y2, conf, class, ...extra] where the extra suffix
        contains keypoints (if num_keypoints > 0) or mask coefficients
        (if nc is provided explicitly).
    """
    num_keypoint_values = num_keypoints * keypoint_dims
    if nc is None:
        nc = pred.shape[1] - 4 - num_keypoint_values
    else:
        num_keypoint_values = pred.shape[1] - 4 - nc
    if nc <= 0:
        raise ValueError(f"Expected at least one class channel, got shape {pred.shape}")

    mi = 4 + nc
    candidate_mask = np.amax(pred[:, 4:mi], axis=1) > nms_score_threshold
    pred = pred.transpose(0, -1, -2)

    max_wh = 7680
    max_nms = 30000
    multi_label &= nc > 1
    outputs: List[np.ndarray] = [
        np.zeros((0, 6 + num_keypoint_values), dtype=np.float32)
        for _ in range(pred.shape[0])
    ]

    for batch_idx, prediction in enumerate(pred):
        prediction = prediction[candidate_mask[batch_idx]]
        if prediction.size == 0:
            continue

        boxes = xywh2xyxy(prediction[:, :4])
        class_scores = prediction[:, 4:mi]
        keypoints = prediction[:, mi:]

        if multi_label:
            row_indices, class_indices = (class_scores > nms_score_threshold).nonzero()
            detections = np.concatenate(
                (
                    boxes[row_indices],
                    prediction[row_indices, class_indices + 4, None],
                    class_indices[:, None].astype(np.float32),
                    keypoints[row_indices],
                ),
                axis=1,
            )
        else:
            confidences = class_scores.max(axis=1, keepdims=True)
            class_ids = class_scores.argmax(axis=1, keepdims=True).astype(np.float32)
            detections = np.concatenate(
                (boxes, confidences, class_ids, keypoints),
                axis=1,
            )

        detections = detections[(-detections[:, 4]).argsort(kind="mergesort")[:max_nms]]
        class_offsets = detections[:, 5:6] * (0 if is_agnostic_nms else max_wh)
        keep = np.asarray(
            numpy_nms(
                detections[:, :4] + class_offsets,
                detections[:, 4],
                iou_threshold,
            ),
            dtype=np.int32,
        )
        if keep.size == 0:
            continue
        if keep.size > max_det:
            keep = keep[:max_det]

        outputs[batch_idx] = detections[keep]

    return outputs


def nms_yolov10(
    pred: np.ndarray,
    nms_score_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_det: int = 300,
    is_agnostic_nms: bool = False,
) -> List[np.ndarray]:
    """
    Apply class-aware Non-Maximum Suppression (NMS) to YOLOv10 predictions.
    Args:
        pred (np.ndarray): Predictions with shape [batch, 4 + num_classes, N].
        nms_score_threshold (float): Confidence threshold for candidate filtering.
        iou_threshold (float): IoU threshold used for NMS suppression.
        max_det (int): Maximum number of detections per image.
        is_agnostic_nms (bool): Whether to apply class-agnostic NMS.

    Returns:
        List[np.ndarray]: List of detections per image. Each detection has
        format [x1, y1, x2, y2, conf, class].
    """
    batch_size = pred.shape[0]
    max_wh = 7680  # per-class box offset so NMS never merges across classes
    max_nms = 30000  # maximum number of boxes into NMS

    # [batch, channels, N] -> [batch, N, channels] for row-per-anchor access.
    pred = pred.transpose(0, 2, 1)

    outputs: List[np.ndarray] = [
        np.zeros((0, 6), dtype=np.float32) for _ in range(batch_size)
    ]

    for batch_idx, anchors in enumerate(pred):
        boxes, class_scores = anchors[:, :4], anchors[:, 4:]

        # One detection row [x1, y1, x2, y2, conf, class] per (anchor, class)
        # pair above the threshold.
        anchor_idx, class_idx = np.where(class_scores > nms_score_threshold)
        if anchor_idx.size == 0:
            continue

        detections = np.concatenate(
            [
                boxes[anchor_idx],
                class_scores[anchor_idx, class_idx, None],
                class_idx[:, None].astype(np.float32),
            ],
            axis=1,
        )

        #  Sort by score and cap candidates.
        detections = detections[(-detections[:, 4]).argsort(kind="mergesort")[:max_nms]]

        # Class-aware NMS via per-class box offsets.
        class_offsets = detections[:, 5:6] * (0 if is_agnostic_nms else max_wh)

        keep = np.asarray(
            numpy_nms(
                detections[:, :4] + class_offsets,
                detections[:, 4],
                iou_threshold,
            ),
            dtype=np.int32,
        )

        if keep.size == 0:
            continue
        if keep.size > max_det:
            keep = keep[:max_det]

        outputs[batch_idx] = detections[keep]

    return outputs
