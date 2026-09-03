# Copyright 2025-2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import numpy.typing as npt
import numpy as np

from core.python.config.config import Config
from core.python.inference import ModelOutput
from core.python.postprocess.detection_utils import scale_boxes, nms_yolov8

from core.python.postprocess.interfaces import ObjectDetectionOutput


def postprocess(
    model_output: "ModelOutput",
    config: Config,
    original_image: npt.NDArray,
    image_name: str,
) -> ObjectDetectionOutput | None:
    """
    Convert raw model predictions into structured object detection results.

    This function performs postprocessing on the model outputs to produce
    finalized detections in the coordinate space of the original image.
    The following steps are applied:

    1. Apply Non-Maximum Suppression per class (NMS) to remove redundant detections.
    2. Scale predicted bounding boxes from model input resolution to the
       original image resolution.
    3. Return a `ObjectDetectionOutput` instance or None if no detections are found.

    Parameters
    ----------
    model_output : ModelOutput
        Structured predictions produced by the model.
        `model_output.outputs` is a dictionary mapping output node names
        to NumPy arrays. Only the primary output is used.

    config : Config
        Model configuration containing both preprocessing and postprocessing
        settings. The following fields are used:

        - ``config.preprocess.input_shape``: Model input tensor shape.
        - ``config.postprocess.nms_score_threshold``: Score threshold for NMS.
        - ``config.postprocess.iou_threshold``: IoU threshold for NMS.
        - ``config.postprocess.top_k``: Maximum number of detections to keep.

    original_image : npt.NDArray
        Original input image as a NumPy array with shape ``(H, W, C)``.
        This image is used to scale model predictions (bounding boxes)
        from the model input coordinate space to the original
        image coordinate space.

    image_name : str
        Primarily used to display the image name on the console

    Returns
    -------
    ObjectDetectionOutput | None
        Dataclass containing final detection results. Returns `None` if
        no detections are found.

        Attributes:
        - boxes : ndarray of shape (N, 4)
            Bounding boxes in (x1, y1, x2, y2) format, scaled to the original image.
        - scores : ndarray of shape (N,)
            Confidence scores for each detection.
        - classes : ndarray of shape (N,)
            Predicted class IDs.
        - image_name : str
            Name of the image file.
    """
    (predictions,) = model_output.outputs.values()

    config_preprocess, config_postprocess = config.preprocess, config.postprocess

    model_input_shape = (
        config_preprocess.input_shape.channels,
        config_preprocess.input_shape.height,
        config_preprocess.input_shape.width,
    )

    # Validate postprocessing configuration
    assert config_postprocess.nms_score_threshold is not None
    assert config_postprocess.iou_threshold is not None
    assert config_postprocess.top_k is not None

    box_coords = []
    scores = []
    classes = []
    num_classes = 80
    img = original_image

    input_image_shape = (img.shape[0], img.shape[1])

    boxes = decode_yolox_netout_to_xyxy(predictions, num_classes, model_input_shape)

    # Apply NMS
    yolox_scores = (
        boxes[:, 4:5] * boxes[:, 5:]
    )  # obj_conf * cls_conf → [N, num_classes]

    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    boxes_xywh = np.stack([cx, cy, w, h], axis=1)

    prediction = np.concatenate([boxes_xywh, yolox_scores], axis=1).T[np.newaxis, :, :]

    detections = nms_yolov8(
        pred=prediction,
        nms_score_threshold=config_postprocess.nms_score_threshold,
        iou_threshold=config_postprocess.iou_threshold,
        max_det=config_postprocess.top_k,
        is_agnostic_nms=False,
        multi_label=True,
    )[0]

    detections = detections if detections.shape[0] > 0 else None

    boxes_scaled = []

    if detections is not None and len(detections) > 0:
        box_coords = detections[:, :4]
        scores = detections[:, 4]
        classes = detections[:, 5]

        boxes_scaled = scale_boxes(
            (model_input_shape[1], model_input_shape[2]), box_coords, input_image_shape
        )

    else:
        return None

    return ObjectDetectionOutput(
        boxes=boxes_scaled,
        scores=scores,
        classes=classes,
        image_name=image_name,
    )


def decode_yolox_netout_to_xyxy(
    dequantized_output: np.ndarray,
    num_classes: int,
    model_input_shape: tuple[int, int, int],
) -> np.ndarray:
    """
    Decode raw YOLOX network output into bounding boxes in (x1, y1, x2, y2) format.

    Applies grid and stride corrections across the three YOLOX detection heads
    (strides 8, 16, 32), then converts from centre/size (cx, cy, w, h) to
    corner coordinates (x1, y1, x2, y2).

    Args:
        dequantized_output: Raw dequantized model output, where the first element
            has shape [1, num_anchors, num_classes + 5].
        num_classes: Number of object classes the model was trained on.
        model_input_shape: Model input shape as (batch, height, width).

    Returns:
        Array of shape [num_anchors, num_classes + 5] with the first four columns
        as decoded (x1, y1, x2, y2) bounding box coordinates in input-image pixel space.
    """
    boxes = dequantized_output.reshape((1, -1, num_classes + 5))

    grids = []
    expanded_strides = []
    strides = [8, 16, 32]
    hsizes = [model_input_shape[1] // stride for stride in strides]
    wsizes = [model_input_shape[2] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    boxes[..., :2] = (boxes[..., :2] + grids) * expanded_strides
    boxes[..., 2:4] = np.exp(boxes[..., 2:4]) * expanded_strides
    box_corner = np.zeros_like(boxes)
    box_corner[:, :, 0] = boxes[:, :, 0] - boxes[:, :, 2] / 2
    box_corner[:, :, 1] = boxes[:, :, 1] - boxes[:, :, 3] / 2
    box_corner[:, :, 2] = boxes[:, :, 0] + boxes[:, :, 2] / 2
    box_corner[:, :, 3] = boxes[:, :, 1] + boxes[:, :, 3] / 2
    boxes[:, :, :4] = box_corner[:, :, :4]

    boxes = boxes.squeeze()

    return boxes
