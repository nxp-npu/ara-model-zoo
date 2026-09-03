# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import numpy.typing as npt

from core.python.config.config import Config
from core.python.inference import ModelOutput
from core.python.postprocess.detection_utils import (
    nms_yolov8,
    scale_pose_keypoints,
    scale_boxes,
)
from core.python.postprocess.interfaces import FaceDetectionOutput


def postprocess(
    model_output: "ModelOutput",
    config: Config,
    original_image: npt.NDArray,
    image_name: str,
) -> FaceDetectionOutput | None:
    """
    Convert raw model predictions into structured face detection results.

    This function performs postprocessing on the model outputs to produce
    finalized detections in the coordinate space of the original image.
    The following steps are applied:

    1. Apply Non-Maximum Suppression (NMS) to remove redundant detections.
    2. Scale predicted bounding boxes from model input resolution to the
       original image resolution.
    3. Scale predicted keypoints to match the original image coordinates.
    4. Return a `FaceDetectionOutput` instance or None if no detections are found.

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
        This image is used to scale model predictions (bounding boxes and
        keypoints) from the model input coordinate space to the original
        image coordinate space.

    image_name : str
        Primarily used to display the image name on the console

    Returns
    -------
    FaceDetectionOutput | None
        Dataclass containing final detection results. Returns `None` if
        no detections are found.

        Attributes:
        - boxes : ndarray of shape (N, 4)
            Bounding boxes in (x1, y1, x2, y2) format, scaled to the original image.
        - scores : ndarray of shape (N,)
            Confidence scores for each detection.
        - classes : ndarray of shape (N,)
            Predicted class IDs.
        - keypoints : ndarray of shape (N, 5, 3)
            Keypoints in (x, y, confidence) format.
    """
    # Extract primary output (exactly one expected)
    (predictions,) = model_output.outputs.values()
    config_preprocess, config_postprocess = config.preprocess, config.postprocess
    assert config_preprocess is not None, "Preprocess config is not set"
    assert config_postprocess is not None, "Postprocess config is not set"

    model_input_shape = (
        config_preprocess.input_shape.channels,
        config_preprocess.input_shape.height,
        config_preprocess.input_shape.width,
    )
    input_image_shape = original_image.shape[:2]  # (H, W)

    # Validate postprocessing configuration
    assert config_postprocess.nms_score_threshold is not None
    assert config_postprocess.iou_threshold is not None
    assert config_postprocess.top_k is not None

    # Apply NMS
    boxes_kpts_list = nms_yolov8(
        predictions,
        nms_score_threshold=config_postprocess.nms_score_threshold,
        iou_threshold=config_postprocess.iou_threshold,
        max_det=config_postprocess.top_k,
        is_agnostic_nms=False,
        multi_label=True,
        num_keypoints=5,
        keypoint_dims=3,
    )

    boxes_keypoints = boxes_kpts_list[0]

    # Return None if no detections
    if len(boxes_keypoints) == 0:
        return None

    # Extract boxes, scores, classes, and keypoints
    boxes = boxes_keypoints[:, :4]
    scores = boxes_keypoints[:, 4]
    classes = boxes_keypoints[:, 5]
    kpts = boxes_keypoints[:, 6:]

    # Scale keypoints
    keypoints_scaled = scale_pose_keypoints(
        kpts,
        model_input_shape[1:],  # (H, W)
        input_image_shape,
        num_keypoints=5,
    )

    # Scale boxes
    boxes_scaled = scale_boxes(
        model_input_shape[1:],  # (H, W)
        boxes,
        input_image_shape,
    )

    return FaceDetectionOutput(
        boxes=boxes_scaled,
        scores=scores,
        classes=classes,
        keypoints=keypoints_scaled,
        image_name=image_name,
    )
