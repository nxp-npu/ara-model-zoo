# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import numpy as np
import numpy.typing as npt

from core.python.config.config import Config
from core.python.inference import ModelOutput
from core.python.postprocess.interfaces import SegDetectionOutput
from core.python.postprocess.detection_utils import (
    nms_yolov8,
    process_mask,
    scale_boxes,
    scale_mask,
    xyxy2xywh,
)


def postprocess(
    model_output: "ModelOutput",
    config: Config,
    original_image: npt.NDArray,
    image_name: str,
) -> SegDetectionOutput | None:
    h = original_image.shape[0]
    w = original_image.shape[1]

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

    output0 = model_output.outputs["output0"]
    proto = model_output.outputs["output1"].reshape(32, 160, 160)

    pred = nms_yolov8(
        output0,
        nms_score_threshold=config_postprocess.nms_score_threshold,
        iou_threshold=config_postprocess.iou_threshold,
        max_det=config_postprocess.top_k,
        is_agnostic_nms=False,
        multi_label=True,
        nc=80,
    )[0]

    if len(pred) <= 0:
        return None

    masks = process_mask(
        proto,
        pred[:, 6:],
        pred[:, :4],
        (model_input_shape[1], model_input_shape[2]),
    )

    masks = np.stack([scale_mask(mask, [h, w]) for mask in masks])
    boxes = pred[:, :4]
    scores = pred[:, 4]
    classes = pred[:, 5]

    boxes = scale_boxes((model_input_shape[1], model_input_shape[2]), boxes, (h, w))

    boxes = xyxy2xywh(boxes)

    return SegDetectionOutput(
        image_name=image_name, boxes=boxes, scores=scores, classes=classes, masks=masks
    )
