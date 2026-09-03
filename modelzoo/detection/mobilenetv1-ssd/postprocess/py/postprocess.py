# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import numpy as np
import numpy.typing as npt

from core.python.config import Config
from core.python.inference import ModelOutput
from core.python.postprocess.detection.coco_classes_labels import (
    COCO_CLASSES,
    COCO_NAMES,
)
from core.python.postprocess.detection_utils import nms_yolov8
from core.python.postprocess.interfaces import ObjectDetectionOutput


# Box decode and class sigmoid live in the continuation graph,
# so this stage receives decoded boxes (ymin, xmin, ymax, xmax) and sigmoid scores.
_SSD_TO_COCO80 = np.array(
    [
        COCO_CLASSES.index(name) if name in COCO_CLASSES else -1
        for name in COCO_NAMES[:90]
    ],
    dtype=np.int64,
)


def postprocess(
    model_output: ModelOutput,
    config: Config,
    original_image: npt.NDArray,
    image_name: str,
) -> ObjectDetectionOutput | None:
    assert config.postprocess.nms_score_threshold is not None
    assert config.postprocess.iou_threshold is not None
    assert config.postprocess.top_k is not None

    boxes_yxyx = np.asarray(model_output.outputs["boxes"]).reshape(-1, 4)
    scores = np.asarray(model_output.outputs["scores"]).reshape(boxes_yxyx.shape[0], -1)

    boxes_xyxy = boxes_yxyx[:, [1, 0, 3, 2]]
    cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2
    cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2
    bw = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
    bh = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
    boxes_xywh = np.stack([cx, cy, bw, bh], axis=1)
    pred = np.concatenate([boxes_xywh, scores], axis=1).T[np.newaxis, :, :]

    dets = nms_yolov8(
        pred,
        nms_score_threshold=config.postprocess.nms_score_threshold,
        iou_threshold=config.postprocess.iou_threshold,
        max_det=config.postprocess.top_k,
        is_agnostic_nms=False,
    )[0]
    if dets.shape[0] == 0:
        return None

    classes = _SSD_TO_COCO80[dets[:, 5].astype(np.int64)]
    keep = classes >= 0
    dets, classes = dets[keep], classes[keep]
    if len(dets) == 0:
        return None

    img_h, img_w = original_image.shape[:2]
    boxes = dets[:, :4].copy()
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] * img_w, 0, img_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] * img_h, 0, img_h)

    return ObjectDetectionOutput(
        boxes=boxes,
        scores=dets[:, 4],
        classes=classes,
        image_name=image_name,
    )
