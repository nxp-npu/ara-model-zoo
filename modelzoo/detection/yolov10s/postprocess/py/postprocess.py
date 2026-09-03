# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import numpy as np
import numpy.typing as npt

from core.python.config import Config
from core.python.inference import ModelOutput
from core.python.postprocess.detection.coco_classes_labels import COCO_CLASSES
from core.python.postprocess.detection_utils import nms_yolov10, scale_boxes
from core.python.postprocess.interfaces import ObjectDetectionOutput


def _as_channel_first(output: np.ndarray) -> np.ndarray:
    """
    Return a [batch, channels, N] view of an HW backbone output.
    """
    output = np.asarray(output)
    return output.reshape(output.shape[:3]) if output.ndim > 3 else output


def _extract_prediction_tensor(model_output: ModelOutput) -> np.ndarray:
    """
    Normalise HW outputs into a single [1, 4 + num_classes, N] tensor.
    """
    tensors = [_as_channel_first(output) for output in model_output.outputs.values()]

    if len(tensors) != 2:
        raise ValueError(
            f"Expected 2 output tensors (boxes + scores), received {len(tensors)}"
        )

    # Smaller channel count is the 4-channel box tensor, the other is scores.
    boxes, scores = sorted(tensors, key=lambda tensor: tensor.shape[1])
    if boxes.shape[1] != 4 or scores.shape[1] <= 4:
        raise ValueError(
            "Unable to identify YOLOv10 box/score tensors from outputs "
            f"with shapes {[tensor.shape for tensor in tensors]}"
        )
    return np.concatenate([boxes, scores], axis=1)


def _decode_native_end2end(
    model_output: ModelOutput,
    nms_score_threshold: float,
) -> np.ndarray:
    """
    Decode the native end2end head output.

    Returns [k, 6] rows of [x1, y1, x2, y2, conf, class].
    """
    tensor = np.asarray(next(iter(model_output.outputs.values())))
    detections = tensor.reshape(-1, tensor.shape[-1])  # [max_det, 6]
    return detections[detections[:, 4] > nms_score_threshold]


def postprocess(
    model_output: ModelOutput,
    config: Config,
    original_image: npt.NDArray,
    image_name: str,
) -> ObjectDetectionOutput | None:
    # Float ONNX runs the full native graph -> a single [1, max_det, 6] end2end
    # tensor (no NMS). HW is cut at the dvconvert onodes -> two backbone tensors
    # decoded with multi-label NMS.
    assert config.postprocess.nms_score_threshold is not None
    assert config.postprocess.iou_threshold is not None
    assert config.postprocess.top_k is not None

    outputs = list(model_output.outputs.values())
    if len(outputs) == 1 and np.asarray(outputs[0]).shape[-1] == 6:
        detections = _decode_native_end2end(
            model_output,
            nms_score_threshold=config.postprocess.nms_score_threshold,
        )
    else:
        predictions = _extract_prediction_tensor(model_output)
        detections = nms_yolov10(
            pred=predictions,
            nms_score_threshold=config.postprocess.nms_score_threshold,
            iou_threshold=config.postprocess.iou_threshold,
            max_det=config.postprocess.top_k,
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
