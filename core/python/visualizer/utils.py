# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import numpy as np
import cv2
import os

from typing import Tuple, List
from core.python.logger import logger
from core.python.postprocess.detection.coco_classes_labels import COCO_COLOR, COCO_LIMB
from core.python.postprocess.classification import imagenet_classes_labels


def draw_detections(
    img: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    colors: np.ndarray | None = None,
    class_labels: List[str] | None = None,
    score_format: str = "{:.1f}%",
    box_thickness: int = 2,
    font_scale: float = 0.4,
) -> np.ndarray:
    img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(class_ids[i]) if class_ids is not None else 0
        score = float(scores[i])
        color = get_color(cls_id, colors)
        label = get_label(cls_id, class_labels)
        text = "{}:{}".format(label, score_format.format(score * 100))

        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thickness)

        text_size = cv2.getTextSize(text, font, font_scale, 1)[0]
        cv2.putText(
            img,
            text,
            (x1, y1 + text_size[1]),
            font,
            font_scale,
            get_text_color(color),
            thickness=1,
        )

    return img


def draw_keypoints(
    img: np.ndarray,
    keypoints: np.ndarray,
    score_threshold: float = 0.5,
) -> np.ndarray:
    img = img.copy()
    if keypoints.ndim == 2:
        keypoints = keypoints[np.newaxis, ...]
    height, width = img.shape[:2]

    for det_keypoints in keypoints:
        for point in det_keypoints:
            x, y = point[:2]
            score = point[2] if len(point) > 2 else 1.0
            if score < score_threshold or not (0 <= x < width and 0 <= y < height):
                continue
            cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1, cv2.LINE_AA)

    return img


def draw_pose_keypoints(
    img: np.ndarray,
    keypoints: np.ndarray,
    skeleton: List[List[int]] | None = None,
    score_threshold: float = 0.5,
    line_thickness: int = 2,
) -> np.ndarray:
    img = draw_keypoints(img, keypoints, score_threshold=score_threshold)
    if keypoints.ndim == 2:
        keypoints = keypoints[np.newaxis, ...]

    skeleton = skeleton or COCO_LIMB
    height, width = img.shape[:2]

    for det_keypoints in keypoints:
        for limb_idx, (start_idx, end_idx) in enumerate(skeleton):
            if start_idx >= len(det_keypoints) or end_idx >= len(det_keypoints):
                continue

            start = det_keypoints[start_idx]
            end = det_keypoints[end_idx]
            if len(start) > 2 and (
                start[2] < score_threshold or end[2] < score_threshold
            ):
                continue

            x1, y1 = start[:2]
            x2, y2 = end[:2]
            if not (
                0 <= x1 < width
                and 0 <= y1 < height
                and 0 <= x2 < width
                and 0 <= y2 < height
            ):
                continue

            color = tuple(
                int(channel) for channel in COCO_COLOR[limb_idx % len(COCO_COLOR)]
            )
            cv2.line(
                img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                thickness=line_thickness,
                lineType=cv2.LINE_AA,
            )

    return img


def get_color(cls_id: int, colors: np.ndarray | None = None) -> Tuple[int, int, int]:
    if colors is not None and cls_id < len(colors):
        color = np.asarray(colors[cls_id], dtype=np.float32)
        if color.max() <= 1.0:
            color = color * 255.0
        color = color.astype(np.uint8)
        return int(color[2]), int(color[1]), int(color[0])
    return 0, 255, 0


def get_label(cls_id: int, class_labels: List[str] | None = None) -> str:
    if class_labels is not None and cls_id < len(class_labels):
        return str(class_labels[cls_id])
    return str(cls_id)


def get_text_color(bg_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return (0, 0, 0) if np.mean(bg_color) > 127 else (255, 255, 255)


def masks2segments(masks):
    """
    Takes a list of masks(n,h,w) and returns a list of segments(n,xy), from
    https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py.

    Args:
        masks (numpy.ndarray): the output of the model, which is a tensor of shape (batch_size, 160, 160).

    Returns:
        segments (List): list of segment masks.
    """
    segments = []
    for x in masks.astype("uint8"):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[
            0
        ]  # CHAIN_APPROX_SIMPLE
        if c:
            c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))  # no segments found
        segments.append(c.astype("float32"))
    return segments


def draw_segmentations(
    img: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    masks: np.ndarray,
    class_ids: np.ndarray,
    colors: np.ndarray | None = None,
    class_labels: List[str] | None = None,
    score_format: str = "{:.1f}%",
    box_thickness: int = 2,
    font_scale: float = 0.4,
) -> np.ndarray:

    if img is None:
        raise FileNotFoundError
    img = img.copy()

    numpy_masks = np.concatenate([np.expand_dims(x[:, :, 0], 0) for x in masks])
    segments = masks2segments(numpy_masks)

    for box, conf, cls_id, segment in zip(boxes, scores, class_ids, segments):
        # draw contour and fill mask
        color = get_color(int(cls_id) if cls_id is not None else 0, colors)
        label = get_label(int(cls_id) if cls_id is not None else 0, class_labels)

        cv2.polylines(
            img,
            np.int32([segment]),
            True,
            (255, 255, 255),
            1,
        )  # white borderline
        cv2.fillPoly(img, np.int32([segment]), color)

        # draw bbox rectangle
        cv2.rectangle(
            img,
            (int(box[0]), int(box[1])),
            (int(box[0] + box[2]), int(box[1] + box[[3]])),
            color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"{label}: {conf:.3f}",
            (int(box[0]), int(box[1] - 9)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    # Mix image
    img = cv2.addWeighted(img, 0.3, img, 0.7, 0)

    return img


def draw_classification_output(
    top_k: list[dict[str, np.int64 | np.float32]], image: np.ndarray
):
    text = (
        str(imagenet_classes_labels.IMAGENET_CLASSES_LABELS[int(top_k[0]["label"])])
        + "-"
        + str(top_k[0]["score"])
    )
    overlayed_image = cv2.putText(
        img=image,
        text=text,
        org=(30, 30),
        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
        fontScale=1,
        color=(0, 255, 255),
        thickness=1,
    )

    return overlayed_image


def save(image: np.ndarray, filename: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, image)
    logger.info("Saved visualization to %s", output_path)
    return output_path
