# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import tabulate
import os
import numpy as np


from core.python.postprocess.classification import imagenet_classes_labels


# Face detection output console display
def face_estimation_console_output(
    image_name: str,
    kpts: np.ndarray,
    boxes: np.ndarray,
    classes: np.ndarray,
    scores: np.ndarray,
) -> None:
    """
    Print formatted console output for face detection results.

    This function prints a tabulated summary of detections for a given image,
    including class ID, detection confidence score, bounding box coordinates
    in `xyxy` format, and facial keypoints. The console output format is
    designed for readability.

    Parameters
    ----------
    image_name : str
        Name of the processed image.

    kpts : np.ndarray
        Array of keypoints with shape (N, 5, 3), where N is the number of
        detections and each keypoint is represented as (x, y, score).

    boxes : np.ndarray
        Array of bounding boxes with shape (N, 4) in `xyxy` format:
        (x1, y1, x2, y2), corresponding to the top-left and bottom-right
        corners of the bounding box.

    classes : np.ndarray
        Array of predicted class IDs with shape (N,).

    scores : np.ndarray
        Array of detection confidence scores with shape (N,).

    Returns
    -------
    None
    """

    num_detections = len(kpts)
    print(f"\nImage: {image_name}\t\tDetections: {num_detections}")

    header = [
        "Class",
        "Score (%)",
        "Bounding Box (x1, y1, x2, y2)",
        "Keypoints (x, y, score)",
    ]
    rows = []

    for kpt, box, cls, score in zip(kpts, boxes, classes, scores):
        # Format xyxy bounding box
        bbox_str = f"[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]"

        # Format keypoints
        kpt_str = "\n".join(f"[{k[0]:.1f}, {k[1]:.1f}, {k[2] * 100:.2f}%]" for k in kpt)

        rows.append(
            [
                int(cls),
                f"{score * 100:.2f}",
                bbox_str,
                kpt_str,
            ]
        )

    print(
        tabulate.tabulate(
            rows,
            headers=header,
            tablefmt="grid",
            numalign="left",
            stralign="left",
        )
        + "\n"
    )


def classification_console_output(
    top_k: list[dict[str, np.int64 | np.float32]], image_path: str
):
    row = []
    header = ["image", "label", "score"]
    labels = ""
    scores = ""
    for i in range(0, len(top_k)):
        labels += (
            str(imagenet_classes_labels.IMAGENET_CLASSES_LABELS[int(top_k[i]["label"])])
            + "\n"
        )
        scores += str(round(top_k[i]["score"], 6)) + "\n"
    row.append([os.path.basename(image_path), labels, scores])
    print(tabulate.tabulate(row, header, tablefmt="grid") + "\n")


def pose_estimation_console_output(
    image_name: str,
    kpts: np.ndarray,
    boxes: np.ndarray,
    classes: np.ndarray,
    scores: np.ndarray,
) -> None:
    """
    Print formatted console output for pose estimation results.

    Parameters
    ----------
    image_name : str
        Name of the processed image.
    kpts : np.ndarray
        Array of keypoints with shape (N, 17, 3).
    boxes : np.ndarray
        Array of bounding boxes with shape (N, 4) in `xyxy` format.
    classes : np.ndarray
        Array of predicted class IDs with shape (N,).
    scores : np.ndarray
        Array of detection confidence scores with shape (N,).
    """

    num_detections = len(kpts)
    print(f"\nImage: {image_name}\t\tDetections: {num_detections}")

    header = [
        "Class",
        "Score (%)",
        "Bounding Box (x1, y1, x2, y2)",
        "Keypoints (x, y, score)",
    ]
    rows = []

    for kpt, box, cls, score in zip(kpts, boxes, classes, scores):
        bbox_str = f"[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]"
        kpt_str = "\n".join(f"[{k[0]:.1f}, {k[1]:.1f}, {k[2] * 100:.2f}%]" for k in kpt)
        rows.append(
            [
                int(cls),
                f"{score * 100:.2f}",
                bbox_str,
                kpt_str,
            ]
        )

    print(
        tabulate.tabulate(
            rows,
            headers=header,
            tablefmt="grid",
            numalign="left",
            stralign="left",
        )
        + "\n"
    )
