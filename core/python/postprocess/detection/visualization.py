# Copyright 2025-2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import os
import tabulate
import logging

logger = logging.getLogger(__name__)


def object_detection_console_output(
    image_path, model_input_shape, boxes, scores, cls_ids, class_labels=None
):
    print(
        "\nimage: "
        + os.path.basename(image_path)
        + "\t\t"
        + "detections: "
        + str(len(boxes))
    )
    row = []
    header = ["image", "label", "confidence", "coordinates \n[xmin, ymin, xmax, ymax ]"]
    labels = ""
    confidence = ""
    coordinates = ""
    for i in range(0, len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = float(scores[i]) * 100
        lbl = (
            class_labels[cls_id]
            if class_labels is not None and cls_id < len(class_labels)
            else str(cls_id)
        )
        labels += lbl + "\n"
        confidence += f"{score:.2f}" + "%\n"
        coordinates += "[{:<5s} {:<5s} {:<5s} {:<5s}] \n".format(
            str(int(box[0])), str(int(box[1])), str(int(box[2])), str(int(box[3]))
        )
    row.append([os.path.basename(image_path), labels, confidence, coordinates])
    print(tabulate.tabulate(row, header, tablefmt="grid") + "\n")
