# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from core.python.inference.interfaces import ModelOutput

if TYPE_CHECKING:
    from core.python.config import Config


_SSD_FEATURE_MAP_SHAPES = (19, 10, 5, 3, 2, 1)
_SSD_ASPECT_RATIOS = (
    (1.0, 2.0, 0.5),
    (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0),
    (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0),
    (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0),
    (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0),
    (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0),
)


def _generate_ssd_anchors() -> np.ndarray:
    scales = [0.2 + (0.95 - 0.2) * index / 5 for index in range(6)] + [1.0]
    anchors: list[tuple[float, float, float, float]] = []

    for layer, grid_size in enumerate(_SSD_FEATURE_MAP_SHAPES):
        scale = scales[layer]
        if layer == 0:
            specs = ((0.1, 1.0), (scale, 2.0), (scale, 0.5))
        else:
            specs = tuple(
                (scale, aspect_ratio) for aspect_ratio in _SSD_ASPECT_RATIOS[layer]
            ) + ((np.sqrt(scale * scales[layer + 1]), 1.0),)

        for grid_y in range(grid_size):
            center_y = (grid_y + 0.5) / grid_size
            for grid_x in range(grid_size):
                center_x = (grid_x + 0.5) / grid_size
                for anchor_scale, aspect_ratio in specs:
                    ratio_sqrt = np.sqrt(aspect_ratio)
                    height = anchor_scale / ratio_sqrt
                    width = anchor_scale * ratio_sqrt
                    anchors.append(
                        (
                            center_y - height / 2,
                            center_x - width / 2,
                            center_y + height / 2,
                            center_x + width / 2,
                        )
                    )

    return np.asarray(anchors, dtype=np.float32)


def _select_ssd_cutoff_tensors(
    outputs: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (box_encodings, class_logits) from ARA cutoff outputs.

    Preferred names match ``dvconvert.onode`` (`concat` boxes, `concat_1` scores).
    Falls back to selecting by tensor size when names differ.
    """
    if "concat" in outputs and "concat_1" in outputs:
        return np.asarray(outputs["concat"]), np.asarray(outputs["concat_1"])

    if len(outputs) != 2:
        raise ValueError(
            f"Expected two MobileNet SSD cutoff tensors, got {len(outputs)}: "
            f"{list(outputs)}"
        )

    smaller, larger = sorted(outputs.values(), key=np.size)
    return np.asarray(smaller), np.asarray(larger)


def continue_mobilenet_ssd(
    model_output: ModelOutput,
    config: Config,
) -> ModelOutput:
    del config
    box_tensor, score_tensor = _select_ssd_cutoff_tensors(model_output.outputs)
    anchors = _generate_ssd_anchors()
    num_anchors = anchors.shape[0]

    # Cutoff boxes arrive as (1, N, 1, 4) / (N, 4). Reshape to (4, N) via
    # (N, 4).T — not reshape(4, N), which scrambles the encoding channels.
    encoded_boxes = box_tensor.astype(np.float32, copy=False).reshape(num_anchors, 4).T
    scores = score_tensor.astype(np.float32, copy=False).reshape(num_anchors, -1)
    scores = 1.0 / (1.0 + np.exp(-scores))
    # Match postprocessing.onnx: drop background class (index 0) → 90 COCO scores.
    if scores.shape[1] == 91:
        scores = scores[:, 1:]

    anchor_width = anchors[:, 3] - anchors[:, 1]
    anchor_height = anchors[:, 2] - anchors[:, 0]
    anchor_x = anchors[:, 1] + anchor_width / 2
    anchor_y = anchors[:, 0] + anchor_height / 2

    center_y = (encoded_boxes[0] / 10.0) * anchor_height + anchor_y
    center_x = (encoded_boxes[1] / 10.0) * anchor_width + anchor_x
    half_height = np.exp(encoded_boxes[2] / 5.0) * anchor_height / 2
    half_width = np.exp(encoded_boxes[3] / 5.0) * anchor_width / 2

    boxes = np.stack(
        (
            center_y - half_height,
            center_x - half_width,
            center_y + half_height,
            center_x + half_width,
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    return ModelOutput(
        outputs={"boxes": boxes, "scores": scores.astype(np.float32, copy=False)}
    )
