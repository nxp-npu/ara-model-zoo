# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

from core.python.inference.interfaces import ModelOutput

if TYPE_CHECKING:
    from core.python.config import Config


_YOLOV7_ANCHORS: tuple[tuple[tuple[float, float], ...], ...] = (
    ((48, 64), (76, 144), (160, 112)),
    ((144, 300), (304, 220), (288, 584)),
    ((568, 440), (768, 972), (1836, 1604)),
)


def continue_yolov7(
    model_output: ModelOutput,
    config: Config,
    *,
    output_name: str = "output",
    anchors: Sequence[Sequence[Sequence[float]]] = _YOLOV7_ANCHORS,
) -> ModelOutput:
    input_height = config.preprocess.input_shape.height
    input_width = config.preprocess.input_shape.width
    outputs = sorted(model_output.outputs.values(), key=np.size, reverse=True)
    strides = (8, 16, 32)
    if len(outputs) != len(strides):
        raise ValueError(
            f"Expected {len(strides)} YOLOv7 cutoff tensors, got {len(outputs)}"
        )

    decoded_heads: list[np.ndarray] = []
    for tensor, stride, head_anchors in zip(outputs, strides, anchors):
        grid_height = input_height // stride
        grid_width = input_width // stride
        predictions = np.asarray(tensor, dtype=np.float32).reshape(
            1, 3, grid_height, grid_width, 85
        )

        grid_x, grid_y = np.meshgrid(
            np.arange(grid_width, dtype=np.float32),
            np.arange(grid_height, dtype=np.float32),
        )
        grid = np.stack((grid_x, grid_y), axis=-1)[None, None]
        anchor_grid = np.asarray(head_anchors, dtype=np.float32).reshape(1, 3, 1, 1, 2)

        xy = predictions[..., :2] * (2.0 * stride) + (grid - 0.5) * stride
        wh = predictions[..., 2:4] ** 2 * anchor_grid
        decoded = np.concatenate((xy, wh, predictions[..., 4:]), axis=-1)
        decoded_heads.append(decoded.reshape(1, -1, 85))

    return ModelOutput(
        outputs={
            output_name: np.concatenate(decoded_heads, axis=1).astype(
                np.float32, copy=False
            )
        }
    )
