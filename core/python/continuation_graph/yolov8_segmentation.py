# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from core.python.continuation_graph.common import as_channel_first, find_output
from core.python.inference.interfaces import ModelOutput

if TYPE_CHECKING:
    from core.python.config import Config


def continue_yolov8_segmentation(
    model_output: ModelOutput,
    config: Config,
) -> ModelOutput:
    del config
    boxes = as_channel_first(find_output(model_output.outputs, "Mul_2_output"), 4)
    scores = as_channel_first(find_output(model_output.outputs, "Sigmoid_output"), 80)
    masks = as_channel_first(find_output(model_output.outputs, "Concat_output"), 32)
    proto = find_output(model_output.outputs, "output1").astype(np.float32, copy=False)
    predictions = np.concatenate((boxes, scores, masks), axis=1).astype(
        np.float32, copy=False
    )
    return ModelOutput(outputs={"output0": predictions, "output1": proto})
