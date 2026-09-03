# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from core.python.inference.interfaces import ModelOutput

if TYPE_CHECKING:
    from core.python.config import Config

_SCRFD_NUM_OUTPUTS = 9
_SCRFD_HEAD_ORDER = {2: 0, 8: 1, 20: 2}


def _head_sort_key(item: tuple[str, np.ndarray]) -> tuple[int, int]:
    name, value = item
    shape = np.asarray(value).shape
    if len(shape) == 4 and shape[1] in _SCRFD_HEAD_ORDER:
        channels, height, width = shape[1:]
    elif len(shape) == 4 and shape[-1] in _SCRFD_HEAD_ORDER:
        height, width, channels = shape[1:]
    else:
        raise ValueError(f"Unexpected SCRFD output shape for {name}: {shape}")

    return -(height * width), _SCRFD_HEAD_ORDER[channels]


def continue_scrfd(
    model_output: ModelOutput,
    config: Config,
) -> ModelOutput:
    """Pass through SCRFD cutoff heads for Python postprocess.

    SCRFD is pruned so the on-device graph already ends at the nine
    classification / box / keypoint heads (with sigmoid applied on-device).
    There is no additional arithmetic tail; postprocess.py performs decode
    and NMS on these tensors directly.
    """
    del config
    unordered_outputs = {
        name: np.asarray(value, dtype=np.float32)
        for name, value in model_output.outputs.items()
    }
    outputs = dict(sorted(unordered_outputs.items(), key=_head_sort_key))
    if len(outputs) != _SCRFD_NUM_OUTPUTS:
        raise ValueError(
            f"Expected {_SCRFD_NUM_OUTPUTS} SCRFD cutoff tensors, got {len(outputs)}: "
            f"{list(outputs)}"
        )
    return ModelOutput(outputs=outputs)
