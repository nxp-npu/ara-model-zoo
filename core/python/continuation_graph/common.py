# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from __future__ import annotations

from typing import Mapping

import numpy as np


def normalized_name(name: str) -> str:
    return name.replace("/", "_").replace(".", "_").replace(":", "_")


def find_output(outputs: Mapping[str, np.ndarray], marker: str) -> np.ndarray:
    matches = [
        np.asarray(value)
        for name, value in outputs.items()
        if marker in normalized_name(name)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one cutoff tensor matching '{marker}', found {len(matches)} "
            f"in {list(outputs)}"
        )
    return matches[0]


def as_channel_first(tensor: np.ndarray, channels: int) -> np.ndarray:
    array = np.asarray(tensor, dtype=np.float32)
    core = array[0] if array.ndim > 1 and array.shape[0] == 1 else array
    channel_axes = [axis for axis, size in enumerate(core.shape) if size == channels]
    if not channel_axes:
        if array.size % channels != 0:
            raise ValueError(
                f"Tensor shape {array.shape} cannot be normalized to {channels} channels"
            )
        return array.reshape(1, channels, -1)

    channel_axis = 0 if 0 in channel_axes else channel_axes[-1]
    core = np.moveaxis(core, channel_axis, 0)
    return core.reshape(1, channels, -1)
