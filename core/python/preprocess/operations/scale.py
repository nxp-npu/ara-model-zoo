# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import numpy as np

from core.python.preprocess.operations.base import Base
from .utils import _validate_channels, _validate_channel_consistency


class Scale(Base):
    """
    Scale image channels by per-channel factors.

    Supported image formats:
    - Grayscale: (H, W)
    - RGB:       (H, W, 3)

    Parameters
    ----------
    scale : list[float]
        Scale values per channel.

    channels : int
        Expected number of image channels.
    """

    def __init__(self, scale: list[float], channels: int):
        super().__init__()

        # -------------------------
        # Validate channel configuration and scale-channel consistency.
        # -------------------------
        _validate_channels(channels)
        _validate_channel_consistency(len(scale), channels, context="scale")

        self.channels = channels
        self.scale = np.asarray(scale, dtype=np.float32)

    def run(self, image: np.ndarray) -> np.ndarray:
        """
        Apply per-channel scaling in-place.
        """

        # -------------------------
        # Validate dtype
        # -------------------------
        if not np.issubdtype(image.dtype, np.floating):
            raise TypeError(f"Expected floating-point image, got {image.dtype}")

        # -------------------------
        # Validate dimensions
        # -------------------------
        if image.ndim not in (2, 3):
            raise ValueError("Expected image shape (H, W) or (H, W, C)")

        # -------------------------
        # Infer + Validate image channels
        # -------------------------
        image_channels = 1 if image.ndim == 2 else image.shape[2]
        _validate_channel_consistency(self.channels, image_channels)

        # -------------------------
        # Scale image
        # -------------------------
        if self.channels == 1:
            image *= self.scale[0]
        else:
            image *= self.scale.reshape(1, 1, self.channels).astype(image.dtype)

        return image
