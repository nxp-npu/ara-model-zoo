# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import numpy as np

from core.python.preprocess.operations.base import Base
from .utils import _validate_channels, _validate_channel_consistency


class MeanSubtraction(Base):
    """
    Subtract per-channel mean from an image.

    Supported image formats:
    - Grayscale: (H, W)
    - RGB:       (H, W, 3)

    Parameters
    ----------
    mean : list[float]
        Mean values per channel.

    channels : int
        Expected number of image channels.
    """

    def __init__(self, mean: list[float], channels: int):
        super().__init__()

        # -------------------------
        # Validate channel configuration and mean-channel consistency.
        # -------------------------
        _validate_channels(channels)
        _validate_channel_consistency(len(mean), channels, context="mean")

        self.channels = channels
        self.mean = np.asarray(mean, dtype=np.float32)

    def run(self, image: np.ndarray) -> np.ndarray:
        """
        Apply per-channel mean subtraction in-place.
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
        # Mean subtraction
        # -------------------------
        if self.channels == 1:
            image -= self.mean[0]
        else:
            image -= self.mean.reshape(1, 1, self.channels).astype(image.dtype)

        return image
