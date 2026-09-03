# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from typing import Callable

import numpy as np

from core.python.preprocess.operations.base import Base
from core.python.config.config import CropShape


class CenterCrop(Base):
    """
    Center crop implementation.

    Precedence:
    - (h, w) → absolute crop (highest priority)
    - fraction → MobileNet-style fractional crop
    """

    def __init__(self, cropshape: CropShape):
        super().__init__()

        self.crop_h: int | None = None
        self.crop_w: int | None = None
        self.fraction: float | None = None

        self._geometry_fn: Callable[[int, int], tuple[int, int]]

        self._compile(cropshape)

    # =====================================================
    # Compile configuration
    # =====================================================
    def _compile(self, cfg: CropShape) -> None:
        """Validates configuration and binds MobileNet-style geometry."""

        has_h = cfg.height is not None
        has_w = cfg.width is not None

        # -------------------------
        # Absolute crop (highest priority)
        # -------------------------
        if has_h and has_w:
            self.crop_h = cfg.height
            self.crop_w = cfg.width
            self._geometry_fn = self._absolute_geometry
            return

        # -------------------------
        # Partial absolute crop → invalid
        # -------------------------
        if has_h or has_w:
            raise ValueError(
                "Both height and width must be provided for absolute crop."
            )

        # -------------------------
        # Fractional (MobileNet-style) crop
        # -------------------------
        if cfg.fraction is None:
            raise ValueError("fraction is not initialized")

        self.fraction = cfg.fraction
        self._geometry_fn = self._compute_fractional_geometry

    # =====================================================
    # Geometry functions
    # =====================================================
    def _absolute_geometry(self, image_h: int, image_w: int) -> tuple[int, int]:
        """Returns fixed crop size."""
        return self.crop_h, self.crop_w  # type: ignore

    def _compute_fractional_geometry(
        self, image_h: int, image_w: int
    ) -> tuple[int, int]:
        """Computes MobileNet-style fractional crop size with +1 pixel bias."""
        assert self.fraction is not None
        return (
            int(image_h * self.fraction + 1),
            int(image_w * self.fraction + 1),
        )

    # =====================================================
    # Execution
    # =====================================================
    def run(self, image: np.ndarray) -> np.ndarray:
        image_h, image_w = image.shape[:2]

        crop_h, crop_w = self._geometry_fn(image_h, image_w)

        top = (image_h - crop_h) // 2
        left = (image_w - crop_w) // 2

        return image[top : top + crop_h, left : left + crop_w]
