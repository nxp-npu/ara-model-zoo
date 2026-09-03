# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import numpy as np
import cv2


from core.python.preprocess.operations.base import Base
from core.python.config.config import InterpolationValues


class Resize(Base):
    """This class implements the resize operation"""

    def __init__(
        self,
        shape: tuple[int, int],
        interpolation: str | None = None,
        maintain_aspect_ratio: bool = False,
        padding: bool | None = False,
        padding_value: int | None = 0,
        scale: float = 1.0,
        resize: int | None = None,
    ):
        super().__init__()
        self.output_shape = shape
        self.interpolation = self._get_interpolation(interpolation)
        self.maintain_aspect_ratio = maintain_aspect_ratio
        self.padding = padding
        self.padding_value = padding_value
        self.scale = scale
        self.resize = resize

    @staticmethod
    def _get_interpolation(interpolation: str | None = None):
        if interpolation is None:
            return cv2.INTER_LINEAR

        if interpolation.upper() in [
            InterpolationValues.LINEAR.value,
            InterpolationValues.BILINEAR.value,
            InterpolationValues.INTER_LINEAR.value,
        ]:
            return cv2.INTER_LINEAR

        if interpolation.upper() in [
            InterpolationValues.NEAREST.value,
            InterpolationValues.INTER_NEAREST.value,
        ]:
            return cv2.INTER_NEAREST

        if interpolation.upper() in [
            InterpolationValues.CUBIC.value,
            InterpolationValues.BICUBIC.value,
            InterpolationValues.INTER_CUBIC.value,
        ]:
            return cv2.INTER_CUBIC

        if interpolation.upper() in [
            InterpolationValues.AREA.value,
            InterpolationValues.INTER_AREA.value,
        ]:
            return cv2.INTER_AREA

        if interpolation.upper() in [
            InterpolationValues.LANCZOS.value,
            InterpolationValues.LANCZOS4.value,
            InterpolationValues.INTER_LANCZOS4.value,
        ]:
            return cv2.INTER_LANCZOS4

        raise ValueError(f"Invalid Interpolation type: {interpolation}")

    @staticmethod
    def compute_keep_ratio_size(
        image_h: int,
        image_w: int,
        out_h: int,
        out_w: int,
    ) -> tuple[int, int]:
        ratio = min(out_h / image_h, out_w / image_w)

        resized_h = int(round(image_h * ratio))
        resized_w = int(round(image_w * ratio))

        return resized_h, resized_w

    def _resize_with_aspectratio(self, image: np.ndarray) -> np.ndarray:
        image_h, image_w = image.shape[:2]

        resized_h, resized_w = self.compute_keep_ratio_size(
            image_h=image_h,
            image_w=image_w,
            out_h=self.output_shape[1],
            out_w=self.output_shape[0],
        )

        return cv2.resize(
            image,
            (resized_w, resized_h),
            interpolation=self.interpolation,
        )

    def _resize_with_aspectratio_pad(self, image: np.ndarray) -> np.ndarray:
        image = self._resize_with_aspectratio(image)

        resized_h, resized_w = image.shape[:2]
        out_w, out_h = self.output_shape

        dh = out_h - resized_h
        dw = out_w - resized_w

        dh /= 2
        dw /= 2
        left = int(round(dw - 0.1))
        top = int(round(dh - 0.1))

        if image.ndim == 2:
            padded_img = np.ones((out_h, out_w), dtype=np.float32) * self.padding_value
        else:
            channels = image.shape[2]
            padded_img = (
                np.ones((out_h, out_w, channels), dtype=np.float32) * self.padding_value
            )

        padded_img[top : resized_h + top, left : resized_w + left] = image
        image = padded_img
        return image

    def run(self, image: np.ndarray) -> np.ndarray:
        if self.padding and self.maintain_aspect_ratio:
            image = self._resize_with_aspectratio_pad(image)
        elif self.maintain_aspect_ratio:
            image = self._resize_with_aspectratio(image)
        else:
            image = cv2.resize(
                image, self.output_shape, interpolation=self.interpolation
            )

        return image
