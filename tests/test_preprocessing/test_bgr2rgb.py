# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import cv2
import numpy as np
import pytest

from core.python.preprocess.operations.bgrtorgb import BgrToRgb


@pytest.fixture
def transform():
    """
    Provides a fresh transform instance for each test.
    """
    return BgrToRgb()


@pytest.fixture
def sample_bgr_image():
    """
    Canonical deterministic 2x2 BGR image.
    """
    return np.array(
        [
            [
                [255, 0, 0],
                [0, 255, 0],
            ],
            [
                [0, 0, 255],
                [10, 20, 30],
            ],
        ],
        dtype=np.uint8,
    )


class TestBgrToRgbCorrectness:
    """
    Tests semantic correctness of the transformation.
    """

    def test_run_correctly_swaps_channels(
        self,
        transform,
        sample_bgr_image,
    ):
        # Arrange
        expected = np.array(
            [
                [
                    [0, 0, 255],
                    [0, 255, 0],
                ],
                [
                    [255, 0, 0],
                    [30, 20, 10],
                ],
            ],
            dtype=np.uint8,
        )

        # Act
        output = transform.run(sample_bgr_image)

        # Assert
        assert np.array_equal(output, expected)

    def test_specific_pixel_mapping(self, transform):
        image = np.array([[[10, 20, 30]]], dtype=np.uint8)

        output = transform.run(image)

        expected = np.array([30, 20, 10], dtype=np.uint8)

        assert np.array_equal(output[0, 0], expected), (
            f"Pixel mapping failed:\n"
            f"INPUT(BGR)={image[0, 0]}\n"
            f"OUTPUT(RGB)={output[0, 0]}\n"
            f"EXPECTED={expected}"
        )

    def test_run_preserves_channel_count(self, transform):
        image = np.random.randint(
            0,
            255,
            (32, 32, 3),
            dtype=np.uint8,
        )

        output = transform.run(image)

        assert output.ndim == 3, (
            f"Expected 3D output (H, W, C), got shape {output.shape}"
        )

        assert output.shape[2] == 3, (
            f"Channel count mismatch:\n"
            f"INPUT shape={image.shape}\n"
            f"OUTPUT shape={output.shape}"
        )

    @pytest.mark.parametrize(
        "shape",
        [
            (1, 1, 3),
            (32, 32, 3),
            (224, 224, 3),
        ],
    )
    def test_run_preserves_shape(
        self,
        transform,
        shape,
    ):
        # Arrange
        image = np.random.randint(
            0,
            255,
            shape,
            dtype=np.uint8,
        )

        # Act
        output = transform.run(image)

        # Assert
        assert output.shape == image.shape

    @pytest.mark.parametrize(
        "dtype",
        [
            np.uint8,
            np.float32,
        ],
    )
    def test_run_preserves_dtype(
        self,
        transform,
        dtype,
    ):
        # Arrange
        image = np.zeros((32, 32, 3), dtype=dtype)

        # Act
        output = transform.run(image)

        # Assert
        assert output.dtype == image.dtype


class TestBgrToRgbInvariants:
    """
    Tests transformation invariants and safety guarantees.
    """

    def test_run_preserves_all_pixel_values(
        self,
        transform,
    ):
        # Arrange
        image = np.random.randint(
            0,
            255,
            (64, 64, 3),
            dtype=np.uint8,
        )

        # Act
        output = transform.run(image)

        # Assert
        assert np.array_equal(
            np.sort(image.flatten()),
            np.sort(output.flatten()),
        )

    def test_run_returns_numpy_array(self, transform, sample_bgr_image):
        output = transform.run(sample_bgr_image)

        assert isinstance(output, np.ndarray), (
            f"Expected np.ndarray output, got {type(output)}"
        )

    def test_output_is_not_view_of_input(self, transform, sample_bgr_image):
        output = transform.run(sample_bgr_image)

        assert not np.shares_memory(output, sample_bgr_image), (
            "Memory safety violation:\n"
            "Output shares memory with input image.\n"
            "This can lead to unintended side effects in downstream pipelines."
        )

    def test_run_does_not_modify_input_inplace(
        self,
        transform,
    ):
        # Arrange
        image = np.random.randint(
            0,
            255,
            (32, 32, 3),
            dtype=np.uint8,
        )

        original = image.copy()

        # Act
        transform.run(image)

        # Assert
        assert np.array_equal(image, original)

    def test_run_is_deterministic(
        self,
        transform,
    ):
        # Arrange
        image = np.random.randint(
            0,
            255,
            (32, 32, 3),
            dtype=np.uint8,
        )

        # Act
        output_1 = transform.run(image)
        output_2 = transform.run(image)

        # Assert
        assert np.array_equal(output_1, output_2)


class TestBgrToRgbValidation:
    """
    Tests invalid inputs and failure behavior.
    """

    def test_run_raises_for_empty_input(
        self,
        transform,
    ):
        # Arrange
        image = np.empty((0, 0, 3), dtype=np.uint8)

        # Act / Assert
        with pytest.raises(cv2.error):
            transform.run(image)
