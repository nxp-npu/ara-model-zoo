# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import numpy as np
import pytest

from core.python.preprocess.operations.tofloat import ToFloat


@pytest.fixture
def transform():
    """
    Provides a fresh default transform instance
    for each test.
    """
    return ToFloat()


@pytest.fixture
def sample_uint8_image():
    """
    Canonical deterministic uint8 image.
    """
    return np.array(
        [
            [
                [0, 127, 255],
                [10, 20, 30],
            ]
        ],
        dtype=np.uint8,
    )


class TestToFloatCorrectness:
    """
    Tests semantic correctness of dtype conversion.
    """

    def test_run_converts_uint8_to_float32(
        self,
        transform,
        sample_uint8_image,
    ):
        # Act
        output = transform.run(sample_uint8_image)

        # Assert
        assert output.dtype == np.float32

    def test_run_preserves_shape(
        self,
        transform,
        sample_uint8_image,
    ):
        # Act
        output = transform.run(sample_uint8_image)

        # Assert
        assert output.shape == sample_uint8_image.shape

    def test_run_preserves_values_numerically(
        self,
        transform,
        sample_uint8_image,
    ):
        # Arrange
        expected = sample_uint8_image.astype(np.float32)

        # Act
        output = transform.run(sample_uint8_image)

        # Assert
        assert np.array_equal(output, expected)

    @pytest.mark.parametrize(
        "target_dtype",
        [
            np.float16,
            np.float32,
            np.float64,
        ],
    )
    def test_run_supports_multiple_float_dtypes(
        self,
        sample_uint8_image,
        target_dtype,
    ):
        # Arrange
        transform = ToFloat(dtype=target_dtype)

        # Act
        output = transform.run(sample_uint8_image)

        # Assert
        assert output.dtype == target_dtype


class TestToFloatInvariants:
    """
    Tests transformation invariants and safety guarantees.
    """

    def test_run_supports_grayscale_input(
        self,
        transform,
    ):
        # Arrange
        image = np.array(
            [
                [0, 127],
                [255, 64],
            ],
            dtype=np.uint8,
        )

        # Act
        output = transform.run(image)

        # Assert
        assert output.ndim == 2
        assert output.shape == image.shape
        assert output.dtype == np.float32
        assert np.array_equal(image, output)

    def test_run_does_not_mutate_input(
        self,
        transform,
        sample_uint8_image,
    ):
        # Arrange
        original = sample_uint8_image.copy()

        # Act
        transform.run(sample_uint8_image)

        # Assert
        assert np.array_equal(
            sample_uint8_image,
            original,
        )

    def test_run_returns_ndarray(
        self,
        transform,
        sample_uint8_image,
    ):
        # Act
        output = transform.run(sample_uint8_image)

        # Assert
        assert isinstance(output, np.ndarray)

    def test_run_is_deterministic(
        self,
        transform,
        sample_uint8_image,
    ):
        # Act
        output_1 = transform.run(sample_uint8_image)
        output_2 = transform.run(sample_uint8_image)

        # Assert
        assert np.array_equal(output_1, output_2)


class TestToFloatValidation:
    """
    Tests invalid input and failure behavior.
    """

    def test_run_raises_for_invalid_dtype(
        self,
        sample_uint8_image,
    ):
        # Arrange
        transform = ToFloat(dtype="invalid_dtype")

        # Act / Assert
        with pytest.raises(TypeError):
            transform.run(sample_uint8_image)

    def test_run_handles_empty_input(
        self,
        transform,
    ):
        # Arrange
        image = np.empty((0, 0, 3), dtype=np.uint8)

        # Act
        output = transform.run(image)

        # Assert
        assert output.shape == (0, 0, 3)
        assert output.dtype == np.float32
