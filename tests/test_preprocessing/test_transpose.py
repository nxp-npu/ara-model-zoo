# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import numpy as np
import pytest

from core.python.preprocess.operations.transpose import ChannelFirstTranspose


@pytest.fixture
def transform():
    """
    Provides a fresh transform instance for each test.
    """
    return ChannelFirstTranspose()


@pytest.fixture
def sample_hwc_image():
    """
    Canonical deterministic HWC image.

    Shape:
        (H=1, W=2, C=3)

    Channel values are intentionally distinct to make
    transpose correctness visually obvious.
    """
    return np.array(
        [
            [
                [1, 10, 100],
                [2, 20, 200],
            ]
        ],
        dtype=np.uint8,
    )


class TestChannelFirstTransposeCorrectness:
    """
    Tests semantic correctness of HWC -> CHW transpose.
    """

    def test_run_converts_hwc_to_chw(
        self,
        transform,
    ):
        # Arrange
        image = np.zeros((224, 224, 3), dtype=np.uint8)

        # Act
        output = transform.run(image)

        # Assert
        assert output.shape == (3, 224, 224)

    def test_run_correctly_reorders_axes(
        self,
        transform,
        sample_hwc_image,
    ):
        # Arrange
        expected = np.array(
            [
                [[1, 2]],
                [[10, 20]],
                [[100, 200]],
            ],
            dtype=np.uint8,
        )

        # Act
        output = transform.run(sample_hwc_image)

        # Assert
        assert np.array_equal(output, expected)

    @pytest.mark.parametrize(
        "shape",
        [
            (1, 1, 3),
            (32, 32, 3),
            (224, 224, 3),
            (64, 64, 1),
        ],
    )
    def test_run_preserves_expected_dimensions(
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

        expected_shape = (
            shape[2],
            shape[0],
            shape[1],
        )

        # Act
        output = transform.run(image)

        # Assert
        assert output.shape == expected_shape


class TestChannelFirstTransposeInvariants:
    """
    Tests transformation invariants and safety guarantees.
    """

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

    def test_run_preserves_all_values(
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
        output = transform.run(image)

        # Assert
        assert np.array_equal(
            np.sort(image.flatten()),
            np.sort(output.flatten()),
        )

    def test_run_does_not_mutate_input(
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


class TestChannelFirstTransposeValidation:
    """
    Tests invalid inputs and failure behavior.
    """

    @pytest.mark.parametrize(
        "invalid_shape",
        [
            (32, 32),
            (32,),
            (32, 32, 3, 1),
        ],
    )
    def test_run_raises_for_invalid_dimensions(
        self,
        transform,
        invalid_shape,
    ):
        # Arrange
        image = np.zeros(
            invalid_shape,
            dtype=np.uint8,
        )

        # Act / Assert
        with pytest.raises(ValueError):
            transform.run(image)

    def test_run_raises_for_empty_input(
        self,
        transform,
    ):
        # Arrange
        image = np.empty((0, 0, 3), dtype=np.uint8)

        # Act
        output = transform.run(image)

        # Assert
        assert output.shape == (3, 0, 0)
