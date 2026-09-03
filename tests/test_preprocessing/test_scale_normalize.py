# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import numpy as np
import pytest

from core.python.preprocess.operations.scale import Scale


# =========================================================
# FIXTURES
# =========================================================


@pytest.fixture
def rgb_transform():
    return Scale(
        scale=[0.5, 2.0, 4.0],
        channels=3,
    )


@pytest.fixture
def grayscale_transform():
    return Scale(
        scale=[0.5],
        channels=1,
    )


@pytest.fixture
def sample_rgb_image():
    return np.array(
        [
            [
                [10.0, 20.0, 30.0],
                [40.0, 50.0, 60.0],
            ]
        ],
        dtype=np.float32,
    )


@pytest.fixture
def sample_gray_image():
    return np.array(
        [
            [10.0, 20.0],
            [30.0, 40.0],
        ],
        dtype=np.float32,
    )


# =========================================================
# CONSTRUCTOR TESTS
# =========================================================


class TestScaleConstructor:
    @pytest.mark.parametrize(
        "channels",
        [
            0,
            2,
            4,
            -1,
        ],
    )
    def test_constructor_rejects_invalid_channels(
        self,
        channels,
    ):
        with pytest.raises(ValueError):
            Scale(
                scale=[1.0],
                channels=channels,
            )

    @pytest.mark.parametrize(
        ("scale", "channels"),
        [
            ([], 1),
            ([1.0, 2.0], 1),
            ([1.0], 3),
            ([1.0, 2.0], 3),
            ([1.0, 2.0, 3.0, 4.0], 3),
        ],
    )
    def test_constructor_rejects_scale_channel_mismatch(
        self,
        scale,
        channels,
    ):
        with pytest.raises(ValueError):
            Scale(
                scale=scale,
                channels=channels,
            )

    def test_constructor_accepts_valid_rgb_configuration(
        self,
    ):
        transform = Scale(
            scale=[1.0, 2.0, 3.0],
            channels=3,
        )

        assert transform.channels == 3

        assert np.array_equal(
            transform.scale,
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
        )

    def test_constructor_accepts_valid_grayscale_configuration(
        self,
    ):
        transform = Scale(
            scale=[0.5],
            channels=1,
        )

        assert transform.channels == 1

        assert np.array_equal(
            transform.scale,
            np.array([0.5], dtype=np.float32),
        )


# =========================================================
# VALIDATION TESTS
# =========================================================


class TestScaleValidation:
    def test_rejects_non_floating_input(
        self,
        rgb_transform,
    ):
        image = np.ones(
            (2, 2, 3),
            dtype=np.uint8,
        )

        with pytest.raises(TypeError):
            rgb_transform.run(image)

    @pytest.mark.parametrize(
        "shape",
        [
            (2,),
            (2, 2, 2, 2),
        ],
    )
    def test_rejects_invalid_ndim(
        self,
        rgb_transform,
        shape,
    ):
        image = np.ones(
            shape,
            dtype=np.float32,
        )

        with pytest.raises(ValueError):
            rgb_transform.run(image)

    def test_rejects_rgb_transform_for_grayscale_image(
        self,
    ):
        transform = Scale(
            scale=[1.0, 2.0, 3.0],
            channels=3,
        )

        image = np.ones(
            (2, 2),
            dtype=np.float32,
        )

        with pytest.raises(ValueError):
            transform.run(image)

    def test_rejects_grayscale_transform_for_rgb_image(
        self,
    ):
        transform = Scale(
            scale=[1.0],
            channels=1,
        )

        image = np.ones(
            (2, 2, 3),
            dtype=np.float32,
        )

        with pytest.raises(ValueError):
            transform.run(image)


# =========================================================
# CORRECTNESS TESTS
# =========================================================


class TestScaleCorrectness:
    def test_rgb_scaling(
        self,
        rgb_transform,
        sample_rgb_image,
    ):
        output = rgb_transform.run(sample_rgb_image)

        expected = np.array(
            [
                [
                    [5.0, 40.0, 120.0],
                    [20.0, 100.0, 240.0],
                ]
            ],
            dtype=np.float32,
        )

        assert np.array_equal(output, expected)

    def test_grayscale_scaling(
        self,
        grayscale_transform,
        sample_gray_image,
    ):
        output = grayscale_transform.run(sample_gray_image)

        expected = np.array(
            [
                [5.0, 10.0],
                [15.0, 20.0],
            ],
            dtype=np.float32,
        )

        assert np.array_equal(output, expected)

    @pytest.mark.parametrize(
        "shape",
        [
            (1, 1, 3),
            (10, 10, 3),
            (224, 224, 3),
        ],
    )
    def test_rgb_shape_preserved(
        self,
        rgb_transform,
        shape,
    ):
        image = np.random.rand(*shape).astype(np.float32)

        output = rgb_transform.run(image)

        assert output.shape == shape

    @pytest.mark.parametrize(
        "shape",
        [
            (1, 1),
            (10, 10),
            (224, 224),
        ],
    )
    def test_grayscale_shape_preserved(
        self,
        grayscale_transform,
        shape,
    ):
        image = np.random.rand(*shape).astype(np.float32)

        output = grayscale_transform.run(image)

        assert output.shape == shape


# =========================================================
# INVARIANTS
# =========================================================


class TestScaleInvariants:
    def test_is_inplace_operation_rgb(
        self,
        rgb_transform,
        sample_rgb_image,
    ):
        original = sample_rgb_image.copy()

        output = rgb_transform.run(sample_rgb_image)

        assert output is sample_rgb_image

        assert not np.array_equal(
            original,
            output,
        )

    def test_is_inplace_operation_grayscale(
        self,
        grayscale_transform,
        sample_gray_image,
    ):
        original = sample_gray_image.copy()

        output = grayscale_transform.run(sample_gray_image)

        assert output is sample_gray_image

        assert not np.array_equal(
            original,
            output,
        )

    def test_deterministic_behavior_rgb(
        self,
        rgb_transform,
        sample_rgb_image,
    ):
        img1 = sample_rgb_image.copy()
        img2 = sample_rgb_image.copy()

        out1 = rgb_transform.run(img1)
        out2 = rgb_transform.run(img2)

        assert np.array_equal(out1, out2)

    def test_deterministic_behavior_grayscale(
        self,
        grayscale_transform,
        sample_gray_image,
    ):
        img1 = sample_gray_image.copy()
        img2 = sample_gray_image.copy()

        out1 = grayscale_transform.run(img1)
        out2 = grayscale_transform.run(img2)

        assert np.array_equal(out1, out2)

    def test_returns_numpy_array_rgb(
        self,
        rgb_transform,
        sample_rgb_image,
    ):
        output = rgb_transform.run(sample_rgb_image)

        assert isinstance(output, np.ndarray)

    def test_returns_numpy_array_grayscale(
        self,
        grayscale_transform,
        sample_gray_image,
    ):
        output = grayscale_transform.run(sample_gray_image)

        assert isinstance(output, np.ndarray)
