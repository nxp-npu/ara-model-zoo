# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from itertools import product

import pytest
import numpy as np

from core.python.config import CropShape
from core.python.preprocess.operations import CenterCrop

# =========================================================
# CONFIGURATION COMPILATION TESTS
# =========================================================


class TestCenterCropCompile:
    """
    Tests configuration compilation.

    Contract:
    - Absolute crop selected when height+width provided.
    - Fractional crop selected when fraction provided.
    - Partial absolute crop rejected.
    - Fraction required when absolute crop absent.
    - Correct geometry function bound.
    """

    def test_compile_absolute_crop_binds_absolute_geometry(
        self,
    ):
        crop = CenterCrop(
            CropShape(
                h=224,
                w=224,
            )
        )

        assert crop.crop_h == 224
        assert crop.crop_w == 224

        assert crop._geometry_fn == crop._absolute_geometry

    def test_compile_fractional_crop_binds_fractional_geometry(
        self,
    ):
        crop = CenterCrop(
            CropShape(
                fraction=0.875,
            )
        )

        assert crop.fraction == 0.875

        assert crop._geometry_fn == crop._compute_fractional_geometry

    def test_compile_rejects_missing_width(
        self,
    ):
        with pytest.raises(
            ValueError,
            match=("Both height and width must be provided"),
        ):
            CenterCrop(
                CropShape(
                    h=224,
                )
            )

    def test_compile_rejects_missing_height(
        self,
    ):
        with pytest.raises(
            ValueError,
            match=("Both height and width must be provided"),
        ):
            CenterCrop(
                CropShape(
                    w=224,
                )
            )

    def test_compile_rejects_missing_fraction(
        self,
    ):
        with pytest.raises(
            ValueError,
            match="fraction is not initialized",
        ):
            CenterCrop(CropShape())

    def test_absolute_crop_takes_precedence_over_fraction(self):
        crop = CenterCrop(
            CropShape(
                h=224,
                w=224,
                fraction=0.875,
            )
        )

        assert crop._geometry_fn == crop._absolute_geometry


# =========================================================
# ABSOLUTE CENTER CROP TESTS
# =========================================================


class TestAbsoluteCenterCrop:
    """
    Tests absolute center crop behavior.

    Contract:
    - Output shape equals requested crop shape.
    - Crop extracted from image center.
    - RGB supported.
    - Grayscale supported.
    - Deterministic.
    - Works across arbitrary geometries.
    """

    ABSOLUTE_CASES = [
        ((100, 200, 3), (50, 100)),
        ((200, 100, 3), (50, 50)),
        ((301, 503, 3), (127, 211)),
        ((1000, 1000, 3), (999, 999)),
        ((137, 293), (71, 149)),
    ]

    @pytest.mark.parametrize(
        ("input_shape", "crop_shape"),
        ABSOLUTE_CASES,
    )
    def test_output_matches_requested_crop_shape(
        self,
        input_shape,
        crop_shape,
    ):
        image = np.random.randint(
            0,
            255,
            input_shape,
            dtype=np.uint8,
        )

        crop_h, crop_w = crop_shape

        crop = CenterCrop(
            CropShape(
                h=crop_h,
                w=crop_w,
            )
        )

        output = crop.run(image)

        expected_shape = (
            crop_h,
            crop_w,
            *image.shape[2:],
        )

        assert output.shape == expected_shape, (
            f"Shape mismatch:\n"
            f"INPUT={input_shape}\n"
            f"CROP(H,W)={crop_shape}\n"
            f"OUTPUT={output.shape}\n"
            f"EXPECTED={expected_shape}"
        )

    @pytest.mark.parametrize(
        ("input_shape", "crop_shape"),
        [
            ((100, 200, 3), (50, 100)),
            ((301, 503, 3), (127, 211)),
        ],
    )
    def test_crop_is_extracted_from_image_center(
        self,
        input_shape,
        crop_shape,
    ):
        image = np.arange(
            np.prod(input_shape),
            dtype=np.int64,
        ).reshape(input_shape)

        crop_h, crop_w = crop_shape

        crop = CenterCrop(
            CropShape(
                h=crop_h,
                w=crop_w,
            )
        )

        output = crop.run(image)

        image_h, image_w = image.shape[:2]

        top = (image_h - crop_h) // 2
        left = (image_w - crop_w) // 2

        expected = image[
            top : top + crop_h,
            left : left + crop_w,
        ]

        assert np.array_equal(
            output,
            expected,
        ), (
            f"Center crop mismatch:\n"
            f"INPUT={input_shape}\n"
            f"CROP(H,W)={crop_shape}\n"
            f"TOP={top}\n"
            f"LEFT={left}"
        )

    @pytest.mark.parametrize(
        ("input_shape", "crop_shape"),
        [
            ((100, 200, 3), (50, 100)),
            ((301, 503, 3), (127, 211)),
        ],
    )
    def test_crop_preserves_channels(
        self,
        input_shape,
        crop_shape,
    ):
        image = np.random.randint(
            0,
            255,
            input_shape,
            dtype=np.uint8,
        )

        crop_h, crop_w = crop_shape

        crop = CenterCrop(
            CropShape(
                h=crop_h,
                w=crop_w,
            )
        )

        output = crop.run(image)

        assert output.shape[2] == image.shape[2]

    @pytest.mark.parametrize(
        ("input_shape", "crop_shape"),
        [
            ((100, 200), (50, 100)),
            ((301, 503), (127, 211)),
            ((137, 293), (71, 149)),
        ],
    )
    def test_crop_supports_grayscale(
        self,
        input_shape,
        crop_shape,
    ):
        image = np.random.randint(
            0,
            255,
            input_shape,
            dtype=np.uint8,
        )

        crop_h, crop_w = crop_shape

        crop = CenterCrop(
            CropShape(
                h=crop_h,
                w=crop_w,
            )
        )

        output = crop.run(image)

        assert output.ndim == 2

        assert output.shape == (
            crop_h,
            crop_w,
        )

    def test_crop_is_deterministic(
        self,
    ):
        image = np.random.randint(
            0,
            255,
            (301, 503, 3),
            dtype=np.uint8,
        )

        crop = CenterCrop(
            CropShape(
                h=127,
                w=211,
            )
        )

        output_1 = crop.run(
            image.copy(),
        )

        output_2 = crop.run(
            image.copy(),
        )

        assert np.array_equal(
            output_1,
            output_2,
        )


# =========================================================
# FRACTIONAL CENTER CROP TESTS
# =========================================================


class TestFractionalCenterCrop:
    """
    Tests MobileNet-style fractional center crop behavior.

    Contract:
    - Crop dimensions follow MobileNet formula:
        int(H * fraction + 1), int(W * fraction + 1)
    - +1 bias must be applied correctly.
    - Output shape matches computed geometry.
    - Works for RGB and grayscale.
    - Deterministic.
    """

    # =========================================================
    # STRESS INPUTS
    # =========================================================

    FRACTIONS = [
        0.95,
        0.875,
        0.8,
        0.5,
        0.25,
    ]

    IMAGE_SHAPES = [
        (120, 240, 3),
        (240, 120, 3),
        (333, 777, 3),
        (777, 333, 3),
        (120, 240),
    ]

    # =========================================================
    # CORE GEOMETRY TEST
    # =========================================================

    @pytest.mark.parametrize(
        ("image_shape", "fraction"),
        tuple(product(IMAGE_SHAPES, FRACTIONS)),
    )
    def test_output_matches_mobilenet_fractional_geometry(
        self,
        image_shape,
        fraction,
    ):
        image = np.random.randint(
            0,
            255,
            image_shape,
            dtype=np.uint8,
        )

        crop = CenterCrop(
            CropShape(
                fraction=fraction,
            )
        )

        input_h, input_w = image.shape[:2]

        expected_h = int(input_h * fraction + 1)
        expected_w = int(input_w * fraction + 1)

        output = crop.run(image)

        output_h, output_w = output.shape[:2]

        assert (output_h, output_w) == (
            expected_h,
            expected_w,
        ), (
            f"Fractional crop mismatch:\n"
            f"INPUT={image_shape[:2]}\n"
            f"FRACTION={fraction}\n"
            f"EXPECTED={(expected_h, expected_w)}\n"
            f"OUTPUT={(output_h, output_w)}"
        )

    # =========================================================
    # GRAYSCALE SUPPORT
    # =========================================================

    @pytest.mark.parametrize(
        ("image_shape", "fraction"),
        [
            ((120, 240), 0.875),
            ((240, 120), 0.5),
            ((333, 777), 0.25),
        ],
    )
    def test_grayscale_support(
        self,
        image_shape,
        fraction,
    ):
        image = np.random.randint(
            0,
            255,
            image_shape,
            dtype=np.uint8,
        )

        crop = CenterCrop(
            CropShape(
                fraction=fraction,
            )
        )

        output = crop.run(image)

        assert output.ndim == 2

        input_h, input_w = image_shape

        expected_h = int(input_h * fraction + 1)
        expected_w = int(input_w * fraction + 1)

        assert output.shape == (
            expected_h,
            expected_w,
        )

    # =========================================================
    # DETERMINISM
    # =========================================================

    def test_deterministic_output(self):
        image = np.random.randint(
            0,
            255,
            (333, 777, 3),
            dtype=np.uint8,
        )

        crop = CenterCrop(
            CropShape(
                fraction=0.875,
            )
        )

        out1 = crop.run(image.copy())
        out2 = crop.run(image.copy())

        assert np.array_equal(out1, out2)


# =========================================================
# CENTER CROP POSITIONING TESTS
# =========================================================


class TestCenterCropPositioning:
    """
    Tests correctness of crop centering logic.

    This is the most critical crop test suite.

    Contract:
    - Crop must be centered horizontally.
    - Crop must be centered vertically.
    - Floor division must define correct top-left origin.
    - Odd/even dimension handling must be correct.
    - No off-by-one drift allowed.

    Strategy:
    - Use coordinate-based synthetic images:
        image[y, x] = y OR image[y, x] = x
    - Validate extracted crop origin exactly.
    """

    # =========================================================
    # STRESS SHAPES (designed for off-by-one detection)
    # =========================================================

    STRESS_CASES = [
        ((100, 100), (50, 50)),
        ((101, 101), (50, 50)),
        ((100, 101), (50, 50)),
        ((101, 100), (50, 50)),
        ((137, 293), (71, 149)),
        ((293, 137), (149, 71)),
    ]

    # =========================================================
    # COORDINATE IMAGE (Y-MAP TEST)
    # =========================================================

    @pytest.mark.parametrize(
        ("image_shape", "crop_shape"),
        STRESS_CASES,
    )
    def test_crop_is_centered_vertically_and_horizontally_y_map(
        self,
        image_shape,
        crop_shape,
    ):
        image_h, image_w = image_shape
        crop_h, crop_w = crop_shape

        # coordinate image: value = row index
        image = np.tile(
            np.arange(image_h, dtype=np.int32).reshape(image_h, 1),
            (1, image_w),
        )

        crop = CenterCrop(
            CropShape(
                h=crop_h,
                w=crop_w,
            )
        )

        output = crop.run(image)

        expected_top = (image_h - crop_h) // 2
        expected_left = (image_w - crop_w) // 2

        expected = image[
            expected_top : expected_top + crop_h,
            expected_left : expected_left + crop_w,
        ]

        assert np.array_equal(output, expected), (
            f"Vertical centering mismatch:\n"
            f"INPUT={image_shape}\n"
            f"CROP={crop_shape}\n"
            f"EXPECTED_TOP={expected_top}\n"
            f"EXPECTED_LEFT={expected_left}"
        )

    # =========================================================
    # COORDINATE IMAGE (X-MAP TEST)
    # =========================================================

    @pytest.mark.parametrize(
        ("image_shape", "crop_shape"),
        STRESS_CASES,
    )
    def test_crop_is_centered_horizontally_x_map(
        self,
        image_shape,
        crop_shape,
    ):
        image_h, image_w = image_shape
        crop_h, crop_w = crop_shape

        # coordinate image: value = column index
        image = np.tile(
            np.arange(image_w, dtype=np.int32),
            (image_h, 1),
        )

        crop = CenterCrop(
            CropShape(
                h=crop_h,
                w=crop_w,
            )
        )

        output = crop.run(image)

        expected_top = (image_h - crop_h) // 2
        expected_left = (image_w - crop_w) // 2

        expected = image[
            expected_top : expected_top + crop_h,
            expected_left : expected_left + crop_w,
        ]

        assert np.array_equal(output, expected), (
            f"Horizonal centering mismatch:\n"
            f"INPUT={image_shape}\n"
            f"CROP={crop_shape}\n"
            f"EXPECTED_TOP={expected_top}\n"
            f"EXPECTED_LEFT={expected_left}"
        )

    # =========================================================
    # EXPLICIT ORIGIN VALIDATION (DIRECT BOUNDARY TEST)
    # =========================================================

    @pytest.mark.parametrize(
        ("image_shape", "crop_shape"),
        STRESS_CASES,
    )
    def test_crop_origin_is_exact_floor_division(
        self,
        image_shape,
        crop_shape,
    ):
        image_h, image_w = image_shape
        crop_h, crop_w = crop_shape

        image = np.random.randint(
            0,
            255,
            (image_h, image_w),
            dtype=np.uint8,
        )

        crop = CenterCrop(
            CropShape(
                h=crop_h,
                w=crop_w,
            )
        )

        output = crop.run(image)

        expected_top = (image_h - crop_h) // 2
        expected_left = (image_w - crop_w) // 2

        # Verify top-left pixel matches exact image origin
        assert np.array_equal(
            output[0, 0],
            image[expected_top, expected_left],
        ), (
            f"Origin mismatch:\n"
            f"INPUT={image_shape}\n"
            f"CROP={crop_shape}\n"
            f"EXPECTED_TOP={expected_top}\n"
            f"EXPECTED_LEFT={expected_left}"
        )

    # =========================================================
    # ODD/EVEN ASYMMETRY DETECTION
    # =========================================================

    @pytest.mark.parametrize(
        ("image_shape", "crop_shape"),
        [
            ((101, 101), (50, 50)),
            ((100, 101), (50, 50)),
            ((101, 100), (50, 50)),
        ],
    )
    def test_odd_even_centering_consistency(
        self,
        image_shape,
        crop_shape,
    ):
        image_h, image_w = image_shape
        crop_h, crop_w = crop_shape

        image = np.arange(image_h * image_w).reshape(image_h, image_w)

        crop = CenterCrop(
            CropShape(
                h=crop_h,
                w=crop_w,
            )
        )

        output = crop.run(image)

        expected_top = (image_h - crop_h) // 2
        expected_left = (image_w - crop_w) // 2

        expected = image[
            expected_top : expected_top + crop_h,
            expected_left : expected_left + crop_w,
        ]

        assert np.array_equal(output, expected)


# =========================================================
# CENTER CROP INVARIANTS TESTS
# =========================================================


class TestCenterCropInvariants:
    """
    Tests generic invariants for CenterCrop.

    These tests are NOT about correctness of cropping logic.

    They ensure:
    - type safety
    - API stability
    - deterministic behavior
    - no hidden dtype coercion
    """

    # =========================================================
    # FIXTURE IMAGE
    # =========================================================

    @pytest.fixture
    def image_rgb(self):
        return np.random.randint(
            0,
            255,
            (120, 240, 3),
            dtype=np.uint8,
        )

    @pytest.fixture
    def image_gray(self):
        return np.random.randint(
            0,
            255,
            (120, 240),
            dtype=np.uint8,
        )

    @pytest.fixture
    def crop(self):
        return CenterCrop(
            CropShape(
                h=60,
                w=100,
            )
        )

    # =========================================================
    # TYPE INVARIANTS
    # =========================================================

    def test_returns_ndarray_rgb(self, crop, image_rgb):
        output = crop.run(image_rgb)

        assert isinstance(output, np.ndarray)

    def test_returns_ndarray_grayscale(self, crop, image_gray):
        output = crop.run(image_gray)

        assert isinstance(output, np.ndarray)

    # =========================================================
    # DTYPE INVARIANT
    # =========================================================

    def test_preserves_dtype_rgb(self, crop, image_rgb):
        output = crop.run(image_rgb)

        assert output.dtype == image_rgb.dtype

    def test_preserves_dtype_grayscale(self, crop, image_gray):
        output = crop.run(image_gray)

        assert output.dtype == image_gray.dtype

    # =========================================================
    # DETERMINISM INVARIANT
    # =========================================================

    def test_is_deterministic_rgb(self, crop, image_rgb):
        out1 = crop.run(image_rgb.copy())
        out2 = crop.run(image_rgb.copy())

        assert np.array_equal(out1, out2)

    def test_is_deterministic_grayscale(self, crop, image_gray):
        out1 = crop.run(image_gray.copy())
        out2 = crop.run(image_gray.copy())

        assert np.array_equal(out1, out2)
