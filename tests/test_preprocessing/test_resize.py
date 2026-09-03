# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import numpy as np
import pytest
from unittest.mock import patch, sentinel

import cv2

from core.python.config.config import InterpolationValues
from core.python.preprocess.operations.resize import Resize


# =========================================================
# DISPATCH TESTS
# =========================================================


class TestResizeDispatch:
    @pytest.mark.parametrize(
        (
            "maintain_aspect_ratio",
            "pad",
            "expected_strategy",
            "expected_output",
        ),
        [
            (
                True,
                True,
                "pad",
                sentinel.pad_output,
            ),
            (
                True,
                False,
                "keep_ratio",
                sentinel.keep_ratio_output,
            ),
            (
                False,
                False,
                "simple_resize",
                sentinel.simple_output,
            ),
            (
                False,
                True,
                "simple_resize",
                sentinel.simple_output,
            ),
        ],
    )
    def test_run_dispatches_to_expected_resize_strategy(
        self,
        maintain_aspect_ratio,
        pad,
        expected_strategy,
        expected_output,
    ):
        # Arrange
        transform = Resize(
            shape=(640, 480),
            interpolation=None,
            maintain_aspect_ratio=maintain_aspect_ratio,
            padding=pad,
            padding_value=None,
            scale=1,
            resize=256,
        )

        image = np.zeros(
            (224, 224, 3),
            dtype=np.uint8,
        )

        with (
            patch.object(
                transform,
                "_resize_with_aspectratio_pad",
                return_value=sentinel.pad_output,
            ) as mock_pad,
            patch.object(
                transform,
                "_resize_with_aspectratio",
                return_value=sentinel.keep_ratio_output,
            ) as mock_keep_ratio,
            patch(
                "core.python.preprocess.operations.resize.cv2.resize",
                return_value=sentinel.simple_output,
            ) as mock_cv_resize,
        ):
            # Act
            output = transform.run(image)

            # Assert dispatch behavior
            if expected_strategy == "pad":
                mock_pad.assert_called_once()
                mock_keep_ratio.assert_not_called()
                mock_cv_resize.assert_not_called()

            elif expected_strategy == "keep_ratio":
                mock_keep_ratio.assert_called_once()
                mock_pad.assert_not_called()
                mock_cv_resize.assert_not_called()

            else:
                mock_cv_resize.assert_called_once()
                mock_pad.assert_not_called()
                mock_keep_ratio.assert_not_called()

            # Assert return propagation
            assert output is expected_output


# =========================================================
# INTERPOLATION MAPPING TESTS
# =========================================================


class TestInterpolationMapping:
    """
    Tests interpolation string normalization and mapping.

    Contract:
    - Supported aliases map to the correct OpenCV constant.
    - Mapping is case-insensitive.
    - None defaults to INTER_LINEAR.
    - Invalid interpolation names raise ValueError.
    """

    @pytest.mark.parametrize(
        ("interpolation", "expected"),
        [
            # Default behavior
            (
                None,
                cv2.INTER_LINEAR,
            ),
            # Linear aliases
            (
                InterpolationValues.LINEAR.value,
                cv2.INTER_LINEAR,
            ),
            (
                InterpolationValues.BILINEAR.value,
                cv2.INTER_LINEAR,
            ),
            (
                InterpolationValues.INTER_LINEAR.value,
                cv2.INTER_LINEAR,
            ),
            # Nearest aliases
            (
                InterpolationValues.NEAREST.value,
                cv2.INTER_NEAREST,
            ),
            (
                InterpolationValues.INTER_NEAREST.value,
                cv2.INTER_NEAREST,
            ),
            # Cubic aliases
            (
                InterpolationValues.CUBIC.value,
                cv2.INTER_CUBIC,
            ),
            (
                InterpolationValues.BICUBIC.value,
                cv2.INTER_CUBIC,
            ),
            (
                InterpolationValues.INTER_CUBIC.value,
                cv2.INTER_CUBIC,
            ),
            # Area aliases
            (
                InterpolationValues.AREA.value,
                cv2.INTER_AREA,
            ),
            (
                InterpolationValues.INTER_AREA.value,
                cv2.INTER_AREA,
            ),
            # Lanczos aliases
            (
                InterpolationValues.LANCZOS.value,
                cv2.INTER_LANCZOS4,
            ),
            (
                InterpolationValues.LANCZOS4.value,
                cv2.INTER_LANCZOS4,
            ),
            (
                InterpolationValues.INTER_LANCZOS4.value,
                cv2.INTER_LANCZOS4,
            ),
        ],
    )
    def test_alias_maps_to_expected_cv2_constant(
        self,
        interpolation,
        expected,
    ):
        output = Resize._get_interpolation(
            interpolation,
        )

        assert output == expected

    @pytest.mark.parametrize(
        ("interpolation", "expected"),
        [
            ("linear", cv2.INTER_LINEAR),
            ("LiNeAr", cv2.INTER_LINEAR),
            ("nearest", cv2.INTER_NEAREST),
            ("NeArEsT", cv2.INTER_NEAREST),
            ("cubic", cv2.INTER_CUBIC),
            ("CuBiC", cv2.INTER_CUBIC),
            ("area", cv2.INTER_AREA),
            ("ArEa", cv2.INTER_AREA),
            ("lanczos", cv2.INTER_LANCZOS4),
            ("LaNcZoS", cv2.INTER_LANCZOS4),
        ],
    )
    def test_mapping_is_case_insensitive(
        self,
        interpolation,
        expected,
    ):
        output = Resize._get_interpolation(
            interpolation,
        )

        assert output == expected

    @pytest.mark.parametrize(
        "interpolation",
        [
            "",
            "foo",
            "random",
            "inter_magic",
        ],
    )
    def test_invalid_interpolation_raises_value_error(
        self,
        interpolation,
    ):
        with pytest.raises(
            ValueError,
            match="Invalid Interpolation type",
        ):
            Resize._get_interpolation(
                interpolation,
            )


# =========================================================
# SIMPLE RESIZE TESTS
# =========================================================


class TestSimpleResize:
    """
    Tests resize behavior when:

        maintain_aspect_ratio=False
        padding= Don't Care (X)

    Contract:
    - Output must exactly match target shape
    - Aspect ratio preservation is NOT required
    - RGB and grayscale images must be supported
    - Input dtype must be preserved
    - Resize operation must be deterministic
    """

    @pytest.fixture
    def rgb_image(self):
        """
        Canonical RGB image.
        Shape: (H, W, C)
        """
        return np.random.randint(
            0,
            255,
            (120, 240, 3),
            dtype=np.uint8,
        )

    @pytest.fixture
    def grayscale_image(self):
        """
        Canonical grayscale image.
        Shape: (H, W)
        """
        return np.random.randint(
            0,
            255,
            (120, 240),
            dtype=np.uint8,
        )

    @pytest.fixture
    def simple_resize_transform(self):
        """
        Resize transform configured to use
        direct/simple resize path.
        """
        return Resize(
            shape=(640, 480),
            interpolation=None,
            maintain_aspect_ratio=False,
            padding=False,
            padding_value=None,
        )

    def test_simple_resize_returns_expected_rgb_shape(
        self,
        simple_resize_transform,
        rgb_image,
    ):
        output = simple_resize_transform.run(
            rgb_image,
        )

        assert output.shape == (480, 640, 3)

    def test_simple_resize_returns_expected_grayscale_shape(
        self,
        simple_resize_transform,
        grayscale_image,
    ):
        output = simple_resize_transform.run(
            grayscale_image,
        )

        assert output.shape == (480, 640)

    def test_simple_resize_preserves_dtype(
        self,
        simple_resize_transform,
        rgb_image,
    ):
        output = simple_resize_transform.run(
            rgb_image,
        )

        assert output.dtype == rgb_image.dtype

    def test_simple_resize_is_deterministic(
        self,
        simple_resize_transform,
        rgb_image,
    ):
        output_1 = simple_resize_transform.run(
            rgb_image,
        )

        output_2 = simple_resize_transform.run(
            rgb_image,
        )

        assert np.array_equal(
            output_1,
            output_2,
        )


# =========================================================
# MAINTAIN ASPECT RATIO BUT NO PAD RESIZE TESTS
# =========================================================


class TestKeepRatioResizeNoPad:
    """
    Tests resize behavior when:

        maintain_aspect_ratio=True
        padding=False

    Contract:
    - Aspect ratio must be preserved
    - Output must fit within target bounds
    - Output must NOT be forced to target shape
    - No distortion across arbitrary geometries
    """

    @pytest.mark.parametrize(
        ("input_shape", "target_shape"),
        [
            # RGB cases
            ((100, 200, 3), (640, 640)),
            ((200, 100, 3), (640, 640)),
            ((200, 200, 3), (640, 640)),
            # grayscale equivalents
            ((100, 200), (640, 640)),
            ((200, 100), (640, 640)),
            ((200, 200), (640, 640)),
            # non-square targets (RGB + grayscale behavior parity)
            ((200, 200, 3), (400, 800)),
            ((200, 200, 3), (800, 400)),
            ((200, 200), (400, 800)),
            ((200, 200), (800, 400)),
            # extreme aspect ratios (both modalities)
            ((50, 500, 3), (640, 640)),
            ((500, 50, 3), (640, 640)),
            ((50, 500), (640, 640)),
            ((500, 50), (640, 640)),
            # stress tests
            ((137, 293, 3), (517, 811)),
            ((101, 307, 3), (601, 997)),
            ((137, 293), (517, 811)),
            ((101, 307), (601, 997)),
        ],
    )
    def test_keep_ratio_preserves_aspect_ratio(
        self,
        input_shape,
        target_shape,
    ):
        image = np.random.randint(
            0,
            255,
            input_shape,
            dtype=np.uint8,
        )

        transform = Resize(
            shape=target_shape,
            interpolation=None,
            maintain_aspect_ratio=True,
            padding=False,
        )

        input_h, input_w = image.shape[:2]
        input_ratio = input_w / input_h

        output = transform.run(image)

        output_h, output_w = output.shape[:2]
        output_ratio = output_w / output_h

        geometry = (
            f"INPUT={input_shape[:2]} "
            f"(ratio={input_ratio:.4f}), "
            f"TARGET(W,H)={target_shape}, "
            f"OUTPUT={output.shape[:2]} "
            f"(ratio={output_ratio:.4f})"
        )

        assert output_ratio == pytest.approx(
            input_ratio,
            rel=1e-2,
        ), f"Aspect ratio violation: {geometry}"

    @pytest.mark.parametrize(
        ("input_shape", "target_shape"),
        [
            ((100, 200, 3), (640, 640)),
            ((200, 100, 3), (640, 640)),
            ((50, 500, 3), (640, 640)),
            ((500, 50, 3), (640, 640)),
            ((137, 293, 3), (517, 811)),
        ],
    )
    def test_keep_ratio_output_within_target_bounds(
        self,
        input_shape,
        target_shape,
    ):
        image = np.random.randint(
            0,
            255,
            input_shape,
            dtype=np.uint8,
        )

        transform = Resize(
            shape=target_shape,
            interpolation=None,
            maintain_aspect_ratio=True,
            padding=False,
        )

        output = transform.run(image)

        output_h, output_w = output.shape[:2]

        target_w, target_h = target_shape

        assert output_h <= target_h, (
            f"Height overflow: INPUT={input_shape[:2]}, "
            f"TARGET(W,H)={target_shape}, OUTPUT={output.shape[:2]}"
        )

        assert output_w <= target_w, (
            f"Width overflow: INPUT={input_shape[:2]}, "
            f"TARGET(W,H)={target_shape}, OUTPUT={output.shape[:2]}"
        )

    @pytest.mark.parametrize(
        ("input_shape", "target_shape"),
        [
            ((100, 200, 3), (640, 640)),
            ((200, 100, 3), (640, 640)),
            ((200, 200, 3), (400, 800)),
            ((200, 200, 3), (800, 400)),
        ],
    )
    def test_keep_ratio_does_not_force_target_shape(
        self,
        input_shape,
        target_shape,
    ):
        image = np.random.randint(
            0,
            255,
            input_shape,
            dtype=np.uint8,
        )

        transform = Resize(
            shape=target_shape,
            interpolation=None,
            maintain_aspect_ratio=True,
            padding=False,
        )

        input_ratio = input_shape[1] / input_shape[0]

        target_ratio = target_shape[0] / target_shape[1]

        output = transform.run(image)

        if input_ratio != pytest.approx(target_ratio, rel=1e-6):
            assert output.shape[:2] != (
                target_shape[1],
                target_shape[0],
            ), (
                f"FORCED RESIZE BUG DETECTED: "
                f"INPUT={input_shape[:2]} (ratio={input_ratio:.4f}), "
                f"TARGET(W,H)={target_shape} (ratio={target_ratio:.4f}), "
                f"OUTPUT={output.shape[:2]}"
            )

    def test_keep_ratio_square_input_non_square_target(
        self,
    ):
        image = np.random.randint(
            0,
            255,
            (200, 200, 3),
            dtype=np.uint8,
        )

        transform = Resize(
            shape=(400, 800),
            interpolation=None,
            maintain_aspect_ratio=True,
            padding=False,
        )

        output = transform.run(image)

        out_h, out_w = output.shape[:2]

        assert out_w / out_h == pytest.approx(
            1.0,
            rel=1e-2,
        ), f"Square invariant broken. OUTPUT={output.shape[:2]}"

        target_w, target_h = (400, 800)

        assert out_h <= target_h
        assert out_w <= target_w

        assert output.shape[:2] != (target_h, target_w)

    def test_keep_ratio_preserves_dtype(
        self,
    ):
        image = np.random.randint(
            0,
            255,
            (100, 200, 3),
            dtype=np.uint8,
        )

        transform = Resize(
            shape=(640, 640),
            interpolation=None,
            maintain_aspect_ratio=True,
            padding=False,
        )

        output = transform.run(image)

        assert output.dtype == image.dtype

    def test_keep_ratio_preserves_channels_rgb(
        self,
    ):
        image = np.random.randint(
            0,
            255,
            (100, 200, 3),
            dtype=np.uint8,
        )

        transform = Resize(
            shape=(640, 640),
            interpolation=None,
            maintain_aspect_ratio=True,
            padding=False,
        )

        output = transform.run(image)

        assert output.shape[2] == 3

    def test_keep_ratio_supports_grayscale(
        self,
    ):
        image = np.random.randint(
            0,
            255,
            (100, 200),
            dtype=np.uint8,
        )

        transform = Resize(
            shape=(640, 640),
            interpolation=None,
            maintain_aspect_ratio=True,
            padding=False,
        )

        output = transform.run(image)

        assert output.ndim == 2

    def test_keep_ratio_is_deterministic(
        self,
    ):
        image = np.random.randint(
            0,
            255,
            (100, 200, 3),
            dtype=np.uint8,
        )

        transform = Resize(
            shape=(640, 640),
            interpolation=None,
            maintain_aspect_ratio=True,
            padding=False,
        )

        out1 = transform.run(image.copy())
        out2 = transform.run(image.copy())

        assert np.array_equal(out1, out2)


# =========================================================
# MAINTAIN ASPECT RATIO WITH PAD RESIZE TESTS
# =========================================================


class TestKeepRatioResizeWithPad:
    """
    Tests resize behavior when:
    - maintain_aspect_ratio=True
    - padding=True

    Contract:
    - Output must match target shape exactly
    - Aspect ratio preserved in resized content
    - Padding compensates mismatch
    - Content is centered
    - Padding value is correct
    """

    TARGET_SHAPE = (640, 640)  # (width, height)
    PAD_VALUE = 0

    # =========================================================
    # WEIRD TARGET SHAPES (EDGE CASE STRESS SET)
    # =========================================================
    WEIRD_TARGET_SHAPES = [
        (641, 639),
        (799, 401),
        (513, 777),
        (777, 513),
        (200, 2000),  # ultra-wide
        (2000, 200),  # ultra-tall
        (317, 521),
        (643, 991),
        (419, 827),
        # (32, 64),
        (15, 27),
        # (33, 17),
        (480, 853),  # mobile portrait
        (1080, 1920),  # HD portrait
        (1920, 1080),  # HD landscape
    ]

    # =========================================================
    # FIXTURES
    # =========================================================

    @pytest.fixture
    def rgb_square(self):
        return np.full((200, 200, 3), 255, dtype=np.uint8)

    def _content_bbox(self, output):
        """Extract non-padding region bounding box."""
        if output.ndim == 3:
            mask = np.any(output != self.PAD_VALUE, axis=-1)
        else:
            mask = output != self.PAD_VALUE

        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]

        return rows, cols

    # =========================================================
    # CORE SHAPE VALIDATION
    # =========================================================

    @pytest.mark.parametrize("target_shape", WEIRD_TARGET_SHAPES)
    def test_output_matches_target_shape(self, target_shape, rgb_square):
        transform = Resize(
            shape=target_shape,
            interpolation=None,
            maintain_aspect_ratio=True,
            padding=True,
            padding_value=self.PAD_VALUE,
        )

        output = transform.run(rgb_square)

        assert output.shape[:2] == target_shape[::-1], (
            f"Shape mismatch:\n"
            f"INPUT={rgb_square.shape[:2]}\n"
            f"TARGET(W,H)={target_shape}\n"
            f"OUTPUT={output.shape[:2]}"
        )

    # =========================================================
    # ASPECT RATIO PRESERVATION (CONTENT DOMAIN)
    # =========================================================

    @pytest.mark.parametrize("target_shape", WEIRD_TARGET_SHAPES)
    def test_aspect_ratio_preserved_across_weird_targets(self, target_shape):
        image = np.full((120, 300, 3), 255, dtype=np.uint8)

        transform = Resize(
            shape=target_shape,
            interpolation=None,
            maintain_aspect_ratio=True,
            padding=True,
            padding_value=self.PAD_VALUE,
        )

        input_h, input_w = image.shape[:2]
        input_ratio = input_w / input_h

        output = transform.run(image)

        rows, cols = self._content_bbox(output)

        content_h = len(rows)
        content_w = len(cols)

        output_ratio = content_w / content_h

        assert output_ratio == pytest.approx(input_ratio, rel=1e-2), (
            f"Aspect ratio violation:\n"
            f"INPUT={image.shape[:2]} ratio={input_ratio:.4f}\n"
            f"CONTENT(W,H)={(content_w, content_h)}\n"
            f"TARGET(W,H)={target_shape}\n"
            f"OUTPUT={output.shape[:2]} ratio={output_ratio:.4f}"
        )

    # =========================================================
    # PADDING MUST EXIST (NON-DEGENERATE CASES)
    # =========================================================

    @pytest.mark.parametrize("target_shape", WEIRD_TARGET_SHAPES)
    def test_padding_applied_only_when_aspect_ratio_mismatch(
        self,
        target_shape,
    ):
        # Arrange
        image = np.full(
            (120, 300, 3),
            fill_value=255,
            dtype=np.uint8,
        )

        transform = Resize(
            shape=target_shape,
            interpolation=None,
            maintain_aspect_ratio=True,
            padding=True,
            padding_value=self.PAD_VALUE,
        )

        input_h, input_w = image.shape[:2]
        target_w, target_h = target_shape

        input_ratio = input_w / input_h
        target_ratio = target_w / target_h

        # Act
        output = transform.run(image)

        # Input is entirely 255 and pad value is 0.
        # Therefore any occurrence of PAD_VALUE
        # indicates padding was introduced.
        padding_exists = np.any(output == self.PAD_VALUE)

        geometry = (
            f"INPUT={image.shape[:2]} "
            f"(ratio={input_ratio:.4f}), "
            f"TARGET(W,H)={target_shape} "
            f"(ratio={target_ratio:.4f}), "
            f"OUTPUT={output.shape[:2]}"
        )

        # Assert
        if input_ratio == pytest.approx(
            target_ratio,
            rel=1e-6,
        ):
            assert not padding_exists, f"Padding unexpectedly detected. {geometry}"
        else:
            assert padding_exists, f"Expected padding was not detected. {geometry}"

    # =========================================================
    # CENTERING CHECK (CRITICAL FOR LETTERBOXING)
    # =========================================================

    @pytest.mark.parametrize("target_shape", WEIRD_TARGET_SHAPES)
    def test_content_is_centered(self, target_shape):
        image = np.full((120, 300, 3), 255, dtype=np.uint8)

        transform = Resize(
            shape=target_shape,
            interpolation=None,
            maintain_aspect_ratio=True,
            padding=True,
            padding_value=self.PAD_VALUE,
        )

        output = transform.run(image)

        rows, cols = self._content_bbox(output)

        top = rows[0]
        bottom = output.shape[0] - rows[-1] - 1

        left = cols[0]
        right = output.shape[1] - cols[-1] - 1

        assert abs(top - bottom) <= 1
        assert abs(left - right) <= 1

    # =========================================================
    # GRAYSCALE SUPPORT
    # =========================================================

    @pytest.mark.parametrize("target_shape", WEIRD_TARGET_SHAPES)
    def test_grayscale_support(self, target_shape):
        image = np.full((120, 240), 255, dtype=np.uint8)

        transform = Resize(
            shape=target_shape,
            interpolation=None,
            maintain_aspect_ratio=True,
            padding=True,
            padding_value=self.PAD_VALUE,
        )

        output = transform.run(image)

        assert output.ndim == 2
        assert output.shape == target_shape[::-1]

    # =========================================================
    # INVARIANTS
    # =========================================================

    def test_deterministic(self):
        image = np.full((120, 300, 3), 255, dtype=np.uint8)

        transform = Resize(
            shape=self.TARGET_SHAPE,
            interpolation=None,
            maintain_aspect_ratio=True,
            padding=True,
            padding_value=self.PAD_VALUE,
        )

        out1 = transform.run(image.copy())
        out2 = transform.run(image.copy())

        assert np.array_equal(out1, out2)

    def test_returns_numpy_array(self):
        image = np.full((120, 300, 3), 255, dtype=np.uint8)

        transform = Resize(
            shape=self.TARGET_SHAPE,
            interpolation=None,
            maintain_aspect_ratio=True,
            padding=True,
            padding_value=self.PAD_VALUE,
        )

        output = transform.run(image)

        assert isinstance(output, np.ndarray)
