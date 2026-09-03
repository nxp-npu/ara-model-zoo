# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from unittest.mock import patch

import numpy as np
import pytest

from core.python.postprocess.detection_utils import (
    scale_pose_keypoints,
    scale_coords,
    scale_boxes,
    scale_mask,
    crop_mask,
    xyxy2xywh,
    xywh2xyxy,
    process_mask,
    numpy_nms,
    nms_yolov10,
    nms_yolov8,
)


# =========================================================
# TEST scale_pose_keypoints
# =========================================================


class TestScalePoseKeypoints:
    """
    Contract tests for scale_pose_keypoints.

    Current implementation assumptions:

    - Input is iterable over poses.
    - Each pose must be reshapeable to
      (num_keypoints, 3).
    - Output dtype is float32.
    - Empty input returns shape
      (0, num_keypoints, 3).
    """

    # ---------------------------------------------------------
    # Basic behavior
    # ---------------------------------------------------------

    def test_single_pose_output_shape(self):
        keypoints = np.ones((1, 51), dtype=np.float32)

        result = scale_pose_keypoints(
            keypoints,
            model_input_shape=(640, 640),
            image_shape=(480, 640),
        )

        assert result.shape == (1, 17, 3)

    def test_multiple_pose_output_shape(self):
        keypoints = np.ones((3, 51), dtype=np.float32)

        result = scale_pose_keypoints(
            keypoints,
            model_input_shape=(640, 640),
            image_shape=(480, 640),
        )

        assert result.shape == (3, 17, 3)

    def test_output_dtype_is_float32(self):
        keypoints = np.ones((2, 51), dtype=np.float64)

        result = scale_pose_keypoints(
            keypoints,
            model_input_shape=(640, 640),
            image_shape=(480, 640),
        )

        assert result.dtype == np.float32

    # ---------------------------------------------------------
    # Empty input behavior
    # ---------------------------------------------------------

    def test_empty_input_returns_empty_pose_tensor(self):
        keypoints = np.empty((0, 51), dtype=np.float32)

        result = scale_pose_keypoints(
            keypoints,
            model_input_shape=(640, 640),
            image_shape=(480, 640),
        )

        assert result.shape == (0, 17, 3)
        assert result.dtype == np.float32

    def test_empty_input_respects_custom_num_keypoints(self):
        keypoints = np.empty((0, 15), dtype=np.float32)

        result = scale_pose_keypoints(
            keypoints,
            model_input_shape=(640, 640),
            image_shape=(480, 640),
            num_keypoints=5,
        )

        assert result.shape == (0, 5, 3)

    # ---------------------------------------------------------
    # num_keypoints behavior
    # ---------------------------------------------------------

    def test_custom_num_keypoints(self):
        keypoints = np.ones((2, 15), dtype=np.float32)

        result = scale_pose_keypoints(
            keypoints,
            model_input_shape=(640, 640),
            image_shape=(480, 640),
            num_keypoints=5,
        )

        assert result.shape == (2, 5, 3)

    # -----------------------------------------------------
    # Characterization: negative num_keypoints
    # -----------------------------------------------------
    @pytest.mark.parametrize(
        ("num_keypoints", "expected_shape"),
        [
            (-1, (1, 17, 3)),
            (-5, (1, 17, 3)),
            (-999, (1, 17, 3)),
        ],
    )
    def test_negative_num_keypoints_uses_numpy_inference(
        self,
        num_keypoints,
        expected_shape,
    ):
        """
        Characterization test.

        Documents current behavior when num_keypoints is
        negative. NumPy reshape treats a single negative
        dimension as an inferred dimension rather than a
        literal keypoint count.
        """

        keypoints = np.ones(
            (1, 51),
            dtype=np.float32,
        )

        result = scale_pose_keypoints(
            keypoints,
            model_input_shape=(640, 640),
            image_shape=(480, 640),
            num_keypoints=num_keypoints,
        )

        assert result.shape == expected_shape

    @pytest.mark.parametrize(
        ("num_keypoints", "expected_shape"),
        [
            (-1, (1, 17, 3)),
            (-5, (1, 17, 3)),
            (-999, (1, 17, 3)),
        ],
    )
    def test_negative_num_keypoints_3d_input_behavior(
        self,
        num_keypoints,
        expected_shape,
    ):
        """
        Characterization test.

        Documents current behavior when the input is already
        shaped as (N, num_keypoints, 3) and num_keypoints is
        negative.
        """

        keypoints = np.ones(
            (1, 17, 3),
            dtype=np.float32,
        )

        result = scale_pose_keypoints(
            keypoints,
            model_input_shape=(640, 640),
            image_shape=(480, 640),
            num_keypoints=num_keypoints,
        )

        assert result.shape == expected_shape

    # -----------------------------------------------------
    # Characterization: zero num_keypoints
    # -----------------------------------------------------
    def test_zero_num_keypoints_current_behavior(self):
        """
        Documents current NumPy reshape behavior for
        num_keypoints=0.
        """

        keypoints = np.ones(
            (1, 51),
            dtype=np.float32,
        )

        with pytest.raises(ValueError):
            scale_pose_keypoints(
                keypoints,
                model_input_shape=(640, 640),
                image_shape=(480, 640),
                num_keypoints=0,
            )

    # ---------------------------------------------------------
    # Shape requirements
    # ---------------------------------------------------------

    def test_invalid_flattened_keypoint_length_raises(self):
        keypoints = np.ones((1, 50), dtype=np.float32)

        with pytest.raises(ValueError):
            scale_pose_keypoints(
                keypoints,
                model_input_shape=(640, 640),
                image_shape=(480, 640),
            )

    def test_custom_num_keypoints_shape_mismatch_raises(self):
        keypoints = np.ones((1, 16), dtype=np.float32)

        with pytest.raises(ValueError):
            scale_pose_keypoints(
                keypoints,
                model_input_shape=(640, 640),
                image_shape=(480, 640),
                num_keypoints=5,
            )

    # ---------------------------------------------------------
    # Undocumented rank behavior
    # ---------------------------------------------------------

    def test_1d_input_behavior(self):
        """
        Documents current behavior for
        rank-1 input.
        """

        keypoints = np.ones((51,), dtype=np.float32)

        with pytest.raises(Exception):
            scale_pose_keypoints(
                keypoints,
                model_input_shape=(640, 640),
                image_shape=(480, 640),
            )

    def test_scalar_input_behavior(self):
        """
        Documents current behavior for
        scalar input.
        """

        keypoints = np.array(42.0, dtype=np.float32)

        with pytest.raises(Exception):
            scale_pose_keypoints(
                keypoints,
                model_input_shape=(640, 640),
                image_shape=(480, 640),
            )

    def test_3d_input_behavior(self):
        """
        Current implementation appears
        to accept already-reshaped poses.
        """

        keypoints = np.ones((2, 17, 3), dtype=np.float32)

        result = scale_pose_keypoints(
            keypoints,
            model_input_shape=(640, 640),
            image_shape=(480, 640),
        )

        assert result.shape == (2, 17, 3)

    # ---------------------------------------------------------
    # scale_coords integration
    # ---------------------------------------------------------

    @patch(
        "core.python.postprocess.detection_utils.scale_coords",
        side_effect=lambda model_shape, kpts, image_shape: kpts,
    )
    def test_scales_each_pose_independently(
        self,
        mock_scale_coords,
    ):
        keypoints = np.ones((4, 51), dtype=np.float32)

        scale_pose_keypoints(
            keypoints,
            model_input_shape=(640, 640),
            image_shape=(480, 640),
        )

        assert mock_scale_coords.call_count == 4

    @patch(
        "core.python.postprocess.detection_utils.scale_coords",
        side_effect=lambda model_shape, kpts, image_shape: kpts,
    )
    def test_forwards_expected_arguments(
        self,
        mock_scale_coords,
    ):
        keypoints = np.ones((1, 51), dtype=np.float32)

        model_input_shape = (640, 640)
        image_shape = (480, 640)

        scale_pose_keypoints(
            keypoints,
            model_input_shape=model_input_shape,
            image_shape=image_shape,
        )

        args, _ = mock_scale_coords.call_args

        assert args[0] == model_input_shape
        assert args[2] == image_shape
        assert args[1].shape == (17, 3)

    @patch("core.python.postprocess.detection_utils.scale_coords")
    def test_preserves_pose_order(
        self,
        mock_scale_coords,
    ):
        mock_scale_coords.side_effect = [
            np.full((17, 3), 1, dtype=np.float32),
            np.full((17, 3), 2, dtype=np.float32),
            np.full((17, 3), 3, dtype=np.float32),
        ]

        keypoints = np.ones((3, 51), dtype=np.float32)

        result = scale_pose_keypoints(
            keypoints,
            model_input_shape=(640, 640),
            image_shape=(480, 640),
        )

        assert np.all(result[0] == 1)
        assert np.all(result[1] == 2)
        assert np.all(result[2] == 3)


# -----------------------------------------------------
# Shared test cases
# -----------------------------------------------------

SCALE_FUNCTION_CASES = [
    pytest.param(
        scale_coords,
        2,
        id="scale_coords",
    ),
    pytest.param(
        scale_boxes,
        4,
        id="scale_boxes",
    ),
]


# =====================================================
# COMMON SCALING CONTRACTS
# =====================================================


class TestScalingUtilitiesCommon:
    """
    Shared contract tests for coordinate scaling utilities.

    Covers behavior common to:
        - scale_coords
        - scale_boxes
    """

    # -------------------------------------------------
    # Shape preservation
    # -------------------------------------------------

    @pytest.mark.parametrize(
        ("scale_fn", "num_columns"),
        SCALE_FUNCTION_CASES,
    )
    def test_preserves_shape(
        self,
        scale_fn,
        num_columns,
    ):
        coords = np.ones(
            (5, num_columns),
            dtype=np.float32,
        )

        result = scale_fn(
            model_input_shape=(640, 640),
            coords=coords.copy(),
            img_shape=(640, 640),
        )

        assert result.shape == coords.shape

    # -------------------------------------------------
    # ratio_pad override
    # -------------------------------------------------

    @pytest.mark.parametrize(
        ("scale_fn", "coords"),
        [
            pytest.param(
                scale_coords,
                np.array([[30.0, 40.0]], dtype=np.float32),
                id="scale_coords",
            ),
            pytest.param(
                scale_boxes,
                np.array([[30.0, 40.0, 50.0, 60.0]], dtype=np.float32),
                id="scale_boxes",
            ),
        ],
    )
    def test_uses_provided_ratio_pad(
        self,
        scale_fn,
        coords,
    ):
        result = scale_fn(
            model_input_shape=(640, 640),
            coords=coords.copy(),
            img_shape=(320, 320),
            ratio_pad=((2.0,), (10.0, 20.0)),
        )

        assert result.shape == coords.shape

    # -------------------------------------------------
    # Lower clipping
    # -------------------------------------------------

    @pytest.mark.parametrize(
        ("scale_fn", "coords"),
        [
            pytest.param(
                scale_coords,
                np.array([[-100.0, -200.0]], dtype=np.float32),
                id="scale_coords",
            ),
            pytest.param(
                scale_boxes,
                np.array(
                    [[-100.0, -200.0, -50.0, -60.0]],
                    dtype=np.float32,
                ),
                id="scale_boxes",
            ),
        ],
    )
    def test_clips_negative_values(
        self,
        scale_fn,
        coords,
    ):
        result = scale_fn(
            model_input_shape=(640, 640),
            coords=coords.copy(),
            img_shape=(480, 640),
        )

        assert np.all(result >= 0)

    # -------------------------------------------------
    # Upper clipping
    # -------------------------------------------------

    @pytest.mark.parametrize(
        ("scale_fn", "coords"),
        [
            pytest.param(
                scale_coords,
                np.array([[1e6, 1e6]], dtype=np.float32),
                id="scale_coords",
            ),
            pytest.param(
                scale_boxes,
                np.array(
                    [[1e6, 1e6, 1e6, 1e6]],
                    dtype=np.float32,
                ),
                id="scale_boxes",
            ),
        ],
    )
    def test_clips_to_image_boundaries(
        self,
        scale_fn,
        coords,
    ):
        result = scale_fn(
            model_input_shape=(640, 640),
            coords=coords.copy(),
            img_shape=(480, 640),
        )

        assert np.all(result >= 0)

        assert np.all(result[..., 0] <= 640)
        assert np.all(result[..., 1] <= 480)

    # -------------------------------------------------
    # Empty input
    # -------------------------------------------------

    @pytest.mark.parametrize(
        ("scale_fn", "num_columns"),
        SCALE_FUNCTION_CASES,
    )
    def test_handles_empty_input(
        self,
        scale_fn,
        num_columns,
    ):
        coords = np.empty(
            (0, num_columns),
            dtype=np.float32,
        )

        result = scale_fn(
            model_input_shape=(640, 640),
            coords=coords.copy(),
            img_shape=(480, 640),
        )

        assert result.shape == coords.shape

    # -------------------------------------------------
    # In-place modification
    # -------------------------------------------------

    @pytest.mark.parametrize(
        ("scale_fn", "coords"),
        [
            pytest.param(
                scale_coords,
                np.array([[320.0, 320.0]], dtype=np.float32),
                id="scale_coords",
            ),
            pytest.param(
                scale_boxes,
                np.array(
                    [[100.0, 100.0, 200.0, 200.0]],
                    dtype=np.float32,
                ),
                id="scale_boxes",
            ),
        ],
    )
    def test_modifies_input_array_in_place(
        self,
        scale_fn,
        coords,
    ):
        result = scale_fn(
            model_input_shape=(640, 640),
            coords=coords,
            img_shape=(320, 320),
        )

        assert result is coords

    # -------------------------------------------------
    # Rank-1 behavior
    # -------------------------------------------------

    @pytest.mark.parametrize(
        ("scale_fn", "coords"),
        [
            pytest.param(
                scale_coords,
                np.array([1.0, 2.0], dtype=np.float32),
                id="scale_coords",
            ),
            pytest.param(
                scale_boxes,
                np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                id="scale_boxes",
            ),
        ],
    )
    def test_rank1_input_current_behavior(
        self,
        scale_fn,
        coords,
    ):
        with pytest.raises(Exception):
            scale_fn(
                model_input_shape=(640, 640),
                coords=coords,
                img_shape=(480, 640),
            )

    # -------------------------------------------------
    # Scalar behavior
    # -------------------------------------------------

    @pytest.mark.parametrize(
        "scale_fn",
        [
            scale_coords,
            scale_boxes,
        ],
    )
    def test_scalar_input_current_behavior(
        self,
        scale_fn,
    ):
        coords = np.array(
            42.0,
            dtype=np.float32,
        )

        with pytest.raises(Exception):
            scale_fn(
                model_input_shape=(640, 640),
                coords=coords,
                img_shape=(480, 640),
            )


# =====================================================
# SCALE_COORDS
# =====================================================


class TestScaleCoords:
    """
    Contracts specific to scale_coords.
    """

    def test_preserves_third_column(self):
        coords = np.array(
            [
                [100.0, 200.0, 0.75],
                [300.0, 400.0, 0.90],
            ],
            dtype=np.float32,
        )

        result = scale_coords(
            model_input_shape=(640, 640),
            coords=coords.copy(),
            img_shape=(640, 640),
        )

        np.testing.assert_array_equal(
            result[:, 2],
            coords[:, 2],
        )

    def test_single_column_input_current_behavior(self):
        coords = np.ones(
            (5, 1),
            dtype=np.float32,
        )

        with pytest.raises(Exception):
            scale_coords(
                model_input_shape=(640, 640),
                coords=coords,
                img_shape=(480, 640),
            )

    def test_higher_dimensional_input_behavior(self):
        """
        Current implementation may accept certain
        higher-dimensional inputs due to NumPy indexing
        semantics. This behavior is documented here for
        visibility but is not part of the documented contract.
        """

        coords = np.ones(
            (2, 3, 2, 2),
            dtype=np.float32,
        )

        result = scale_coords(
            model_input_shape=(640, 640),
            coords=coords.copy(),
            img_shape=(640, 640),
        )

        assert result.shape == coords.shape


# =====================================================
# SCALE_BOXES
# =====================================================


class TestScaleBoxes:
    """
    Contracts specific to scale_boxes.
    """

    def test_insufficient_box_columns_current_behavior(self):
        coords = np.ones(
            (5, 3),
            dtype=np.float32,
        )

        with pytest.raises(Exception):
            scale_boxes(
                model_input_shape=(640, 640),
                coords=coords,
                img_shape=(480, 640),
            )

    def test_higher_dimensional_input_behavior(self):
        """
        Current implementation may accept certain
        higher-dimensional inputs due to NumPy indexing
        semantics. This behavior is documented here for
        visibility but is not part of the documented contract.
        """

        coords = np.ones(
            (2, 4, 2),
            dtype=np.float32,
        )

        result = scale_boxes(
            model_input_shape=(640, 640),
            coords=coords.copy(),
            img_shape=(640, 640),
        )

        assert result.shape == coords.shape

    def test_higher_dimensional_insufficient_second_dimension(self):
        coords = np.ones(
            (2, 3, 4),
            dtype=np.float32,
        )

        with pytest.raises(IndexError):
            scale_boxes(
                model_input_shape=(640, 640),
                coords=coords,
                img_shape=(640, 640),
            )


# =====================================================
# TEST SCALE_MASK
# =====================================================


class TestScaleMask:
    """
    Contract and characterization tests for scale_mask.

    Contract:
        - Accepts a 2D binary mask.
        - shape must contain exactly (height, width).
        - Returns a uint8 BGR image of shape (height, width, 3).
        - Output values are binary (0 or 255).
        - Does not mutate the caller's input mask.
        - Rejects non-binary masks.

    Characterization tests document current behavior for
    unsupported inputs that are not explicitly validated.
    """

    # -----------------------------------------------------
    # Basic behavior
    # -----------------------------------------------------

    @pytest.mark.parametrize(
        ("input_shape", "output_shape"),
        [
            ((640, 640), [320, 320]),
            ((320, 320), [320, 320]),
        ],
        ids=[
            "resize",
            "identity",
        ],
    )
    def test_returns_expected_shape_and_dtype(
        self,
        input_shape,
        output_shape,
    ):
        mask = np.ones(
            input_shape,
            dtype=np.uint8,
        )

        result = scale_mask(
            mask=mask,
            shape=output_shape,
        )

        assert result.shape == (*output_shape, 3)
        assert result.dtype == np.uint8

    # -----------------------------------------------------
    # Output contract
    # -----------------------------------------------------

    def test_output_contains_only_binary_values(self):
        mask = np.random.randint(
            0,
            2,
            size=(640, 640),
            dtype=np.uint8,
        )

        result = scale_mask(
            mask=mask,
            shape=[320, 320],
        )

        assert set(np.unique(result).tolist()) <= {0, 255}

    def test_output_is_three_channel_bgr(self):
        mask = np.ones(
            (640, 640),
            dtype=np.uint8,
        )

        result = scale_mask(
            mask=mask,
            shape=[320, 320],
        )

        assert result.ndim == 3
        assert result.shape[2] == 3

    # -----------------------------------------------------
    # Edge cases
    # -----------------------------------------------------

    def test_all_zero_mask_remains_zero(self):
        mask = np.zeros(
            (640, 640),
            dtype=np.uint8,
        )

        result = scale_mask(
            mask=mask,
            shape=[320, 320],
        )

        assert np.all(result == 0)

    def test_all_one_mask_remains_foreground(self):
        mask = np.ones(
            (640, 640),
            dtype=np.uint8,
        )

        result = scale_mask(
            mask=mask,
            shape=[320, 320],
        )

        assert np.all(result == 255)

    @pytest.mark.parametrize("foreground_ratio", [0.1, 0.25, 0.5, 0.75, 0.9])
    def test_preserves_foreground_ratio_when_no_padding(
        self,
        foreground_ratio: float,
    ):
        mask = np.zeros((100, 100), dtype=np.uint8)

        foreground_width = round(mask.shape[1] * foreground_ratio)
        mask[:, :foreground_width] = 1

        expected_ratio = np.mean(mask == 1)

        result = scale_mask(
            mask=mask,
            shape=[100, 100],  # Same shape => no padding removal or resizing.
        )

        actual_ratio = np.mean(result == 255)

        assert actual_ratio == pytest.approx(expected_ratio)

    # -----------------------------------------------------
    # Caller input is not mutated
    # -----------------------------------------------------

    def test_does_not_mutate_input_mask(self):
        mask = np.ones(
            (640, 640),
            dtype=np.uint8,
        )

        original = mask.copy()

        result = scale_mask(
            mask=mask,
            shape=[320, 320],
        )

        assert np.array_equal(mask, original)

        assert result is not mask

    # -----------------------------------------------------
    # Contract enforcement: binary mask
    # -----------------------------------------------------

    @pytest.mark.parametrize(
        "mask",
        [
            np.array(
                [
                    [0, 1, 2],
                    [1, 0, 1],
                ],
                dtype=np.uint8,
            ),
            np.array(
                [
                    [0.0, 1.0],
                    [0.5, 1.0],
                ],
                dtype=np.float32,
            ),
        ],
        ids=[
            "uint8_non_binary",
            "float_non_binary",
        ],
    )
    def test_rejects_non_binary_masks(
        self,
        mask,
    ):
        with pytest.raises(
            ValueError,
            match="binary mask",
        ):
            scale_mask(
                mask=mask,
                shape=[2, 2],
            )

    # -----------------------------------------------------
    # Contract enforcement: mask rank
    # -----------------------------------------------------

    @pytest.mark.parametrize(
        "mask",
        [
            np.array(
                [1, 1, 1],
                dtype=np.uint8,
            ),
            np.ones(
                (10, 10, 3),
                dtype=np.uint8,
            ),
        ],
        ids=[
            "rank1",
            "rank3",
        ],
    )
    def test_rejects_invalid_mask_dimensions(
        self,
        mask,
    ):
        with pytest.raises(AssertionError):
            scale_mask(
                mask=mask,
                shape=[100, 100],
            )

    # -----------------------------------------------------
    # Contract enforcement: shape length
    # -----------------------------------------------------

    @pytest.mark.parametrize(
        "shape",
        [
            [],
            [100],
            [100, 200, 300],
        ],
        ids=[
            "empty",
            "one_dimension",
            "three_dimensions",
        ],
    )
    def test_rejects_invalid_shape_length(
        self,
        shape,
    ):
        mask = np.ones(
            (100, 100),
            dtype=np.uint8,
        )

        with pytest.raises(AssertionError):
            scale_mask(
                mask=mask,
                shape=shape,
            )

    # -----------------------------------------------------
    # Characterization: zero target dimensions
    # -----------------------------------------------------

    @pytest.mark.parametrize(
        "shape",
        [
            [0, 100],
            [100, 0],
            [0, 0],
        ],
        ids=[
            "zero_height",
            "zero_width",
            "zero_both",
        ],
    )
    def test_zero_target_dimension_current_behavior(
        self,
        shape,
    ):
        """
        Documents current behavior.

        Zero target dimensions are not explicitly validated.
        Division by zero currently occurs while computing
        the resize ratio.
        """

        mask = np.ones(
            (100, 100),
            dtype=np.uint8,
        )

        with pytest.raises(ZeroDivisionError):
            scale_mask(
                mask=mask,
                shape=shape,
            )

    # -----------------------------------------------------
    # Characterization: negative target dimensions
    # -----------------------------------------------------

    @pytest.mark.parametrize(
        "shape",
        [
            [-100, 100],
            [100, -100],
            [-100, -100],
        ],
        ids=[
            "negative_height",
            "negative_width",
            "negative_both",
        ],
    )
    def test_negative_target_dimension_current_behavior(
        self,
        shape,
    ):
        """
        Documents current behavior.

        Negative target dimensions are not validated.
        The exception is currently propagated from OpenCV.
        """

        mask = np.ones(
            (100, 100),
            dtype=np.uint8,
        )

        with pytest.raises(Exception):
            scale_mask(
                mask=mask,
                shape=shape,
            )

    # -----------------------------------------------------
    # Characterization: empty mask
    # -----------------------------------------------------

    def test_empty_mask_current_behavior(self):
        """
        Documents current behavior.

        Empty masks are not explicitly validated.
        The exception is currently propagated from OpenCV.
        """

        mask = np.empty(
            (0, 0),
            dtype=np.uint8,
        )

        with pytest.raises(Exception):
            scale_mask(
                mask=mask,
                shape=[100, 100],
            )


# =====================================================
# TEST CROP_MASK
# =====================================================


class TestCropMask:
    """
    Contract tests for crop_mask.

    Contract:
        - Accepts masks of shape (N, H, W).
        - Accepts one bounding box per mask of shape (N, 4).
        - Preserves the input mask shape.
        - Pixels outside each bounding box are set to zero.
        - Boxes may lie partially or completely outside the image.
        - Empty batches are supported.
        - Input arrays are not mutated.

    Contract enforcement:
        - Reject invalid mask dimensions.
        - Reject invalid boxes dimensions.
        - Reject batch-size mismatch.
        - Reject inverted bounding boxes.
        - Reject non-finite box coordinates.
    """

    # -----------------------------------------------------
    # Basic correctness
    # -----------------------------------------------------

    def test_crops_single_mask(self):
        masks = np.ones(
            (1, 5, 5),
            dtype=np.uint8,
        )

        boxes = np.array(
            [[1, 1, 4, 3]],
            dtype=np.float32,
        )

        result = crop_mask(
            masks=masks,
            boxes=boxes,
        )

        expected = np.array(
            [
                [
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ],
            dtype=np.uint8,
        )

        np.testing.assert_array_equal(
            result,
            expected,
        )

    # -----------------------------------------------------
    # One box per mask
    # -----------------------------------------------------

    def test_applies_each_box_to_corresponding_mask(self):
        masks = np.ones(
            (2, 5, 5),
            dtype=np.uint8,
        )

        boxes = np.array(
            [
                [0, 0, 2, 2],
                [3, 3, 5, 5],
            ],
            dtype=np.float32,
        )

        result = crop_mask(
            masks=masks,
            boxes=boxes,
        )

        expected = np.array(
            [
                [
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1],
                ],
            ],
            dtype=np.uint8,
        )

        np.testing.assert_array_equal(
            result,
            expected,
        )

    # -----------------------------------------------------
    # Shape preservation
    # -----------------------------------------------------

    @pytest.mark.parametrize(
        "shape",
        [
            (1, 10, 10),
            (4, 20, 30),
            (8, 160, 160),
        ],
    )
    def test_preserves_shape(
        self,
        shape,
    ):
        n, h, w = shape

        masks = np.ones(
            shape,
            dtype=np.float32,
        )

        boxes = np.tile(
            np.array(
                [0, 0, w, h],
                dtype=np.float32,
            ),
            (n, 1),
        )

        result = crop_mask(
            masks=masks,
            boxes=boxes,
        )

        assert result.shape == shape

    # -----------------------------------------------------
    # Empty batch
    # -----------------------------------------------------

    def test_supports_empty_batch(self):
        masks = np.empty(
            (0, 5, 5),
            dtype=np.float32,
        )

        boxes = np.empty(
            (0, 4),
            dtype=np.float32,
        )

        result = crop_mask(
            masks=masks,
            boxes=boxes,
        )

        assert result.shape == (0, 5, 5)

    # -----------------------------------------------------
    # Full-image crop
    # -----------------------------------------------------

    def test_full_image_box_preserves_mask(self):
        masks = np.random.rand(
            2,
            10,
            10,
        ).astype(np.float32)

        boxes = np.array(
            [
                [0, 0, 10, 10],
                [0, 0, 10, 10],
            ],
            dtype=np.float32,
        )

        result = crop_mask(
            masks=masks,
            boxes=boxes,
        )

        np.testing.assert_array_equal(
            result,
            masks,
        )

    # -----------------------------------------------------
    # Zero-area boxes
    # -----------------------------------------------------

    @pytest.mark.parametrize(
        "box",
        [
            [2, 1, 2, 4],
            [1, 3, 4, 3],
        ],
        ids=[
            "zero_width",
            "zero_height",
        ],
    )
    def test_zero_area_box_returns_empty_mask(
        self,
        box,
    ):
        masks = np.ones(
            (1, 5, 5),
            dtype=np.uint8,
        )

        result = crop_mask(
            masks=masks,
            boxes=np.array([box], dtype=np.float32),
        )

        assert np.count_nonzero(result) == 0

    # -----------------------------------------------------
    # Boxes outside image
    # -----------------------------------------------------

    def test_box_outside_image_returns_empty_mask(self):
        masks = np.ones(
            (1, 5, 5),
            dtype=np.uint8,
        )

        boxes = np.array(
            [[10, 10, 20, 20]],
            dtype=np.float32,
        )

        result = crop_mask(
            masks=masks,
            boxes=boxes,
        )

        assert np.all(result == 0)

    def test_box_partially_outside_image(self):
        masks = np.ones(
            (1, 5, 5),
            dtype=np.uint8,
        )

        boxes = np.array(
            [[-2, -2, 3, 3]],
            dtype=np.float32,
        )

        result = crop_mask(
            masks=masks,
            boxes=boxes,
        )

        assert np.count_nonzero(result) == 9

        expected = np.array(
            [
                [
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ],
            dtype=np.uint8,
        )

        np.testing.assert_array_equal(result, expected)

    # -----------------------------------------------------
    # Float coordinates
    # -----------------------------------------------------

    def test_float_box_coordinates_current_behavior(self):
        """
        Characterization test.

        Floating-point box coordinates are accepted and
        interpreted using NumPy comparison semantics.
        """

        masks = np.ones((1, 5, 5), dtype=np.uint8)

        boxes = np.array(
            [[1.5, 1.5, 4.5, 4.5]],
            dtype=np.float32,
        )

        result = crop_mask(
            masks=masks,
            boxes=boxes,
        )

        expected = np.array(
            [
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                ]
            ],
            dtype=np.uint8,
        )

        np.testing.assert_array_equal(result, expected)

    # -----------------------------------------------------
    # Caller inputs are not mutated
    # -----------------------------------------------------

    def test_does_not_mutate_inputs(self):
        masks = np.ones(
            (2, 5, 5),
            dtype=np.float32,
        )

        boxes = np.array(
            [
                [1, 1, 4, 4],
                [0, 0, 2, 2],
            ],
            dtype=np.float32,
        )

        masks_original = masks.copy()
        boxes_original = boxes.copy()

        _ = crop_mask(
            masks=masks,
            boxes=boxes,
        )

        np.testing.assert_array_equal(
            masks,
            masks_original,
        )

        np.testing.assert_array_equal(
            boxes,
            boxes_original,
        )

    # -----------------------------------------------------
    # Contract enforcement: invalid mask dimensions
    # -----------------------------------------------------

    @pytest.mark.parametrize(
        "masks",
        [
            np.ones((5, 5), dtype=np.float32),
            np.ones((1, 2, 3, 4), dtype=np.float32),
        ],
        ids=[
            "rank2",
            "rank4",
        ],
    )
    def test_rejects_invalid_mask_rank(
        self,
        masks,
    ):
        boxes = np.ones(
            (1, 4),
            dtype=np.float32,
        )

        with pytest.raises(ValueError):
            crop_mask(
                masks=masks,
                boxes=boxes,
            )

    # -----------------------------------------------------
    # Contract enforcement: invalid box shape
    # -----------------------------------------------------

    @pytest.mark.parametrize(
        "boxes",
        [
            np.ones((1, 3), dtype=np.float32),
            np.ones((1, 5), dtype=np.float32),
            np.ones((4,), dtype=np.float32),
            np.ones((1, 4, 1), dtype=np.float32),
        ],
        ids=[
            "three_columns",
            "five_columns",
            "rank1",
            "rank3",
        ],
    )
    def test_rejects_invalid_box_shape(
        self,
        boxes,
    ):
        masks = np.ones(
            (1, 5, 5),
            dtype=np.float32,
        )

        with pytest.raises(ValueError):
            crop_mask(
                masks=masks,
                boxes=boxes,
            )

    # -----------------------------------------------------
    # Contract enforcement: batch mismatch
    # -----------------------------------------------------

    def test_rejects_batch_size_mismatch(self):
        masks = np.ones(
            (2, 5, 5),
            dtype=np.float32,
        )

        boxes = np.ones(
            (1, 4),
            dtype=np.float32,
        )

        with pytest.raises(ValueError):
            crop_mask(
                masks=masks,
                boxes=boxes,
            )

    # -----------------------------------------------------
    # Contract enforcement: inverted boxes
    # -----------------------------------------------------

    @pytest.mark.parametrize(
        "box",
        [
            [4, 1, 2, 3],
            [1, 4, 3, 2],
        ],
        ids=[
            "x2_less_than_x1",
            "y2_less_than_y1",
        ],
    )
    def test_rejects_inverted_boxes(
        self,
        box,
    ):
        masks = np.ones(
            (1, 5, 5),
            dtype=np.float32,
        )

        with pytest.raises(ValueError):
            crop_mask(
                masks=masks,
                boxes=np.array([box], dtype=np.float32),
            )

    # # -----------------------------------------------------
    # # Contract enforcement: finite coordinates
    # # -----------------------------------------------------

    # @pytest.mark.parametrize(
    #     "box",
    #     [
    #         [np.nan, 0, 4, 4],
    #         [0, np.inf, 4, 4],
    #         [0, 0, np.nan, 4],
    #     ],
    #     ids=[
    #         "nan",
    #         "positive_inf",
    #         "nan_x2",
    #     ],
    # )
    # def test_rejects_non_finite_boxes(
    #     self,
    #     box,
    # ):
    #     masks = np.ones(
    #         (1, 5, 5),
    #         dtype=np.float32,
    #     )

    #     with pytest.raises(ValueError):
    #         crop_mask(
    #             masks=masks,
    #             boxes=np.array([box], dtype=np.float32),
    #         )


# =====================================================
# TEST XYXY -> XYWH
# =====================================================


class TestXYXY2XYWH:
    """
    Contract and characterization tests for xyxy2xywh.

    Guarantees:
        - Converts width and height correctly.
        - Preserves x and y coordinates.
        - Preserves dtype.
        - Preserves shape.
        - Returns a new array.
        - Handles empty input.
        - Explicitly enforces the supported input shape.

    Characterization tests document current behavior for
    unsupported inputs.
    """

    # -----------------------------------------------------
    # Basic conversion
    # -----------------------------------------------------

    def test_converts_single_box(self):
        boxes = np.array(
            [[10.0, 20.0, 30.0, 50.0]],
            dtype=np.float32,
        )

        result = xyxy2xywh(boxes)

        expected = np.array(
            [[10.0, 20.0, 20.0, 30.0]],
            dtype=np.float32,
        )

        np.testing.assert_allclose(result, expected)

    def test_converts_multiple_boxes(self):
        boxes = np.array(
            [
                [1, 2, 6, 8],
                [10, 20, 15, 30],
            ],
            dtype=np.float32,
        )

        result = xyxy2xywh(boxes)

        expected = np.array(
            [
                [1, 2, 5, 6],
                [10, 20, 5, 10],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(result, expected)

    # -----------------------------------------------------
    # Preserve x and y
    # -----------------------------------------------------

    def test_preserves_top_left_coordinates(self):
        boxes = np.array(
            [
                [5, 6, 20, 40],
            ],
            dtype=np.float32,
        )

        result = xyxy2xywh(boxes)

        np.testing.assert_array_equal(
            result[:, :2],
            boxes[:, :2],
        )

    # -----------------------------------------------------
    # Shape and dtype
    # -----------------------------------------------------

    @pytest.mark.parametrize(
        "shape",
        [
            (1, 4),
            (5, 4),
            (100, 4),
        ],
    )
    def test_preserves_shape(self, shape):
        boxes = np.ones(
            shape,
            dtype=np.float32,
        )

        result = xyxy2xywh(boxes)

        assert result.shape == shape

    @pytest.mark.parametrize(
        "dtype",
        [
            np.float32,
            np.float64,
            np.int32,
            np.int64,
        ],
    )
    def test_preserves_dtype(self, dtype):
        boxes = np.ones(
            (5, 4),
            dtype=dtype,
        )

        result = xyxy2xywh(boxes)

        assert result.dtype == dtype

    # -----------------------------------------------------
    # Copy semantics
    # -----------------------------------------------------

    def test_returns_new_array(self):
        boxes = np.array(
            [[1, 2, 3, 4]],
            dtype=np.float32,
        )

        result = xyxy2xywh(boxes)

        assert result is not boxes

    def test_does_not_modify_input(self):
        boxes = np.array(
            [[10, 20, 30, 40]],
            dtype=np.float32,
        )

        original = boxes.copy()

        xyxy2xywh(boxes)

        np.testing.assert_array_equal(
            boxes,
            original,
        )

    # -----------------------------------------------------
    # Empty input
    # -----------------------------------------------------

    def test_handles_empty_input(self):
        boxes = np.empty(
            (0, 4),
            dtype=np.float32,
        )

        result = xyxy2xywh(boxes)

        assert result.shape == (0, 4)

    # -----------------------------------------------------
    # Floating-point coordinates
    # -----------------------------------------------------

    def test_supports_float_coordinates(self):
        boxes = np.array(
            [
                [1.5, 2.5, 8.5, 10.5],
            ],
            dtype=np.float32,
        )

        result = xyxy2xywh(boxes)

        expected = np.array(
            [
                [1.5, 2.5, 7.0, 8.0],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(result, expected)

    # -----------------------------------------------------
    # Contract enforcement: inverted boxes
    # -----------------------------------------------------

    @pytest.mark.parametrize(
        "box",
        [
            [10, 10, 5, 6],  # x2 < x1
            [10, 10, 15, 6],  # y2 < y1
            [10, 10, 5, 5],  # both inverted
        ],
        ids=[
            "negative_width",
            "negative_height",
            "negative_width_and_height",
        ],
    )
    def test_rejects_inverted_boxes(
        self,
        box,
    ):
        """
        Bounding boxes must satisfy:

            x2 >= x1
            y2 >= y1
        """

        boxes = np.array(
            [box],
            dtype=np.float32,
        )

        with pytest.raises(
            ValueError,
            match=r"x2 >= x1 and y2 >= y1",
        ):
            xyxy2xywh(boxes)

    # -----------------------------------------------------
    # Contract enforcement: input shape
    # -----------------------------------------------------

    @pytest.mark.parametrize(
        "boxes",
        [
            np.ones((4,), dtype=np.float32),
            np.ones((5, 3), dtype=np.float32),
            np.ones((5, 5), dtype=np.float32),
            np.ones((2, 3, 4), dtype=np.float32),
        ],
        ids=[
            "rank1",
            "three_columns",
            "five_columns",
            "rank3",
        ],
    )
    def test_rejects_invalid_input_shape(
        self,
        boxes,
    ):
        with pytest.raises(ValueError):
            xyxy2xywh(boxes)


# =====================================================
# TEST XYWH -> XYXY
# =====================================================


class TestXYWH2XYXY:
    """
    Contract tests for xywh2xyxy.

    Guarantees:
        - Converts (x, y, w, h) to (x1, y1, x2, y2).
        - Preserves input shape.
        - Preserves input dtype.
        - Returns a copy.
        - Does not modify the input array.
        - Supports empty inputs.
        - Enforces the documented input contract.
    """

    # -----------------------------------------------------
    # Basic conversion
    # -----------------------------------------------------

    def test_converts_xywh_to_xyxy(self):
        boxes = np.array(
            [
                [20.0, 30.0, 10.0, 8.0],
            ],
            dtype=np.float32,
        )

        result = xywh2xyxy(boxes)

        expected = np.array(
            [
                [15.0, 26.0, 25.0, 34.0],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(
            result,
            expected,
            atol=1e-6,
        )

    # -----------------------------------------------------
    # Shape preservation
    # -----------------------------------------------------

    @pytest.mark.parametrize(
        "shape",
        [
            (1, 4),
            (5, 4),
            (100, 4),
            (0, 4),
        ],
    )
    def test_preserves_shape(
        self,
        shape,
    ):
        boxes = np.ones(
            shape,
            dtype=np.float32,
        )

        result = xywh2xyxy(boxes)

        assert result.shape == shape

    # -----------------------------------------------------
    # Dtype preservation
    # -----------------------------------------------------

    @pytest.mark.parametrize(
        "dtype",
        [
            np.float32,
            np.float64,
        ],
    )
    def test_preserves_dtype(
        self,
        dtype,
    ):
        boxes = np.ones(
            (5, 4),
            dtype=dtype,
        )

        result = xywh2xyxy(boxes)

        assert result.dtype == dtype

    # -----------------------------------------------------
    # Copy semantics
    # -----------------------------------------------------

    def test_returns_copy(self):
        boxes = np.ones(
            (5, 4),
            dtype=np.float32,
        )

        result = xywh2xyxy(boxes)

        assert result is not boxes

    def test_does_not_modify_input(self):
        boxes = np.array(
            [
                [10.0, 20.0, 6.0, 4.0],
            ],
            dtype=np.float32,
        )

        original = boxes.copy()

        xywh2xyxy(boxes)

        np.testing.assert_array_equal(
            boxes,
            original,
        )

    # -----------------------------------------------------
    # Empty input
    # -----------------------------------------------------

    def test_handles_empty_input(self):
        boxes = np.empty(
            (0, 4),
            dtype=np.float32,
        )

        result = xywh2xyxy(boxes)

        assert result.shape == (0, 4)

    # -----------------------------------------------------
    # Contract enforcement: input shape
    # -----------------------------------------------------

    @pytest.mark.parametrize(
        "boxes",
        [
            np.ones((4,), dtype=np.float32),
            np.ones((5, 3), dtype=np.float32),
            np.ones((5, 5), dtype=np.float32),
            np.ones((2, 3, 4), dtype=np.float32),
        ],
        ids=[
            "rank1",
            "three_columns",
            "five_columns",
            "rank3",
        ],
    )
    def test_rejects_invalid_input_shape(
        self,
        boxes,
    ):
        with pytest.raises(
            ValueError,
            match=r"Expected boxes with shape \(N, 4\)\.",
        ):
            xywh2xyxy(boxes)

    # # -----------------------------------------------------
    # # Contract enforcement: finite coordinates
    # # -----------------------------------------------------

    # @pytest.mark.parametrize(
    #     "box",
    #     [
    #         [np.nan, 10, 5, 5],
    #         [10, np.inf, 5, 5],
    #         [10, 10, np.nan, 5],
    #         [10, 10, 5, np.inf],
    #     ],
    #     ids=[
    #         "nan_x",
    #         "inf_y",
    #         "nan_width",
    #         "inf_height",
    #     ],
    # )
    # def test_rejects_non_finite_boxes(
    #     self,
    #     box,
    # ):
    #     boxes = np.array(
    #         [box],
    #         dtype=np.float32,
    #     )

    #     with pytest.raises(
    #         ValueError,
    #         match="finite",
    #     ):
    #         xywh2xyxy(boxes)

    # -----------------------------------------------------
    # Contract enforcement: negative width/height
    # -----------------------------------------------------

    @pytest.mark.parametrize(
        "box",
        [
            [10, 10, -5, 4],
            [10, 10, 5, -4],
            [10, 10, -5, -4],
        ],
        ids=[
            "negative_width",
            "negative_height",
            "negative_width_and_height",
        ],
    )
    def test_rejects_negative_dimensions(
        self,
        box,
    ):
        """
        Width and height must be non-negative.
        """

        boxes = np.array(
            [box],
            dtype=np.float32,
        )

        with pytest.raises(
            ValueError,
            match=r"width >= 0 and height >= 0",
        ):
            xywh2xyxy(boxes)

    # -----------------------------------------------------
    # Degenerate boxes
    # -----------------------------------------------------

    @pytest.mark.parametrize(
        ("box", "expected"),
        [
            (
                [10, 10, 0, 5],
                [10, 7.5, 10, 12.5],
            ),
            (
                [10, 10, 5, 0],
                [7.5, 10, 12.5, 10],
            ),
            (
                [10, 10, 0, 0],
                [10, 10, 10, 10],
            ),
        ],
        ids=[
            "zero_width",
            "zero_height",
            "zero_both",
        ],
    )
    def test_allows_zero_dimensions(
        self,
        box,
        expected,
    ):
        """
        Zero-width and/or zero-height boxes are valid and
        produce coincident corner coordinates.
        """

        boxes = np.array(
            [box],
            dtype=np.float32,
        )

        result = xywh2xyxy(boxes)

        np.testing.assert_allclose(
            result,
            np.array([expected], dtype=np.float32),
            atol=1e-6,
        )


# =====================================================
# PROCESS_MASK
# =====================================================


class TestProcessMask:
    """
    Contract tests for process_mask.

    Guarantees:
        - Projects mask coefficients onto prototypes.
        - Resizes masks to the requested image shape.
        - Applies crop_mask to each mask.
        - Returns boolean masks.
        - Preserves the number of detections.
        - Enforces the documented input contract.
    """

    # -----------------------------------------------------
    # Basic behavior
    # -----------------------------------------------------

    @patch("core.python.postprocess.detection_utils.crop_mask")
    def test_returns_boolean_masks(
        self,
        mock_crop_mask,
    ):
        mock_crop_mask.return_value = np.array(
            [[[1.0, -1.0], [0.0, 2.0]]],
            dtype=np.float32,
        )

        protos = np.ones((1, 2, 2), dtype=np.float32)
        masks_in = np.ones((1, 1), dtype=np.float32)
        bboxes = np.array([[0, 0, 2, 2]], dtype=np.float32)

        result = process_mask(
            protos,
            masks_in,
            bboxes,
            shape=(2, 2),
        )

        assert result.dtype == np.bool_

    @patch("core.python.postprocess.detection_utils.crop_mask")
    @pytest.mark.parametrize(
        "shape",
        [
            (1, 1),
            (8, 8),
            (32, 64),
            (64, 32),
            (320, 640),
            (640, 320),
            (640, 640),
        ],
        ids=[
            "minimum",
            "small_square",
            "small_landscape",
            "small_portrait",
            "landscape",
            "portrait",
            "square",
        ],
    )
    def test_output_shape_matches_requested_shape(
        self,
        mock_crop_mask,
        shape,
    ):
        num_masks = 2

        mock_crop_mask.return_value = np.ones(
            (num_masks, *shape),
            dtype=np.float32,
        )

        protos = np.ones((4, 8, 8), dtype=np.float32)
        masks_in = np.ones((num_masks, 4), dtype=np.float32)
        bboxes = np.ones((num_masks, 4), dtype=np.float32)

        result = process_mask(
            protos,
            masks_in,
            bboxes,
            shape=shape,
        )

        assert result.shape == (num_masks, *shape)

    # -----------------------------------------------------
    # Thresholding
    # -----------------------------------------------------

    @patch("core.python.postprocess.detection_utils.crop_mask")
    @pytest.mark.parametrize(
        ("crop_output", "expected"),
        [
            (
                np.array(
                    [[[-1.0, 0.0], [0.1, 2.0]]],
                    dtype=np.float32,
                ),
                np.array(
                    [[[False, False], [True, True]]],
                    dtype=bool,
                ),
            ),
            (
                np.array(
                    [[[-100.0, -1e-6], [1e-6, 100.0]]],
                    dtype=np.float32,
                ),
                np.array(
                    [[[False, False], [True, True]]],
                    dtype=bool,
                ),
            ),
            (
                np.zeros(
                    (1, 2, 2),
                    dtype=np.float32,
                ),
                np.zeros(
                    (1, 2, 2),
                    dtype=bool,
                ),
            ),
            (
                np.ones(
                    (1, 2, 2),
                    dtype=np.float32,
                ),
                np.ones(
                    (1, 2, 2),
                    dtype=bool,
                ),
            ),
        ],
        ids=[
            "negative_zero_positive",
            "epsilon_boundary",
            "all_zero",
            "all_positive",
        ],
    )
    def test_thresholds_crop_mask_output(
        self,
        mock_crop_mask,
        crop_output,
        expected,
    ):
        mock_crop_mask.return_value = crop_output

        protos = np.ones((1, 2, 2), dtype=np.float32)
        masks_in = np.ones((1, 1), dtype=np.float32)
        bboxes = np.ones((1, 4), dtype=np.float32)

        result = process_mask(
            protos,
            masks_in,
            bboxes,
            shape=(2, 2),
        )

        np.testing.assert_array_equal(
            result,
            expected,
        )

    # -----------------------------------------------------
    # crop_mask receives resized masks
    # -----------------------------------------------------

    @patch("core.python.postprocess.detection_utils.crop_mask")
    @pytest.mark.parametrize(
        "shape",
        [
            (1, 1),
            (32, 64),
            (64, 32),
            (320, 640),
        ],
        ids=[
            "minimum",
            "landscape_small",
            "portrait_small",
            "landscape",
        ],
    )
    def test_passes_resized_masks_to_crop_mask(
        self,
        mock_crop_mask,
        shape,
    ):
        num_masks = 3
        mask_dim = 8

        mock_crop_mask.return_value = np.ones(
            (num_masks, *shape),
            dtype=np.float32,
        )

        protos = np.ones(
            (mask_dim, 4, 4),
            dtype=np.float32,
        )

        masks_in = np.ones(
            (num_masks, mask_dim),
            dtype=np.float32,
        )

        bboxes = np.arange(
            num_masks * 4,
            dtype=np.float32,
        ).reshape(num_masks, 4)

        process_mask(
            protos=protos,
            masks_in=masks_in,
            bboxes=bboxes,
            shape=shape,
        )

        mock_crop_mask.assert_called_once()

        resized_masks = mock_crop_mask.call_args.kwargs["masks"]
        passed_boxes = mock_crop_mask.call_args.kwargs["boxes"]

        # One resized mask per detection.
        assert resized_masks.shape == (num_masks, *shape)

        # Matrix multiplication and resize produce floating-point masks.
        assert np.issubdtype(
            resized_masks.dtype,
            np.floating,
        )

        # Bounding boxes are forwarded unchanged.
        np.testing.assert_array_equal(
            passed_boxes,
            bboxes,
        )

    # -----------------------------------------------------
    # Empty detections
    # -----------------------------------------------------

    @patch("core.python.postprocess.detection_utils.crop_mask")
    def test_handles_empty_detections(
        self,
        mock_crop_mask,
    ):
        mock_crop_mask.return_value = np.empty(
            (0, 64, 64),
            dtype=np.float32,
        )

        protos = np.ones((4, 8, 8), dtype=np.float32)
        masks_in = np.empty((0, 4), dtype=np.float32)
        bboxes = np.empty((0, 4), dtype=np.float32)

        result = process_mask(
            protos,
            masks_in,
            bboxes,
            shape=(64, 64),
        )

        assert result.shape == (0, 64, 64)

    # -----------------------------------------------------
    # Contract enforcement: protos shape
    # -----------------------------------------------------

    @pytest.mark.parametrize(
        "protos",
        [
            np.ones((8, 8), dtype=np.float32),
            np.ones((1, 2, 3, 4), dtype=np.float32),
        ],
        ids=[
            "rank2",
            "rank4",
        ],
    )
    def test_rejects_invalid_protos_rank(
        self,
        protos,
    ):
        with pytest.raises(ValueError):
            process_mask(
                protos,
                np.ones((1, 8), dtype=np.float32),
                np.ones((1, 4), dtype=np.float32),
                shape=(64, 64),
            )

    # -----------------------------------------------------
    # Contract enforcement: masks_in shape
    # -----------------------------------------------------

    @pytest.mark.parametrize(
        "masks_in",
        [
            np.ones((8,), dtype=np.float32),
            np.ones((1, 8, 1), dtype=np.float32),
        ],
        ids=[
            "rank1",
            "rank3",
        ],
    )
    def test_rejects_invalid_masks_in_rank(
        self,
        masks_in,
    ):
        with pytest.raises(ValueError):
            process_mask(
                np.ones((8, 4, 4), dtype=np.float32),
                masks_in,
                np.ones((1, 4), dtype=np.float32),
                shape=(64, 64),
            )

    # -----------------------------------------------------
    # Contract enforcement: mask dimension mismatch
    # -----------------------------------------------------

    def test_rejects_mask_dimension_mismatch(self):
        with pytest.raises(ValueError):
            process_mask(
                np.ones((32, 8, 8), dtype=np.float32),
                np.ones((5, 31), dtype=np.float32),
                np.ones((5, 4), dtype=np.float32),
                shape=(64, 64),
            )

    # -----------------------------------------------------
    # Contract enforcement: one box per detection
    # -----------------------------------------------------

    def test_rejects_detection_count_mismatch(self):
        with pytest.raises(ValueError):
            process_mask(
                np.ones((32, 8, 8), dtype=np.float32),
                np.ones((5, 32), dtype=np.float32),
                np.ones((4, 4), dtype=np.float32),
                shape=(64, 64),
            )

    # -----------------------------------------------------
    # Contract enforcement: shape length
    # -----------------------------------------------------

    @pytest.mark.parametrize(
        "shape",
        [
            (),
            (64,),
            (64, 64, 3),
        ],
    )
    def test_rejects_invalid_shape_length(
        self,
        shape,
    ):
        with pytest.raises(ValueError):
            process_mask(
                np.ones((8, 4, 4), dtype=np.float32),
                np.ones((1, 8), dtype=np.float32),
                np.ones((1, 4), dtype=np.float32),
                shape=shape,
            )

    # -----------------------------------------------------
    # Contract enforcement: non-positive shape
    # -----------------------------------------------------

    @pytest.mark.parametrize(
        "shape",
        [
            (0, 64),
            (64, 0),
            (-1, 64),
            (64, -1),
        ],
    )
    def test_rejects_non_positive_shape(
        self,
        shape,
    ):
        with pytest.raises(ValueError):
            process_mask(
                np.ones((8, 4, 4), dtype=np.float32),
                np.ones((1, 8), dtype=np.float32),
                np.ones((1, 4), dtype=np.float32),
                shape=shape,
            )

    @pytest.mark.parametrize(
        "protos",
        [
            np.empty((0, 8, 8), dtype=np.float32),
            np.empty((8, 0, 8), dtype=np.float32),
            np.empty((8, 8, 0), dtype=np.float32),
            np.empty((0, 0, 0), dtype=np.float32),
        ],
        ids=[
            "zero_channels",
            "zero_height",
            "zero_width",
            "all_zero",
        ],
    )
    def test_rejects_empty_prototype_tensor(
        self,
        protos,
    ):
        with pytest.raises(
            ValueError,
            match="Expected non-empty prototype tensor.",
        ):
            process_mask(
                protos,
                np.ones((1, max(protos.shape[0], 1)), dtype=np.float32),
                np.ones((1, 4), dtype=np.float32),
                shape=(64, 64),
            )


# =====================================================
# NUMPY_NMS
# =====================================================


class TestNumpyNMS:
    """
    Contract and behavioral tests for numpy_nms.

    Supported contract:
        - dets must have shape (N, 4).
        - scores must have shape (N,).
        - One score must exist per detection.
        - Bounding boxes must be finite.
        - Scores must be finite.
        - Bounding boxes must satisfy
          x2 >= x1 and y2 >= y1.

    The returned indices correspond to the detections kept
    after Non-Maximum Suppression and are ordered by
    descending confidence score.
    """

    # -----------------------------------------------------
    # Basic behavior
    # -----------------------------------------------------

    def test_empty_input_returns_empty_list(self):
        result = numpy_nms(
            dets=np.empty((0, 4), dtype=np.float32),
            scores=np.empty((0,), dtype=np.float32),
            thresh=0.5,
        )

        assert result == []

    def test_single_detection_is_kept(self):
        dets = np.array(
            [[10, 10, 20, 20]],
            dtype=np.float32,
        )

        scores = np.array(
            [0.9],
            dtype=np.float32,
        )

        assert numpy_nms(
            dets=dets,
            scores=scores,
            thresh=0.5,
        ) == [0]

    def test_non_overlapping_boxes_keep_all(self):
        dets = np.array(
            [
                [0, 0, 10, 10],
                [20, 20, 30, 30],
                [40, 40, 50, 50],
            ],
            dtype=np.float32,
        )

        scores = np.array(
            [0.7, 0.9, 0.8],
            dtype=np.float32,
        )

        result = numpy_nms(
            dets=dets,
            scores=scores,
            thresh=0.5,
        )

        assert result == [1, 2, 0]

    def test_identical_boxes_keep_highest_score(self):
        dets = np.array(
            [
                [10, 10, 20, 20],
                [10, 10, 20, 20],
            ],
            dtype=np.float32,
        )

        scores = np.array(
            [0.7, 0.9],
            dtype=np.float32,
        )

        assert numpy_nms(
            dets=dets,
            scores=scores,
            thresh=0.5,
        ) == [1]

    def test_overlapping_boxes_below_threshold_are_kept(self):
        dets = np.array(
            [
                [0, 0, 10, 10],
                [5, 5, 15, 15],
            ],
            dtype=np.float32,
        )

        scores = np.array(
            [0.9, 0.8],
            dtype=np.float32,
        )

        result = numpy_nms(
            dets=dets,
            scores=scores,
            thresh=0.8,
        )

        assert result == [0, 1]

    def test_overlapping_boxes_above_threshold_are_suppressed(self):
        dets = np.array(
            [
                [0, 0, 10, 10],
                [1, 1, 11, 11],
            ],
            dtype=np.float32,
        )

        scores = np.array(
            [0.9, 0.8],
            dtype=np.float32,
        )

        result = numpy_nms(
            dets=dets,
            scores=scores,
            thresh=0.5,
        )

        assert result == [0]

    # -----------------------------------------------------
    # Ordering
    # -----------------------------------------------------

    def test_returns_indices_in_descending_score_order(self):
        dets = np.array(
            [
                [0, 0, 5, 5],
                [10, 10, 15, 15],
                [20, 20, 25, 25],
            ],
            dtype=np.float32,
        )

        scores = np.array(
            [0.2, 0.9, 0.5],
            dtype=np.float32,
        )

        result = numpy_nms(
            dets=dets,
            scores=scores,
            thresh=0.5,
        )

        assert result == [1, 2, 0]

    def test_equal_scores_are_deterministic(self):
        dets = np.array(
            [
                [0, 0, 10, 10],
                [20, 20, 30, 30],
                [40, 40, 50, 50],
            ],
            dtype=np.float32,
        )

        scores = np.array(
            [0.8, 0.8, 0.8],
            dtype=np.float32,
        )

        result1 = numpy_nms(
            dets=dets,
            scores=scores,
            thresh=0.5,
        )

        result2 = numpy_nms(
            dets=dets,
            scores=scores,
            thresh=0.5,
        )

        assert result1 == result2

    # -----------------------------------------------------
    # Threshold behavior
    # -----------------------------------------------------

    def test_threshold_zero(self):
        dets = np.array(
            [
                [0, 0, 10, 10],
                [9, 9, 20, 20],
            ],
            dtype=np.float32,
        )

        scores = np.array(
            [0.9, 0.8],
            dtype=np.float32,
        )

        result = numpy_nms(
            dets=dets,
            scores=scores,
            thresh=0.0,
        )

        assert result == [0]

    def test_threshold_one(self):
        dets = np.array(
            [
                [0, 0, 10, 10],
                [0, 0, 10, 10],
            ],
            dtype=np.float32,
        )

        scores = np.array(
            [0.9, 0.8],
            dtype=np.float32,
        )

        result = numpy_nms(
            dets=dets,
            scores=scores,
            thresh=1.0,
        )

        assert result == [0, 1]

    # -----------------------------------------------------
    # Degenerate boxes
    # -----------------------------------------------------

    @pytest.mark.parametrize(
        "box",
        [
            [10, 10, 10, 20],
            [10, 10, 20, 10],
            [10, 10, 10, 10],
        ],
        ids=[
            "zero_width",
            "zero_height",
            "point",
        ],
    )
    def test_allows_zero_area_boxes(
        self,
        box,
    ):
        dets = np.array(
            [box],
            dtype=np.float32,
        )

        scores = np.array(
            [1.0],
            dtype=np.float32,
        )

        assert numpy_nms(
            dets=dets,
            scores=scores,
            thresh=0.5,
        ) == [0]

    # -----------------------------------------------------
    # Contract enforcement: dets shape
    # -----------------------------------------------------

    @pytest.mark.parametrize(
        "dets",
        [
            np.ones((4,), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
            np.ones((1, 5), dtype=np.float32),
            np.ones((1, 2, 4), dtype=np.float32),
        ],
        ids=[
            "rank1",
            "three_columns",
            "five_columns",
            "rank3",
        ],
    )
    def test_rejects_invalid_dets_shape(
        self,
        dets,
    ):
        scores = np.ones(
            (1,),
            dtype=np.float32,
        )

        with pytest.raises(ValueError):
            numpy_nms(
                dets=dets,
                scores=scores,
                thresh=0.5,
            )

    # -----------------------------------------------------
    # Contract enforcement: scores shape
    # -----------------------------------------------------

    @pytest.mark.parametrize(
        "scores",
        [
            np.ones((1, 1), dtype=np.float32),
            np.ones((1, 1, 1), dtype=np.float32),
        ],
        ids=[
            "rank2",
            "rank3",
        ],
    )
    def test_rejects_invalid_scores_shape(
        self,
        scores,
    ):
        dets = np.ones(
            (1, 4),
            dtype=np.float32,
        )

        with pytest.raises(ValueError):
            numpy_nms(
                dets=dets,
                scores=scores,
                thresh=0.5,
            )

    # -----------------------------------------------------
    # Contract enforcement: detection count
    # -----------------------------------------------------

    def test_rejects_detection_score_count_mismatch(self):
        dets = np.ones(
            (2, 4),
            dtype=np.float32,
        )

        scores = np.ones(
            (3,),
            dtype=np.float32,
        )

        with pytest.raises(ValueError):
            numpy_nms(
                dets=dets,
                scores=scores,
                thresh=0.5,
            )

    # -----------------------------------------------------
    # Contract enforcement: inverted boxes
    # -----------------------------------------------------

    @pytest.mark.parametrize(
        "box",
        [
            [5, 0, 4, 10],
            [0, 5, 10, 4],
        ],
        ids=[
            "negative_width",
            "negative_height",
        ],
    )
    def test_rejects_inverted_boxes(
        self,
        box,
    ):
        with pytest.raises(ValueError):
            numpy_nms(
                dets=np.array([box], dtype=np.float32),
                scores=np.array([1.0], dtype=np.float32),
                thresh=0.5,
            )


# =====================================================
# YOLOV10_NMS
# =====================================================


class TestNMSYOLOV10:
    # -----------------------------------------------------
    # Basic behavior
    # -----------------------------------------------------

    def test_returns_one_output_per_batch(self):
        pred = np.zeros(
            (3, 5, 1),
            dtype=np.float32,
        )

        result = nms_yolov10(pred)

        assert len(result) == 3

        for detections in result:
            assert detections.shape == (0, 6)

    def test_returns_empty_detections_when_no_scores_above_threshold(self):
        pred = np.zeros(
            (1, 5, 2),
            dtype=np.float32,
        )

        result = nms_yolov10(
            pred,
            nms_score_threshold=0.25,
        )

        assert len(result) == 1
        assert result[0].shape == (0, 6)

    def test_single_detection_is_preserved(self):
        pred = np.array(
            [
                [
                    [10],  # x1
                    [20],  # y1
                    [30],  # x2
                    [40],  # y2
                    [0.90],  # class 0
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov10(pred)

        expected = np.array(
            [
                [10, 20, 30, 40, 0.90, 0],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(
            result[0],
            expected,
        )

    def test_multiple_images_processed_independently(self):
        pred = np.array(
            [
                [
                    [100],
                    [100],
                    [120],
                    [120],
                    [0.80],
                ],
                [
                    [100],
                    [100],
                    [120],
                    [120],
                    [0.80],
                ],
            ],
            dtype=np.float32,
        )

        result = nms_yolov10(pred)

        assert len(result) == 2

        np.testing.assert_allclose(
            result[0],
            np.array(
                [[100, 100, 120, 120, 0.80, 0]],
                dtype=np.float32,
            ),
        )

        np.testing.assert_allclose(
            result[1],
            np.array(
                [[100, 100, 120, 120, 0.80, 0]],
                dtype=np.float32,
            ),
        )

    # -----------------------------------------------------
    # Confidence threshold
    # -----------------------------------------------------

    def test_filters_detections_below_threshold(self):
        pred = np.array(
            [
                [
                    [0, 20],
                    [0, 20],
                    [10, 30],
                    [10, 30],
                    [0.90, 0.20],
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov10(
            pred,
            nms_score_threshold=0.25,
        )

        expected = np.array(
            [
                [0, 0, 10, 10, 0.90, 0],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(
            result[0],
            expected,
        )

    def test_threshold_is_strictly_greater_than(self):
        threshold = 0.25

        pred = np.array(
            [
                [
                    [0],
                    [0],
                    [10],
                    [10],
                    [threshold],
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov10(
            pred,
            nms_score_threshold=threshold,
        )

        assert result[0].shape == (0, 6)

    def test_keeps_multiple_classes_above_threshold_for_same_anchor(self):
        pred = np.array(
            [
                [
                    [10],  # x1
                    [20],  # y1
                    [30],  # x2
                    [40],  # y2
                    [0.90],  # class 0
                    [0.75],  # class 1
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov10(pred)

        expected = np.array(
            [
                [10, 20, 30, 40, 0.90, 0],
                [10, 20, 30, 40, 0.75, 1],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(
            result[0],
            expected,
        )

    # -----------------------------------------------------
    # NMS behavior
    # -----------------------------------------------------

    def test_suppresses_overlapping_boxes_same_class(self):
        pred = np.array(
            [
                [
                    [0, 1],  # x1
                    [0, 1],  # y1
                    [10, 11],  # x2
                    [10, 11],  # y2
                    [0.95, 0.80],  # class 0
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov10(
            pred,
            iou_threshold=0.5,
        )

        expected = np.array(
            [
                [0, 0, 10, 10, 0.95, 0],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(
            result[0],
            expected,
        )

    def test_keeps_non_overlapping_boxes_same_class(self):
        pred = np.array(
            [
                [
                    [0, 100],
                    [0, 100],
                    [10, 110],
                    [10, 110],
                    [0.95, 0.80],
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov10(
            pred,
            iou_threshold=0.5,
        )

        expected = np.array(
            [
                [0, 0, 10, 10, 0.95, 0],
                [100, 100, 110, 110, 0.80, 0],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(
            result[0],
            expected,
        )

    # -----------------------------------------------------
    # Class-aware vs. agnostic NMS
    # -----------------------------------------------------

    @pytest.mark.parametrize(
        "is_agnostic_nms,expected",
        [
            (
                False,
                np.array(
                    [
                        [0, 0, 10, 10, 0.95, 0],
                        [0, 0, 10, 10, 0.85, 1],
                    ],
                    dtype=np.float32,
                ),
            ),
            (
                True,
                np.array(
                    [
                        [0, 0, 10, 10, 0.95, 0],
                    ],
                    dtype=np.float32,
                ),
            ),
        ],
        ids=[
            "class_aware",
            "agnostic",
        ],
    )
    def test_is_agnostic_nms_controls_cross_class_suppression(
        self,
        is_agnostic_nms,
        expected,
    ):
        pred = np.array(
            [
                [
                    # x1
                    [0, 0, 0],
                    # y1
                    [0, 0, 0],
                    # x2
                    [10, 10, 10],
                    # y2
                    [10, 10, 10],
                    # class 0 scores
                    [0.95, 0.90, 0.00],
                    # class 1 scores
                    [0.00, 0.00, 0.85],
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov10(
            pred,
            is_agnostic_nms=is_agnostic_nms,
        )

        np.testing.assert_allclose(
            result[0],
            expected,
        )

    # -----------------------------------------------------
    # Ordering
    # -----------------------------------------------------

    def test_outputs_sorted_by_confidence(self):
        pred = np.array(
            [
                [
                    [20, 40, 60],
                    [20, 40, 60],
                    [30, 50, 70],
                    [30, 50, 70],
                    [0.60, 0.95, 0.80],
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov10(pred)

        expected_scores = np.array(
            [0.95, 0.80, 0.60],
            dtype=np.float32,
        )

        np.testing.assert_allclose(
            result[0][:, 4],
            expected_scores,
        )

    # -----------------------------------------------------
    # max_det
    # -----------------------------------------------------

    def test_limits_output_to_max_det(self):
        pred = np.array(
            [
                [
                    [0, 20, 40],
                    [0, 20, 40],
                    [10, 30, 50],
                    [10, 30, 50],
                    [0.95, 0.90, 0.85],
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov10(
            pred,
            max_det=2,
        )

        assert result[0].shape == (2, 6)

        np.testing.assert_allclose(
            result[0][:, 4],
            np.array(
                [0.95, 0.90],
                dtype=np.float32,
            ),
        )

    def test_keeps_all_detections_when_max_det_exceeds_candidate_count(self):
        pred = np.array(
            [
                [
                    [0, 20],
                    [0, 20],
                    [10, 30],
                    [10, 30],
                    [0.90, 0.80],
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov10(
            pred,
            max_det=100,
        )

        assert result[0].shape == (2, 6)

    # -----------------------------------------------------
    # Edge cases and output contract
    # -----------------------------------------------------

    @patch("core.python.postprocess.detection_utils.numpy_nms")
    def test_handles_empty_keep_indices_from_numpy_nms(
        self,
        mock_numpy_nms,
    ):
        mock_numpy_nms.return_value = []

        pred = np.array(
            [
                [
                    [0],
                    [0],
                    [10],
                    [10],
                    [0.90],
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov10(pred)

        assert len(result) == 1
        assert result[0].shape == (0, 6)

    def test_empty_batch_returns_empty_list(self):
        pred = np.empty(
            (0, 5, 10),
            dtype=np.float32,
        )

        result = nms_yolov10(pred)

        assert result == []

    def test_image_with_no_candidates_returns_empty_detection_array(self):
        pred = np.array(
            [
                [
                    [0],
                    [0],
                    [10],
                    [10],
                    [0.10],  # Below default threshold (0.25)
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov10(pred)

        assert len(result) == 1
        assert result[0].shape == (0, 6)
        assert result[0].dtype == np.float32

    def test_output_dtype_is_float32(self):
        pred = np.array(
            [
                [
                    [0],
                    [0],
                    [10],
                    [10],
                    [0.90],
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov10(pred)

        assert result[0].dtype == np.float32

    def test_output_has_expected_detection_format(self):
        pred = np.array(
            [
                [
                    [0, 10],  # x1
                    [0, 10],  # y1
                    [5, 15],  # x2
                    [5, 15],  # y2
                    [0.90, 0.10],  # class 0
                    [0.20, 0.80],  # class 1
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov10(pred)

        detections = result[0]

        assert detections.ndim == 2
        assert detections.shape == (2, 6)

        np.testing.assert_allclose(
            detections[:, 5],
            np.array([0, 1], dtype=np.float32),
        )

    # -----------------------------------------------------
    # IoU-threshold boundary values
    # -----------------------------------------------------

    @pytest.mark.parametrize(
        "iou_threshold,expected",
        [
            (
                0.0,
                np.array(
                    [
                        [0, 0, 10, 10, 0.90, 0],
                    ],
                    dtype=np.float32,
                ),
            ),
            (
                1.0,
                np.array(
                    [
                        [0, 0, 10, 10, 0.90, 0],
                        [1, 1, 11, 11, 0.80, 0],
                    ],
                    dtype=np.float32,
                ),
            ),
        ],
        ids=[
            "minimum",
            "maximum",
        ],
    )
    def test_iou_threshold_boundary_values(
        self,
        iou_threshold,
        expected,
    ):
        pred = np.array(
            [
                [
                    [0, 1],  # x1
                    [0, 1],  # y1
                    [10, 11],  # x2
                    [10, 11],  # y2
                    [0.90, 0.80],  # class 0 scores
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov10(
            pred,
            iou_threshold=iou_threshold,
        )

        np.testing.assert_allclose(
            result[0],
            expected,
        )

    # -----------------------------------------------------
    # Score-threshold boundary values
    # -----------------------------------------------------

    @pytest.mark.parametrize(
        "score_threshold,expected",
        [
            (
                0.0,
                np.array(
                    [
                        [0, 0, 10, 10, 0.60, 0],
                        [20, 20, 30, 30, 0.40, 0],
                    ],
                    dtype=np.float32,
                ),
            ),
            (
                1.0,
                np.empty((0, 6), dtype=np.float32),
            ),
        ],
        ids=[
            "minimum",
            "maximum",
        ],
    )
    def test_score_threshold_boundary_values(
        self,
        score_threshold,
        expected,
    ):
        pred = np.array(
            [
                [
                    [0, 20],  # x1
                    [0, 20],  # y1
                    [10, 30],  # x2
                    [10, 30],  # y2
                    [0.60, 0.40],  # class 0 scores
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov10(
            pred,
            nms_score_threshold=score_threshold,
        )

        np.testing.assert_allclose(
            result[0],
            expected,
        )


# =====================================================
# YOLOV8_NMS
# =====================================================


class TestNMSYOLOV8:
    # -----------------------------------------------------
    # Basic behavior
    # -----------------------------------------------------

    def test_returns_one_output_per_batch(self):
        pred = np.zeros(
            (3, 5, 1),
            dtype=np.float32,
        )

        result = nms_yolov8(pred)

        assert len(result) == 3

        for detections in result:
            assert detections.shape == (0, 6)

    def test_returns_empty_detections_when_no_scores_above_threshold(self):
        pred = np.zeros(
            (1, 5, 2),
            dtype=np.float32,
        )

        result = nms_yolov8(
            pred,
            nms_score_threshold=0.25,
        )

        assert len(result) == 1
        assert result[0].shape == (0, 6)

    def test_single_detection_is_preserved(self):
        pred = np.array(
            [
                [
                    [20],
                    [30],
                    [20],
                    [20],
                    [0.90],
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov8(pred)

        expected = np.array(
            [
                [10, 20, 30, 40, 0.90, 0],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(
            result[0],
            expected,
        )

    def test_multiple_images_processed_independently(self):
        pred = np.array(
            [
                [
                    [20],
                    [30],
                    [20],
                    [20],
                    [0.90],
                ],
                [
                    [20],
                    [30],
                    [20],
                    [20],
                    [0.90],
                ],
            ],
            dtype=np.float32,
        )

        result = nms_yolov8(pred)

        assert len(result) == 2

        np.testing.assert_allclose(
            result[0],
            np.array(
                [
                    [10, 20, 30, 40, 0.90, 0],
                ],
                dtype=np.float32,
            ),
        )

        np.testing.assert_allclose(
            result[1],
            np.array(
                [
                    [10, 20, 30, 40, 0.90, 0],
                ],
                dtype=np.float32,
            ),
        )

    # -----------------------------------------------------
    # Confidence threshold
    # -----------------------------------------------------

    def test_filters_detections_below_threshold(self):
        pred = np.array(
            [
                [
                    [20, 40],
                    [20, 40],
                    [20, 20],
                    [20, 20],
                    [0.90, 0.20],
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov8(
            pred,
            nms_score_threshold=0.25,
        )

        expected = np.array(
            [
                [10, 10, 30, 30, 0.90, 0],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(
            result[0],
            expected,
        )

    def test_threshold_is_strictly_greater_than(self):
        threshold = 0.25

        pred = np.array(
            [
                [
                    [20],
                    [20],
                    [20],
                    [20],
                    [threshold],
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov8(
            pred,
            nms_score_threshold=threshold,
        )

        assert result[0].shape == (0, 6)

    def test_keeps_highest_scoring_class_per_anchor_by_default(self):
        pred = np.array(
            [
                [
                    [20],
                    [30],
                    [20],
                    [20],
                    [0.90],
                    [0.75],
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov8(pred)

        expected = np.array(
            [
                [10, 20, 30, 40, 0.90, 0],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(
            result[0],
            expected,
        )

    # -----------------------------------------------------
    # NMS behavior
    # -----------------------------------------------------

    def test_suppresses_overlapping_boxes_same_class(self):
        pred = np.array(
            [
                [
                    [5.0, 6.0],
                    [5.0, 6.0],
                    [10.0, 10.0],
                    [10.0, 10.0],
                    [0.95, 0.80],
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov8(
            pred,
            iou_threshold=0.5,
        )

        expected = np.array(
            [
                [0.0, 0.0, 10.0, 10.0, 0.95, 0.0],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(
            result[0],
            expected,
        )

    def test_keeps_non_overlapping_boxes_same_class(self):
        pred = np.array(
            [
                [
                    [5.0, 105.0],
                    [5.0, 105.0],
                    [10.0, 10.0],
                    [10.0, 10.0],
                    [0.95, 0.80],
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov8(
            pred,
            iou_threshold=0.5,
        )

        expected = np.array(
            [
                [0.0, 0.0, 10.0, 10.0, 0.95, 0.0],
                [100.0, 100.0, 110.0, 110.0, 0.80, 0.0],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(
            result[0],
            expected,
        )

    # -----------------------------------------------------
    # Class-aware vs. agnostic NMS
    # -----------------------------------------------------

    @pytest.mark.parametrize(
        "is_agnostic_nms,expected",
        [
            (
                False,
                np.array(
                    [
                        [0, 0, 10, 10, 0.95, 0],
                        [0, 0, 10, 10, 0.85, 1],
                    ],
                    dtype=np.float32,
                ),
            ),
            (
                True,
                np.array(
                    [
                        [0, 0, 10, 10, 0.95, 0],
                    ],
                    dtype=np.float32,
                ),
            ),
        ],
        ids=[
            "class_aware",
            "agnostic",
        ],
    )
    def test_is_agnostic_nms_controls_cross_class_suppression(
        self,
        is_agnostic_nms,
        expected,
    ):
        pred = np.array(
            [
                [
                    [5, 5, 5],
                    [5, 5, 5],
                    [10, 10, 10],
                    [10, 10, 10],
                    [0.95, 0.90, 0.00],
                    [0.00, 0.00, 0.85],
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov8(
            pred,
            is_agnostic_nms=is_agnostic_nms,
        )

        np.testing.assert_allclose(
            result[0],
            expected,
        )

    # -----------------------------------------------------
    # Ordering
    # -----------------------------------------------------

    def test_outputs_sorted_by_confidence(self):
        pred = np.array(
            [
                [
                    [25, 45, 65],
                    [25, 45, 65],
                    [10, 10, 10],
                    [10, 10, 10],
                    [0.60, 0.95, 0.80],
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov8(pred)

        expected_scores = np.array(
            [0.95, 0.80, 0.60],
            dtype=np.float32,
        )

        np.testing.assert_allclose(
            result[0][:, 4],
            expected_scores,
        )

    # -----------------------------------------------------
    # max_det
    # -----------------------------------------------------

    def test_limits_output_to_max_det(self):
        pred = np.array(
            [
                [
                    [5, 25, 45],
                    [5, 25, 45],
                    [10, 10, 10],
                    [10, 10, 10],
                    [0.95, 0.90, 0.85],
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov8(
            pred,
            max_det=2,
        )

        assert result[0].shape == (2, 6)

        np.testing.assert_allclose(
            result[0][:, 4],
            np.array(
                [0.95, 0.90],
                dtype=np.float32,
            ),
        )

    def test_keeps_all_detections_when_max_det_exceeds_candidate_count(self):
        pred = np.array(
            [
                [
                    [5, 25],
                    [5, 25],
                    [10, 10],
                    [10, 10],
                    [0.90, 0.80],
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov8(
            pred,
            max_det=100,
        )

        assert result[0].shape == (2, 6)

    # -----------------------------------------------------
    # Edge cases and output contract
    # -----------------------------------------------------

    @patch("core.python.postprocess.detection_utils.numpy_nms")
    def test_handles_empty_keep_indices_from_numpy_nms(
        self,
        mock_numpy_nms,
    ):
        mock_numpy_nms.return_value = []

        pred = np.array(
            [
                [
                    [5],
                    [5],
                    [10],
                    [10],
                    [0.90],
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov8(pred)

        assert len(result) == 1
        assert result[0].shape == (0, 6)

    def test_empty_batch_returns_empty_list(self):
        pred = np.empty(
            (0, 5, 10),
            dtype=np.float32,
        )

        result = nms_yolov8(pred)

        assert result == []

    def test_image_with_no_candidates_returns_empty_detection_array(self):
        pred = np.array(
            [
                [
                    [5],
                    [5],
                    [10],
                    [10],
                    [0.10],  # Below default threshold.
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov8(pred)

        assert len(result) == 1
        assert result[0].shape == (0, 6)
        assert result[0].dtype == np.float32

    def test_output_dtype_is_float32(self):
        pred = np.array(
            [
                [
                    [5],
                    [5],
                    [10],
                    [10],
                    [0.90],
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov8(pred)

        assert result[0].dtype == np.float32

    def test_output_has_expected_detection_format(self):
        pred = np.array(
            [
                [
                    [5, 15],
                    [5, 15],
                    [10, 10],
                    [10, 10],
                    [0.90, 0.10],
                    [0.20, 0.80],
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov8(pred)

        detections = result[0]

        assert detections.ndim == 2
        assert detections.shape == (2, 6)

        np.testing.assert_allclose(
            detections[:, 5],
            np.array(
                [0, 1],
                dtype=np.float32,
            ),
        )

    def test_candidate_mask_is_applied_per_image(self):
        pred = np.array(
            [
                [
                    [5],
                    [5],
                    [10],
                    [10],
                    [0.90],
                ],
                [
                    [5],
                    [5],
                    [10],
                    [10],
                    [0.10],
                ],
            ],
            dtype=np.float32,
        )

        result = nms_yolov8(pred, nms_score_threshold=0.25)

        np.testing.assert_allclose(
            result[0],
            np.array(
                [
                    [0, 0, 10, 10, 0.90, 0],
                ],
                dtype=np.float32,
            ),
        )

        assert result[1].shape == (0, 6)

    # -----------------------------------------------------
    # IoU-threshold boundary values
    # -----------------------------------------------------

    @pytest.mark.parametrize(
        "iou_threshold,expected",
        [
            (
                0.0,
                np.array(
                    [
                        [0, 0, 10, 10, 0.90, 0],
                    ],
                    dtype=np.float32,
                ),
            ),
            (
                1.0,
                np.array(
                    [
                        [0, 0, 10, 10, 0.90, 0],
                        [0, 0, 10, 10, 0.80, 0],
                    ],
                    dtype=np.float32,
                ),
            ),
        ],
        ids=[
            "minimum",
            "maximum",
        ],
    )
    def test_iou_threshold_boundary_values(
        self,
        iou_threshold,
        expected,
    ):
        pred = np.array(
            [
                [
                    [5, 5],
                    [5, 5],
                    [10, 10],
                    [10, 10],
                    [0.90, 0.80],
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov8(
            pred,
            iou_threshold=iou_threshold,
        )

        np.testing.assert_allclose(
            result[0],
            expected,
        )

    def test_preserves_mask_coefficients_after_nms(self):
        pred = np.array(
            [
                [
                    [5, 6],
                    [5, 6],
                    [10, 10],
                    [10, 10],
                    # class 0
                    [0.95, 0.80],
                    # mask coefficient 1
                    [11, 21],
                    # mask coefficient 2
                    [12, 22],
                    # mask coefficient 3
                    [13, 23],
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov8(
            pred,
            nc=1,
            iou_threshold=0.5,
        )

        expected = np.array(
            [
                [
                    0,
                    0,
                    10,
                    10,
                    0.95,
                    0,
                    11,
                    12,
                    13,
                ],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(
            result[0],
            expected,
        )

    # -----------------------------------------------------
    # Score-threshold boundary values
    # -----------------------------------------------------

    @pytest.mark.parametrize(
        "score_threshold,expected",
        [
            (
                0.0,
                np.array(
                    [
                        [0, 0, 10, 10, 0.60, 0],
                        [20, 20, 30, 30, 0.40, 0],
                    ],
                    dtype=np.float32,
                ),
            ),
            (
                1.0,
                np.empty((0, 6), dtype=np.float32),
            ),
        ],
        ids=[
            "minimum",
            "maximum",
        ],
    )
    def test_score_threshold_boundary_values(
        self,
        score_threshold,
        expected,
    ):
        pred = np.array(
            [
                [
                    [5, 25],
                    [5, 25],
                    [10, 10],
                    [10, 10],
                    [0.60, 0.40],
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov8(
            pred,
            nms_score_threshold=score_threshold,
        )

        np.testing.assert_allclose(
            result[0],
            expected,
        )

    # -----------------------------------------------------
    # multi_label behavior
    # -----------------------------------------------------

    @pytest.mark.parametrize(
        "multi_label,expected",
        [
            (
                False,
                np.array(
                    [
                        [0, 0, 10, 10, 0.95, 0],
                    ],
                    dtype=np.float32,
                ),
            ),
            (
                True,
                np.array(
                    [
                        [0, 0, 10, 10, 0.95, 0],
                        [0, 0, 10, 10, 0.85, 1],
                    ],
                    dtype=np.float32,
                ),
            ),
        ],
        ids=[
            "single_label",
            "multi_label",
        ],
    )
    def test_multi_label_controls_multiple_class_predictions_per_anchor(
        self,
        multi_label,
        expected,
    ):
        pred = np.array(
            [
                [
                    [5],
                    [5],
                    [10],
                    [10],
                    [0.95],
                    [0.85],
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov8(
            pred,
            nms_score_threshold=0.25,
            iou_threshold=0.45,
            multi_label=multi_label,
            is_agnostic_nms=False,
        )

        np.testing.assert_allclose(
            result[0],
            expected,
        )

    @pytest.mark.parametrize(
        "multi_label",
        [
            False,
            True,
        ],
        ids=[
            "single_label",
            "multi_label",
        ],
    )
    def test_agnostic_nms_suppresses_same_anchor_multi_label_detections(
        self,
        multi_label,
    ):
        pred = np.array(
            [
                [
                    [5],
                    [5],
                    [10],
                    [10],
                    [0.95],
                    [0.85],
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov8(
            pred,
            nms_score_threshold=0.25,
            iou_threshold=0.45,
            multi_label=multi_label,
            is_agnostic_nms=True,
        )

        expected = np.array(
            [
                [0, 0, 10, 10, 0.95, 0],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(
            result[0],
            expected,
        )

    @pytest.mark.parametrize(
        "pred,multi_label,expected",
        [
            (
                # Single-class model (nc=1)
                np.array(
                    [
                        [
                            [5],
                            [5],
                            [10],
                            [10],
                            [0.95],
                        ]
                    ],
                    dtype=np.float32,
                ),
                True,
                np.array(
                    [
                        [0, 0, 10, 10, 0.95, 0],
                    ],
                    dtype=np.float32,
                ),
            ),
            (
                # Two-class model (nc=2)
                np.array(
                    [
                        [
                            [5],
                            [5],
                            [10],
                            [10],
                            [0.95],
                            [0.85],
                        ]
                    ],
                    dtype=np.float32,
                ),
                True,
                np.array(
                    [
                        [0, 0, 10, 10, 0.95, 0],
                        [0, 0, 10, 10, 0.85, 1],
                    ],
                    dtype=np.float32,
                ),
            ),
        ],
        ids=[
            "single_class_model",
            "multi_class_model",
        ],
    )
    def test_multi_label_behavior_depends_on_number_of_classes(
        self,
        pred,
        multi_label,
        expected,
    ):
        result = nms_yolov8(
            pred,
            nms_score_threshold=0.25,
            iou_threshold=0.45,
            multi_label=multi_label,
            is_agnostic_nms=False,
        )

        np.testing.assert_allclose(
            result[0],
            expected,
        )

    # -----------------------------------------------------
    # nc
    # -----------------------------------------------------

    def test_explicit_nc_preserves_extra_channels(self):
        """
        Explicitly configured nc should correctly separate class scores from
        extra channels (e.g. segmentation mask coefficients), preserving the
        extra channels in the output.
        """
        pred = np.array(
            [
                [
                    [20],
                    [30],
                    [20],
                    [20],
                    # class 0
                    [0.90],
                    # class 1
                    [0.10],
                    # extra channels (e.g. mask coefficients)
                    [0.11],
                    [0.22],
                    [0.33],
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov8(
            pred,
            nc=2,
            nms_score_threshold=0.25,
        )

        expected = np.array(
            [
                [10, 20, 30, 40, 0.90, 0, 0.11, 0.22, 0.33],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(
            result[0],
            expected,
        )

    def test_nc_is_inferred_when_not_provided(self):
        """
        When nc is omitted, it should be inferred from the prediction tensor.
        """
        pred = np.array(
            [
                [
                    [20],
                    [30],
                    [20],
                    [20],
                    [0.10],
                    [0.90],
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov8(
            pred,
            nms_score_threshold=0.25,
        )

        expected = np.array(
            [
                [10, 20, 30, 40, 0.90, 1],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(
            result[0],
            expected,
        )

    @pytest.mark.parametrize(
        "nc",
        [
            0,
            -1,
        ],
        ids=[
            "zero",
            "negative",
        ],
    )
    def test_invalid_nc_raises_value_error(
        self,
        nc,
    ):
        pred = np.zeros(
            (1, 6, 1),
            dtype=np.float32,
        )

        with pytest.raises(
            ValueError,
            match=r"Expected at least one class channel",
        ):
            nms_yolov8(
                pred,
                nc=nc,
            )

    def test_explicit_nc_ignores_num_keypoints_configuration(self):
        """
        Explicit nc takes precedence over pose configuration. All remaining
        channels are treated as extra channels regardless of the supplied
        num_keypoints/keypoint_dims values.
        """
        pred = np.array(
            [
                [
                    [20],
                    [30],
                    [20],
                    [20],
                    # class 0
                    [0.90],
                    # class 1
                    [0.10],
                    # extra channels
                    [0.11],
                    [0.22],
                    [0.33],
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov8(
            pred,
            nc=2,
            num_keypoints=17,  # intentionally incorrect
            keypoint_dims=3,
        )

        expected = np.array(
            [
                [10, 20, 30, 40, 0.90, 0, 0.11, 0.22, 0.33],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(result[0], expected)

    # -----------------------------------------------------
    # Pose (num_keypoints / keypoint_dims)
    # -----------------------------------------------------

    def test_preserves_keypoints_when_num_keypoints_configured(self):
        """
        Pose keypoints should be preserved unchanged in the output.
        """
        pred = np.array(
            [
                [
                    [20],
                    [30],
                    [20],
                    [20],
                    # class 0
                    [0.90],
                    # class 1
                    [0.10],
                    # kp1 (x, y, v)
                    [1],
                    [2],
                    [3],
                    # kp2 (x, y, v)
                    [4],
                    [5],
                    [6],
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov8(
            pred,
            num_keypoints=2,
            nms_score_threshold=0.25,
        )

        expected = np.array(
            [
                [10, 20, 30, 40, 0.90, 0, 1, 2, 3, 4, 5, 6],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(
            result[0],
            expected,
        )

    def test_supports_custom_keypoint_dimensions(self):
        """
        keypoint_dims should determine how many values belong to each keypoint.
        """
        pred = np.array(
            [
                [
                    [20],
                    [30],
                    [20],
                    [20],
                    # class 0
                    [0.90],
                    # kp1 (x, y)
                    [11],
                    [12],
                    # kp2 (x, y)
                    [21],
                    [22],
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov8(
            pred,
            num_keypoints=2,
            keypoint_dims=2,
        )

        expected = np.array(
            [
                [10, 20, 30, 40, 0.90, 0, 11, 12, 21, 22],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(
            result[0],
            expected,
        )

    def test_num_keypoints_zero_returns_detection_format(self):
        """
        Detection models (num_keypoints=0) should return the standard
        six-column detection format.
        """
        pred = np.array(
            [
                [
                    [20],
                    [30],
                    [20],
                    [20],
                    [0.90],
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov8(
            pred,
            num_keypoints=0,
        )

        assert result[0].shape == (1, 6)

    @pytest.mark.parametrize(
        ("num_keypoints", "keypoint_dims"),
        [
            (3, 3),
            (2, 5),
        ],
        ids=[
            "num_keypoints",
            "keypoint_dims",
        ],
    )
    def test_invalid_inferred_nc_raises_value_error(
        self,
        num_keypoints,
        keypoint_dims,
    ):
        """
        If the inferred number of classes is non-positive, a ValueError
        should be raised.
        """
        pred = np.zeros(
            (1, 10, 1),
            dtype=np.float32,
        )

        with pytest.raises(
            ValueError,
            match=r"Expected at least one class channel",
        ):
            nms_yolov8(
                pred,
                num_keypoints=num_keypoints,
                keypoint_dims=keypoint_dims,
            )

    def test_num_keypoints_changes_class_inference(self):
        """
        num_keypoints should participate in inferring the number of class
        channels when nc is not provided.
        """
        pred = np.array(
            [
                [
                    [20],
                    [30],
                    [20],
                    [20],
                    # class 0
                    [0.10],
                    # class 1
                    [0.90],
                    # kp1
                    [1],
                    [2],
                    [3],
                ]
            ],
            dtype=np.float32,
        )

        result = nms_yolov8(
            pred, num_keypoints=1, keypoint_dims=3, nms_score_threshold=0.25
        )

        expected = np.array(
            [
                [10, 20, 30, 40, 0.90, 1, 1, 2, 3],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(
            result[0],
            expected,
        )
