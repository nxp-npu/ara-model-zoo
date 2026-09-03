# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from enum import Enum
from itertools import product

import pytest
import numpy as np

from core.python.config import Config
from core.python.preprocess.interfaces import PreprocessOutput
from core.python.preprocess.preprocessing_builder import PreprocessingBuilder

# Transformation Classes
from core.python.preprocess.operations.scale import Scale
from core.python.preprocess.operations.resize import Resize
from core.python.preprocess.operations.tofloat import ToFloat
from core.python.preprocess.operations.bgrtorgb import BgrToRgb
from core.python.preprocess.operations.centercrop import CenterCrop
from core.python.preprocess.operations.transpose import ChannelFirstTranspose
from core.python.preprocess.operations.meansubtraction import MeanSubtraction


class TestPipelineExecution:
    """
    Pipeline execution contract tests.

    Core guarantees:
    - input image is never mutated
    - original image is an exact snapshot at pipeline entry
    - processed image is derived from config pipeline
    - processed image is numpy.ndarray
    - no shared memory between input and outputs
    - behavior is stable across all config combinations

    Notes:
    - Empty pipeline execution is intentionally not tested.
    - Current implementation always includes mandatory
      normalization transforms (e.g. Scale, Mean),
      therefore a zero-transform pipeline cannot be
      produced through any valid configuration.
    """

    CONFIG_FLAGS = (
        "bgr_to_rgb",
        "resize",
        "centercrop",
        "hwc_to_chw",
    )

    # =========================================================
    # HELPERS
    # =========================================================

    @staticmethod
    def _create_test_image():
        return np.random.randint(
            0,
            255,
            (480, 640, 3),
            dtype=np.uint8,
        )

    @staticmethod
    def _build_config(
        bgr_to_rgb=True,
        resize=True,
        centercrop=True,
        hwc_to_chw=False,
    ):
        cfg = Config.from_file("modelzoo/classification/mobilenetv1/config/run.yaml")

        if not resize:
            cfg.preprocess.resize = None

        if not centercrop:
            cfg.preprocess.centercrop = None

        cfg.preprocess.bgr_to_rgb = bgr_to_rgb
        cfg.preprocess.hwc_to_chw = hwc_to_chw

        return cfg

    @staticmethod
    def _assert_pipeline_contracts(
        image: np.ndarray,
        snapshot: np.ndarray,
        output: PreprocessOutput,
        config_description: str = "",
    ):
        """
        Verifies all pipeline execution guarantees.
        """

        assert np.array_equal(image, snapshot), (
            "IN-PLACE MUTATION DETECTED:\n"
            "Input image was modified during pipeline execution.\n"
            f"{config_description}"
        )

        assert np.array_equal(output.original_image, snapshot), (
            "ORIGINAL SNAPSHOT VIOLATION:\n"
            "original_image does not match pipeline entry state.\n"
            f"{config_description}"
        )

        assert output.processed_image is not None, (
            "PIPELINE FAILURE:\n"
            "Pre-processing returned no processed image.\n"
            f"{config_description}"
        )

        assert isinstance(output.processed_image, np.ndarray), (
            "TYPE CONTRACT VIOLATION:\n"
            "processed_image must be a numpy.ndarray.\n"
            f"Got: {type(output.processed_image).__name__}\n"
            f"{config_description}"
        )

        assert not np.shares_memory(
            image,
            output.processed_image,
        ), (
            "MEMORY ALIASING DETECTED:\n"
            "processed_image shares memory with input image.\n"
            f"{config_description}"
        )

    # =========================================================
    # DEFAULT CONFIG SANITY CHECK
    # =========================================================

    def test_pipeline_contracts_default_configuration(self):
        image = self._create_test_image()
        snapshot = image.copy()

        cfg = self._build_config()

        output: PreprocessOutput = PreprocessingBuilder(cfg).execute_pipeline(image)

        self._assert_pipeline_contracts(
            image=image,
            snapshot=snapshot,
            output=output,
            config_description="default configuration",
        )

    # =========================================================
    # CONFIGURATION MATRIX VALIDATION
    # =========================================================

    @pytest.mark.parametrize(
        CONFIG_FLAGS,
        tuple(product([False, True], repeat=len(CONFIG_FLAGS))),
    )
    def test_pipeline_contracts_across_all_configurations(
        self,
        bgr_to_rgb,
        resize,
        centercrop,
        hwc_to_chw,
    ):
        image = self._create_test_image()
        snapshot = image.copy()

        cfg = self._build_config(
            bgr_to_rgb=bgr_to_rgb,
            resize=resize,
            centercrop=centercrop,
            hwc_to_chw=hwc_to_chw,
        )

        output: PreprocessOutput = PreprocessingBuilder(cfg).execute_pipeline(image)

        config_description = (
            f"bgr_to_rgb={bgr_to_rgb}, "
            f"resize={resize}, "
            f"centercrop={centercrop}, "
            f"hwc_to_chw={hwc_to_chw}"
        )

        self._assert_pipeline_contracts(
            image=image,
            snapshot=snapshot,
            output=output,
            config_description=config_description,
        )

    # =========================================================
    # DETERMINISM GUARANTEE
    # =========================================================

    def test_pipeline_is_deterministic(self):
        image = self._create_test_image()

        cfg = self._build_config()

        out1: PreprocessOutput = PreprocessingBuilder(cfg).execute_pipeline(image)

        out2: PreprocessOutput = PreprocessingBuilder(cfg).execute_pipeline(image)

        assert np.array_equal(
            out1.processed_image,
            out2.processed_image,
        ), "NON-DETERMINISM DETECTED:\nPipeline output differs across identical runs."

        assert np.array_equal(
            out1.original_image,
            out2.original_image,
        ), "NON-DETERMINISM DETECTED:\noriginal_image differs across identical runs."


class TestTransformDispatch:
    """
    Tests routing from TransformKey → Transform class.

    Contract:
    - Each valid enum key maps to correct transform class
    - No cross-wiring between transforms
    - No silent fallback behavior
    - Invalid keys must fail fast
    - Each dispatch must return a fresh instance
    """

    # =========================================================
    # TEST SETUP
    # =========================================================

    class InvalidValues(Enum):
        UNSUPPORTED = "unsupported"

    @staticmethod
    def _build_config():
        return Config.from_file("modelzoo/classification/mobilenetv1/config/run.yaml")

    @classmethod
    def _build_engine(cls):
        return PreprocessingBuilder(cls._build_config())

    # =========================================================
    # VALID DISPATCH CASES
    # =========================================================

    @pytest.mark.parametrize(
        ("key", "expected_class"),
        [
            (PreprocessingBuilder.VALUES.SCALE, Scale),
            (PreprocessingBuilder.VALUES.RESIZE, Resize),
            (PreprocessingBuilder.VALUES.TO_FLOAT, ToFloat),
            (PreprocessingBuilder.VALUES.BGR_TO_RGB, BgrToRgb),
            (PreprocessingBuilder.VALUES.MEAN, MeanSubtraction),
            (PreprocessingBuilder.VALUES.CENTER_CROP, CenterCrop),
            (
                PreprocessingBuilder.VALUES.CHANNEL_FIRST_TRANSPOSE,
                ChannelFirstTranspose,
            ),
        ],
    )
    def test_valid_transform_key_maps_to_correct_class(
        self,
        key,
        expected_class,
    ):
        # Arrange
        engine = self._build_engine()

        # Act
        transform = engine._get_module(key)

        # Assert
        assert isinstance(transform, expected_class), (
            f"Dispatch mismatch:\n"
            f"KEY={key}\n"
            f"EXPECTED={expected_class.__name__}\n"
            f"ACTUAL={type(transform).__name__}"
        )

    # =========================================================
    # INVALID KEY HANDLING
    # =========================================================

    @pytest.mark.parametrize(
        "invalid_key",
        [
            InvalidValues.UNSUPPORTED,
        ],
    )
    def test_invalid_key_raises_not_implemented_error(
        self,
        invalid_key,
    ):
        # Arrange
        engine = self._build_engine()

        # Act / Assert
        with pytest.raises(NotImplementedError):
            engine._get_module(invalid_key)

    # =========================================================
    # SAFETY: NO INSTANCE REUSE
    # =========================================================

    @pytest.mark.parametrize(
        "key",
        [
            PreprocessingBuilder.VALUES.SCALE,
            PreprocessingBuilder.VALUES.RESIZE,
            PreprocessingBuilder.VALUES.TO_FLOAT,
            PreprocessingBuilder.VALUES.BGR_TO_RGB,
            PreprocessingBuilder.VALUES.MEAN,
            PreprocessingBuilder.VALUES.CENTER_CROP,
            PreprocessingBuilder.VALUES.CHANNEL_FIRST_TRANSPOSE,
        ],
    )
    def test_dispatch_returns_fresh_instance_each_time(self, key):
        # Arrange
        engine = self._build_engine()

        # Act
        transform_1 = engine._get_module(key)
        transform_2 = engine._get_module(key)

        # Assert
        assert transform_1 is not transform_2, (
            "Transform instances are reused unexpectedly. "
            "Expected fresh instance per dispatch call."
        )
