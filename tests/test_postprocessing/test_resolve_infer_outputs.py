# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from pathlib import Path
from copy import deepcopy

import pytest
import numpy as np
from typing import cast
from unittest.mock import sentinel

import core.python.utils as utils
from core.python.config import Config, InferenceConfig
import core.python.postprocess.resolve_infer_outputs as rio
from core.python.inference import ModelOutput


# =========================================================
# FAKE CONTINUATION SESSION
# =========================================================


class FakeSession:
    input_names = ["node1", "node2"]

    def get_input_metadata(self):
        return [
            type("N", (), {"name": "cont_node_1", "shape": (1,)})(),
            type("N", (), {"name": "cont_node_2", "shape": (1,)})(),
        ]

    def infer(self, inputs):
        return type("R", (), {"outputs": {"final": np.zeros((1, 3))}})()


def fake_session(*args, **kwargs):
    return FakeSession()


# ====================================
# Fixtures
# ====================================


@pytest.fixture
def valid_cfg(tmp_path):
    """
    Minimal valid configuration for constructor tests.

    Continuation session is NOT created because
    postprocessing.onnx does not exist by default, and modelname has no
    registered PyTorch continue_graph.
    """

    cfg = sentinel.cfg

    cfg.out = str(tmp_path)
    # Unknown model → get_model_graph_continuation fails → cutoff_exists False
    # when postprocessing.onnx is absent (or ORT is unavailable).
    cfg.modelname = "not_a_real_model"

    cfg.dvconvert = sentinel.cvt
    cfg.dvconvert.onode = "node1:node2"

    cfg.inference = InferenceConfig.model_construct()

    return cfg


@pytest.fixture
def cutoff_model(
    tmp_path: Path,
) -> None:
    """
    Creates dummy continuation model.

    Only file existence matters because session
    creation is mocked.
    """

    path = tmp_path / "compiled_model" / "postprocessing.onnx"

    path.parent.mkdir()

    path.touch()


# =========================================================
# CUTOFF DETECTION TESTS
# =========================================================


class TestCutoffDetection:
    """
    Tests cutoff graph activation logic.

    Contract:

    cutoff_exists is True iff session_type == "ara" AND either:

        - onnxruntime is available and postprocessing.onnx exists, or
        - a PyTorch continue_graph is registered for cfg.modelname

    Otherwise cutoff_exists must be False.
    """

    @pytest.mark.parametrize(
        (
            "session_type",
            "create_cutoff_model",
            "expected_cutoff_exists",
        ),
        [
            ("CPU", True, False),
            ("CPU", False, False),
            ("ara", True, True),
            ("ara", False, False),
            ("RANDOM_BACKEND", True, False),
            ("RANDOM_BACKEND", False, False),
        ],
    )
    def test_cutoff_detection_truth_table(
        self,
        tmp_path,
        valid_cfg,
        monkeypatch,
        session_type,
        create_cutoff_model,
        expected_cutoff_exists,
    ):
        # Arrange
        cfg = deepcopy(valid_cfg)

        if create_cutoff_model:
            (
                cutoff_path := tmp_path / "compiled_model" / "postprocessing.onnx"
            ).parent.mkdir()
            cutoff_path.touch()

        # Prefer ONNX when the cutoff file exists so this table does not depend
        # on whether onnxruntime is installed in the test environment.
        monkeypatch.setattr(
            rio,
            "is_onnxruntime_available",
            lambda: bool(create_cutoff_model),
        )

        # -----------------------------
        # FULL MOCK OF CONTINUATION LAYER
        # -----------------------------
        monkeypatch.setattr(
            rio,
            "create_inference_session",
            fake_session,
        )

        monkeypatch.setattr(
            utils,
            "map_device_to_continuation_nodes",
            lambda _: {
                "node1": "cont_node_1",
                "node2": "cont_node_2",
            },
        )

        # Act
        resolver = rio.ResolveInferenceOutputs(
            cfg,
            session_type=session_type,
        )

        # Assert
        assert resolver.cutoff_exists == expected_cutoff_exists, (
            f"Incorrect cutoff detection.\n"
            f"session_type={session_type}\n"
            f"cutt-off exists={create_cutoff_model}\n"
            f"expected={expected_cutoff_exists}\n"
            f"actual={resolver.cutoff_exists}"
        )


# =========================================================
# CONTINUATION GRAPH INITIALIZATION
# =========================================================


class TestContinuationGraphInitialization:
    """
    Tests ResolveInferenceOutputs constructor behavior when
    continuation graph exists.

    Covers:

    1. Building device_to_continuation_map
    2. Validating session input names against configured nodes
    """

    # =====================================================
    # HELPERS
    # =====================================================

    @pytest.fixture
    def fake_session_factory(
        self,
    ):
        """
        Creates fake continuation session.

        The returned object mimics the subset of the
        inference session interface used by
        ResolveInferenceOutputs.
        """

        def _factory(
            input_names: list[str],
        ):

            class FakeSession:
                def __init__(self):

                    self.input_names = input_names

                def get_input_metadata(self):

                    return [
                        type(
                            "Node",
                            (),
                            {
                                "name": name,
                                "shape": (1,),
                            },
                        )()
                        for name in self.input_names
                    ]

            return FakeSession()

        return _factory

    @pytest.fixture
    def patch_session(
        self,
        monkeypatch,
        fake_session_factory,
    ):
        """
        Patch continuation session creation.
        """

        def _patch(
            session_input_names: list[str],
        ):

            monkeypatch.setattr(
                rio,
                "create_inference_session",
                lambda *a, **k: fake_session_factory(session_input_names),
            )

        return _patch

    # =====================================================
    # PROPERTIES INITIALIZATIONS VALIDATION
    # =====================================================

    def test_none_cfg_raises(self):
        with pytest.raises(
            ValueError,
            match="cfg must not be None",
        ):
            rio.ResolveInferenceOutputs(
                cast(Config, None),
                session_type="ara",
            )

    def test_missing_output_directory_raises(
        self,
        valid_cfg,
    ):
        cfg = deepcopy(valid_cfg)

        cfg.out = "/definitely/does/not/exist"

        with pytest.raises(
            ValueError,
            match="Output directory does not exist",
        ):
            rio.ResolveInferenceOutputs(
                cfg,
                session_type="ara",
            )

    def test_non_string_onode_raises(
        self,
        tmp_path,
        valid_cfg,
        cutoff_model,
    ):
        cfg = deepcopy(valid_cfg)

        cfg.dvconvert.onode = 123

        with pytest.raises(
            TypeError,
            match="onode must be a string",
        ):
            rio.ResolveInferenceOutputs(
                cfg,
                session_type="ara",
            )

    def test_missing_inference_cfg_raises(
        self,
        valid_cfg,
        tmp_path,
        cutoff_model,
    ):
        cfg = deepcopy(valid_cfg)

        cfg.inference = None

        with pytest.raises(
            ValueError,
            match="cfg.inference must be provided",
        ):
            rio.ResolveInferenceOutputs(
                cfg,
                session_type="ara",
            )

    def test_continuation_model_path_is_constructed_correctly(
        self,
        valid_cfg,
        tmp_path,
    ):
        cfg = deepcopy(valid_cfg)

        resolver = rio.ResolveInferenceOutputs(
            cfg,
            session_type="cpu",
        )

        assert (
            resolver.continuation_model_path
            == tmp_path / "compiled_model" / "postprocessing.onnx"
        )

    # =====================================================
    # DEVICE -> CONTINUATION MAPPING
    # =====================================================

    @pytest.mark.parametrize(
        ("onode_str", "expected_mapping"),
        [
            (
                "/model.22/Mul_output_0#100:/model.22/Sigmoid_output_0#100",
                {
                    "_model_22_Mul_output_0:100": "/model.22/Mul_output_0:100",
                    "_model_22_Sigmoid_output_0:100": "/model.22/Sigmoid_output_0:100",
                },
            ),
            (
                "/a.b/c.d",
                {
                    "_a_b_c_d": "/a.b/c.d",
                },
            ),
            (
                "/a.b/c.d#1",
                {
                    "_a_b_c_d:1": "/a.b/c.d:1",
                },
            ),
            (
                "/x.y#10:/p.q/r#20",
                {
                    "_x_y:10": "/x.y:10",
                    "_p_q_r:20": "/p.q/r:20",
                },
            ),
            (
                "/layer.1/block.2/output#abc:/layer.3/block.4/output#xyz",
                {
                    "_layer_1_block_2_output:abc": "/layer.1/block.2/output:abc",
                    "_layer_3_block_4_output:xyz": "/layer.3/block.4/output:xyz",
                },
            ),
            (
                "/layer_1/block.2/output#abc:/layer_3/block.4/output#xyz",
                {
                    "_layer_1_block_2_output:abc": "/layer_1/block.2/output:abc",
                    "_layer_3_block_4_output:xyz": "/layer_3/block.4/output:xyz",
                },
            ),
        ],
        ids=[
            "hash_replaced_by_colon",
            "slash_dot_translation",
            "single_hash",
            "multiple_nodes",
            "deep_nested_names",
            "underscored_names",
        ],
    )
    def test_constructor_builds_device_to_continuation_mapping(
        self,
        valid_cfg,
        tmp_path: Path,
        cutoff_model,
        patch_session,
        onode_str,
        expected_mapping,
    ):
        # Arrange

        cfg = deepcopy(valid_cfg)

        cfg.dvconvert.onode = onode_str

        patch_session(list(expected_mapping.values()))

        # Act

        resolver = rio.ResolveInferenceOutputs(
            cfg,
            session_type="ara",
        )

        # Assert

        assert resolver.device_to_continuation_map == expected_mapping

    # =====================================================
    # SESSION INPUT VALIDATION
    # =====================================================

    @pytest.mark.parametrize(
        (
            "device_to_continuation_map",
            "session_input_names",
            "should_raise",
            "missing",
            "unexpected",
        ),
        [
            # ----------------------------------------
            # Exact match
            # ----------------------------------------
            (
                {
                    "device_preprocess": "model/preprocess/output",
                    "device_logits": "model/classifier/logits",
                },
                [
                    "model/preprocess/output",
                    "model/classifier/logits",
                ],
                False,
                None,
                None,
            ),
            # ----------------------------------------
            # Missing input
            # ----------------------------------------
            (
                {
                    "device_preprocess": "model/preprocess/output",
                    "device_logits": "model/classifier/logits",
                },
                [
                    "model/preprocess/output",
                ],
                True,
                ["model/classifier/logits"],
                None,
            ),
            # ----------------------------------------
            # Unexpected input
            # ----------------------------------------
            (
                {
                    "device_logits": "model/classifier/logits",
                },
                [
                    "model/classifier/logits",
                    "model/aux/output",
                ],
                True,
                None,
                ["model/aux/output"],
            ),
            # ----------------------------------------
            # Missing + unexpected
            # ----------------------------------------
            (
                {
                    "device_logits": "model/classifier/logits",
                    "device_aux": "model/aux/output",
                },
                [
                    "model/classifier/logits",
                    "model/extra/output",
                ],
                True,
                ["model/aux/output"],
                ["model/extra/output"],
            ),
            # ----------------------------------------
            # Order independence
            # ----------------------------------------
            (
                {
                    "device_a": "model/a/output",
                    "device_b": "model/b/output",
                    "device_c": "model/c/output",
                },
                [
                    "model/c/output",
                    "model/a/output",
                    "model/b/output",
                ],
                False,
                None,
                None,
            ),
        ],
        ids=[
            "exact_match",
            "missing_input",
            "unexpected_input",
            "missing_and_unexpected",
            "order_independent",
        ],
    )
    def test_constructor_validates_session_inputs_against_config(
        self,
        valid_cfg,
        tmp_path: Path,
        cutoff_model,
        monkeypatch,
        patch_session,
        device_to_continuation_map,
        session_input_names,
        should_raise,
        missing,
        unexpected,
    ):
        # Arrange

        cfg = deepcopy(valid_cfg)

        patch_session(session_input_names)

        monkeypatch.setattr(
            rio,
            "map_device_to_continuation_nodes",
            lambda _: device_to_continuation_map,
        )

        # Act / Assert

        if should_raise:
            with pytest.raises(
                RuntimeError,
            ) as exc_info:
                rio.ResolveInferenceOutputs(
                    cfg,
                    session_type="ara",
                )

            msg = str(exc_info.value)

            assert "Mismatch between configured continuation graph nodes" in msg

            if missing:
                for node in missing:
                    assert node in msg

            if unexpected:
                for node in unexpected:
                    assert node in msg

        else:
            resolver = rio.ResolveInferenceOutputs(
                cfg,
                session_type="ara",
            )

            assert resolver.cutoff_exists is True

            assert set(resolver.device_to_continuation_map.values()) == set(
                session_input_names
            )

    # -----------------------------------------------------
    # session.input_names raises
    # -----------------------------------------------------

    def test_session_input_names_access_failure(
        self,
        valid_cfg,
        cutoff_model,
        monkeypatch,
    ):
        cfg = deepcopy(valid_cfg)

        class BadSession:
            @property
            def input_names(self):
                raise RuntimeError("session crash")

            def get_input_metadata(self):
                return []

        monkeypatch.setattr(
            rio,
            "create_inference_session",
            lambda *args, **kwargs: BadSession(),
        )

        monkeypatch.setattr(
            rio,
            "map_device_to_continuation_nodes",
            lambda _: {"dev1": "node1"},
        )

        with pytest.raises(
            RuntimeError,
            match="Failed to retrieve input node names",
        ):
            rio.ResolveInferenceOutputs(
                cfg,
                session_type="ara",
            )

    # -----------------------------------------------------
    # session.input_names is empty
    # -----------------------------------------------------

    def test_empty_session_input_names_raises(
        self,
        valid_cfg,
        cutoff_model,
        monkeypatch,
    ):
        cfg = deepcopy(valid_cfg)

        class EmptySession:
            input_names = []

            def get_input_metadata(self):
                return []

        monkeypatch.setattr(
            rio,
            "create_inference_session",
            lambda *args, **kwargs: EmptySession(),
        )

        monkeypatch.setattr(
            rio,
            "map_device_to_continuation_nodes",
            lambda _: {"dev1": "node1"},
        )

        with pytest.raises(
            RuntimeError,
            match="Continuation model contains no input nodes",
        ):
            rio.ResolveInferenceOutputs(
                cfg,
                session_type="ara",
            )


# =========================================================
# CONTINUATION INFERENCE EXECUTION TESTS
# =========================================================


class TestRunContinuationInference:
    """
    Tests full execution behavior of run_continuation_inference.

    Covers:
    - early return mode
    - validation errors
    - mapping + alignment + inference path
    - empty output failure
    """

    # -----------------------------------------------------
    # FIXTURE: base resolver in continuation mode
    # -----------------------------------------------------

    @pytest.fixture
    def make_resolver(
        self,
        valid_cfg,
        tmp_path,
        monkeypatch,
        cutoff_model,
    ):
        def _make(session=None):
            cfg = deepcopy(valid_cfg)

            # minimal mapping
            monkeypatch.setattr(
                rio,
                "map_device_to_continuation_nodes",
                lambda _: {
                    "_layer_1_block_2_output:abc": "/layer.1/block.2/output:abc",
                    "_layer_3_block_4_output:xyz": "/layer.3/block.4/output:xyz",
                },
            )

            # fake session with infer
            class FakeSession:
                input_names = [
                    "/layer.1/block.2/output:abc",
                    "/layer.3/block.4/output:xyz",
                ]

                def get_input_metadata(self):
                    return [
                        type(
                            "Node",
                            (),
                            {
                                "name": n,
                                "shape": (1,),
                            },
                        )()
                        for n in self.input_names
                    ]

                def infer(self, inputs):
                    class Result:
                        outputs = {
                            "final": np.array(
                                [1, 2, 3],
                                dtype=np.float32,
                            )
                        }

                    return Result()

            monkeypatch.setattr(
                rio,
                "create_inference_session",
                lambda *a, **k: session if session is not None else FakeSession(),
            )

            return rio.ResolveInferenceOutputs(
                cfg,
                session_type="ara",
            )

        return _make

    # -----------------------------------------------------
    # EARLY RETURN PATH
    # -----------------------------------------------------

    def test_early_return_when_no_cutoff_or_wrong_backend(self, valid_cfg):
        cfg = deepcopy(valid_cfg)

        resolver = rio.ResolveInferenceOutputs(cfg, session_type="CPU")

        input_data = ModelOutput(outputs={"a": np.array([1])})

        out = resolver.run_continuation_inference(input_data)

        assert out is input_data  # identity return

    # -----------------------------------------------------
    # TYPE VALIDATION
    # -----------------------------------------------------

    def test_invalid_input_type_raises(self, make_resolver):
        with pytest.raises(TypeError):
            make_resolver().run_continuation_inference(cast(ModelOutput, object()))

    # -----------------------------------------------------
    # EMPTY INPUT EDGE CASE
    # -----------------------------------------------------

    def test_empty_outputs_raises_runtime_error_when_no_cutoff(self, valid_cfg):
        cfg = deepcopy(valid_cfg)

        resolver = rio.ResolveInferenceOutputs(cfg, session_type="CPU")

        input_data = ModelOutput(outputs={})

        with pytest.raises(RuntimeError, match="empty"):
            resolver.run_continuation_inference(input_data)

    # -----------------------------------------------------
    # DEVICE COUNT MISMATCH
    # -----------------------------------------------------

    def test_device_output_count_mismatch(self, make_resolver):
        input_data = ModelOutput(
            outputs={
                "_layer_1_block_2_output:abc": np.array([1]),
            }
        )

        with pytest.raises(ValueError):
            make_resolver().run_continuation_inference(input_data)

    # -----------------------------------------------------
    # MISSING DEVICE NODE
    # -----------------------------------------------------

    def test_missing_device_node_raises_keyerror(
        self,
        make_resolver,
    ):
        # Arrange
        resolver = make_resolver()

        input_data = ModelOutput(
            outputs={
                "_layer_1_block_2_output:abc": np.array([1]),
                "_layer_3_block_4_output:xyz": np.array([2]),
            }
        )

        # Break mapping intentionally
        resolver.device_to_continuation_map = {
            "_layer_1_block_2_output:abc": "/layer.1/block.2/output:abc",
            "dev_node_missing": "dev/node.missing",
        }

        # Act / Assert
        with pytest.raises(KeyError):
            resolver.run_continuation_inference(input_data)

    # -----------------------------------------------------
    # EMPTY INFERENCE OUTPUT
    # -----------------------------------------------------

    def test_empty_inference_output_raises(self, make_resolver):
        class FakeSession:
            input_names = ["/layer.1/block.2/output:abc", "/layer.3/block.4/output:xyz"]

            def get_input_metadata(self):
                return [
                    type("Node", (), {"name": n, "shape": (1,)})()
                    for n in self.input_names
                ]

            def infer(self, inputs):
                class Result:
                    outputs = {}

                return Result()

        resolver = make_resolver(session=FakeSession())

        input_data = ModelOutput(
            outputs={
                "_layer_1_block_2_output:abc": np.array([1]),
                "_layer_3_block_4_output:xyz": np.array([2]),
            }
        )

        with pytest.raises(RuntimeError, match="no outputs"):
            resolver.run_continuation_inference(input_data)

    # -----------------------------------------------------
    # HAPPY PATH
    # -----------------------------------------------------

    def test_successful_continuation_inference(self, make_resolver):
        input_data = ModelOutput(
            outputs={
                "_layer_1_block_2_output:abc": np.array([1]),
                "_layer_3_block_4_output:xyz": np.array([2]),
            }
        )

        result = make_resolver().run_continuation_inference(input_data)

        assert hasattr(result, "outputs")
        assert "final" in result.outputs
