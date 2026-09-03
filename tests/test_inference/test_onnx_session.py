# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.python.inference.interfaces import ModelOutput, ParameterMetaData
from core.python.inference.sessions.onnx_session import OnnxInferenceSession
from core.python.inference.utils import (
    _find_names_from_shape,
    generate_input_feed_from_list,
)


@pytest.fixture
def mock_input_meta():
    return [
        ParameterMetaData(
            name="input1",
            shape=[1, 3, 224, 224],
            dtype=np.float32,
            size=1 * 3 * 224 * 224,
        ),
        ParameterMetaData(name="input2", shape=[1, 10], dtype=np.int64, size=1 * 10),
        ParameterMetaData(
            name="input3",
            shape=[1, 3, 224, 224],
            dtype=np.float32,
            size=1 * 3 * 224 * 224,
        ),
    ]


@pytest.fixture
def inference_config():
    from core.python.config.config import InferenceConfig

    return InferenceConfig()


class TestHelperFunctions:
    def test_find_names_from_shape_exact_match(self, mock_input_meta):
        names = _find_names_from_shape([1, 10], mock_input_meta)
        assert names == ["input2"]

    def test_find_names_from_shape_multiple_matches(self, mock_input_meta):
        names = _find_names_from_shape([1, 3, 224, 224], mock_input_meta)
        assert names == ["input1", "input3"]

    def test_find_names_from_shape_no_match(self, mock_input_meta):
        with pytest.raises(ValueError, match="does not match any model input shapes"):
            _find_names_from_shape([1, 5], mock_input_meta)

    def test_generate_input_feed_from_list_success(self, mock_input_meta):
        inputs = [
            np.zeros((1, 10), dtype=np.int64),
            np.zeros((1, 3, 224, 224), dtype=np.float32),
            np.zeros((1, 3, 224, 224), dtype=np.float32),
        ]
        feed = generate_input_feed_from_list(inputs, mock_input_meta)
        assert len(feed) == 3
        assert "input2" in feed
        assert "input1" in feed
        assert "input3" in feed
        assert feed["input2"].shape == (1, 10)

    def test_generate_input_feed_from_list_mismatch_length(self, mock_input_meta):
        inputs = [np.zeros((1, 10), dtype=np.int64)]
        with pytest.raises(ValueError, match="Expected 3 inputs, but got 1"):
            generate_input_feed_from_list(inputs, mock_input_meta)

    def test_generate_input_feed_from_list_no_available_name(self, mock_input_meta):
        inputs = [
            np.zeros((1, 10), dtype=np.int64),
            np.zeros((1, 10), dtype=np.int64),  # Too many of this shape
            np.zeros((1, 3, 224, 224), dtype=np.float32),
        ]
        with pytest.raises(ValueError, match="No available name for input shape"):
            generate_input_feed_from_list(inputs, mock_input_meta)


class TestOnnxInferenceSession:
    @pytest.fixture
    def mock_ort_session(self):
        with patch("onnxruntime.InferenceSession") as mock_session:
            session_instance = MagicMock()

            # Mock inputs
            input1 = MagicMock()
            input1.name = "input1"
            input1.shape = [1, 3, 224, 224]
            input1.type = "tensor(float)"

            # Mock outputs
            output1 = MagicMock()
            output1.name = "output1"
            output1.shape = [1, 1000]
            output1.type = "tensor(float)"

            session_instance.get_inputs.return_value = [input1]
            session_instance.get_outputs.return_value = [output1]

            mock_session.return_value = session_instance
            yield mock_session, session_instance

    def test_init_and_properties(self, mock_ort_session, inference_config):
        mock_session_cls, mock_session_instance = mock_ort_session

        session = OnnxInferenceSession("dummy_model.onnx", inference_config)

        mock_session_cls.assert_called_once_with(
            "dummy_model.onnx", providers=["CPUExecutionProvider"]
        )

        assert session.model_path == Path("dummy_model.onnx")
        assert session.session_type == "onnx"
        assert session.input_names == ["input1"]
        assert session.output_names == ["output1"]

        input_meta = session.get_input_metadata()
        assert len(input_meta) == 1
        assert input_meta[0].name == "input1"
        assert input_meta[0].shape == [1, 3, 224, 224]
        assert input_meta[0].dtype == np.float32

        output_meta = session.get_output_metadata()
        assert len(output_meta) == 1
        assert output_meta[0].name == "output1"
        assert output_meta[0].shape == [1, 1000]
        assert output_meta[0].dtype == np.float32

    def test_parse_metadata_unsupported_dtype(self, mock_ort_session, inference_config):
        _, mock_session_instance = mock_ort_session

        bad_node = MagicMock()
        bad_node.name = "bad_input"
        bad_node.shape = [1]
        bad_node.type = "tensor(unknown)"

        mock_session_instance.get_inputs.return_value = [bad_node]

        with pytest.raises(
            ValueError, match="Could not parse dtype tensor\\(unknown\\)"
        ):
            OnnxInferenceSession("dummy_model.onnx", inference_config)

    def test_preprocess_inputs_dict(self, mock_ort_session, inference_config):
        session = OnnxInferenceSession("dummy_model.onnx", inference_config)

        inputs = {"input1": np.zeros((1, 3, 224, 224), dtype=np.float32)}
        feed = session._preprocess_inputs(inputs)

        assert feed == inputs

    def test_preprocess_inputs_list(self, mock_ort_session, inference_config):
        session = OnnxInferenceSession("dummy_model.onnx", inference_config)

        input_arr = np.zeros((1, 3, 224, 224), dtype=np.float32)
        inputs = [input_arr]
        feed = session._preprocess_inputs(inputs)

        assert "input1" in feed
        np.testing.assert_array_equal(feed["input1"], input_arr)

    def test_preprocess_inputs_invalid_type(self, mock_ort_session, inference_config):
        session = OnnxInferenceSession("dummy_model.onnx", inference_config)

        with pytest.raises(
            TypeError, match="Inputs must be a list of ndarrays or a dict"
        ):
            session._preprocess_inputs(np.zeros((1, 3, 224, 224)))  # type: ignore

    def test_infer(self, mock_ort_session, inference_config):
        _, mock_session_instance = mock_ort_session

        # Setup mock run return value
        mock_output_data = np.ones((1, 1000), dtype=np.float32)
        mock_session_instance.run.return_value = [mock_output_data]

        session = OnnxInferenceSession("dummy_model.onnx", inference_config)

        inputs = {"input1": np.zeros((1, 3, 224, 224), dtype=np.float32)}
        output = session.infer(inputs)

        # Verify run was called correctly
        mock_session_instance.run.assert_called_once_with(
            output_names=["output1"],
            input_feed=inputs,
        )

        assert isinstance(output, ModelOutput)
        assert "output1" in output.outputs
        np.testing.assert_array_equal(output.outputs["output1"], mock_output_data)
