# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.python.inference import create_inference_session


@pytest.fixture
def inference_config():
    from core.python.config import InferenceConfig

    return InferenceConfig()


class TestCreateInferenceSession:
    @pytest.fixture
    def mock_file_exists(self):
        with (
            patch("core.python.inference.factory.Path.exists") as mock_exists,
            patch("core.python.inference.factory.Path.is_file") as mock_is_file,
        ):
            mock_exists.return_value = True
            mock_is_file.return_value = True
            yield

    @patch("onnxruntime.InferenceSession")
    def test_create_onnx_session(self, mock_ort, mock_file_exists, inference_config):
        mock_ort_instance = MagicMock()
        mock_ort_instance.get_inputs.return_value = []
        mock_ort_instance.get_outputs.return_value = []
        mock_ort.return_value = mock_ort_instance

        result = create_inference_session("model.onnx", inference_config)

        mock_ort.assert_called_once()
        assert result.model_path == Path("model.onnx")

    @patch("onnxruntime.InferenceSession")
    def test_create_onnx_session_with_kwargs(
        self, mock_ort, mock_file_exists, inference_config
    ):
        mock_ort_instance = MagicMock()
        mock_ort_instance.get_inputs.return_value = []
        mock_ort_instance.get_outputs.return_value = []
        mock_ort.return_value = mock_ort_instance

        result = create_inference_session(
            "model.onnx",
            inference_config,
            providers=["CPUExecutionProvider"],
        )

        mock_ort.assert_called_once_with(
            "model.onnx",
            providers=["CPUExecutionProvider"],
        )
        assert result.model_path == Path("model.onnx")

    @patch("onnxruntime.InferenceSession")
    def test_create_onnx_session_from_path_object(
        self, mock_ort, mock_file_exists, inference_config
    ):
        mock_ort_instance = MagicMock()
        mock_ort_instance.get_inputs.return_value = []
        mock_ort_instance.get_outputs.return_value = []
        mock_ort.return_value = mock_ort_instance

        result = create_inference_session(Path("model.onnx"), inference_config)

        mock_ort.assert_called_once()
        assert result.model_path == Path("model.onnx")

    def test_unsupported_model_type(self, mock_file_exists, inference_config):
        with pytest.raises(ValueError, match="Unsupported model type '.pt'"):
            create_inference_session("model.pt", inference_config)

    def test_unsupported_model_type_error_message(
        self, mock_file_exists, inference_config
    ):
        with pytest.raises(ValueError, match="Supported types:"):
            create_inference_session("model.unknown", inference_config)

    def test_file_not_found(self, inference_config):
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            create_inference_session("nonexistent.onnx", inference_config)

    def test_path_is_directory(self, inference_config):
        with (
            patch("core.python.inference.factory.Path.exists") as mock_exists,
            patch("core.python.inference.factory.Path.is_file") as mock_is_file,
        ):
            mock_exists.return_value = True
            mock_is_file.return_value = False

            with pytest.raises(IsADirectoryError, match="is a directory"):
                create_inference_session("some/directory", inference_config)
