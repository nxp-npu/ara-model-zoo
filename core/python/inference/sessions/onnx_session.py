# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from ..interfaces import ModelOutput, ParameterMetaData
from .base import BaseInferenceSession

if TYPE_CHECKING:
    from core.python.config.config import InferenceConfig


_ONNX_DTYPE_MAPPING: dict[str, type[np.generic]] = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(uint8)": np.uint8,
    "tensor(int8)": np.int8,
}


def is_onnxruntime_available() -> bool:
    """Return whether the developer-only ONNX Runtime dependency is installed."""
    return importlib.util.find_spec("onnxruntime") is not None


class OnnxInferenceSession(BaseInferenceSession):
    """
    Inference session for ONNX models using ONNX Runtime.

    This class provides an implementation of BaseInferenceSession for ONNX
    models. It uses ONNX Runtime to load and execute models, supporting
    various execution providers (CPU, CUDA, TensorRT, etc.).

    Attributes:
        model_path: Path to the ONNX model file.
        _session: Internal ONNX Runtime inference session.
        _input_metadata: Cached metadata for model inputs.
        _output_metadata: Cached metadata for model outputs.
    """

    def __init__(
        self,
        model_path: str | Path,
        config: InferenceConfig,
        *,
        providers: list[str] = ["CPUExecutionProvider"],
    ):
        """
        Initialize an ONNX inference session.

        Args:
            model_path: Path to the ONNX model file.
            providers: List of execution providers to use, in order of preference.
                      Defaults to CPUExecutionProvider.

        Raises:
            FileNotFoundError: If the model file doesn't exist.
            RuntimeError: If ONNX Runtime fails to load the model.
        """
        super().__init__(model_path, config)

        try:
            ort: Any = importlib.import_module("onnxruntime")
        except ModuleNotFoundError as exc:
            raise ImportError(
                "ONNX inference is available only with the development dependencies. "
                "Install Ara-model-zoo with the 'dev' extra to use ONNX Runtime."
            ) from exc

        self._session = ort.InferenceSession(str(self.model_path), providers=providers)

        # Cache metadata
        self._input_metadata: list[ParameterMetaData] = self._parse_metadata(
            self._session.get_inputs()
        )
        self._output_metadata: list[ParameterMetaData] = self._parse_metadata(
            self._session.get_outputs()
        )

    def _parse_metadata(self, nodes: Sequence[Any]) -> list[ParameterMetaData]:
        """
        Parse ONNX Runtime node information into ParameterMetaData objects.

        Args:
            nodes: List of ONNX Runtime node objects (from get_inputs() or get_outputs()).

        Returns:
            A list of ParameterMetaData objects describing the nodes.

        Raises:
            ValueError: If an unsupported ONNX data type is encountered.
        """
        metadata = []

        for node in nodes:
            numpy_dtype = _ONNX_DTYPE_MAPPING.get(node.type, None)

            if numpy_dtype is None:
                raise ValueError(
                    f"Could not parse dtype {node.type} for node {node.name}"
                )

            size = np.prod(node.shape)

            metadata.append(
                ParameterMetaData(
                    name=node.name, shape=node.shape, dtype=numpy_dtype, size=size
                )
            )
        return metadata

    @property
    def input_names(self) -> list[str]:
        """Get the names of all model inputs."""
        return [m.name for m in self._input_metadata]

    @property
    def output_names(self) -> list[str]:
        """Get the names of all model outputs."""
        return [m.name for m in self._output_metadata]

    @property
    def session_type(self) -> str:
        """Get the type of inference session (always 'onnx' for this class)."""
        return "onnx"

    def get_input_metadata(self) -> list[ParameterMetaData]:
        """Get metadata for all model inputs."""
        return self._input_metadata

    def get_output_metadata(self) -> list[ParameterMetaData]:
        """Get metadata for all model outputs."""
        return self._output_metadata

    def infer(self, inputs: list[np.ndarray] | dict[str, np.ndarray]) -> ModelOutput:
        """
        Run inference on the provided inputs using the ONNX model.

        Args:
            inputs: Either a list of numpy arrays (automatically matched by shape)
                   or a dictionary mapping input names to numpy arrays.

        Returns:
            A ModelOutput object containing the inference results.

        Raises:
            ValueError: If inputs don't match expected shapes or types.
            RuntimeError: If ONNX Runtime fails to execute the model.
        """
        input_feed = self._preprocess_inputs(inputs)

        raw_outputs = self._session.run(
            output_names=self.output_names,
            input_feed=input_feed,
        )

        outputs_dict = {
            name: np.asarray(data) for name, data in zip(self.output_names, raw_outputs)
        }

        return ModelOutput(outputs=outputs_dict)
