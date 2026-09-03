# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from numpy import ndarray

from core.python.config.config import InferenceConfig

from ..interfaces import ModelOutput, ParameterMetaData
from ..utils import generate_input_feed_from_list

if TYPE_CHECKING:
    from numpy import ndarray


class BaseInferenceSession(ABC):
    """
    Abstract base class for all inference session implementations.

    This class defines the common interface that all inference session types
    must implement. It provides a standardized way to interact with different
    inference backends (ONNX, TensorFlow, PyTorch, etc.) through a unified API.

    Attributes:
        model_path: Path to the model file used by this session.

    Subclasses must implement all abstract methods and properties to provide
    concrete inference functionality for specific model formats.
    """

    def __init__(
        self, model_path: str | Path, config: InferenceConfig | None, **kwargs
    ):
        """
        Initialize the inference session with a model file.

        Args:
            model_path: Path to the model file to load.
            **kwargs: Additional implementation-specific keyword arguments.
        """
        self.model_path = Path(model_path)
        self.config = config

    @property
    @abstractmethod
    def session_type(self) -> str:
        """
        Get the type of inference session.

        Returns:
            A string identifier for the session type (e.g., "onnx", "tensorflow").
        """
        raise NotImplementedError

    @abstractmethod
    def get_input_metadata(self) -> list[ParameterMetaData]:
        """
        Get metadata for all model inputs.

        Returns:
            A list of ParameterMetaData objects describing each input parameter,
            including name, shape, and data type.
        """
        pass

    @abstractmethod
    def get_output_metadata(self) -> list[ParameterMetaData]:
        """
        Get metadata for all model outputs.

        Returns:
            A list of ParameterMetaData objects describing each output parameter,
            including name, shape, and data type.
        """
        pass

    @property
    @abstractmethod
    def input_names(self) -> list[str]:
        """
        Get the names of all model inputs.

        Returns:
            A list of input parameter names in the order they appear in the model.
        """
        pass

    @property
    @abstractmethod
    def output_names(self) -> list[str]:
        """
        Get the names of all model outputs.

        Returns:
            A list of output parameter names in the order they appear in the model.
        """
        pass

    @abstractmethod
    def infer(self, inputs: list[ndarray] | dict[str, ndarray]) -> ModelOutput:
        """
        Run inference on the provided inputs.

        Args:
            inputs: Either a list of numpy arrays (in the same order as model inputs)
                   or a dictionary mapping input names to numpy arrays.

        Returns:
            A ModelOutput object containing the inference results.

        Raises:
            ValueError: If inputs don't match expected shapes or types.
            RuntimeError: If inference fails for any reason.
        """
        pass

    def _preprocess_inputs(
        self, inputs: list[ndarray] | dict[str, ndarray]
    ) -> dict[str, ndarray]:
        """
        Preprocess inputs into the format expected by ONNX Runtime.

        This method converts either a list of numpy arrays or a dictionary
        of named numpy arrays into the input feed dictionary format required
        by ONNX Runtime.

        Args:
            inputs: Either a list of numpy arrays (automatically matched by shape)
                   or a dictionary mapping input names to numpy arrays.

        Returns:
            A dictionary mapping input names to numpy arrays.

        Raises:
            TypeError: If inputs is neither a list nor a dictionary.
            ValueError: If list inputs cannot be matched to model parameters.
        """
        if isinstance(inputs, list):
            # logger.warning(
            #     "Input feed is automatically generated from list and could potentially be incorrect. "
            #     "If unexpected behavior occurs, provide an explicit input feed as a dictionary."
            # )
            input_feed = generate_input_feed_from_list(
                inputs, self.get_input_metadata()
            )

        elif isinstance(inputs, dict):
            # onnx runtime will validate the input feed,
            # no need to manually validate
            input_feed = inputs

        else:
            raise TypeError(
                "Inputs must be a list of ndarrays or a dict of name: ndarray"
            )

        return input_feed
