# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from __future__ import annotations

from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from core.python.config.config import AraProxyInterfaceType

from ..interfaces import (
    ModelOutput,
    ParameterMetaData,
    QuantizationParams,
)
from .base import BaseInferenceSession
from .dvapi_helper import dvapi

if TYPE_CHECKING:
    from core.python.config.config import InferenceConfig

_DEFAULT_SOCKET_PATH = "/var/run/proxy.sock"
_DEFAULT_IPV4_PATH = "127.0.0.1:4096"
_DEFAULT_Q_MODE = 9

_INFERENCE_TIMEOUT = 3600000


class DvApiException(BaseException):
    """Exception raised for DVAPI-related errors.

    This exception is raised when DVAPI operations fail, such as session creation,
    model loading, or inference execution failures.
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


# Mapping from bpp and signed flag to numpy dtype
_BITS_PER_PIXEL_DTYPE_MAPPING = {
    (1, False): np.uint8,
    (1, True): np.int8,
    (2, False): np.uint16,
    (2, True): np.int16,
    (4, False): np.uint32,
    (4, True): np.int32,
}


class ValidQmodes(Enum):
    QMODE_9 = 9
    QMODE_0 = 0


def _determine_array_min_max(dtype: type[np.generic]):
    """Determine the minimum and maximum values for a given integer dtype.

    Args:
        dtype: A numpy integer dtype (e.g., np.int8, np.uint16).

    Returns:
        Tuple of (min_value, max_value) for the given dtype.

    Raises:
        ValueError: If dtype is None or not an integer dtype.

    Note:
        We assume that within ara_session handling, parameter dtype will always be
        an integer. This assumption is validated based on quantization implementation
        in the original model zoo, and from the valid options in dvapi documentation.
        params.dtype should also not be None.
    """
    if dtype is None or not np.issubdtype(dtype, np.integer):
        raise ValueError(f"Expected integer dtype for quantization, got {dtype}")

    dtype_info = np.iinfo(dtype)
    min_val, max_val = dtype_info.min, dtype_info.max

    return min_val, max_val


def quantize_array(array: np.ndarray, params: ParameterMetaData) -> np.ndarray:
    """Quantize a floating-point array to integer representation.

    Args:
        array: Input floating-point numpy array to quantize.
        params: Parameter metadata containing quantization parameters.

    Returns:
        Quantized integer array.

    Raises:
        AssertionError: If q_params or dtype in params is None.

    Note:
        Uses different quantization formulas based on qmode:
        - QMODE_9: output = floor(((array * (1/qn)) + offset) + 0.5)
        - QMODE_0: output = floor(array * qn * scale + 0.5)
        The result is clipped to the dtype's valid range and converted to the target dtype.
    """
    q_params = params.q_params
    assert q_params is not None, "Quantization requires that `q_params` is not None"
    assert params.dtype is not None, (
        "Quantization requires that `params.dtype` is not None"
    )

    qn, offset, scale = q_params.qn, q_params.offset, q_params.scale

    if q_params.qmode == ValidQmodes.QMODE_9.value:
        output = np.floor(((array * (1 / qn)) + offset) + 0.5)
    else:
        output = np.floor(array * qn * scale + 0.5)

    min_val, max_val = _determine_array_min_max(params.dtype)

    output = np.clip(output, min_val, max_val)
    output = output.astype(params.dtype)
    return output


def dequantize_array(array: np.ndarray, params: ParameterMetaData) -> np.ndarray:
    """Dequantize an integer array back to floating-point representation.

    Args:
        array: Input integer numpy array to dequantize.
        params: Parameter metadata containing quantization parameters.

    Returns:
        Dequantized floating-point array (np.float32).

    Raises:
        AssertionError: If q_params or dtype in params is None.

    Note:
        Uses different dequantization formulas based on qmode:
        - QMODE_9: output = (array - offset) * qn
        - QMODE_0: output = (array + offset) / (qn * scale)
        The array is first converted to bytes and back to ensure correct dtype handling.
    """
    q_params = params.q_params
    assert q_params is not None, "Quantization requires that `q_params` is not None"
    assert params.dtype is not None, (
        "Quantization requires that `params.dtype` is not None"
    )

    qn, offset, scale = q_params.qn, q_params.offset, q_params.scale

    array_buffer = array.tobytes()
    output = np.frombuffer(array_buffer, dtype=params.dtype).astype(int)

    if q_params.qmode == ValidQmodes.QMODE_9.value:
        output = (output - offset) * qn
    else:
        output = (output + offset) / (qn * scale)

    output = output.astype(np.float32)
    return output


def _get_dtype_from_bpp(bits_per_pixel: int, is_signed: bool) -> type[np.generic]:
    """Get numpy dtype from bpp and signed flag."""
    dtype = _BITS_PER_PIXEL_DTYPE_MAPPING.get((bits_per_pixel, is_signed))
    if dtype is None:
        raise ValueError(
            f"Unsupported bpp={bits_per_pixel}, is_signed={is_signed} combination"
        )
    return dtype


class AraInferenceSession(BaseInferenceSession):
    """
    Inference session for Ara hardware models using DVAPI.

    This class provides an implementation of BaseInferenceSession for Ara
    hardware models. It uses the DVAPI to load and execute models on Ara
    hardware endpoints.

    Attributes:
        model_path: Path to the model file.
        _session: Internal DVAPI session.
        _model_handle: Loaded model handle.
        _input_metadata: Cached metadata for model inputs.
        _output_metadata: Cached metadata for model outputs.
    """

    def __init__(self, model_path: str | Path, config: InferenceConfig, **kwargs):
        """
        Initialize an Ara hardware inference session.

        Args:
            model_path: Path to the model file.
            **kwargs: Additional keyword arguments:
                - interface_type: Communication interface type (default: system default)
                - socket_path: Unix socket path (default: /var/run/dvproxy.sock)
                - endpoint_index: Endpoint index to use (default: 0)
                - ipv4_ip: IPv4 address and port (default: 127.0.0.1:4096)

        Raises:
            FileNotFoundError: If the model file doesn't exist.
            DvApiException: If DVAPI fails to create session or load model.
        """
        super().__init__(model_path, config)

        self.model_name = self.model_path.name

        self.__init_config_attributes()

        self._session: dvapi.DVSession = self.__create_session()
        self._endpoint_list: list = []
        self._endpoint_list = self.__get_endpoint_list()
        self._model_handle = self.__load_model()

        self.num_inputs: int = self._model_handle.num_inputs
        self.num_outputs: int = self._model_handle.num_outputs

        self._input_metadata: list[ParameterMetaData] = self.__parse_input_metadata()

        if self._input_metadata:
            assert self._input_metadata[0].q_params is not None
            self._qmode = self._input_metadata[0].q_params.qmode
        else:
            self._qmode = _DEFAULT_Q_MODE

        self._output_metadata: list[ParameterMetaData] = self.__parse_output_metadata()

    def __init_config_attributes(self) -> None:
        """Parse configuration items from the inference configuration.

        This method extracts interface settings, socket information, and endpoint
        configuration from the provided InferenceConfig object. It handles both
        IPv4 and socket interface types, validating and parsing the socket string
        for IPv4 connections.

        Raises:
            ValueError: If the IPv4 socket format is invalid or port is not a valid integer.
        """
        self._interface_type: AraProxyInterfaceType = self.config.interface
        self._socket: str = self.config.socket

        if self._interface_type == AraProxyInterfaceType.IPV4:
            # Assume that the socket string is in "ip:port" format
            try:
                ip, port = self._socket.split(":")
            except ValueError:
                raise ValueError(
                    f"Invalid IPv4 socket format: {self._socket}. Expected 'ip:port'"
                )

            self._ipv4_address: str = ip
            try:
                self._ipv4_port: int = int(port)
            except ValueError:
                raise ValueError(f"Invalid port number: {port}")

        self._endpoint_index: int = self.config.endpoint

    def __parse_input_metadata(self) -> list[ParameterMetaData]:
        """Parse input metadata from the loaded model.

        Returns:
            List of ParameterMetaData objects containing metadata for each model input.
            Each metadata includes name, size, shape, dtype, and quantization parameters.

        Note:
            Extracts information from DVAPI input parameters including bits per pixel (bpp),
            signedness, shape dimensions, and quantization parameters.
        """
        param_meta = []
        for i in range(self.num_inputs):
            input_params = self._model_handle.input_param[i]

            bpp = input_params.bpp
            is_signed = input_params.preprocess_param.is_signed
            dtype = _get_dtype_from_bpp(bpp, is_signed)

            shape = [
                input_params.batch_size,
                input_params.nch,
                input_params.height,
                input_params.width,
            ]

            assert hasattr(input_params, "preprocess_param")
            q_params = QuantizationParams(
                qn=input_params.preprocess_param.qn,
                scale=input_params.preprocess_param.output_scale,
                is_signed=input_params.preprocess_param.is_signed,
                offset=input_params.preprocess_param.offset,
                qmode=input_params.preprocess_param.qmode,
            )

            metadata = ParameterMetaData(
                name=input_params.layer_name,
                size=input_params.size,
                shape=shape,
                dtype=dtype,
                q_params=q_params,
            )
            param_meta.append(metadata)
        return param_meta

    def __parse_output_metadata(self) -> list[ParameterMetaData]:
        """Parse output metadata from the loaded model.

        Returns:
            List of ParameterMetaData objects containing metadata for each model output.
            Each metadata includes name, size, shape, dtype, and quantization parameters.

        Note:
            Extracts information from DVAPI output parameters including bits per pixel (bpp),
            signedness, shape dimensions, and quantization parameters. Handles both old
            and new DVAPI versions for signedness detection.
        """
        param_meta = []
        for i in range(self.num_outputs):
            output_params = self._model_handle.output_param[i]

            bits_per_pixel = output_params.bpp
            if hasattr(output_params, "is_output_signed"):
                is_signed = output_params.is_output_signed
            else:
                is_signed = output_params.postprocess_param.is_signed

            dtype = _get_dtype_from_bpp(bits_per_pixel, is_signed)

            # Create shape from model parameters
            if output_params.width != 1:
                shape = [
                    output_params.num,  # batch size
                    output_params.nch,
                    output_params.height,
                    output_params.width,
                ]
            else:
                if output_params.height != 1:
                    shape = [
                        output_params.num,  # batch size
                        output_params.nch,
                        output_params.height,
                    ]
                else:
                    shape = [
                        output_params.num,  # batch size
                        output_params.nch,
                    ]

            q_params = QuantizationParams(
                qn=output_params.postprocess_param.qn,
                scale=output_params.postprocess_param.output_scale,
                offset=output_params.postprocess_param.offset,
                is_signed=is_signed,
                qmode=self._qmode,
                is_float=output_params.postprocess_param.is_float,
            )

            metadata = ParameterMetaData(
                name=output_params.layer_name,
                size=output_params.size,
                shape=shape,
                dtype=dtype,
                q_params=q_params,
            )
            param_meta.append(metadata)
        return param_meta

    def __create_session(self) -> dvapi.DVSession:
        """Create a DVAPI session based on interface type.

        Returns:
            DVSession object for communication with Ara hardware.

        Raises:
            ValueError: If unsupported interface type is specified.
            DvApiException: If DVAPI fails to create the session.

        Note:
            Supports both TCP/IPv4 and Unix socket interfaces based on configuration.
        """
        if self._interface_type == AraProxyInterfaceType.IPV4:
            ip = self._ipv4_address
            port = self._ipv4_port
            ret, session = dvapi.DVSession.create_via_tcp_ipv4_socket(ip, port)

        elif self._interface_type == AraProxyInterfaceType.SOCKET:
            ret, session = dvapi.DVSession.create_via_unix_socket(self._socket)

        else:
            raise ValueError(f"Unsupported interface type: {self._interface_type}")

        if ret != dvapi.dv_status_code.DV_SUCCESS:
            raise DvApiException(
                f"Could not create session: {dvapi.dv_stringify_status_code(ret)}"
            )

        assert session is not None, (
            "DVSession can not be None if dvapi did not return error code"
        )
        return session

    def __get_endpoint_list(self) -> list:
        """Get list of available endpoints.

        Returns:
            List of available Ara hardware endpoints.

        Raises:
            DvApiException: If DVAPI fails to retrieve the endpoint list.
        """
        ret, endpoint_list = self._session.get_endpoint_list()

        if ret != dvapi.dv_status_code.DV_SUCCESS:
            raise DvApiException(
                f"Could not load endpoints: {dvapi.dv_stringify_status_code(ret)}"
            )

        assert endpoint_list is not None, (
            "Endpoint List can not be None if dvapi did not return error code"
        )
        return endpoint_list[:]

    def _get_endpoint(self):
        """Get the endpoint to use for inference."""
        if not self._endpoint_list:
            raise RuntimeError("No endpoints available")

        if self._endpoint_index is None:
            return self._endpoint_list[0]

        if 0 <= self._endpoint_index < len(self._endpoint_list):
            return self._endpoint_list[self._endpoint_index]

        raise IndexError(
            f"Endpoint index {self._endpoint_index} is out of bounds. "
            f"Only {len(self._endpoint_list)} endpoints available"
        )

    def __load_model(self) -> dvapi.DVModel:
        """Load the model onto the endpoint."""
        endpoint = self._get_endpoint()
        model_path = str(self.model_path)

        ret, model_handle = self._session.load_model_from_file(
            endpoint, model_path, self.model_name
        )

        if ret != dvapi.dv_status_code.DV_SUCCESS:
            raise DvApiException(
                f"Could not load model: {dvapi.dv_stringify_status_code(ret)}"
            )

        return model_handle

    def _create_input_tensors(
        self, input_feed: dict[str, np.ndarray]
    ) -> list[dvapi.DVTensor]:
        """Create DVTensor objects from input feed."""
        input_tensors = []

        for i, params in enumerate(self.get_input_metadata()):
            input_name = params.name

            if input_name not in input_feed:
                raise ValueError(f"Missing input for parameter: {input_name}")
            input_data = input_feed[input_name]

            quantized_array = quantize_array(input_data, params)
            quantized_buffer = quantized_array.flatten()

            if quantized_buffer.shape[0] != params.size:
                raise ValueError(
                    f"Quantized buffer size mismatch for {input_name}"
                    f"Expected {params.size}, got {quantized_buffer.shape[0]}"
                )

            assert quantized_buffer.dtype == params.dtype

            dv_tensor = dvapi.DVTensor(
                quantized_buffer, self._model_handle.input_param[i]
            )

            input_tensors.append(dv_tensor)

        return input_tensors

    def _create_output_tensors(self) -> list[dvapi.DVTensor]:
        """Create DVTensor objects for model outputs.

        Returns:
            List of DVTensor objects initialized with zero buffers for each output.
            Each tensor is created with the appropriate dtype and size from the output metadata.
        """
        output_tensors = []

        for i, params in enumerate(self.get_output_metadata()):
            output_buffer = np.zeros(params.size, np.int8)
            output_tensors.append(
                dvapi.DVTensor(output_buffer, self._model_handle.output_param[i])
            )
        return output_tensors

    @property
    def session_type(self) -> str:
        """Get the type of inference session (always 'ara' for this class)."""
        return "ara"

    def get_input_metadata(self) -> list[ParameterMetaData]:
        """Get metadata for all model inputs."""
        return self._input_metadata

    def get_output_metadata(self) -> list[ParameterMetaData]:
        """Get metadata for all model outputs."""
        return self._output_metadata

    @property
    def input_names(self) -> list[str]:
        """Get the names of all model inputs."""
        return [m.name for m in self._input_metadata]

    @property
    def output_names(self) -> list[str]:
        """Get the names of all model outputs."""
        return [m.name for m in self._output_metadata]

    def infer(self, inputs: list[np.ndarray] | dict[str, np.ndarray]) -> ModelOutput:
        """
        Run inference on the provided inputs using the Ara hardware.

        Args:
            inputs: Either a list of numpy arrays (automatically matched by shape)
                   or a dictionary mapping input names to numpy arrays.

        Returns:
            A ModelOutput object containing the inference results.

        Raises:
            ValueError: If inputs don't match expected shapes or types.
            DvApiException: If DVAPI fails to execute the inference.
        """

        input_feed = self._preprocess_inputs(inputs)
        input_tensors = self._create_input_tensors(input_feed)
        output_tensors = self._create_output_tensors()

        # Run synchronous inference
        status, infer_request = self._model_handle.infer_async(
            input_tensors=input_tensors, output_tensors=output_tensors
        )

        if status != dvapi.dv_status_code.DV_SUCCESS:
            raise DvApiException(
                f"Inference failed: {dvapi.dv_stringify_status_code(status)}"
            )

        # Wait for completion
        status = infer_request.wait_for_completion(timeout=_INFERENCE_TIMEOUT)
        if status != dvapi.dv_status_code.DV_SUCCESS:
            raise DvApiException(
                f"Inference wait failed: {dvapi.dv_stringify_status_code(status)}"
            )

        dequant_output_dict = {}
        quant_output_dict = {}
        for i, (output_tensor, param) in enumerate(
            zip(output_tensors, self.get_output_metadata())
        ):
            output_name = param.name
            assert output_tensor.params.layer_name == output_name
            assert output_name == output_tensor.params.layer_name, (
                "Assumed output_tensor and metadata follow the same Error."
                " If this assertion is hit than this assumption has failed"
            )

            quant_output_array = deepcopy(output_tensor.numpy_data)
            dequant_output_array = dequantize_array(quant_output_array, param)
            dequant_output_array = dequant_output_array.reshape(param.shape)  # type: ignore

            dequant_output_dict[output_name] = dequant_output_array
            quant_output_dict[output_name] = quant_output_array

        return ModelOutput(
            outputs=dequant_output_dict,
            quant_outputs=quant_output_dict,
        )

    def __del__(self):
        """Cleanup resources when session is destroyed."""
        if self._model_handle is not None:
            self._model_handle.__del__()
        if self._session is not None:
            self._session.__del__()
