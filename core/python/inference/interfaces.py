# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy import ndarray


@dataclass
class QuantizationParams:
    """Quantization parameters for model inputs or outputs.

    This class stores quantization parameters used for quantizing or dequantizing
    tensors. It supports both symmetric and asymmetric quantization schemes.

    Attributes:
        qn: Quantization normalization factor.
        scale: Scale factor for quantization (float).
        offset: Offset for asymmetric quantization.
        is_signed: Whether the quantized values are signed.
        is_float: Whether the original data is floating point.
        qmode: Quantization mode identifier.
    """

    qn: float
    scale: float
    offset: int
    is_signed: bool
    is_float: bool = False
    qmode: int | None = None


@dataclass
class ParameterMetaData:
    """
    Metadata for model parameters (inputs or outputs).

    This class stores metadata about model parameters including name, shape,
    data type, size, and quantization parameters. It is used to describe both
    input and output parameters of inference models.

    Attributes:
        name: The name of the parameter as defined in the model.
        shape: The shape of the parameter tensor. None values indicate dynamic
               dimensions that can vary at runtime.
        dtype: The numpy data type of the parameter. None indicates the data
               type is not specified or unknown.
        size: The size of the parameter tensor in bytes. Optional.
        q_params: Quantization parameters for the parameter. Optional.
    """

    name: str
    shape: list[int | None]
    size: int
    dtype: type[np.generic] | None = None
    q_params: QuantizationParams | None = None


@dataclass
class ModelOutput:
    """
    Container for model inference outputs.

    This class holds the results from a model inference session as a dictionary
    mapping output names to their corresponding numpy arrays. The structure is
    intentionally simple to allow extension for future inference session types
    and additional output metadata as needed.

    Attributes:
        outputs: Dictionary with output names as keys and numpy arrays as values.
        quant_outputs: Optional dictionary with output names as keys and quantized
                       numpy arrays as values. Used when outputs are quantized.

    Properties:
        output_names: List of output names in the order they appear in the model.
    """

    outputs: dict[str, ndarray]
    quant_outputs: dict[str, ndarray] | None = None

    @property
    def output_names(self) -> list[str]:
        """Return the output names as a list of strings."""
        return list(self.outputs.keys())
