# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

"""
Inference module for Ara-model-zoo.

This module provides a unified interface for running inference with various
model formats. It includes base classes for inference sessions, implementations
for specific model types, and common data structures for handling model inputs
and outputs.

Modules:
    interfaces: Common data structures and enums for inference.
    sessions: Inference session implementations for different model formats.
    factory: Factory function for creating inference sessions.

Classes:
    ParameterMetaData: Metadata for model parameters (inputs or outputs).
    ModelOutput: Container for model inference outputs.
    AraProxyInterfaceType: Communication interface types for AraProxy sessions.
    OnnxInferenceSession: Inference session for ONNX models.

Functions:
    create_inference_session: Factory function to create inference sessions.

Constants:
    DEFAULT_INTERFACE_TYPE: Default communication interface based on OS.

Examples:
    Create an inference session automatically from model path:

    >>> session = create_inference_session("model.onnx", inference_config)

    `inference_config` is an instance of `core.python.config.InferenceConfig`

    >>> result = session.infer({"input_name": numpy_array})

    Or explicitly use a specific session type:

    >>> session = OnnxInferenceSession("model.onnx", providers=["CPUExecutionProvider"])
    >>> result = session.infer([numpy_array])
"""

from .factory import create_inference_session
from .interfaces import (
    ModelOutput,
    ParameterMetaData,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sessions import AraInferenceSession, OnnxInferenceSession

__all__ = [
    "ModelOutput",
    "ParameterMetaData",
    "create_inference_session",
]

if TYPE_CHECKING:
    __all__ += ["OnnxInferenceSession", "AraInferenceSession"]
