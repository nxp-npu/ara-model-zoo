# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

"""
Inference session implementations for Ara-model-zoo.

This subpackage contains concrete implementations of BaseInferenceSession
for different model formats. Each session type provides specialized
functionality for loading and executing models in a specific format.

Modules:
    base: Abstract base class for inference sessions.
    onnx_session: ONNX model inference session using ONNX Runtime.
    ara_session: ARA model inference session.

Classes:
    BaseInferenceSession: Abstract base class for all inference sessions.
    OnnxInferenceSession: Inference session for ONNX models.
    AraInferenceSession: Inference session for ARA models.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ara_session import AraInferenceSession
    from .onnx_session import OnnxInferenceSession

__all__ = ["OnnxInferenceSession", "AraInferenceSession"]
