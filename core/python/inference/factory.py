# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import importlib

if TYPE_CHECKING:
    from core.python.config import InferenceConfig

    from .sessions.base import BaseInferenceSession


SESSION_REGISTRY: dict[str, dict[str, str]] = {
    ".onnx": {
        "module": "core.python.inference.sessions.onnx_session",
        "class": "OnnxInferenceSession",
    },
    ".dvm": {
        "module": "core.python.inference.sessions.ara_session",
        "class": "AraInferenceSession",
    },
}


def create_inference_session(
    model_path: str | Path,
    config: InferenceConfig | None,
    **kwargs,
) -> "BaseInferenceSession":
    """
    Factory function to create an inference session based on model type.

    Automatically detects the model type from the file extension and returns
    the appropriate BaseInferenceSession subclass instance.

    Args:
        model_path: Path to the model file (str or Path).
        **kwargs: Additional keyword arguments forwarded to the session constructor.

    Returns:
        An instance of a BaseInferenceSession subclass for the detected model type.

    Raises:
        FileNotFoundError: If the model file does not exist.
        IsADirectoryError: If the model path points to a directory.
        ValueError: If the file extension is not in SESSION_REGISTRY.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: '{model_path}'")

    if not model_path.is_file():
        raise IsADirectoryError(f"Model path '{model_path}' is a directory, not a file")

    suffix = model_path.suffix.lower()
    session_info = SESSION_REGISTRY.get(suffix)

    if session_info is None:
        supported = ", ".join(f"'{ext}'" for ext in SESSION_REGISTRY)
        raise ValueError(
            f"Unsupported model type '{suffix}'. Supported types: {supported}"
        )

    module = importlib.import_module(session_info["module"])
    session_class = getattr(module, session_info["class"])

    return session_class(model_path, config, **kwargs)
