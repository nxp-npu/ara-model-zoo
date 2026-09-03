# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import numpy as np
import numpy.typing as npt

from core.python.config.config import Config
from core.python.inference import ModelOutput
from core.python.postprocess.classification import extract_topk_predictions
from core.python.postprocess.interfaces import ClassificationOutput


def postprocess(
    model_output: "ModelOutput",
    config: "Config",
    original_image: npt.NDArray,
    image_name: str,
) -> "ClassificationOutput | None":

    # ---------------------------------------------------------
    # Extract model probability tensor
    # Single-head classification assumption
    # ---------------------------------------------------------
    (probabilities,) = model_output.outputs.values()

    # Remove trailing spatial singleton dimensions:
    if (
        probabilities.ndim == 4
        and probabilities.shape[-1] == 1
        and probabilities.shape[-2] == 1
    ):
        probabilities = probabilities.squeeze(axis=(-1, -2))

    # Ensure shape is (num_classes,)
    if probabilities.ndim == 2:
        probabilities = probabilities[0]

    # ---------------------------------------------------------
    # Extract Top-K predictions
    # ---------------------------------------------------------
    topk_config = config.postprocess.top_k
    if topk_config is None:
        raise ValueError("top_k must be set in config.postprocess")

    topk_predictions = extract_topk_predictions(
        probabilities=probabilities,
        topk=topk_config,
    )

    # ---------------------------------------------------------
    # Apply legacy label alignment shift
    # ---------------------------------------------------------
    for pred in topk_predictions:
        pred["label"] = np.int64(pred["label"] - 1)

    # ---------------------------------------------------------
    # Final structured output
    # ---------------------------------------------------------
    return ClassificationOutput(
        top_n=topk_predictions,
        image_name=image_name,
    )
