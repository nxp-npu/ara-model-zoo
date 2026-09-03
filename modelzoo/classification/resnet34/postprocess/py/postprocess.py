# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import numpy.typing as npt

from core.python.config.config import Config
from core.python.inference import ModelOutput
from core.python.postprocess.classification import generate_topk_predictions
from core.python.postprocess.interfaces import ClassificationOutput


def postprocess(
    model_output: "ModelOutput",
    config: Config,
    original_image: npt.NDArray,
    image_name: str,
) -> ClassificationOutput | None:

    # Extract logits
    logits = list(model_output.outputs.values())[0].squeeze()

    # Only extract Top-K predictions from logits because that is configured in run.yaml
    topk_config = config.postprocess.top_k
    if topk_config is None:
        raise ValueError("top_k must be set in run.yaml")

    topk_predictions = generate_topk_predictions(
        output_tensor=logits,
        topk=topk_config,
    )

    # Wrap probabilities in appropriate output interface
    return ClassificationOutput(top_n=topk_predictions, image_name=image_name)
