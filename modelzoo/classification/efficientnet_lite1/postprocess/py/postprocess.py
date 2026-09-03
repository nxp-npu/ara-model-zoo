# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import numpy as np
import numpy.typing as npt

from core.python.config.config import Config
from core.python.inference import ModelOutput
from core.python.postprocess.classification.classification_utils import (
    extract_topk_predictions,
)
from core.python.postprocess.classification.imagenet_classes_labels import (
    IMAGENET_CLASSES_LABELS,
)
from core.python.postprocess.interfaces import ClassificationOutput


def postprocess(
    model_output: "ModelOutput",
    config: Config,
    original_image: npt.NDArray,
    image_name: str,
) -> ClassificationOutput:
    """Turn the raw model output into a structured classification result.

    The model outputs a single softmax tensor of shape (1, 1000), one score
    per ImageNet class. This function picks the top-K class indices by score
    and wraps them in a ClassificationOutput along with optional human-readable
    class names.
    """
    output_tensor = next(iter(model_output.outputs.values()))

    assert config.postprocess.top_k is not None
    top_n = extract_topk_predictions(
        probabilities=np.asarray(output_tensor).flatten(),
        topk=config.postprocess.top_k,
    )

    for entry in top_n:
        idx = int(entry["label"])
        if idx in IMAGENET_CLASSES_LABELS:
            entry["class_name"] = IMAGENET_CLASSES_LABELS[idx]  # type: ignore

    return ClassificationOutput(
        image_name=image_name,
        top_n=top_n,
    )
