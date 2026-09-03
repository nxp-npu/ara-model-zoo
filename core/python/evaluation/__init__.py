# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from .base import DatasetSample
from .factory import (
    DATASET_REGISTRY,
    EVALUATOR_REGISTRY,
    get_evaluation_dataset,
    create_evaluator,
)
from .widerface import WiderFaceDataset, WiderFaceEvaluator, evaluate_widerface

__all__ = [
    "DATASET_REGISTRY",
    "DatasetSample",
    "EVALUATOR_REGISTRY",
    "WiderFaceDataset",
    "WiderFaceEvaluator",
    "get_evaluation_dataset",
    "create_evaluator",
    "evaluate_widerface",
]
