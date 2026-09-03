# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from __future__ import annotations

from pathlib import Path

from core.python.config.config import EvaluationConfig

from .base import BaseEvaluationDataset, BaseEvaluator
from .coco import (
    CocoDataset,
    CocoEvaluator,
    CocoKeypointsDataset,
    CocoKeypointsEvaluator,
)
from .seg_coco import SegCocoDataset, SegCocoEvaluator
from .widerface import WiderFaceDataset, WiderFaceEvaluator
from .imagenet import ImagenetDataset, ImagenetEvaluator


DATASET_REGISTRY: dict[str, type[BaseEvaluationDataset]] = {
    SegCocoDataset.dataset_type: SegCocoDataset,
    CocoDataset.dataset_type: CocoDataset,
    CocoKeypointsDataset.dataset_type: CocoKeypointsDataset,
    WiderFaceDataset.dataset_type: WiderFaceDataset,
    ImagenetDataset.dataset_type: ImagenetDataset,
}

EVALUATOR_REGISTRY: dict[str, type[BaseEvaluator]] = {
    SegCocoEvaluator.metric_type: SegCocoEvaluator,
    CocoEvaluator.metric_type: CocoEvaluator,
    CocoKeypointsEvaluator.metric_type: CocoKeypointsEvaluator,
    WiderFaceEvaluator.metric_type: WiderFaceEvaluator,
    ImagenetEvaluator.metric_type: ImagenetEvaluator,
}


def get_evaluation_dataset(spec: EvaluationConfig) -> type[BaseEvaluationDataset]:
    if not isinstance(spec, EvaluationConfig):
        raise TypeError("Evaluation spec must be an EvaluationConfig instance")

    dataset_type = spec.type
    dataset_cls = DATASET_REGISTRY.get(dataset_type)
    if dataset_cls is None:
        supported = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(
            f"Unsupported evaluation dataset type '{dataset_type}'. Supported types: {supported}"
        )

    return dataset_cls


def create_evaluator(
    spec: EvaluationConfig,
    output_dir: str | Path | None = None,
) -> BaseEvaluator:
    if not isinstance(spec, EvaluationConfig):
        raise TypeError("Evaluation spec must be an EvaluationConfig instance")

    metric_type = spec.type
    evaluator_cls = EVALUATOR_REGISTRY.get(metric_type)
    if evaluator_cls is None:
        supported = ", ".join(sorted(EVALUATOR_REGISTRY))
        raise ValueError(
            f"Unsupported evaluation metric type '{metric_type}'. Supported types: {supported}"
        )

    resolved_output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (Path.cwd() / "evaluation_output").resolve()
    )
    return evaluator_cls(spec, output_dir=resolved_output_dir)
