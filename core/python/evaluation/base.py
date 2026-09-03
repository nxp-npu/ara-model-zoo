# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Callable

from core.python.config.config import EvaluationConfig
from core.python.config import Config

from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetSample:
    sample_id: str
    image_path: Path
    ground_truth: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)


class BaseEvaluationDataset(ABC, Dataset):
    dataset_type = "base"
    task = "generic"

    def __init__(
        self,
        config: Config,
        image_dir: str | None = None,
        label_dir: str | None = None,
        transform: Optional[Callable] | None = None,
        sample_size: int | None = None,
    ):
        self._samples: list[DatasetSample] = []
        self._config = config
        self._transform = transform

    @abstractmethod
    def validate(self) -> None:
        """Validate the dataset specification and on-disk layout."""

    def __iter__(self) -> Iterator[DatasetSample]:
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        return len(self._samples)

    def image_paths(self) -> list[str]:
        return [str(sample.image_path) for sample in self._samples]


class BaseEvaluator(ABC):
    metric_type = "base"

    def __init__(self, spec: EvaluationConfig, output_dir: Path):
        self.spec = spec
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def add_sample(self, sample: DatasetSample, prediction: Any) -> None:
        """Accumulate model predictions for one dataset sample."""

    @abstractmethod
    def evaluate(self) -> dict[str, Any]:
        """Run the final metric computation and return a summary."""

    def reset(self) -> None:
        """Clear all internally accumulated samples/predictions.

        Subclasses that accumulate state (counters, prediction lists, etc.)
        should override this to reset to an empty initial state so the
        evaluator can be reused for a fresh evaluation run.
        """
