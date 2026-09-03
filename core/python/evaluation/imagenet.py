# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import json
from typing import cast, Any, Optional, Callable
from pathlib import Path
from tqdm import tqdm

from core.python.evaluation.base import (
    BaseEvaluationDataset,
    BaseEvaluator,
    DatasetSample,
)
from core.python.config.config import EvaluationConfig
from core.python.postprocess.interfaces import (
    PostprocessingOutput,
    ClassificationOutput,
)

from core.python.config import Config
from core.python import read_image

from torchvision import datasets

import torch


def coerce_imagenet_root(root: Path) -> Path:
    nested_root = root / "imagenet"
    return nested_root if nested_root.is_dir() else root


class ImagenetDataset(BaseEvaluationDataset):
    dataset_type = "imagenet"
    task = "classification"

    def __init__(
        self,
        config: Config,
        image_dir: str | None = None,
        label_dir: str | None = None,
        transform: Optional[Callable] | None = None,
        sample_size: int | None = None,
    ):
        spec: EvaluationConfig | None = config.evaluation if config.evaluation else None
        super().__init__(config=config, transform=transform)

        if spec is not None and spec.root is None:
            raise KeyError("IMAGENET dataset spec is missing required key 'root'")

        if spec is not None:
            self.root = (
                coerce_imagenet_root(spec.root.expanduser().resolve())
                if spec.root is not None
                else None
            )
            self.split = spec.split.lower()
            self.images_dir = (
                spec.images_dir or self.root / f"{self.split}"
                if self.root is not None
                else None
            )
            if self.images_dir is not None:
                self.images_dir = self.images_dir.expanduser().resolve()

        self.imagepath = []  # This will be used to store image paths
        self.validate()
        self._samples = self.load_samples()

    def validate(self) -> None:
        if self.split not in {"train", "val"}:
            raise ValueError(
                f"Unsupported Imagenet split '{self.split}'. Supported splits: train, val"
            )
        if self.images_dir is not None and not self.images_dir.is_dir():
            raise FileNotFoundError(
                f"Imagenet images directory not found: {self.images_dir}"
            )

    def load_samples(self) -> list[DatasetSample]:
        samples: list[DatasetSample] = []

        if self.images_dir is not None:
            self.dataset = datasets.ImageFolder(root=self.images_dir)

        total = len(self.dataset)

        for path, label in tqdm(
            self.dataset.samples,
            desc=f"Loading image paths, Total={total}: ",
            unit=" images",
        ):
            samples.append(
                DatasetSample(
                    sample_id=str(label),
                    image_path=Path(path),
                    ground_truth={"value": int(label)},
                )
            )

        return samples

    def __getitem__(self, index: int):
        image = read_image(str(self._samples[index].image_path))

        transformed_image = None
        if self._transform is not None:
            transformed_image = self._transform(
                image.copy(), self._config
            ).processed_image

        return (
            image,
            torch.tensor(transformed_image),
            self._samples[index],
        )


class ImagenetEvaluator(BaseEvaluator):
    metric_type = "imagenet"

    def __init__(self, spec: EvaluationConfig, output_dir: Path):
        super().__init__(spec, output_dir)

        self.top_one_matched = 0
        self.top_five_matched = 0
        self.total = 0
        self.summary_path = self.output_dir / "summary.json"

    def add_sample(
        self,
        sample: DatasetSample,
        prediction: PostprocessingOutput | None,
    ) -> None:
        prediction = cast(ClassificationOutput, prediction)

        if int(prediction.top_n[0]["label"]) == sample.ground_truth["value"]:
            self.top_one_matched += 1

        for i in range(5):
            if int(prediction.top_n[i]["label"]) == sample.ground_truth["value"]:
                self.top_five_matched += 1
                break

        self.total += 1

    def reset(self) -> None:
        self.top_one_matched = 0
        self.top_five_matched = 0
        self.total = 0

    def evaluate(self) -> dict[str, Any]:
        summary: dict[str, Any] = {}

        # Calculate Top_1 and Top_5 accuracy
        top_one_accuracy = (
            0 if self.total == 0 else (self.top_one_matched / self.total) * 100
        )
        top_five_accuracy = (
            0 if self.total == 0 else (self.top_five_matched / self.total) * 100
        )

        # Update summary
        summary.update(
            {
                "Top_1_Accuracy": top_one_accuracy,
                "Top_5_Accuracy": top_five_accuracy,
            }
        )

        self.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary
