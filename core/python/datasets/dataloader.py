# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import os
import torch
import random

from typing import Optional, Callable
from functools import partial
from torch.utils.data import DataLoader, Subset, Dataset

from .cocodataset import COCODataset

from core.python.config import Config

# Model names that use COCO detection-style train images/labels for PTQ calibration.
_COCO_TRAIN_MODELS = (
    "yolov8n",
    "yolov8n_pose",
    "yolov8s_pose",
    "yolov8m_pose",
    "yolov8l_pose",
    "yolov8x_pose",
)


def _coco_train_dataset_paths(dataset_path: str):
    return (
        COCODataset,
        os.path.join(dataset_path, "images/train2017"),
        os.path.join(dataset_path, "labels/train"),
    )


DATASET_REGISTRY: dict[str, Callable[[str], tuple]] = {
    name: _coco_train_dataset_paths for name in _COCO_TRAIN_MODELS
}


def detection_collate_fn(batch):
    orig_images = [item[0] for item in batch]
    images = [item[1] for item in batch]
    targets = [item[2] for item in batch]
    stacked_images = torch.stack(images, dim=0)
    return orig_images, stacked_images, targets


def calibrate_mode_detection_collate_fn(batch):
    images = [item[1] for item in batch]
    return torch.stack(images, dim=0)


def numpy_collate_fn(batch, batch_size, preprocess_config):
    length = len(batch)

    original_images = [item[0] for item in batch]
    transformed_images = [item[1] for item in batch]
    original_names = [item[2].image_path.name for item in batch]
    original_samples = [item[2] for item in batch]

    if length < batch_size:
        # If length of samples is not multiples of batch_size then there were few samples
        # which have not been processed in above loop. Thus run evaluation on those samples
        REMAINING_SAMPLES = length
        DUMMY_IMAGE = [
            torch.zeros(
                (
                    preprocess_config.input_shape.channels,
                    preprocess_config.input_shape.height,
                    preprocess_config.input_shape.width,
                ),
                dtype=torch.float32,
            )
        ]

        # Since model can only accept images equal to batch_size so we have to
        # append some dummy images to batch to account for that
        DUMMY_IMAGES = DUMMY_IMAGE * (batch_size - REMAINING_SAMPLES)

        # Now append DUMMY_IMAGES to batch
        transformed_images.extend(DUMMY_IMAGES)

    return (
        original_images,
        torch.stack(transformed_images, dim=0),
        original_names,
        original_samples,
    )


def dataloader(
    config: Config,
    dataset_class: Callable[..., Dataset],
    image_dir: str | None = None,
    label_dir: str | None = None,
    batch_size: int = 1,
    num_workers: int = 8,
    transform: Optional[Callable] = None,
    sample_size: Optional[int] = None,
    calibration_mode: bool = False,
    is_numpy: bool = False,
):

    dataset = dataset_class(
        config=config,
        image_dir=image_dir,
        label_dir=label_dir,
        transform=transform,
        sample_size=sample_size,
    )

    if sample_size:
        total_size = len(dataset)  # type: ignore
        sample_indices = random.sample(range(total_size), min(sample_size, total_size))
        dataset = Subset(dataset, sample_indices)

    if is_numpy:
        collate_fn = partial(
            numpy_collate_fn, batch_size=batch_size, preprocess_config=config.preprocess
        )
    elif calibration_mode:
        collate_fn = calibrate_mode_detection_collate_fn
    else:
        collate_fn = detection_collate_fn

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return dataloader
