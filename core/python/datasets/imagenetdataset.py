# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from typing import Optional, Callable

import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import Dataset, Subset

from core.python.config import Config
from core.python.logger import logger


class ImagenetDataset(Dataset):
    def __init__(
        self,
        config: Config,
        image_dir: str,
        label_dir: str | None = None,
        transform: Optional[Callable] | None = None,
        sample_size: int | None = None,
    ):
        self.config = config
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        dataset = datasets.ImageFolder(root=self.image_dir)

        if sample_size is not None:
            # sampling in case of Resnet models is different than others as in case of Resnet
            # we have imagenet dataset which contains 1000 folders. 1 folder for each class and we
            # should sample from each folder

            # Create class indices mapping
            class_indices = {}
            for idx, (_, label) in enumerate(dataset.samples):
                class_indices.setdefault(label, []).append(idx)

            num_classes = len(class_indices)
            samples_per_class = sample_size // num_classes
            samples_per_class = max(1, samples_per_class)

            logger.info(
                f"Sampling {samples_per_class} examples per class from {num_classes} classes for a total of {samples_per_class * num_classes}."
            )

            selected_indices = []
            for cls, indices in class_indices.items():
                selected_indices.extend(indices[:samples_per_class])

            dataset = Subset(dataset, selected_indices)

        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image = self.dataset[index][0]
        target = self.dataset[index][1]
        if self.transform:
            image = self.transform(
                np.array(image)[:, :, ::-1].copy(), self.config
            ).processed_image

        return torch.tensor(image), target
