# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import os
from typing import Optional, Callable

import torch
import numpy as np
from torch.utils.data import Dataset

from core.python.config import Config
from core.python import read_image


class COCODataset(Dataset):
    def __init__(
        self,
        config: Config,
        image_dir: str,
        label_dir: str,
        transform: Optional[Callable],
        sample_size: int | None = None,
    ):
        self.config = config
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_files = sorted(os.listdir(image_dir))
        self.transform = transform

    def _parse_annotations(self, path):
        label_filename = os.path.splitext(path)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_filename)

        boxes = []
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found at {label_path}")

        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, cx, cy, w, h = map(float, parts)
                boxes.append([class_id, cx, cy, w, h])
        return boxes

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, index: int):
        img_filename = self.img_files[index]
        img_path = os.path.join(self.image_dir, img_filename)
        image = read_image(img_path)
        if self.transform:
            image = self.transform(np.array(image), self.config).processed_image

        boxes = self._parse_annotations(img_filename)
        target = torch.tensor(boxes) if boxes else torch.zeros((0, 5))

        return torch.tensor(image), target
