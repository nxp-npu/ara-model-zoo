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


class WiderFaceDataset(Dataset):
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
        self.img_files = []
        self.transform = transform
        self._parse_annotations(self.label_dir)

    def _parse_annotations(self, path):
        with open(path, "r") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            img_path = lines[i].strip()
            num_faces = int(lines[i + 1].strip())

            boxes = []
            start_idx = i + 2
            if num_faces == 0:
                i += 3
                continue

            for j in range(num_faces):
                parts = list(map(int, lines[start_idx + j].strip().split()))
                boxes.append(parts[:4])

            self.img_files.append({"img_path": img_path, "boxes": boxes})

            i = start_idx + num_faces

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, index: int):
        info = self.img_files[index]
        img_path = os.path.join(self.image_dir, info["img_path"])
        image = read_image(img_path)
        if self.transform:
            image = self.transform(np.array(image), self.config).processed_image

        return torch.tensor(image), torch.tensor(info["boxes"])
