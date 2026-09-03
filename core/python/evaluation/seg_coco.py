# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Callable

import torch
import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from core.python.config.config import EvaluationConfig
from core.python.config import Config
from core.python import read_image
from core.python.logger import logger
from core.python.postprocess.detection.coco_classes_labels import (
    COCO_CLASSES,
    COCO_NAMES,
)
from core.python.postprocess.interfaces import SegDetectionOutput

from .base import BaseEvaluationDataset, BaseEvaluator, DatasetSample


def binary_mask_to_rle(binary_mask: np.ndarray) -> dict:
    rle = {"counts": [], "size": list(binary_mask.shape)}

    flattened_mask = binary_mask.ravel(order="F")
    diff_arr = np.diff(flattened_mask)
    nonzero_indices = np.where(diff_arr != 0)[0] + 1
    lengths = np.diff(np.concatenate(([0], nonzero_indices, [len(flattened_mask)])))

    # note that the odd counts are always the numbers of zeros
    if flattened_mask[0] == 1:
        lengths = np.concatenate(([0], lengths))

    rle["counts"] = lengths.astype(int).tolist()

    return rle


# def binary_mask_to_rle(mask, compressed=False):
#     """
#     Convert a binary mask to RLE using vectorized NumPy operations.

#     Args:
#         mask:       numpy array of shape (H, W) with binary values (0 or 1)
#         compressed: if True, returns COCO-style compressed RLE string

#     Returns:
#         dict with:
#           'counts' → list of ints (uncompressed) or encoded string (compressed)
#           'size'   → [height, width]
#     """
#     h, w = mask.shape
#     flat = np.asarray(mask, dtype=np.uint8).flatten(order='F')

#     # Find positions where value changes
#     changes = np.diff(flat, prepend=0, append=0)
#     starts  = np.where(changes != 0)[0]
#     counts  = np.diff(starts).tolist()

#     # If mask starts with 1, prepend a 0-length run of zeros
#     if flat[0] == 1:
#         counts.insert(0, 0)

#     if not compressed:
#         return {'counts': counts, 'size': [h, w]}

#     # COCO LEB128-style encoding (vectorized)
#     encoded = []
#     for count in counts:
#         more = True
#         while more:
#             byte = count & 0x1F
#             count >>= 5
#             if count == 0 and (byte & 0x10) == 0:
#                 more = False
#             else:
#                 byte |= 0x20
#             byte += 48
#             encoded.append(chr(byte))

#     return {'counts': ''.join(encoded), 'size': [h, w]}


def _normalize_coco_split(split: str) -> str:

    normalized = split.strip()
    if normalized in {"train", "val", "test"}:
        return f"{normalized}2017"
    return normalized


def _default_coco_images_dir(root: Path, split: str) -> Path:
    normalized_split = _normalize_coco_split(split)
    candidates = (
        root / normalized_split,
        root / "images" / normalized_split,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return candidates[0].resolve()


def _default_coco_annotation_file(root: Path, split: str) -> Path:
    normalized_split = _normalize_coco_split(split)
    candidates = (
        root / "annotations" / f"instances_{normalized_split}.json",
        root / f"instances_{normalized_split}.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def _model_class_to_coco_category_id(class_id: int) -> int:
    if class_id < 0 or class_id >= len(COCO_CLASSES):
        raise ValueError(
            f"Class id {class_id} is outside the supported COCO class range 0-{len(COCO_CLASSES) - 1}"
        )

    class_name = COCO_CLASSES[class_id]
    try:
        return COCO_NAMES.index(class_name) + 1
    except ValueError as exc:
        raise ValueError(
            f"Unable to map model class '{class_name}' to a COCO category id"
        ) from exc


class SegCocoDataset(BaseEvaluationDataset):
    dataset_type = "seg_coco"
    task = "segmentation"

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
        assert spec is not None, "EvaluationConfig is required"

        self.root = spec.root.expanduser().resolve() if spec.root is not None else None
        self.split = _normalize_coco_split(spec.split)
        self.images_dir = spec.images_dir or (
            _default_coco_images_dir(self.root, self.split) if self.root else None
        )
        self.annotation_file = spec.annotation_file or (
            _default_coco_annotation_file(self.root, self.split) if self.root else None
        )

        if self.images_dir is not None:
            self.images_dir = self.images_dir.expanduser().resolve()
        if self.annotation_file is not None:
            self.annotation_file = self.annotation_file.expanduser().resolve()

        self.validate()
        self.coco = COCO(str(self.annotation_file))
        self._samples = self._load_samples()

    def validate(self) -> None:
        if self.images_dir is None:
            raise ValueError(
                "COCO evaluation requires either --dataset-root or --images-dir."
            )
        if self.annotation_file is None:
            raise ValueError(
                "COCO evaluation requires either --dataset-root or --annotation-file."
            )
        if not self.images_dir.is_dir():
            raise FileNotFoundError(
                f"COCO images directory not found: {self.images_dir}"
            )
        if not self.annotation_file.is_file():
            raise FileNotFoundError(
                f"COCO annotation file not found: {self.annotation_file}"
            )

    def class_map(self) -> dict[int, str]:
        return {idx: name for idx, name in enumerate(COCO_CLASSES)}

    def _load_samples(self) -> list[DatasetSample]:
        samples: list[DatasetSample] = []
        for image_id in sorted(self.coco.getImgIds()):
            image_info = self.coco.loadImgs([image_id])[0]
            image_path = (self.images_dir / image_info["file_name"]).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"COCO image '{image_info['file_name']}' not found under {self.images_dir}"
                )

            annotation_ids = self.coco.getAnnIds(imgIds=[image_id])
            annotations = self.coco.loadAnns(annotation_ids)

            boxes = np.zeros((len(annotations), 4), dtype=np.float32)
            segmentations = np.empty((len(annotations),), dtype=object)
            category_ids = np.zeros((len(annotations),), dtype=np.int32)

            for idx, annotation in enumerate(annotations):
                x, y, width, height = annotation["bbox"]
                boxes[idx] = [x, y, x + width, y + height]
                category_ids[idx] = int(annotation["category_id"])
                segmentations[idx] = annotation["segmentation"]

            samples.append(
                DatasetSample(
                    sample_id=str(image_id),
                    image_path=image_path,
                    ground_truth={
                        "segmentation": segmentations,
                        "image_id": int(image_id),
                        "boxes": boxes,
                        "categories": category_ids,
                        "annotations": annotations,
                    },
                    meta={
                        "image_id": int(image_id),
                        "image_name": image_info["file_name"],
                        "relative_path": image_info["file_name"],
                        "width": int(image_info["width"]),
                        "height": int(image_info["height"]),
                    },
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
            torch.tensor(image),
            torch.tensor(transformed_image),
            self._samples[index],
        )


class SegCocoEvaluator(BaseEvaluator):
    metric_type = "seg_coco"
    iou_type = "segm"

    def __init__(self, spec: EvaluationConfig, output_dir: Path):
        super().__init__(spec, output_dir)

        self.root = spec.root.expanduser().resolve() if spec.root is not None else None
        self.split = _normalize_coco_split(spec.split)
        self.annotation_file = spec.annotation_file or (
            _default_coco_annotation_file(self.root, self.split) if self.root else None
        )
        if self.annotation_file is None:
            raise ValueError(
                "COCO evaluator requires either --dataset-root or --annotation-file."
            )

        self.annotation_file = self.annotation_file.expanduser().resolve()
        if not self.annotation_file.is_file():
            raise FileNotFoundError(
                f"COCO annotation file not found: {self.annotation_file}"
            )

        self.prediction_dir = (
            spec.prediction_dir.expanduser().resolve()
            if spec.prediction_dir is not None
            else (self.output_dir / "predictions").resolve()
        )
        self.prediction_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_path = self.prediction_dir / "detections.json"
        self.summary_path = self.output_dir / "summary.json"
        self.coco_gt = COCO(str(self.annotation_file))
        self.predictions: list[dict[str, Any]] = []

    def add_sample(
        self,
        sample: DatasetSample,
        prediction: SegDetectionOutput | None,
    ) -> None:
        if prediction is None:
            return
        if not isinstance(prediction, SegDetectionOutput):
            raise TypeError(
                "COCO evaluation expects the model postprocess step to return ObjectDetectionOutput"
            )

        image_id = int(sample.ground_truth["image_id"])
        boxes = np.asarray(prediction.boxes, dtype=np.float32)
        scores = np.asarray(prediction.scores, dtype=np.float32).reshape(-1)
        classes = np.asarray(prediction.classes, dtype=np.int32).reshape(-1)
        masks = prediction.masks

        if boxes.size == 0 or scores.size == 0 or classes.size == 0:
            return

        for box, score, class_id, mask in zip(boxes, scores, classes, masks):
            x1, y1, width, height = [float(value) for value in box[:4]]
            # width = max(0.0, x2 - x1)
            # height = max(0.0, y2 - y1)

            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            rle_mask = binary_mask_to_rle(mask)
            self.predictions.append(
                {
                    "image_id": image_id,
                    "category_id": _model_class_to_coco_category_id(int(class_id)),
                    "bbox": [x1, y1, width, height],
                    "score": float(score),
                    "segmentation": rle_mask,
                }
            )

    def evaluate(self) -> dict[str, Any]:
        self.predictions_path.write_text(
            json.dumps(self.predictions, indent=2),
            encoding="utf-8",
        )

        if not self.predictions:
            logger.warning("No detections were produced for COCO evaluation.")
            summary = {
                "ap": 0.0,
                "ap50": 0.0,
                "ap75": 0.0,
                "aps": 0.0,
                "apm": 0.0,
                "apl": 0.0,
                "ar1": 0.0,
                "ar10": 0.0,
                "ar100": 0.0,
                "ars": 0.0,
                "arm": 0.0,
                "arl": 0.0,
                "iou_type": self.iou_type,
                "predictions_path": str(self.predictions_path),
            }
            self.summary_path.write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )
            return summary

        results = {}

        for iou_type in ["bbox", "segm"]:
            coco_dt = self.coco_gt.loadRes(str(self.predictions_path))
            coco_eval = COCOeval(self.coco_gt, coco_dt, iou_type)
            coco_eval.params.imgIds = sorted(
                {item["image_id"] for item in self.predictions}
            )
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            stats = coco_eval.stats
            results[iou_type] = {
                "ap": float(stats[0]),
                "ap50": float(stats[1]),
                "ap75": float(stats[2]),
                "aps": float(stats[3]),
                "apm": float(stats[4]),
                "apl": float(stats[5]),
                "ar1": float(stats[6]),
                "ar10": float(stats[7]),
                "ar100": float(stats[8]),
                "ars": float(stats[9]),
                "arm": float(stats[10]),
                "arl": float(stats[11]),
            }

        summary = {
            "bbox": results["bbox"],
            "segm": results["segm"],
            "predictions_path": str(self.predictions_path),
        }

        self.summary_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        return summary
