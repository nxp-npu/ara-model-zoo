# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional, Callable

import torch
import numpy as np

from core.python.config.config import EvaluationConfig
from core.python.config import Config
from core.python import read_image
from core.python.logger import logger
from core.python.postprocess.interfaces import FaceDetectionOutput

from .base import BaseEvaluationDataset, BaseEvaluator, DatasetSample


def _coerce_widerface_root(root: Path) -> Path:
    nested_root = root / "widerface"
    return nested_root if nested_root.is_dir() else root


def _default_widerface_gt_dir(root: Path) -> Path:
    root = _coerce_widerface_root(root)
    candidates = (
        root / "eval_tools" / "eval_tools" / "ground_truth",
        root / "eval_tools" / "ground_truth",
        root / "ground_truth",
        root / "wider_face_split",
    )

    for candidate in candidates:
        if (candidate / "wider_face_val.mat").is_file():
            return candidate.resolve()

    return candidates[0].resolve()


class WiderFaceDataset(BaseEvaluationDataset):
    dataset_type = "widerface"
    task = "face_detection"

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

        if spec.root is None:
            raise KeyError("WIDER FACE dataset spec is missing required key 'root'")

        self.root = _coerce_widerface_root(spec.root.expanduser().resolve())
        self.split = spec.split.lower()
        self.images_dir = (
            (spec.images_dir or self.root / f"WIDER_{self.split}" / "images")
            .expanduser()
            .resolve()
        )
        self.annotation_file = (
            (
                spec.annotation_file
                or self.root
                / "wider_face_split"
                / f"wider_face_{self.split}_bbx_gt.txt"
            )
            .expanduser()
            .resolve()
        )

        self.validate()
        self._samples = self._load_samples()

    def validate(self) -> None:
        if self.split not in {"train", "val"}:
            raise ValueError(
                f"Unsupported WIDER FACE split '{self.split}'. Supported splits: train, val"
            )
        if not self.images_dir.is_dir():
            raise FileNotFoundError(
                f"WIDER FACE images directory not found: {self.images_dir}"
            )
        if not self.annotation_file.is_file():
            raise FileNotFoundError(
                f"WIDER FACE annotation file not found: {self.annotation_file}"
            )

    def class_map(self) -> dict[int, str]:
        return {0: "face"}

    def _load_samples(self) -> list[DatasetSample]:
        lines = self.annotation_file.read_text(encoding="utf-8").splitlines()
        samples: list[DatasetSample] = []
        idx = 0

        while idx < len(lines):
            relative_path = lines[idx].strip()
            idx += 1

            if not relative_path:
                continue

            if idx >= len(lines):
                raise ValueError(
                    f"Malformed WIDER FACE annotation file: missing box count for {relative_path}"
                )

            try:
                num_boxes = int(lines[idx].strip())
            except ValueError as exc:
                raise ValueError(
                    f"Malformed WIDER FACE annotation file: invalid box count for {relative_path}"
                ) from exc
            idx += 1

            boxes = np.zeros((num_boxes, 4), dtype=np.float32)
            annotations: list[list[int]] = []

            for box_idx in range(num_boxes):
                if idx >= len(lines):
                    raise ValueError(
                        f"Malformed WIDER FACE annotation file: truncated annotations for {relative_path}"
                    )

                values = [int(token) for token in lines[idx].split()]
                idx += 1

                if len(values) < 4:
                    raise ValueError(
                        f"Malformed WIDER FACE annotation file: expected at least 4 values for {relative_path}"
                    )

                boxes[box_idx] = values[:4]
                annotations.append(values)

            image_path = (self.images_dir / relative_path).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"WIDER FACE image '{relative_path}' not found under {self.images_dir}"
                )

            path_obj = Path(relative_path)
            event = path_obj.parent.name or "default"

            samples.append(
                DatasetSample(
                    sample_id=relative_path,
                    image_path=image_path,
                    ground_truth={
                        "boxes": boxes,
                        "annotations": annotations,
                    },
                    meta={
                        "event": event,
                        "image_name": path_obj.name,
                        "image_stem": path_obj.stem,
                        "relative_path": relative_path,
                        "split": self.split,
                        "num_faces": num_boxes,
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


def _bbox_overlaps(boxes: np.ndarray, query_boxes: np.ndarray) -> np.ndarray:
    overlaps = np.zeros((boxes.shape[0], query_boxes.shape[0]), dtype=np.float32)

    for query_idx in range(query_boxes.shape[0]):
        query_area = (query_boxes[query_idx, 2] - query_boxes[query_idx, 0] + 1) * (
            query_boxes[query_idx, 3] - query_boxes[query_idx, 1] + 1
        )

        for box_idx in range(boxes.shape[0]):
            inter_w = (
                min(boxes[box_idx, 2], query_boxes[query_idx, 2])
                - max(boxes[box_idx, 0], query_boxes[query_idx, 0])
                + 1
            )
            if inter_w <= 0:
                continue

            inter_h = (
                min(boxes[box_idx, 3], query_boxes[query_idx, 3])
                - max(boxes[box_idx, 1], query_boxes[query_idx, 1])
                + 1
            )
            if inter_h <= 0:
                continue

            union_area = float(
                (boxes[box_idx, 2] - boxes[box_idx, 0] + 1)
                * (boxes[box_idx, 3] - boxes[box_idx, 1] + 1)
                + query_area
                - inter_w * inter_h
            )
            overlaps[box_idx, query_idx] = inter_w * inter_h / union_area

    return overlaps


def _load_widerface_ground_truth(gt_dir: Path):
    from scipy.io import loadmat

    gt_mat = loadmat(gt_dir / "wider_face_val.mat")
    hard_mat = loadmat(gt_dir / "wider_hard_val.mat")
    medium_mat = loadmat(gt_dir / "wider_medium_val.mat")
    easy_mat = loadmat(gt_dir / "wider_easy_val.mat")

    return (
        gt_mat["face_bbx_list"],
        gt_mat["event_list"],
        gt_mat["file_list"],
        hard_mat["gt_list"],
        medium_mat["gt_list"],
        easy_mat["gt_list"],
    )


def _unwrap_mat_value(value: Any) -> Any:
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.reshape(-1)[0]
    return value


def _mat_sequence(value: Any) -> list[Any]:
    value = _unwrap_mat_value(value)
    if isinstance(value, np.ndarray) and value.dtype == object:
        return list(value.reshape(-1))
    return [value]


def _mat_string(value: Any) -> str:
    value = _unwrap_mat_value(value)
    if isinstance(value, np.ndarray):
        return str(value.reshape(-1)[0])
    return str(value)


def _mat_array(value: Any, dtype: np.dtype | type[np.generic]) -> np.ndarray:
    value = _unwrap_mat_value(value)
    return np.asarray(value, dtype=dtype)


def _read_prediction_file(file_path: Path) -> tuple[str, np.ndarray]:
    lines = file_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Malformed prediction file: {file_path}")

    image_name = Path(lines[0].strip()).stem
    box_lines = lines[2:]

    boxes: list[list[float]] = []
    for line in box_lines:
        tokens = [token for token in line.strip().split(" ") if token]
        if len(tokens) < 5:
            continue
        boxes.append([float(token) for token in tokens[:5]])

    if not boxes:
        return image_name, np.zeros((0, 5), dtype=np.float32)

    return image_name, np.asarray(boxes, dtype=np.float32)


def _load_prediction_directory(
    prediction_dir: Path,
) -> dict[str, dict[str, np.ndarray]]:
    predictions: dict[str, dict[str, np.ndarray]] = {}

    if not prediction_dir.exists():
        raise FileNotFoundError(
            f"WIDER FACE prediction directory not found: {prediction_dir}"
        )

    for event_dir in sorted(path for path in prediction_dir.iterdir() if path.is_dir()):
        event_predictions: dict[str, np.ndarray] = {}
        for prediction_file in sorted(event_dir.glob("*.txt")):
            image_name, boxes = _read_prediction_file(prediction_file)
            event_predictions[image_name] = boxes
        predictions[event_dir.name] = event_predictions

    return predictions


def _normalize_scores(predictions: dict[str, dict[str, np.ndarray]]) -> None:
    max_score: float | None = None
    min_score: float | None = None

    for event_predictions in predictions.values():
        for boxes in event_predictions.values():
            if len(boxes) == 0:
                continue

            current_min = float(np.min(boxes[:, -1]))
            current_max = float(np.max(boxes[:, -1]))

            min_score = (
                current_min if min_score is None else min(min_score, current_min)
            )
            max_score = (
                current_max if max_score is None else max(max_score, current_max)
            )

    if min_score is None or max_score is None or max_score <= min_score:
        return

    diff = max_score - min_score
    for event_predictions in predictions.values():
        for boxes in event_predictions.values():
            if len(boxes) == 0:
                continue
            boxes[:, -1] = (boxes[:, -1] - min_score) / diff


def _image_eval(
    pred: np.ndarray,
    gt: np.ndarray,
    ignore: np.ndarray,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    if pred.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    pred_boxes = pred.copy()
    gt_boxes = gt.copy()

    pred_recall = np.zeros(pred_boxes.shape[0], dtype=np.float32)
    recall_list = np.zeros(gt_boxes.shape[0], dtype=np.float32)
    proposal_list = np.ones(pred_boxes.shape[0], dtype=np.float32)

    pred_boxes[:, 2] = pred_boxes[:, 2] + pred_boxes[:, 0]
    pred_boxes[:, 3] = pred_boxes[:, 3] + pred_boxes[:, 1]
    gt_boxes[:, 2] = gt_boxes[:, 2] + gt_boxes[:, 0]
    gt_boxes[:, 3] = gt_boxes[:, 3] + gt_boxes[:, 1]

    overlaps = _bbox_overlaps(pred_boxes[:, :4], gt_boxes)

    for pred_idx in range(pred_boxes.shape[0]):
        gt_overlap = overlaps[pred_idx]
        max_overlap = gt_overlap.max()
        max_idx = gt_overlap.argmax()

        if max_overlap >= iou_threshold:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[pred_idx] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1

        pred_recall[pred_idx] = float(np.sum(recall_list == 1))

    return pred_recall, proposal_list


def _img_pr_info(
    threshold_count: int,
    pred_info: np.ndarray,
    proposal_list: np.ndarray,
    pred_recall: np.ndarray,
) -> np.ndarray:
    pr_info = np.zeros((threshold_count, 2), dtype=np.float32)

    for threshold_idx in range(threshold_count):
        threshold = 1 - (threshold_idx + 1) / threshold_count
        keep_indices = np.where(pred_info[:, 4] >= threshold)[0]

        if len(keep_indices) == 0:
            continue

        last_index = keep_indices[-1]
        proposal_indices = np.where(proposal_list[: last_index + 1] == 1)[0]
        pr_info[threshold_idx, 0] = len(proposal_indices)
        pr_info[threshold_idx, 1] = pred_recall[last_index]

    return pr_info


def _dataset_pr_info(
    threshold_count: int,
    pr_curve: np.ndarray,
    count_face: int,
) -> np.ndarray:
    dataset_curve = np.zeros((threshold_count, 2), dtype=np.float32)

    for threshold_idx in range(threshold_count):
        proposals = pr_curve[threshold_idx, 0]
        recalls = pr_curve[threshold_idx, 1]

        dataset_curve[threshold_idx, 0] = recalls / proposals if proposals > 0 else 0.0
        dataset_curve[threshold_idx, 1] = (
            recalls / count_face if count_face > 0 else 0.0
        )

    return dataset_curve


def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    extended_recall = np.concatenate(([0.0], recall, [1.0]))
    extended_precision = np.concatenate(([0.0], precision, [0.0]))

    for idx in range(extended_precision.size - 1, 0, -1):
        extended_precision[idx - 1] = max(
            extended_precision[idx - 1],
            extended_precision[idx],
        )

    recall_change_indices = np.where(extended_recall[1:] != extended_recall[:-1])[0]
    return float(
        np.sum(
            (
                extended_recall[recall_change_indices + 1]
                - extended_recall[recall_change_indices]
            )
            * extended_precision[recall_change_indices + 1]
        )
    )


def evaluate_widerface(
    prediction_dir: Path,
    gt_dir: Path,
    iou_threshold: float = 0.5,
) -> dict[str, int | float | str]:
    predictions = _load_prediction_directory(prediction_dir)
    _normalize_scores(predictions)

    (
        facebox_list,
        event_list,
        file_list,
        hard_gt_list,
        medium_gt_list,
        easy_gt_list,
    ) = _load_widerface_ground_truth(gt_dir)

    settings = (
        ("easy_ap", easy_gt_list),
        ("medium_ap", medium_gt_list),
        ("hard_ap", hard_gt_list),
    )

    results: dict[str, int | float | str] = {}
    threshold_count = 1000

    for metric_name, gt_list in settings:
        count_face = 0
        pr_curve = np.zeros((threshold_count, 2), dtype=np.float32)

        event_entries = _mat_sequence(event_list)
        file_entries = _mat_sequence(file_list)
        face_entries = _mat_sequence(facebox_list)
        subset_entries = _mat_sequence(gt_list)

        for (
            event_name_entry,
            image_list_entry,
            gt_box_list_entry,
            subset_gt_entry,
        ) in zip(
            event_entries,
            file_entries,
            face_entries,
            subset_entries,
        ):
            event_name = _mat_string(event_name_entry)
            prediction_by_image = predictions.get(event_name, {})

            image_list = _mat_sequence(image_list_entry)
            gt_box_list = _mat_sequence(gt_box_list_entry)
            subset_gt_list = _mat_sequence(subset_gt_entry)

            for image_name_entry, gt_box_entry, keep_index_entry in zip(
                image_list,
                gt_box_list,
                subset_gt_list,
            ):
                image_name = _mat_string(image_name_entry)
                pred_info = prediction_by_image.get(
                    image_name,
                    np.zeros((0, 5), dtype=np.float32),
                )
                gt_boxes = _mat_array(gt_box_entry, np.float32)
                keep_index = _mat_array(keep_index_entry, np.int64).reshape(-1)
                count_face += len(keep_index)

                if len(gt_boxes) == 0 or len(pred_info) == 0:
                    continue

                ignore = np.zeros(gt_boxes.shape[0], dtype=np.float32)
                if len(keep_index) != 0:
                    ignore[keep_index - 1] = 1

                pred_recall, proposal_list = _image_eval(
                    pred_info,
                    gt_boxes,
                    ignore,
                    iou_threshold,
                )
                pr_curve += _img_pr_info(
                    threshold_count,
                    pred_info,
                    proposal_list,
                    pred_recall,
                )

        dataset_curve = _dataset_pr_info(threshold_count, pr_curve, count_face)
        precision = dataset_curve[:, 0]
        recall = dataset_curve[:, 1]
        results[metric_name] = _voc_ap(recall, precision)

    logger.info("==================== WIDER FACE Results ====================")
    logger.info("Easy   Val AP: %.5f", results["easy_ap"])
    logger.info("Medium Val AP: %.5f", results["medium_ap"])
    logger.info("Hard   Val AP: %.5f", results["hard_ap"])
    logger.info("============================================================")

    return results


class WiderFaceEvaluator(BaseEvaluator):
    metric_type = "widerface"

    def __init__(self, spec: EvaluationConfig, output_dir: Path):
        super().__init__(spec, output_dir)

        if spec.gt_dir is not None:
            self.gt_dir = spec.gt_dir.expanduser().resolve()
        elif spec.root is not None:
            self.gt_dir = _default_widerface_gt_dir(spec.root.expanduser().resolve())
        else:
            raise KeyError(
                "WIDER FACE evaluation spec requires either 'gt_dir' or 'root'"
            )

        self.prediction_dir = (
            spec.prediction_dir.expanduser().resolve()
            if spec.prediction_dir is not None
            else (self.output_dir / "predictions").resolve()
        )
        self.iou_threshold = float(spec.iou_threshold)
        self.summary_path = self.output_dir / "summary.json"
        self._predictions: dict[str, dict[str, dict[str, Any]]] = {}

        self.validate()

    def validate(self) -> None:
        if not self.gt_dir.is_dir():
            raise FileNotFoundError(
                f"WIDER FACE ground-truth directory not found: {self.gt_dir}"
            )

        required_files = (
            "wider_face_val.mat",
            "wider_easy_val.mat",
            "wider_medium_val.mat",
            "wider_hard_val.mat",
        )
        for file_name in required_files:
            if not (self.gt_dir / file_name).is_file():
                raise FileNotFoundError(
                    f"Required WIDER FACE ground-truth file not found: {self.gt_dir / file_name}"
                )

    def add_sample(
        self,
        sample: DatasetSample,
        prediction: FaceDetectionOutput | None,
    ) -> None:
        boxes = self._extract_boxes(prediction)
        event = str(sample.meta.get("event") or "default")
        image_stem = str(sample.meta.get("image_stem") or sample.image_path.stem)
        relative_path = str(sample.meta.get("relative_path") or sample.image_path.name)

        event_predictions = self._predictions.setdefault(event, {})
        event_predictions[image_stem] = {
            "relative_path": relative_path,
            "boxes": self._to_widerface_boxes(boxes),
        }

    def evaluate(self) -> dict[str, Any]:
        self._write_predictions()

        summary = evaluate_widerface(
            prediction_dir=self.prediction_dir,
            gt_dir=self.gt_dir,
            iou_threshold=self.iou_threshold,
        )
        summary.update(
            {
                "iou_threshold": self.iou_threshold,
                "prediction_dir": str(self.prediction_dir),
                "ground_truth_dir": str(self.gt_dir),
            }
        )

        self.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary

    def _extract_boxes(self, prediction: FaceDetectionOutput | None) -> np.ndarray:
        if prediction is None:
            return np.zeros((0, 5), dtype=np.float32)

        boxes = np.asarray(prediction.boxes, dtype=np.float32)
        scores = np.asarray(prediction.scores, dtype=np.float32).reshape(-1)

        if boxes.size == 0 or scores.size == 0:
            return np.zeros((0, 5), dtype=np.float32)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(
                "Face-detection predictions must provide a 2D boxes array with 4 columns"
            )
        if boxes.shape[0] != scores.shape[0]:
            raise ValueError(
                "Face-detection predictions must provide the same number of boxes and scores"
            )

        return np.concatenate((boxes, scores[:, None]), axis=1)

    def _to_widerface_boxes(self, boxes: np.ndarray) -> np.ndarray:
        if boxes.size == 0:
            return np.zeros((0, 5), dtype=np.float32)

        boxes = boxes[:, :5].copy()
        order = np.argsort(-boxes[:, 4], kind="stable")
        boxes = boxes[order]

        widerface_boxes = np.empty((boxes.shape[0], 5), dtype=np.float32)
        widerface_boxes[:, 0] = boxes[:, 0]
        widerface_boxes[:, 1] = boxes[:, 1]
        widerface_boxes[:, 2] = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        widerface_boxes[:, 3] = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        widerface_boxes[:, 4] = boxes[:, 4]

        return widerface_boxes

    def _write_predictions(self) -> None:
        if self.prediction_dir.exists():
            shutil.rmtree(self.prediction_dir)

        self.prediction_dir.mkdir(parents=True, exist_ok=True)

        for event, event_predictions in self._predictions.items():
            event_dir = self.prediction_dir / event
            event_dir.mkdir(parents=True, exist_ok=True)

            for image_stem, prediction in event_predictions.items():
                prediction_file = event_dir / f"{image_stem}.txt"
                boxes: np.ndarray = prediction["boxes"]
                lines = [
                    str(prediction["relative_path"]),
                    str(len(boxes)),
                ]

                for x, y, w, h, score in boxes:
                    lines.append(f"{x:.6f} {y:.6f} {w:.6f} {h:.6f} {score:.10f}")

                prediction_file.write_text(
                    "\n".join(lines) + "\n",
                    encoding="utf-8",
                )
