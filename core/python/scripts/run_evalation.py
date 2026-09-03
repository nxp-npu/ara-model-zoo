# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import argparse
from pathlib import Path
from typing import Callable

import numpy.typing as npt
from tqdm import tqdm

from core.python import ProcessorStage, get_model_processor, print_config
from core.python.config import Config, Flow
from core.python.config.config import EvaluationConfig, ModelType
from core.python.datasets import dataloader
from core.python.evaluation import create_evaluator, get_evaluation_dataset
from core.python.evaluation.constants import (
    EVALUATION_NMS_SCORE_THRESHOLD,
    EVALUATION_SCRFD_IOU_THRESHOLD,
    EVALUATION_SCRFD_NMS_SCORE_THRESHOLD,
)
from core.python.inference import ModelOutput, create_inference_session
from core.python.logger import logger
from core.python.postprocess import ResolveInferenceOutputs
from core.python.postprocess.interfaces import PostprocessingOutput
from core.python.preprocess.interfaces import PreprocessOutput


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
    parser.add_argument(
        "--runtime",
        type=str,
        choices=["onnx", "ara"],
        default="onnx",
        help="Device to run evaluation on: 'onnx' or 'ara'",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        help="Root directory for the evaluation dataset.",
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        help="Optional directory containing evaluation ground-truth files.",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        help="Optional explicit directory containing evaluation images.",
    )
    parser.add_argument(
        "--annotation-file",
        type=str,
        help="Optional explicit annotation file for the evaluation dataset.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional number of evaluation samples to process",
    )
    return parser.parse_args()


def apply_evaluation_postprocess_overrides(config: Config) -> Config:
    assert config.postprocess is not None
    if (
        config.postprocess.model_type == ModelType.FACE_DETECTION
        and config.modelname.startswith("scrfd_")
    ):
        updates = {}
        if (
            config.postprocess.nms_score_threshold
            != EVALUATION_SCRFD_NMS_SCORE_THRESHOLD
        ):
            updates["nms_score_threshold"] = EVALUATION_SCRFD_NMS_SCORE_THRESHOLD
        if config.postprocess.iou_threshold != EVALUATION_SCRFD_IOU_THRESHOLD:
            updates["iou_threshold"] = EVALUATION_SCRFD_IOU_THRESHOLD
        if updates:
            logger.info(
                "Overriding SCRFD postprocess thresholds for evaluation: %s",
                updates,
            )
            config.postprocess = config.postprocess.model_copy(update=updates)
        return config

    current_threshold = config.postprocess.nms_score_threshold
    current_threshold = (
        config.postprocess.nms_score_threshold
        if config.postprocess is not None
        else None
    )
    if current_threshold is None or current_threshold == EVALUATION_NMS_SCORE_THRESHOLD:
        return config

    logger.info(
        "Overriding postprocess.nms_score_threshold from %s to %s for evaluation",
        current_threshold,
        EVALUATION_NMS_SCORE_THRESHOLD,
    )

    if config.postprocess is not None:
        config.postprocess = config.postprocess.model_copy(
            update={"nms_score_threshold": EVALUATION_NMS_SCORE_THRESHOLD}
        )
    return config


def build_evaluation_config(
    config: Config,
    dataset_root: str | None = None,
    gt_dir: str | None = None,
    images_dir: str | None = None,
    annotation_file: str | None = None,
) -> EvaluationConfig:
    if config.evaluation is None:
        raise ValueError(
            "Config is missing an 'evaluation' section. Add one before running evaluation."
        )

    evaluation_config = config.evaluation
    updates: dict[str, Path] = {}

    if dataset_root:
        updates["root"] = Path(dataset_root).expanduser().resolve()

    if gt_dir:
        updates["gt_dir"] = Path(gt_dir).expanduser().resolve()

    if images_dir:
        updates["images_dir"] = Path(images_dir).expanduser().resolve()
    if annotation_file:
        updates["annotation_file"] = Path(annotation_file).expanduser().resolve()

    return (
        evaluation_config.model_copy(update=updates) if updates else evaluation_config
    )


def main(
    config_path: str,
    runtime: str = "onnx",
    dataset_root: str | None = None,
    gt_dir: str | None = None,
    images_dir: str | None = None,
    annotation_file: str | None = None,
    limit: int | None = None,
):
    config = Config.from_file(
        config_path, Flow.EVAL_FLOAT if runtime == "onnx" else Flow.EVAL_HW
    )
    config = apply_evaluation_postprocess_overrides(config)
    evaluation_config = build_evaluation_config(
        config=config,
        dataset_root=dataset_root,
        gt_dir=gt_dir,
        images_dir=images_dir,
        annotation_file=annotation_file,
    )
    config.evaluation = evaluation_config
    print_config(config)

    preprocess_func: Callable[[npt.NDArray, Config], PreprocessOutput] = (
        get_model_processor(config.modelname, ProcessorStage.PREPROCESS)
    )
    postprocess_func: Callable[
        [ModelOutput, Config, npt.NDArray, str], PostprocessingOutput | None
    ] = get_model_processor(config.modelname, ProcessorStage.POSTPROCESS)

    evaluation_type = evaluation_config.type
    output_dir = (
        config.out / "evaluation" / f"{evaluation_type}_{evaluation_config.split}"
    )
    dataset = get_evaluation_dataset(evaluation_config)
    evaluator = create_evaluator(evaluation_config, output_dir=output_dir)

    compiled_models_dir = config.out / "compiled_model"
    model_path = (
        compiled_models_dir / "model.dvm"
        if runtime == "ara"
        else compiled_models_dir / "model.onnx"
    )

    session = create_inference_session(model_path, config.inference)
    resolver = ResolveInferenceOutputs(config, session.session_type)

    if config.batch_size != 1:
        logger.warning(f"""Batch size set to 1 (was {config.batch_size})
            Only batch size 1 is supported for now.
            """)
        config.batch_size = 1

    eval_loader = dataloader(
        config,
        dataset,
        image_dir=None,
        label_dir=None,
        batch_size=config.batch_size,
        num_workers=0,
        transform=preprocess_func,
        sample_size=limit,
        calibration_mode=False,
        is_numpy=True,
    )

    for batch in tqdm(eval_loader, desc="Evaluating", unit="image"):
        orig_images, transformed_images, orig_names, targets = batch
        batched_input = transformed_images.cpu().detach().numpy()

        inference_output = session.infer([batched_input])
        model_predictions = resolver.run_continuation_inference(inference_output)
        postprocessed_prediction: PostprocessingOutput | None = postprocess_func(
            model_predictions,
            config,
            orig_images[0],
            orig_names[0],
        )
        if postprocessed_prediction is not None:
            evaluator.add_sample(targets[0], postprocessed_prediction)

    summary = evaluator.evaluate()
    logger.info("Evaluation summary written to %s", evaluator.output_dir)
    for key, value in summary.items():
        logger.info("%s: %s", key, value)

    return summary


if __name__ == "__main__":
    args = parse_args()
    main(
        config_path=args.config,
        runtime=args.runtime,
        dataset_root=args.dataset_root,
        gt_dir=args.gt_dir,
        images_dir=args.images_dir,
        annotation_file=args.annotation_file,
        limit=args.limit,
    )
