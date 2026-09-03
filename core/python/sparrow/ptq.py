# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import datetime
import os
from pathlib import Path
from typing import Callable

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from tools.logger.logger_config import create_logger  # type: ignore[ty:unresolved-import]
from tools.torch.quantization.quantizer_engine import QuantizerEngine  # type: ignore[ty:unresolved-import]
from tools.torch.quantization.utils.data import FuseConfig  # type: ignore[ty:unresolved-import]
from tools.torch.utils.model_utils import model_in_eval_mode  # type: ignore[ty:unresolved-import]
from tqdm import tqdm

from core.python import ProcessorStage, get_model_processor
from core.python.config import Config, QuantizationConfig
from core.python.datasets import dataloader
from core.python.evaluation import create_evaluator, get_evaluation_dataset
from core.python.evaluation.base import BaseEvaluator, DatasetSample
from core.python.inference import ModelOutput
from core.python.logger import logger
from core.python.postprocess.interfaces import PostprocessingOutput
from core.python.scripts.run_evalation import apply_evaluation_postprocess_overrides


def evaluation_loop(
    config: Config,
    evaluator: BaseEvaluator,
    model: nn.Module,
    samples: list[DatasetSample],
    postprocess_func: Callable[
        [ModelOutput, Config, npt.NDArray, str], PostprocessingOutput | None
    ],
):
    assert config.quantization is not None
    model.eval()
    evaluated = 0
    pbar = tqdm(samples, desc="Evaluating the model", total=len(samples))
    with model_in_eval_mode(model), torch.no_grad():
        for sample in pbar:
            input_tensor = torch.Tensor(sample[1]).to(config.quantization.device)

            outputs = model(input_tensor)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            for idx, out in enumerate(outputs):
                mo = ModelOutput(
                    outputs={"preds": np.expand_dims(out.cpu().numpy(), 0)}
                )
                outputs_processed = postprocess_func(
                    mo, config, sample[0][idx], sample[2][idx].image_path.name
                )
                evaluator.add_sample(
                    sample=sample[2][idx], prediction=outputs_processed
                )
                evaluated += 1
                pbar.set_postfix({"Evaluated samples count": evaluated})

    summary = evaluator.evaluate()
    logger.info("Evaluation summary written to %s", evaluator.output_dir)
    for key, value in summary.items():
        logger.info("%s: %s", key, value)
    return summary


def calibration_loop(
    config: Config,
    model: nn.Module,
    samples: list[DatasetSample],
):
    assert config.quantization is not None
    model.eval()

    calibrated = 0
    pbar = tqdm(
        samples, desc="Calibrating the model for PTQ inside the QuantizerEngine"
    )
    with torch.no_grad():
        for model_input in pbar:
            if model_input is None:
                continue
            model_input = model_input.to(config.quantization.device)
            _ = model(model_input)

            input_batch_size = len(model_input)
            calibrated += input_batch_size
            pbar.set_postfix({"Calibrated count": calibrated})
    return model


def run_ptq(
    dataset_path: str,
    config: Config,
    model: nn.Module,
):
    assert config.preprocess is not None, "preprocess config is required"
    assert config.quantization is not None, "quantization config is required"
    input_shape = (1, *config.preprocess.input_shape.model_dump().values())
    q_config: QuantizationConfig = config.quantization

    config = apply_evaluation_postprocess_overrides(config)
    assert config.evaluation is not None, "evaluation config is required"

    config.evaluation = config.evaluation.model_copy(
        update={"root": Path(dataset_path)}
    )

    evaluator = create_evaluator(config.evaluation, config.out)
    preprocess_func = get_model_processor(config.modelname, ProcessorStage.PREPROCESS)
    postprocess_func = get_model_processor(config.modelname, ProcessorStage.POSTPROCESS)
    val_samples = get_evaluation_dataset(config.evaluation)
    val_loader = dataloader(
        config,
        val_samples,
        batch_size=q_config.val_batch_size,
        num_workers=q_config.num_workers,
        transform=preprocess_func,
        calibration_mode=False,
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"ptq_quant_{timestamp}.log"
    log_path = config.out / log_name
    logger = create_logger(log_name, log_path, to_console=True)
    output_dir = config.out.resolve()

    FUSE_CONFIG = FuseConfig(enable=True, fuse_bn=True)

    model = model.to(q_config.device)
    example_inputs = torch.randn(input_shape).to(q_config.device)

    logger.info("Evaluating base model pre-quantization:")
    logger.info(f"Base Model Architecture:\n{model}")
    base_acc = evaluation_loop(config, evaluator, model, val_loader, postprocess_func)
    logger.info(f"Base model accuracy: {base_acc}")
    evaluator.reset()

    engine = QuantizerEngine(
        model=model,
        device=q_config.device,
        example_inputs=example_inputs,
        config_path=q_config.quantization_config,
        fuse_config=FUSE_CONFIG,
        logger=logger,
        export_dir=output_dir,
    )

    logger.info("Preparing QuantizerEngine.")
    engine_simulated_model = engine.prepare()

    config.evaluation = config.evaluation.model_copy(update={"split": "train"})
    calib_samples = get_evaluation_dataset(config.evaluation)
    calib_loader = dataloader(
        config,
        calib_samples,
        batch_size=q_config.calibration_batch_size,
        num_workers=q_config.num_workers,
        transform=preprocess_func,
        sample_size=q_config.calibration_samples,
        calibration_mode=True,
    )

    engine_simulated_model = calibration_loop(
        config, engine_simulated_model, calib_loader
    )

    quant_acc = evaluation_loop(
        config,
        evaluator,
        engine_simulated_model,
        val_loader,
        postprocess_func,
    )
    logger.info(f"Quantized model accuracy: {quant_acc}")
    logger.info(
        f"--- Accuracy Summary ---\nBase Model Accuracy: {base_acc}\nQuantized Model Accuracy: {quant_acc}"
    )

    output_dir = output_dir / "compiled_model"
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Converting and exporting encodings (opset=13)")
    engine.convert_and_export_encoding(opset=13, root_output_path=output_dir)
    logger.info(f"Saving output files at {output_dir}")
