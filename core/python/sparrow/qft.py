# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import os
import copy
import datetime

import torch
import torch.nn as nn

from core.python.config import Config, QuantizationConfig

from models.dataloaders import COCODataloader
from models.objectdetection.yolo.yolo_finetuner import finetuner
from models.objectdetection.yolo.yolo_validator import YoloValidator

from tools.logger.logger_config import create_logger
from tools.torch.utils.utils import get_current_run_dir
from tools.torch.quantization.utils.data import FuseConfig
from tools.torch.quantization.quantizer_engine import QuantizerEngine

kwargs_finetuner = {
    "optimizer": {
        "AdamW": {
            "args": {
                "lr": 1e-5,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.01,
                "amsgrad": False,
            }
        }
    },
    "criterion": {"KLDivLoss": {"args": {}}},
    "lr_scheduler": {
        "CosineAnnealingWarmRestarts": {
            "args": {
                "T_0": 15,
                "eta_min": 0.000002,
                "T_mult": 2,
            }
        }
    },
    "loss_weights": [7.5, 0.5, 1.5],
}


def run_qft(
    dataset_path: str,
    config: Config,
    model: nn.Module,
):
    input_shape = (1, *config.preprocess.input_shape.model_dump().values())
    q_config: QuantizationConfig = config.quantization
    val_images_path = os.path.join(dataset_path, "images/val2017")
    train_images_path = os.path.join(dataset_path, "images/train2017")
    val_annotations_path = os.path.join(
        dataset_path, "annotations/instances_val2017.json"
    )
    train_labels_path = os.path.join(dataset_path, "labels/train2017")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"quant_{timestamp}.log"
    log_path = os.path.join(get_current_run_dir(), log_name)
    logger = create_logger(log_name, log_path, to_console=True)

    FUSE_CONFIG = FuseConfig(enable=True, fuse_bn=True)

    # Copying for base_model for teacher
    teacher_model = copy.deepcopy(model)
    validator = YoloValidator(
        model=model,
        device=q_config.device,  # type: ignore
        val_image_dir=val_images_path,
        val_annotation_json=val_annotations_path,
        batch_size=q_config.val_batch_size,
    )

    # val_acc = validator.validate()  # 0.3732
    model = model.to(q_config.device)
    example_inputs = torch.randn(input_shape).to(q_config.device)

    logger.info("Evaluating base model pre-quantization:")
    logger.info(f"Base Model Architecture:\n{model}")
    base_acc = validator.validate()
    logger.info(f"Base model accuracy: {base_acc}")

    engine = QuantizerEngine(
        model=model,
        device=q_config.device,
        example_inputs=example_inputs,
        config_path=q_config.quantization_config,  # type: ignore
        fuse_config=FUSE_CONFIG,
        logger=logger,
    )

    logger.info("Preparing QuantizerEngine.")
    engine_simulated_model = engine.prepare()

    finetune_loader = COCODataloader(
        image_dir=train_images_path,
        label_dir=train_labels_path,
        batch_size=q_config.calibration_batch_size,
        img_size=input_shape[-1],
        num_workers=q_config.num_workers,
        sample_size=q_config.calibration_samples,
    )

    kwargs_finetuner["epochs"] = q_config.epochs  # type: ignore
    kwargs_finetuner["base_model"] = teacher_model

    engine_simulated_model = finetuner(
        engine_simulated_model,
        finetune_loader,
        device=q_config.device,
        val_callback=validator.validate,
        **kwargs_finetuner,  # type: ignore
    )

    quant_acc = validator.validate(model=engine_simulated_model)
    logger.info(f"Quantized model accuracy: {quant_acc}")
    logger.info(
        f"--- Accuracy Summary ---\nBase Model Accuracy: {base_acc}\nQuantized Model Accuracy: {quant_acc}"
    )
    engine.convert_and_export_encoding(model=engine_simulated_model, opset=13)
