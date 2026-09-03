# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import os
from pathlib import Path
from shutil import rmtree
from typing import Callable

from tqdm import tqdm
import numpy.typing as npt

from core.python import read_image
from core.python.calibration import read_image_list, stage_calib_images
from core.python.config import Config, Flow
from core.python.logger import logger
from core.python import print_config, get_model_processor, ProcessorStage
from core.python.preprocess.interfaces import PreprocessOutput


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess images for model inference"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path to the dataset the calibration images are copied from.",
    )
    return parser.parse_args()


def main(config_path: str, dataset_root: str):
    config = Config.from_file(config_path, Flow.COMPILE)
    print_config(config)

    assert config.calibration_data is not None
    assert config.network is not None

    dataset_path = Path(dataset_root).expanduser().resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    # Load preprocessing and postprocessing functions for the model
    preprocess: Callable[[npt.NDArray, Config], PreprocessOutput] = get_model_processor(
        config.modelname, ProcessorStage.PREPROCESS
    )

    quantize_data_dir = config.calibration_data.quantize
    verify_image_dir = config.calibration_data.verify

    quantize_images = stage_calib_images(
        read_image_list(config.calibration_data.quantize_list),
        dataset_path,
        quantize_data_dir,
    )

    quantize_output_dir: Path = config.network.images.quantize

    # We need to convert "." present inside inode to "_", otherwise dvrun utility throws an error that no valid image found
    if config.dvconvert is not None and "." in config.dvconvert.inode:
        inode = config.dvconvert.inode.replace(".", "_")
        input_file_name = inode + ".bin"
    else:
        input_file_name = (
            config.dvconvert.inode + ".bin" if config.dvconvert is not None else None
        )

    if quantize_output_dir.exists():
        logger.info(f"Removing existing directory: {quantize_output_dir}")
        rmtree(quantize_output_dir)

    quantize_output_dir.mkdir()

    logger.info("Preparing Calibration Processing images")
    for image_name in tqdm(quantize_images):
        image_path = quantize_data_dir / image_name

        output_image_dir = quantize_output_dir / image_name
        output_image_path = output_image_dir / input_file_name

        os.makedirs(output_image_dir, exist_ok=True)
        image = read_image(str(image_path))
        preprocessed_output = preprocess(image, config)
        preprocessed_output.processed_image.tofile(output_image_path)

    verify_output_dir: Path = config.network.images.verify

    if verify_output_dir.exists():
        logger.info(f"Removing existing directory: {verify_output_dir}")
        rmtree(verify_output_dir)
        verify_output_dir.mkdir()

    if verify_image_dir is None:
        logger.info(
            "No Verification images provided. Using 2 quantization images as replacements"
        )
        verify_images = quantize_images[:2]
        verify_image_dir = quantize_data_dir
    else:
        verify_images = os.listdir(verify_image_dir)

    logger.info("Preparing Calibration Verification images")

    for image_name in tqdm(verify_images):
        image_path = verify_image_dir / image_name

        output_image_dir = verify_output_dir / image_name
        output_image_path = output_image_dir / input_file_name

        os.makedirs(output_image_dir, exist_ok=True)

        image = read_image(str(image_path))
        preprocessed_output = preprocess(image, config)
        preprocessed_output.processed_image.tofile(output_image_path)


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.dataset_root)
