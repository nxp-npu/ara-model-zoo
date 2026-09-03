# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import argparse
import os
from typing import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt

from core.python.config import Config, Flow
from core.python.logger import logger
from core.python import read_image, print_config, get_model_processor, ProcessorStage
from core.python.inference import create_inference_session
from core.python.inference import ModelOutput
from core.python.postprocess import ResolveInferenceOutputs
from core.python.preprocess.interfaces import PreprocessOutput
from core.python.postprocess.interfaces import PostprocessingOutput
from core.python.visualizer import Visualizer


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
    parser.add_argument(
        "--runtime",
        type=str,
        choices=["onnx", "ara"],
        default="cpu",
        help="Device to run inference on: 'cpu' or 'ara'",
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Folder containing images to run inference",
    )
    return parser.parse_args()


def main(config_path: str, runtime: str, images_dir: Path):
    config = Config.from_file(
        config_path, Flow.INFER_FLOAT if runtime == "onnx" else Flow.INFER_HW
    )
    print_config(config)

    output_dir: Path = config.out
    compiled_models_dir = output_dir / "compiled_model"

    # Load preprocessing and postprocessing functions for the model
    preprocess_func: Callable[[npt.NDArray, Config], PreprocessOutput] = (
        get_model_processor(config.modelname, ProcessorStage.PREPROCESS)
    )

    postprocess_func: Callable[
        [ModelOutput, Config, npt.NDArray, str], PostprocessingOutput | None
    ] = get_model_processor(config.modelname, ProcessorStage.POSTPROCESS)

    # Resolve model path depending on runtime
    model_path = (
        compiled_models_dir / "model.dvm"
        if runtime == "ara"
        else compiled_models_dir / "model.onnx"
    )

    # Initialize inference session
    session = create_inference_session(model_path, config.inference)

    # Resolve model outputs handler
    resolver = ResolveInferenceOutputs(config, session.session_type)

    # Initialize visualization utility
    viz = Visualizer(
        output_dir=str(output_dir / "postprocessed_output_visualized"),
    )

    for _, image_name in enumerate(os.listdir(images_dir)):
        logger.info(
            "Running inference on %s using session: %s",
            image_name,
            session.session_type,
        )

        image_path = images_dir / image_name
        original_image = read_image(str(image_path))

        # Preprocess
        preprocessed_result = preprocess_func(original_image.copy(), config)
        input_tensor = np.expand_dims(preprocessed_result.processed_image, axis=0)

        # Run inference
        inference_output: ModelOutput = session.infer([input_tensor])

        # Resolve model predictions
        model_predictions: ModelOutput = resolver.run_continuation_inference(
            inference_output
        )

        # Postprocess detections
        output: PostprocessingOutput | None = postprocess_func(
            model_predictions, config, original_image, image_name
        )

        if output:
            # Console Output
            output.print_to_console()

            # Visualize detections
            save_path = viz.visualize(
                image=original_image,
                results=output,
                output_filename=image_name,
            )

            print(f"Saved: {save_path}")
        else:
            logger.info(f"No output for {image_name}")


if __name__ == "__main__":
    args = parse_args()
    config_path: str = args.config
    runtime: str = args.runtime
    images = args.images

    main(config_path, runtime, Path(images))
