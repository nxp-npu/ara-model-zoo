# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import argparse
from typing import Callable

import torch.nn as nn

from core.python.config import Config, Flow, QuantizationSchemes
from core.python import print_config, get_pytorch_model_loader
from core.python.sparrow import set_seed, run_ptq, run_qft


def parse_args():
    parser = argparse.ArgumentParser(description="Quantization script")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
    parser.add_argument("--data_path", required=True, help="Path to the dataset")
    return parser.parse_args()


def main(config_path: str, data_path: str):
    config = Config.from_file(config_path, Flow.QUANTIZATION)
    print_config(config)

    # Get PyTorch model loading function for the requested model_name
    pytorch_model_loader_func: Callable[..., nn.Module] = get_pytorch_model_loader(
        config.modelname
    )

    set_seed()
    model: nn.Module = pytorch_model_loader_func(config)

    # Dispatch based on the Enum
    if (
        config.quantization is not None
        and config.quantization.scheme == QuantizationSchemes.PTQ
    ):
        run_ptq(data_path, config, model)
    elif (
        config.quantization is not None
        and config.quantization.scheme == QuantizationSchemes.QFT
    ):
        run_qft(data_path, config, model)
    else:
        if config.quantization is not None:
            raise ValueError(
                f"Unknown quantization scheme: {config.quantization.scheme}"
            )


if __name__ == "__main__":
    args = parse_args()
    config_path: str = args.config
    data_path: str = args.data_path

    main(config_path, data_path)
