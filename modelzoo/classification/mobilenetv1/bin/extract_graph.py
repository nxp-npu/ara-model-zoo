# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import shutil
import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", required=True, help="Path to original model onnx file"
    )
    parser.add_argument(
        "--output-path", required=True, help="Path to store the generated onnx files"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the model run.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.path)
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_model_path = output_dir / "preprocessing.onnx"

    # copy .onnx file
    shutil.copy2(model_path, output_model_path)

    print(f"Copied model from {model_path} -> {output_model_path}")


if __name__ == "__main__":
    main()
