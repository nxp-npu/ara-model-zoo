# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import argparse
from pathlib import Path

import onnx
import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "run.yaml"


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
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the model run.yaml used to determine dvconvert.onode",
    )
    return parser.parse_args()


def load_cutoff_outputs(config_path: Path) -> list[str]:
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    onode = config.get("dvconvert", {}).get("onode")
    if not isinstance(onode, str) or not onode.strip():
        raise ValueError(
            f"Expected a non-empty dvconvert.onode value in config: {config_path}"
        )

    return [output_name.replace("#", ":") for output_name in onode.split(":")]


def main() -> None:
    args = parse_args()
    model_path = Path(args.path)
    output_dir = Path(args.output_path)
    config_path = Path(args.config)
    output_dir.mkdir(parents=True, exist_ok=True)

    cutoff_outputs = load_cutoff_outputs(config_path)
    model = onnx.load(str(model_path))
    output_names = [output.name for output in model.graph.output]
    if not output_names:
        raise ValueError(f"Model has no graph outputs: {model_path}")

    final_output_name = output_names[0]
    input_names = [input.name for input in model.graph.input]
    if not input_names:
        raise ValueError(f"Model has no graph inputs: {model_path}")

    onnx.utils.extract_model(
        str(model_path),
        str(output_dir / "preprocessing.onnx"),
        input_names,
        cutoff_outputs,
        check_model=True,
    )
    print("Generated preprocessing.onnx")

    onnx.utils.extract_model(
        str(model_path),
        str(output_dir / "postprocessing.onnx"),
        cutoff_outputs,
        [final_output_name],
        check_model=True,
    )
    print(f"Generated postprocessing.onnx with output '{final_output_name}'")


if __name__ == "__main__":
    main()
