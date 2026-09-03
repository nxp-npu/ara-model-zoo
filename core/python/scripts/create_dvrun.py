# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import os
from pathlib import Path
from typing import Any

import yaml

from core.python.config import Config, Flow


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess images for model inference"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file"
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of the Config file used for compilation",
    )
    return parser.parse_args()


# fmt: off
DVRUN_KEYS = [
    "network", "dvconvert", "dvnc",
    "dvsim", "thresholds", "out", "license_key"
]
# fmt: on


def _convert_paths_to_strings(data: dict[str, Any]) -> dict[str, Any]:
    def _convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        elif isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_convert(item) for item in value]
        elif isinstance(value, tuple):
            return tuple(_convert(item) for item in value)
        return value

    return {k: _convert(v) for k, v in data.items()}


def save_file(config: dict[str, Any], output_path: Path):
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def main(config_path: str, name: str):
    config = Config.from_file(config_path, Flow.COMPILE)

    if not config.out.is_dir():
        os.makedirs(config.out, exist_ok=True)

    output_file_name = config.out / name

    config_dict = config.model_dump()

    if config.dvconvert and config.dvconvert.adjust_file is None:
        del config_dict["dvconvert"]["adjust_file"]

    if config.dvconvert and config.dvconvert.encodings_json_file is None:
        del config_dict["dvconvert"]["encodings_json_file"]

    filtered_config = {}
    for key, value in config_dict.items():
        if key in DVRUN_KEYS:
            filtered_config[key] = value

    filtered_config["preprocess"] = False
    filtered_config = _convert_paths_to_strings(filtered_config)

    save_file(filtered_config, output_file_name)


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.name)
