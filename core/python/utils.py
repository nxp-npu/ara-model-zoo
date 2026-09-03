# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from __future__ import annotations

import os
from enum import Enum
import importlib.util
from types import ModuleType
from pathlib import Path
from typing import TYPE_CHECKING, Type, Mapping, Callable, Any

import cv2
import numpy
import numpy as np
import yaml
import torch.nn as nn

from .logger import logger

if TYPE_CHECKING:
    from .config import Config


def load_config(config_path, mode=None):
    try:
        config = yaml.safe_load(open(config_path, "r"))
        if mode and "mode" in config:
            if mode in config["mode"]:
                priority_params = config["mode"][mode]
                config = merge_configs(config, priority_params)
                del config["mode"]
            else:
                raise Exception(
                    "config error: mode " + mode + " does not exist in the config file"
                )
        return config
    except Exception as e:
        raise Exception("config error: unable to load config file - " + str(e))


def merge_configs(config1, config2):
    for k, v in config2.items():
        if isinstance(v, dict):
            if k in config1:
                merge_configs(config1[k], v)
            else:
                config1[k] = v
        else:
            config1[k] = v
    return config1


def print_config(config: Config, name: str | None = None):
    """Print a formatted representation of a configuration object.

    Args:
        config: A Pydantic BaseModel configuration instance to print
    """

    lines: list[str] = []

    def _print_dict(data: dict, indent: int = 0):
        """Recursively build nested dictionary lines with proper indentation."""
        for key, value in data.items():
            if isinstance(value, dict) and not len(value):
                lines.append("  " * indent + f"{key}:" + " {}")
                continue

            if isinstance(value, dict):
                lines.append("  " * indent + f"{key}:")
                _print_dict(value, indent + 1)
            elif isinstance(value, list):
                lines.append("  " * indent + f"{key}:")
                for item in value:
                    if isinstance(item, dict):
                        _print_dict(item, indent + 1)
                        lines.append("  " * (indent + 1) + "---")

                    else:
                        lines.append("  " * (indent + 1) + f"- {item}")
            elif isinstance(value, Type):
                lines.append("  " * indent + f"{key}: {value.__name__}")
            else:
                lines.append("  " * indent + f"{key}: {value}")

    lines.append("")
    lines.append("=" * 60)
    lines.append(
        f"{config.__class__.__name__} Configuration"
        if name is None
        else name.capitalize()
    )
    lines.append("=" * 60)
    _print_dict(config.model_dump())
    lines.append("=" * 60)
    lines.append("")
    logger.info("\n".join(lines))


def get_path(path, config_dir, ignore_error=False):
    if not os.path.isabs(path):
        path = os.path.join(config_dir, path)
        if not os.path.exists(path) and not ignore_error:
            logger.warning("config error: file path not accessible - " + path)
    elif not os.path.exists(path) and not ignore_error:
        logger.warning("config error: file path not accessible - " + path)
    return os.path.abspath(path)


def set_paths(config, config_dir):
    if "datasets" not in config:
        raise Exception(
            "config error: data sets are not specified in yaml configuration"
        )
    if "quantize" in config["datasets"]:
        config["datasets"]["quantize"] = get_path(
            config["datasets"]["quantize"], config_dir
        )
    if "verify" in config["datasets"]:
        config["datasets"]["verify"] = get_path(
            config["datasets"]["verify"], config_dir
        )
    if "quantize_list" in config["datasets"]:
        config["datasets"]["quantize_list"] = (
            get_path(config["datasets"]["quantize_list"], config_dir)
            if config["datasets"]["quantize_list"]
            else None
        )
    if "verify_list" in config["datasets"]:
        config["datasets"]["verify_list"] = (
            get_path(config["datasets"]["verify_list"], config_dir)
            if config["datasets"]["verify_list"]
            else None
        )
    if "prepackaged_model_path" in config:
        config["prepackaged_model_path"] = get_path(
            config["prepackaged_model_path"], config_dir, True
        )
    if config["network"]["srcfw"] != "pytorch" and config["network"]["inet"] != "":
        config["network"]["inet"] = get_path(
            config["network"]["inet"], config_dir, True
        )
    if "pcfg" in config["network"] and not config["network"]["pcfg"]:
        config["network"]["iwt"] = get_path(config["network"]["iwt"], config_dir, True)
    else:
        config["network"]["pcfg"] = get_path(
            config["network"]["pcfg"], config_dir, True
        )
    if "tags" in config["network"] and config["network"]["tags"]:
        config["network"]["tags"] = get_path(
            config["network"]["tags"], config_dir, True
        )
    if "label" in config["network"] and config["network"]["label"]:
        config["network"]["label"] = get_path(
            config["network"]["label"], config_dir, True
        )
    if "adjust" in config["dvnc"] and config["dvnc"]["adjust"]:
        config["dvnc"]["adjust"] = get_path(config["dvnc"]["adjust"], config_dir, True)
    config["network"]["images"]["quantize"] = get_path(
        config["network"]["images"]["quantize"], config_dir, True
    )
    config["network"]["images"]["verify"] = get_path(
        config["network"]["images"]["verify"], config_dir, True
    )
    config["out"] = get_path(config["out"], config_dir, True)
    return config


def get_file_name(params):
    return (
        params.layer_name
        + "-"
        + str(params.fused_parent_id)
        + "-"
        + params.blob_name
        + "-"
        + str(params.blob_id)
    )


def get_output_file(config, mode, params, accuracy_target, image):
    if accuracy_target == "ara":
        return os.path.join(
            get_postprocessed_output_dir(config, mode), image, get_file_name(params)
        )
    elif accuracy_target == "dvnc":
        return os.path.join(
            get_compiler_output_dir(config), image, "quantized", get_file_name(params)
        )
    else:
        raise Exception("unknown accuracy target - " + accuracy_target)


def get_postprocessed_output_dir(config, mode):
    return os.path.join(config["out"], "postprocessed_images_" + mode)


def get_postprocessed_viz_dir(config, mode):
    return os.path.join(config["out"], "postprocessed_images_viz_" + mode)


def get_compiler_output_dir(config):
    return os.path.join(config["out"], "quantizer", "outputs")


def save_to_file(image, output_dir, image_name, layer_name):
    if not os.path.exists(os.path.join(output_dir, image_name)):
        os.makedirs(os.path.join(output_dir, image_name))
    image.tofile(os.path.join(output_dir, image_name, layer_name))


def save_numpy_array_as_file(array: np.ndarray, output_path: Path):
    array.tofile(output_path)


def numpy_read_from_file(file_path, data_type="uint8"):
    return numpy.fromfile(file_path, dtype=data_type)


def get_model_file(config, mode):
    if mode == "prepackaged":
        if "prepackaged_model_path" in config:
            model_path = config["prepackaged_model_path"]
        else:
            raise Exception(
                "path to prepackaged model is not provided in configuration file"
            )
    else:
        model_path = os.path.join(config["out"], "assets", "model.dvm")
    return model_path


def read_image(
    image_path: str,
    color_mode: int = cv2.IMREAD_COLOR,
) -> numpy.ndarray:
    """
    Read an image from disk and produce a numpy array for the transformation pipeline.

    Parameters
    ----------
    image_path : str
        Path to the image file to be read.
    color_mode : int
        OpenCV color mode used for reading the image (e.g., cv2.IMREAD_COLOR).

    Returns
    -------
    numpy.ndarray

    Raises
    ------
    TypeError
        If `image_path` is not a string.
    FileNotFoundError
        If the image cannot be loaded from the given path.
    """

    if not isinstance(image_path, str):
        raise TypeError(
            f"Expected image_path to be str, got {type(image_path).__name__}"
        )

    img = cv2.imread(image_path, color_mode)

    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    return img


def align_shape_trailing_singletons(
    actual_array: np.ndarray, expected_shape: tuple
) -> np.ndarray:
    """
    Aligns an actual NumPy array to an expected ONNX shape by removing only
    extra trailing singleton dimensions, if they exist.

    Algorithm:
    1. Compare leading dimensions (from the start) between actual_array and expected_shape.
        - If any leading dimension mismatches, raise ValueError.
    2. If actual_array has more dimensions than expected_shape:
        - Ensure all extra trailing dimensions are singleton (size=1).
        - Remove (squeeze) the trailing singleton dimensions to match expected_shape.
    3. If actual_array has fewer or equal dimensions and all leading dims match,
        return the array as is.

    Assumptions:
    - Only extra trailing singleton dimensions are allowed.
    - Leading dimensions must always match expected_shape.
    - The input array may have arbitrary trailing 1s that are safe to remove.

    Args:
        actual_array (np.ndarray): The array to align.
        expected_shape (tuple): The expected shape (from ONNX metadata).

    Returns:
        np.ndarray: Aligned array with shape exactly matching expected_shape.

    Raises:
        ValueError: If leading dimensions do not match or
                    if extra trailing dimensions are not singleton.
    """
    actual_shape = actual_array.shape
    n_expected = len(expected_shape)
    n_actual = len(actual_shape)

    # Step 1: Compare trailing dimensions
    for i in range(min(n_expected, n_actual)):
        if actual_shape[n_actual - i - 1] != expected_shape[n_expected - i - 1]:
            raise ValueError(
                f"Leading dimension mismatch at axis {i}: "
                f"actual {actual_shape[n_actual - i - 1]} vs expected {expected_shape[n_expected - i - 1]}"
            )

    # Step 2: Handle extra leading dimensions
    if n_expected > n_actual:
        extra_dims = expected_shape[: n_expected - n_actual]
        if not all(dim == 1 for dim in extra_dims):
            raise ValueError(
                f"Extra leading dimensions must be singleton: actual {extra_dims}"
            )
        aligned_array = actual_array.reshape(expected_shape)
    else:
        aligned_array = actual_array

    return aligned_array


def map_device_to_continuation_nodes(onode_str: str) -> Mapping[str, str]:
    """
    Create a mapping from device output node names (keys) to continuation graph node names (values).

    Key (device_onodes) -> used as-is from the device outputs.
    Value (continuation_graph_inodes) -> sanitized ONNX input names for the continuation graph.

    Transformation rules:
    - Keys (device_onodes):
        - '/' -> '_'
        - '.' -> '_'
    - Values (continuation_graph_inodes):
        - '#' -> ':'

    Parameters
    ----------
    onode_str : str
        Colon-separated device output node names or continuation graph input node names from configuration.

    Returns
    -------
    Mapping[str, str]
        Dictionary mapping device_onodes (keys) to continuation_graph_inodes (values).

    Raises
    ------
    ValueError
        If onode_str contains no valid nodes or duplicate continuation_graph_inodes.

    Example
    -------
    Input: "/model.22/Mul_2_output_0:/model.22/Sigmoid_output_0"
    Output: {
        "_model_22_Mul_2_output_0": "/model.22/Mul_2_output_0",
        "_model_22_Sigmoid_output_0": "/model.22/Sigmoid_output_0"
    }
    """
    key_translation_table = str.maketrans({"/": "_", ".": "_"})

    mapping = {}
    for node_name in onode_str.split(":"):
        node_name = node_name.strip().replace("#", ":")
        if not node_name:
            continue

        continuation_graph_onode = node_name
        device_onode = node_name.translate(key_translation_table)

        if continuation_graph_onode in mapping.values():
            raise ValueError(
                f"Duplicate continuation graph node name detected: {continuation_graph_onode}"
            )

        mapping[device_onode] = continuation_graph_onode

    if not mapping:
        raise ValueError("No valid device node names found in onode_str.")

    return mapping


class ProcessorStage(str, Enum):
    """
    Enumeration of supported model processing stages.
    """

    PREPROCESS = "preprocess"
    POSTPROCESS = "postprocess"


MODELZOO_ROOT = Path("modelzoo")


def locate_model_dir(model_name: str) -> Path:
    """
    Locate the directory corresponding to a given model.

    The function searches recursively under the model zoo root
    and expects exactly one directory matching the given model name.

    Args:
        model_name: Name of the model to search for.

    Returns:
        Path to the resolved model directory.

    Raises:
        ValueError: If no model directory or multiple directories are found.
    """
    model_dirs = list(MODELZOO_ROOT.rglob(model_name))

    if not model_dirs:
        raise ValueError(
            f"No model directory found for '{model_name}' under '{MODELZOO_ROOT}'."
        )

    if len(model_dirs) > 1:
        raise ValueError(
            f"Multiple model directories found for '{model_name}': {model_dirs}"
        )

    return model_dirs[0].resolve()


def locate_processor_file(model_dir: Path, stage: ProcessorStage) -> Path:
    """
    Locate the processor implementation file for a specific stage.

    Each stage is expected to follow the directory layout:

        <model_dir>/
            preprocess/py/preprocess.py
            postprocess/py/postprocess.py

    Args:
        model_dir: Path to the model directory.
        stage: Processing stage to locate.

    Returns:
        Path to the processor Python file.

    Raises:
        FileNotFoundError: If the expected processor file does not exist.
    """
    file_path = model_dir / stage.value / "py" / f"{stage.value}.py"

    if not file_path.is_file():
        raise FileNotFoundError(
            f"{stage.value.capitalize()} file not found: {file_path}"
        )

    return file_path


def locate_pytorch_model_file(model_dir: Path) -> Path:
    """
    Locate the PyTorch model definition file within a model directory.

    The expected structure is:

        <model_dir>/
            model.py

    Args:
        model_dir: Path to the model directory.

    Returns:
        Path to the `model.py` file containing model definition and
        loading utilities.

    Raises:
        FileNotFoundError: If `model.py` does not exist in the given directory.
    """
    file_path = model_dir / "model.py"

    if not file_path.is_file():
        raise FileNotFoundError(f"file not found: {file_path}")

    return file_path


def load_function_from_file(file_path: Path, function_name: str) -> Callable[..., Any]:
    """
    Dynamically load a function from a Python file.

    This enables runtime loading of preprocessing or postprocessing
    logic without requiring the modules to be part of the static
    Python package structure.

    Args:
        file_path: Path to the Python module file.
        function_name: Name of the function to load.

    Returns:
        The callable function loaded from the module.

    Raises:
        FileNotFoundError: If the file does not exist.
        ImportError: If the module cannot be loaded.
        AttributeError: If the function is not present in the module.
        TypeError: If the loaded attribute is not callable.
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    spec = importlib.util.spec_from_file_location(function_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for '{file_path}'")

    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, function_name):
        raise AttributeError(f"Function '{function_name}' not found in '{file_path}'")

    func = getattr(module, function_name)

    if not callable(func):
        raise TypeError(f"Attribute '{function_name}' in '{file_path}' is not callable")

    return func


def get_model_processor(
    model_name: str,
    stage: ProcessorStage,
) -> Callable[..., Any]:
    """
    Retrieve a specific processing function for a model.

    The function loads only the requested stage (preprocess or
    postprocess) to keep the system modular and avoid unnecessary imports.

    Args:
        model_name: Name of the target model.
        stage: Desired processing stage.

    Returns:
        Callable processor function.

    Raises:
        ValueError, FileNotFoundError, ImportError, AttributeError, TypeError:
            Propagated if the model directory, processor file,
            or function cannot be located or loaded.
    """
    model_dir = locate_model_dir(model_name)
    processor_path = locate_processor_file(model_dir, stage)

    return load_function_from_file(processor_path, stage.value)


def get_pytorch_model_loader(
    model_name: str,
) -> Callable[..., nn.Module]:
    """
    Retrieve the PyTorch model loading function for a given model.

    This utility dynamically locates the model implementation file and
    loads a standardized model loader function.

    Contract:
        The target module (e.g., `model.py`) must define a function named
        `load_model_from_checkpoint` with the following signature:

            load_model_from_checkpoint(checkpoint_path: str) -> nn.Module

        The function is expected to load the checkpoint from the given path
        and return an initialized PyTorch model (`nn.Module` instance).

    Args:
        model_name: Name of the target model.

    Returns:
        Callable that loads and returns a PyTorch `nn.Module` instance
        from a checkpoint.

    Raises:
        ValueError, FileNotFoundError, ImportError, AttributeError, TypeError:
            Propagated if the model directory, model file, or required
            loader function cannot be located or loaded.
    """
    model_dir = locate_model_dir(model_name)
    pytorch_model_path = locate_pytorch_model_file(model_dir)

    return load_function_from_file(
        pytorch_model_path,
        function_name="load_model_from_checkpoint",
    )


def get_model_graph_continuation(model_name: str) -> Callable[..., Any]:
    """Load a model's Python implementation of its cut-off graph continuation."""
    model_dir = locate_model_dir(model_name)
    model_path = locate_pytorch_model_file(model_dir)
    return load_function_from_file(model_path, function_name="continue_graph")
