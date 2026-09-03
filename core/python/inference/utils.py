# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from __future__ import annotations

from typing import TYPE_CHECKING

from .interfaces import ParameterMetaData

if TYPE_CHECKING:
    from numpy import ndarray


def _find_names_from_shape(
    input_shape: tuple[int] | list[int], input_meta: list[ParameterMetaData]
) -> list[str]:
    """
    Find input parameter names that match a given shape.

    This helper function searches through input metadata to find all parameter
    names that have a shape matching the provided input_shape.

    Args:
        input_shape: The shape to match against input parameters.
        input_meta: List of ParameterMetaData objects describing model inputs.

    Returns:
        A list of parameter names that match the given shape.

    Raises:
        ValueError: If no parameters match the given shape.
    """
    matched = False

    input_names = []

    for node_args in input_meta:
        node_shape = node_args.shape

        if len(node_shape) == len(input_shape) and all(
            s1 == s2 for s1, s2 in zip(node_shape, input_shape)
        ):
            matched = True
            input_names.append(node_args.name)

    if not matched:
        raise ValueError(
            f"Input shape {input_shape} does not match any model input shapes"
        )

    return input_names


def generate_input_feed_from_list(
    inputs: list[ndarray], input_meta: list[ParameterMetaData]
) -> dict[str, ndarray]:
    """
    Generate an input feed dictionary from a list of numpy arrays.

    This helper function maps a list of input tensors to model input names
    based on shape matching. It automatically assigns each input tensor to
    a parameter name that matches its shape.

    Args:
        inputs: List of numpy arrays to use as model inputs.
        input_meta: List of ParameterMetaData objects describing model inputs.

    Returns:
        A dictionary mapping input names to numpy arrays.

    Raises:
        ValueError: If the number of inputs doesn't match the number of
                   model parameters, or if a tensor cannot be matched to
                   an available parameter name.
    """
    if len(inputs) != len(input_meta):
        raise ValueError(f"Expected {len(input_meta)} inputs, but got {len(inputs)}")

    input_feed = {}
    for input_tensor in inputs:
        potential_names = _find_names_from_shape(input_tensor.shape, input_meta)

        # `_find_names_from_shape` guarantees a non-empty list, but all valid
        # names could already be taken
        name = next((name for name in potential_names if name not in input_feed), None)
        if name is None:
            raise ValueError(f"No available name for input shape {input_tensor.shape}")
        input_feed[name] = input_tensor

    return input_feed
