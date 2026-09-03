# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from collections.abc import Sequence


def _validate_channels(channels: int, valid_channels: Sequence[int] = (1, 3)) -> None:
    """
    Validate that the number of channels is supported.

    Args:
        channels (int): Number of input channels.
        valid_channels (Sequence[int]): Allowed channel values.

    Raises:
        ValueError: If channels is not supported.
    """
    if channels not in valid_channels:
        raise ValueError(
            f"Invalid channels: {channels}. Expected one of {tuple(valid_channels)}."
        )


def _validate_channel_consistency(
    expected_channels: int,
    actual_channels: int,
    *,
    context: str = "input",
) -> None:
    """
    Validate that the number of channels matches the expected value.

    Args:
        expected_channels (int): Expected number of channels.
        actual_channels (int): Observed number of channels.
        context (str): Context for error message (e.g., 'image', 'mean').

    Raises:
        ValueError: If channel counts do not match.
    """
    if actual_channels != expected_channels:
        raise ValueError(
            f"Expected {context} with {expected_channels} channel(s), "
            f"got {actual_channels}."
        )
