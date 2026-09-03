# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import numpy as np
import numpy.typing as npt


def apply_softmax(
    logits: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """
    Apply numerically stable softmax.

    Supported input shapes:
        - (N,)   : single sample
        - (B, N) : batch of samples

    Args:
        logits:
            Logit tensor whose last dimension corresponds
            to class logits.

    Returns:
        Softmax probabilities with the same shape as the input.
    """

    # ---------------------------------------------------------
    # Input validation
    # ---------------------------------------------------------
    if logits.ndim not in (1, 2):
        raise ValueError(
            f"Expected logits with shape (N,) or (B, N), got {logits.shape}."
        )

    if logits.shape[0] == 0:
        raise ValueError(
            f"Empty logits tensor not allowed: received shape {logits.shape}. "
            f"At least one element is required along axis 0."
        )

    axis = -1

    # ---------------------------------------------------------
    # Numerically stable softmax
    # ---------------------------------------------------------
    shifted = logits - np.max(
        logits,
        axis=axis,
        keepdims=True,
    )

    exp_shifted = np.exp(shifted)

    return exp_shifted / np.sum(
        exp_shifted,
        axis=axis,
        keepdims=True,
    )


Prediction = dict[str, np.int64 | np.float32]


def extract_topk_predictions(
    probabilities: npt.NDArray[np.float32],
    topk: int,
) -> list[Prediction] | list[list[Prediction]]:
    """
    Extract Top-K predictions from classification probabilities.

    Supported input shapes:
        - (N,)   : single sample
        - (B, N) : batch of samples

    Args:
        probabilities:
            Probability tensor whose last dimension corresponds
            to class probabilities.

        topk:
            Number of predictions to return.

    Returns:
        For input shape (N,):
            List[Prediction]

        For input shape (B, N):
            List[List[Prediction]]

        Predictions are sorted by descending probability.
    """

    # ---------------------------------------------------------
    # Input validation
    # ---------------------------------------------------------
    if probabilities.ndim not in (1, 2):
        raise ValueError(
            f"Expected probabilities with shape (N,) or (B, N), got {probabilities.shape}."
        )

    if probabilities.shape[0] == 0:
        raise ValueError(
            f"Empty probabilities tensor not allowed: received shape {probabilities.shape}. "
            f"At least one element is required along axis 0."
        )

    axis = -1

    num_classes = probabilities.shape[axis]
    k = min(topk, num_classes)

    # ---------------------------------------------------------
    # Select top-k (unordered)
    # ---------------------------------------------------------
    topk_idx = np.argpartition(
        probabilities,
        kth=-k,
        axis=axis,
    )[..., -k:]

    # ---------------------------------------------------------
    # Gather top-k probabilities
    # ---------------------------------------------------------
    topk_vals = np.take_along_axis(
        probabilities,
        topk_idx,
        axis=axis,
    )

    # ---------------------------------------------------------
    # Sort top-k descending
    # ---------------------------------------------------------
    order = np.argsort(
        topk_vals,
        axis=axis,
    )[..., ::-1]

    topk_idx = np.take_along_axis(
        topk_idx,
        order,
        axis=axis,
    )

    topk_vals = np.take_along_axis(
        topk_vals,
        order,
        axis=axis,
    )

    # ---------------------------------------------------------
    # Format output
    # ---------------------------------------------------------
    def to_predictions(
        indices: npt.NDArray[np.int64],
        scores: npt.NDArray[np.float32],
    ) -> list[Prediction]:
        return [
            {
                "label": np.int64(idx),
                "score": np.float32(score),
            }
            for idx, score in zip(indices, scores)
        ]

    if probabilities.ndim == 1:
        return to_predictions(topk_idx, topk_vals)

    return [
        to_predictions(idx_row, val_row)
        for idx_row, val_row in zip(topk_idx, topk_vals)
    ]


def generate_topk_predictions(
    output_tensor: npt.NDArray[np.float32], topk: int
) -> list[Prediction] | list[list[Prediction]]:
    """
    Generate Top-K classification predictions from model output.

    Supports:
        - (N,)
        - (B, N)

    Works for logits or probabilities.

    Args:
        output_tensor:
            Model output tensor of shape:
            - (num_classes,)
            - (1, num_classes)

        topk:
            Number of predictions to return.

    Returns:
        Top-K predictions sorted by descending confidence.
    """

    return extract_topk_predictions(
        probabilities=apply_softmax(output_tensor),
        topk=topk,
    )
