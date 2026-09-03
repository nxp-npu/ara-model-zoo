# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import pytest
import numpy as np

from core.python.postprocess.interfaces import (
    SegDetectionOutput,
    FaceDetectionOutput,
    ClassificationOutput,
    PoseEstimationOutput,
    ObjectDetectionOutput,
)


@pytest.mark.parametrize(
    "config",
    [
        SegDetectionOutput(
            image_name="image_001.jpg",
            boxes=np.array(
                [
                    [10, 20, 100, 200],
                    [50, 60, 150, 250],
                ],
                dtype=np.float32,
            ),
            scores=np.array([0.95, 0.87], dtype=np.float32),
            classes=np.array([0, 1], dtype=np.int64),
            masks=np.array(
                [
                    [[0, 1], [1, 0]],
                    [[1, 1], [0, 0]],
                ],
                dtype=np.uint8,
            ),
        ),
        FaceDetectionOutput(
            image_name="image_001.jpg",
            boxes=np.array(
                [
                    [10, 20, 100, 200],
                    [50, 60, 150, 250],
                ],
                dtype=np.float32,
            ),
            scores=np.array([0.95, 0.87], dtype=np.float32),
            classes=np.array([0, 1], dtype=np.int64),
            # print_table() currently assumes keypoints.shape == (N, M, 3).
            # Supporting other keypoint formats is outside the scope of this test.
            keypoints=np.array(
                [
                    [
                        [100, 120, 1],
                        [95, 115, 1],
                        [105, 115, 0],
                    ],
                    [
                        [200, 220, 1],
                        [195, 215, 1],
                        [205, 215, 1],
                    ],
                ],
                dtype=np.float32,
            ),
        ),
        ClassificationOutput(
            image_name="image_001.jpg",
            top_n=[
                {
                    "score": np.float32(0.95),
                    "label": np.int64(0),
                },
                {
                    "score": np.float32(0.87),
                    "label": np.int64(1),
                },
            ],
        ),
        PoseEstimationOutput(
            image_name="image_001.jpg",
            boxes=np.array(
                [
                    [10, 20, 100, 200],
                    [50, 60, 150, 250],
                ],
                dtype=np.float32,
            ),
            scores=np.array([0.95, 0.87], dtype=np.float32),
            classes=np.array([0, 1], dtype=np.int64),
            # print_table() currently assumes keypoints.shape == (N, M, 3).
            # Supporting other keypoint formats is outside the scope of this test.
            keypoints=np.array(
                [
                    [
                        [100, 120, 1],
                        [95, 115, 1],
                        [105, 115, 0],
                    ],
                    [
                        [200, 220, 1],
                        [195, 215, 1],
                        [205, 215, 1],
                    ],
                ],
                dtype=np.float32,
            ),
        ),
        ObjectDetectionOutput(
            image_name="image_001.jpg",
            boxes=np.array(
                [
                    [10, 20, 100, 200],
                    [50, 60, 150, 250],
                ],
                dtype=np.float32,
            ),
            scores=np.array([0.95, 0.87], dtype=np.float32),
            classes=np.array([0, 1], dtype=np.int64),
        ),
    ],
)
def test_print_table_variants(capsys, config):
    """Verify that print_to_console() runs without error and writes to stdout."""

    config.print_to_console()

    captured = capsys.readouterr()

    assert captured.out
    assert captured.err == ""
