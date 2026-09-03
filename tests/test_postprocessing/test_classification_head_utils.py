# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

from typing import cast

import pytest
import numpy as np

from core.python.postprocess.classification.classification_utils import (
    apply_softmax,
    extract_topk_predictions,
    generate_topk_predictions,
    Prediction,
)


# =========================================================
# TEST SOFTMAX NORMALIZATION
# =========================================================
class TestSoftmax:
    """
    Contract tests for apply_softmax.

    Supported input shapes:
        - (N,)
        - (B, N)

    Guarantees:
        - Output shape matches input shape.
        - Probabilities are finite and lie in [0, 1].
        - Probabilities sum to 1 along the class axis.
        - Argmax is preserved.
        - Numerically stable for extreme values.
        - Rejects unsupported dimensions.
    """

    # ---------------------------------------------------------
    # Basic correctness
    # ---------------------------------------------------------

    @pytest.mark.parametrize(
        "logits",
        [
            np.array(
                [1.0, 2.0, 3.0],
                dtype=np.float32,
            ),
            np.array(
                [
                    [1.0, 2.0, 3.0],
                    [3.0, 2.0, 1.0],
                ],
                dtype=np.float32,
            ),
        ],
        ids=[
            "1d",
            "2d_batch",
        ],
    )
    def test_preserves_shape_and_probability_sum(
        self,
        logits,
    ):
        result = apply_softmax(logits)

        assert result.shape == logits.shape

        np.testing.assert_allclose(
            result.sum(axis=-1),
            np.ones(logits.shape[:-1] or ()),
            atol=1e-6,
        )

    # ---------------------------------------------------------
    # Probability constraints
    # ---------------------------------------------------------

    @pytest.mark.parametrize(
        "logits",
        [
            np.array(
                [0.5, 2.0, -1.0],
                dtype=np.float32,
            ),
            np.array(
                [
                    [0.5, 2.0, -1.0],
                    [5.0, -3.0, 7.0],
                ],
                dtype=np.float32,
            ),
        ],
        ids=[
            "1d",
            "2d_batch",
        ],
    )
    def test_outputs_are_valid_probabilities(
        self,
        logits,
    ):
        result = apply_softmax(logits)

        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    # ---------------------------------------------------------
    # Ordering preservation
    # ---------------------------------------------------------

    @pytest.mark.parametrize(
        "logits",
        [
            np.array(
                [1.0, 5.0, 3.0],
                dtype=np.float32,
            ),
            np.array(
                [
                    [1.0, 5.0, 3.0],
                    [10.0, -5.0, 2.0],
                ],
                dtype=np.float32,
            ),
        ],
        ids=[
            "1d",
            "2d_batch",
        ],
    )
    def test_argmax_is_preserved(
        self,
        logits,
    ):
        result = apply_softmax(logits)

        np.testing.assert_array_equal(
            np.argmax(result, axis=-1),
            np.argmax(logits, axis=-1),
        )

    # ---------------------------------------------------------
    # Numerical stability
    # ---------------------------------------------------------

    @pytest.mark.parametrize(
        "logits",
        [
            np.array(
                [10000.0, 10001.0, 10002.0],
                dtype=np.float32,
            ),
            np.array(
                [
                    [10000.0, 10001.0, 10002.0],
                    [-10000.0, -9999.0, -9998.0],
                ],
                dtype=np.float32,
            ),
        ],
        ids=[
            "large_positive_1d",
            "large_positive_negative_2d",
        ],
    )
    def test_numerically_stable(
        self,
        logits,
    ):
        result = apply_softmax(logits)

        assert np.isfinite(result).all()

        np.testing.assert_allclose(
            result.sum(axis=-1),
            np.ones(logits.shape[:-1] or ()),
            atol=1e-6,
        )

    def test_extreme_value_dominance(self):
        logits = np.array(
            [
                10000.0,
                10001.0,
                10002.0,
                0.0001,
            ],
            dtype=np.float32,
        )

        result = apply_softmax(logits)

        assert np.isfinite(result).all()

        np.testing.assert_allclose(
            result.sum(),
            1.0,
            atol=1e-6,
        )

        assert result[-1] < 1e-6

    # ---------------------------------------------------------
    # Unsupported dimensions
    # ---------------------------------------------------------

    @pytest.mark.parametrize(
        "logits",
        [
            np.array(
                42.0,
                dtype=np.float32,
            ),
            np.ones(
                (2, 3, 4),
                dtype=np.float32,
            ),
        ],
        ids=[
            "0d_scalar",
            "3d_tensor",
        ],
    )
    def test_rejects_unsupported_dimensions(
        self,
        logits,
    ):
        with pytest.raises(
            ValueError,
            match=r"Expected logits with shape",
        ):
            apply_softmax(logits)

    @pytest.mark.parametrize(
        "empty_logits",
        [
            np.array([], dtype=np.float32),  # (0,)
            np.empty((0, 5), dtype=np.float32),  # (0, N)
        ],
        ids=["empty_1d", "empty_2d"],
    )
    def test_rejects_empty_logits(self, empty_logits):
        with pytest.raises(ValueError, match="Empty"):
            apply_softmax(empty_logits)

    # ---------------------------------------------------------
    # Edge cases
    # ---------------------------------------------------------

    def test_single_element_vector(self):
        logits = np.array(
            [42.0],
            dtype=np.float32,
        )

        result = apply_softmax(logits)

        assert result.shape == (1,)

        np.testing.assert_allclose(
            result,
            np.array([1.0], dtype=np.float32),
        )

    def test_identical_logits_1d(self):
        logits = np.array(
            [1.0, 1.0, 1.0],
            dtype=np.float32,
        )

        result = apply_softmax(logits)

        np.testing.assert_allclose(
            result,
            np.array(
                [1 / 3, 1 / 3, 1 / 3],
                dtype=np.float32,
            ),
            atol=1e-6,
        )

    def test_identical_logits_2d(self):
        logits = np.array(
            [
                [1.0, 1.0, 1.0],
                [5.0, 5.0, 5.0],
            ],
            dtype=np.float32,
        )

        result = apply_softmax(logits)

        expected = np.full(
            (2, 3),
            1 / 3,
            dtype=np.float32,
        )

        np.testing.assert_allclose(
            result,
            expected,
            atol=1e-6,
        )


# =========================================================
# TEST EXTRACT TOP-K
# =========================================================
class TestExtractTopK:
    """
    Contract tests for extract_topk_predictions.

    Guarantees:
    - Works for (N,) and (B, N)
    - Correct top-k ranking
    - Stable index-score consistency
    - Safe behavior under k > num_classes
    - No silent shape bugs across batch dimension
    """

    # -----------------------------------------------------
    # 1D: Basic correctness
    # -----------------------------------------------------
    def test_basic_topk_correctness_1d(self):
        probs = np.array([0.1, 0.8, 0.05, 0.05], dtype=np.float32)

        result = cast(list[Prediction], extract_topk_predictions(probs, topk=2))

        assert len(result) == 2

        assert result[0]["label"] == 1
        assert result[0]["score"] == pytest.approx(0.8)

        assert result[1]["label"] == 0
        assert result[1]["score"] == pytest.approx(0.1)

    # -----------------------------------------------------
    # 1D: Ordering correctness
    # -----------------------------------------------------
    def test_output_sorted_descending_1d(self):
        probs = np.array([0.2, 0.6, 0.9, 0.1], dtype=np.float32)

        result = cast(list[Prediction], extract_topk_predictions(probs, topk=3))

        scores = [x["score"] for x in result]

        assert scores == sorted(scores, reverse=True)

    # -----------------------------------------------------
    # 1D: k clamping behavior
    # -----------------------------------------------------
    def test_topk_clamping_1d(self):
        probs = np.array([0.2, 0.8], dtype=np.float32)

        result = extract_topk_predictions(probs, topk=100)

        assert len(result) == 2

    # -----------------------------------------------------
    # 1D: Duplicate probability handling
    # -----------------------------------------------------
    def test_duplicate_probabilities_1d(self):
        probs = np.array([0.5, 0.5, 0.2], dtype=np.float32)

        result = cast(list[Prediction], extract_topk_predictions(probs, topk=2))

        assert len(result) == 2

        # ordering must still be valid
        assert result[0]["score"] >= result[1]["score"]

        # label-score consistency always holds
        for item in result:
            assert item["score"] == pytest.approx(probs[item["label"]])

    # -----------------------------------------------------
    # 1D: Edge case - single class
    # -----------------------------------------------------
    def test_single_class_1d(self):
        probs = np.array([0.99], dtype=np.float32)

        result = cast(list[Prediction], extract_topk_predictions(probs, topk=5))

        assert result == [{"label": np.int64(0), "score": pytest.approx(0.99)}]

    # -----------------------------------------------------
    # 2D: Batch correctness
    # -----------------------------------------------------
    def test_basic_topk_correctness_2d(self):
        probs = np.array(
            [
                [0.1, 0.8, 0.05, 0.05],
                [0.4, 0.3, 0.2, 0.1],
            ],
            dtype=np.float32,
        )

        result = cast(list[list[Prediction]], extract_topk_predictions(probs, topk=2))

        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[1]) == 2

        # first sample top-1 should be index 1
        assert result[0][0]["label"] == 1

        # second sample top-1 should be index 0
        assert result[1][0]["label"] == 0

    # -----------------------------------------------------
    # 2D: Order correctness per batch row
    # -----------------------------------------------------
    def test_batch_ordering_descending_2d(self):
        probs = np.array(
            [
                [0.2, 0.6, 0.9],
                [0.7, 0.1, 0.2],
            ],
            dtype=np.float32,
        )

        result = cast(list[list[Prediction]], extract_topk_predictions(probs, topk=2))

        for row in result:
            scores = [x["score"] for x in row]
            assert scores == sorted(scores, reverse=True)

    # -----------------------------------------------------
    # 2D: k clamping
    # -----------------------------------------------------
    def test_topk_clamping_2d(self):
        probs = np.array(
            [
                [0.1, 0.2],
                [0.3, 0.4],
            ],
            dtype=np.float32,
        )

        result = cast(list[list[Prediction]], extract_topk_predictions(probs, topk=10))

        assert all(len(row) == 2 for row in result)

    # -----------------------------------------------------
    # 2D: Index-score consistency
    # -----------------------------------------------------
    def test_index_score_consistency_2d(self):
        probs = np.array(
            [
                [0.3, 0.7, 0.2],
                [0.9, 0.05, 0.05],
            ],
            dtype=np.float32,
        )

        result = cast(list[list[Prediction]], extract_topk_predictions(probs, topk=3))

        for batch_idx, row in enumerate(result):
            for item in row:
                label = item["label"]
                score = item["score"]

                assert score == pytest.approx(probs[batch_idx][label])

    # -----------------------------------------------------
    # Contract enforcement: invalid dims
    # -----------------------------------------------------
    @pytest.mark.parametrize(
        "invalid_input",
        [
            np.array([[[[0.1, 0.2]]]], dtype=np.float32),  # (1, 1, 1, N)
            np.array([[[0.1, 0.2]]], dtype=np.float32),  # (1, 1, N)
        ],
    )
    def test_rejects_invalid_dimensions(self, invalid_input):
        with pytest.raises(ValueError):
            extract_topk_predictions(invalid_input, topk=2)

    @pytest.mark.parametrize(
        "empty_input",
        [
            np.array([], dtype=np.float32),  # (0,)
            np.empty((0, 5), dtype=np.float32),  # (0, N)
        ],
    )
    def test_rejects_empty_input(self, empty_input):
        with pytest.raises(ValueError, match="Empty"):
            extract_topk_predictions(empty_input, topk=2)


# =========================================================
# TEST generate_topk_predictions
# =========================================================
class TestGenerateTopKPredictions:
    """
    Contract tests for generate_topk_predictions.

    This is a pipeline test:
        logits → softmax → topk extraction

    We validate:
    - correct integration behavior
    - ranking correctness
    - batching support
    - error propagation from softmax layer
    """

    # -----------------------------------------------------
    # 1. Basic correctness (1D logits)
    # -----------------------------------------------------
    def test_basic_topk_1d_logits(self):
        logits = np.array([1.0, 200.0, 3.0], dtype=np.float32)

        result = cast(list[Prediction], generate_topk_predictions(logits, topk=2))

        assert len(result) == 2

        assert result[0]["label"] == 1
        assert result[1]["label"] == 2

        assert result[0]["score"] > result[1]["score"]

    # -----------------------------------------------------
    # 2. Ordering correctness
    # -----------------------------------------------------
    def test_topk_ordering_descending(self):
        logits = np.array([0.1, 0.5, 0.2, 0.9], dtype=np.float32)

        result = cast(list[Prediction], generate_topk_predictions(logits, topk=3))

        scores = [x["score"] for x in result]

        assert scores == sorted(scores, reverse=True)

    # -----------------------------------------------------
    # 3. topk clamping
    # -----------------------------------------------------
    def test_topk_clamping(self):
        logits = np.array([0.2, 0.8], dtype=np.float32)

        result = cast(list[Prediction], generate_topk_predictions(logits, topk=10))

        assert len(result) == 2

    # -----------------------------------------------------
    # 4. Batch input (1, N)
    # -----------------------------------------------------
    def test_batch_size_one_2d_input(self):
        logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        result = cast(list[list[Prediction]], generate_topk_predictions(logits, topk=2))

        assert len(result) == 1
        assert result[0][0]["label"] == 2

    # -----------------------------------------------------
    # 5. Multi-batch input (B, N)
    # -----------------------------------------------------
    def test_multi_batch_output(self):
        logits = np.array(
            [
                [1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0],
            ],
            dtype=np.float32,
        )

        result = cast(list[list[Prediction]], generate_topk_predictions(logits, topk=1))

        assert len(result) == 2
        assert result[0][0]["label"] == 2
        assert result[1][0]["label"] == 0

    # -----------------------------------------------------
    # 6. Single element edge case
    # -----------------------------------------------------
    def test_single_class(self):
        logits = np.array([42.0], dtype=np.float32)

        result = generate_topk_predictions(logits, topk=1)

        assert result == [{"label": 0, "score": 1.0}]

    # -----------------------------------------------------
    # 7. Invalid shape propagation (from softmax)
    # -----------------------------------------------------
    @pytest.mark.parametrize(
        "invalid_input",
        [
            np.array(5.0, dtype=np.float32),  # scalar (0D)
            np.ones((2, 3, 4), dtype=np.float32),  # 3D tensor
        ],
        ids=["scalar_0d", "invalid_3d"],
    )
    def test_rejects_invalid_shapes(self, invalid_input):
        with pytest.raises(ValueError, match="Expected logits"):
            generate_topk_predictions(invalid_input, topk=2)

    # -----------------------------------------------------
    # 8. Emptiness propagation
    # -----------------------------------------------------
    @pytest.mark.parametrize(
        "empty_input",
        [
            np.array([], dtype=np.float32),  # (0,)
            np.empty((0, 5), dtype=np.float32),  # (0, N)
        ],
        ids=["empty_1d", "empty_2d"],
    )
    def test_rejects_empty_input(self, empty_input):
        with pytest.raises(ValueError, match="Empty"):
            generate_topk_predictions(empty_input, topk=2)
