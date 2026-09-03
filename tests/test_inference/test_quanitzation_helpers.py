# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import numpy as np
import pytest

from core.python.inference.interfaces import ParameterMetaData, QuantizationParams
from core.python.inference.sessions.ara_session import (
    ValidQmodes,
    _determine_array_min_max,
    dequantize_array,
    quantize_array,
)


def make_params(
    array: np.ndarray,
    dtype: type[np.generic],
    qn: float = 1.0,
    scale: float = 1.0,
    offset: int = 0,
    qmode: int = ValidQmodes.QMODE_0.value,
    is_signed: bool = False,
) -> ParameterMetaData:
    q_params = QuantizationParams(
        qn=qn,
        scale=scale,
        offset=offset,
        is_signed=is_signed,
        is_float=False,
        qmode=qmode,
    )
    return ParameterMetaData(
        name="test",
        shape=list(array.shape),
        size=array.nbytes,
        dtype=dtype,
        q_params=q_params,
    )


class TestDetermineArrayMinMax:
    @pytest.mark.parametrize(
        "dtype,expected_min,expected_max",
        [
            (np.int8, -128, 127),
            (np.uint8, 0, 255),
            (np.int16, -32768, 32767),
            (np.uint16, 0, 65535),
            (np.int32, -2147483648, 2147483647),
            (np.uint32, 0, 4294967295),
        ],
    )
    def test_valid_integer_dtypes(self, dtype, expected_min, expected_max):
        min_val, max_val = _determine_array_min_max(dtype)
        assert min_val == expected_min
        assert max_val == expected_max

    def test_none_dtype_raises_error(self):
        with pytest.raises(ValueError, match="Expected integer dtype"):
            _determine_array_min_max(None)  # type: ignore

    @pytest.mark.parametrize(
        "non_integer_dtype",
        [np.float32, np.float64, np.bool_, np.complex64],
    )
    def test_non_integer_dtype_raises_error(self, non_integer_dtype):
        with pytest.raises(ValueError, match="Expected integer dtype"):
            _determine_array_min_max(non_integer_dtype)


class TestQuantizeArray:
    @pytest.mark.parametrize(
        "array,dtype,qn,scale,expected",
        [
            # qn=1, scale=1: floor(x * 1 * 1 + 0.5) = round(x)
            (
                np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                np.int8,
                1.0,
                1.0,
                np.array([0, 1, 2, 3, 4], dtype=np.int8),
            ),
            (
                np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                np.uint8,
                1.0,
                1.0,
                np.array([0, 1, 2, 3, 4], dtype=np.uint8),
            ),
            (
                np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                np.int16,
                1.0,
                1.0,
                np.array([0, 1, 2, 3, 4], dtype=np.int16),
            ),
            # qn=2, scale=1: floor(x * 2 * 1 + 0.5) = floor(2x + 0.5)
            # [0, 2, 4, 6, 8] + 0.5 -> floor -> [0, 2, 4, 6, 8]
            (
                np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                np.uint8,
                2.0,
                1.0,
                np.array([0, 2, 4, 6, 8], dtype=np.uint8),
            ),
            # qn=1, scale=2: floor(x * 1 * 2 + 0.5) = floor(2x + 0.5)
            # same result as above
            (
                np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                np.uint8,
                1.0,
                2.0,
                np.array([0, 2, 4, 6, 8], dtype=np.uint8),
            ),
            # qn=0.5, scale=1: floor(x * 0.5 + 0.5)
            # [0, 0.5, 1, 1.5, 2] + 0.5 -> [0.5, 1.0, 1.5, 2.0, 2.5] -> floor -> [0, 1, 1, 2, 2]
            (
                np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                np.uint8,
                0.5,
                1.0,
                np.array([0, 1, 1, 2, 2], dtype=np.uint8),
            ),
            # fractional inputs: floor(x * 1 * 1 + 0.5) rounds to nearest
            # [0.4+0.5=0.9->0, 0.5+0.5=1.0->1, 1.4+0.5=1.9->1, 1.5+0.5=2.0->2]
            (
                np.array([0.4, 0.5, 1.4, 1.5]),
                np.uint8,
                1.0,
                1.0,
                np.array([0, 1, 1, 2], dtype=np.uint8),
            ),
            # negative values with signed dtype: floor(x * 1 * 1 + 0.5)
            # [-4+0.5=-3.5->-4, -2+0.5=-1.5->-2, 0+0.5=0.5->0, 2+0.5=2.5->2, 4+0.5=4.5->4]
            (
                np.array([-4.0, -2.0, 0.0, 2.0, 4.0]),
                np.int8,
                1.0,
                1.0,
                np.array([-4, -2, 0, 2, 4], dtype=np.int8),
            ),
            # large int16 range
            (
                np.array([0.0, 100.0, 1000.0, 10000.0, 32767.0]),
                np.int16,
                1.0,
                1.0,
                np.array([0, 100, 1000, 10000, 32767], dtype=np.int16),
            ),
        ],
    )
    def test_quantize_default_mode(self, array, dtype, qn, scale, expected):
        params = make_params(array, dtype, qn=qn, scale=scale)
        result = quantize_array(array, params)
        assert result.dtype == dtype
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "array,dtype,qn,offset,expected",
        [
            # qmode_9: floor((array * (1/qn)) + offset + 0.5)
            # qn=1, offset=0: floor(x * 1 + 0 + 0.5) = round(x)
            (
                np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                np.int8,
                1.0,
                0,
                np.array([0, 1, 2, 3, 4], dtype=np.int8),
            ),
            (
                np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                np.uint8,
                1.0,
                0,
                np.array([0, 1, 2, 3, 4], dtype=np.uint8),
            ),
            # qn=2, offset=0: floor(x * 0.5 + 0.5)
            # [0, 0.5, 1, 1.5, 2] + 0.5 -> floor -> [0, 1, 1, 2, 2]
            (
                np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                np.uint8,
                2.0,
                0,
                np.array([0, 1, 1, 2, 2], dtype=np.uint8),
            ),
            # qn=0.5, offset=0: floor(x * 2 + 0.5)
            # [0, 2, 4, 6, 8] + 0.5 -> floor -> [0, 2, 4, 6, 8]
            (
                np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                np.uint8,
                0.5,
                0,
                np.array([0, 2, 4, 6, 8], dtype=np.uint8),
            ),
            # qn=1, offset=5: floor(x * 1 + 5 + 0.5)
            # [5.5, 6.5, 7.5, 8.5, 9.5] -> floor -> [5, 6, 7, 8, 9]
            (
                np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                np.uint8,
                1.0,
                5,
                np.array([5, 6, 7, 8, 9], dtype=np.uint8),
            ),
        ],
    )
    def test_quantize_qmode_9(self, array, dtype, qn, offset, expected):
        params = make_params(
            array, dtype, qn=qn, offset=offset, qmode=ValidQmodes.QMODE_9.value
        )
        result = quantize_array(array, params)
        assert result.dtype == dtype
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "array,dtype,expected",
        [
            # values below uint8 min (0) clip to 0, above uint8 max (255) clip to 255
            (
                np.array([-100.0, 0.0, 100.0, 300.0]),
                np.uint8,
                np.array([0, 0, 100, 255], dtype=np.uint8),
            ),
            # values below int8 min (-128) clip to -128, above int8 max (127) clip to 127
            (
                np.array([-200.0, -128.0, 0.0, 127.0, 200.0]),
                np.int8,
                np.array([-128, -128, 0, 127, 127], dtype=np.int8),
            ),
            # values below int16 min clip to -32768, above max clip to 32767
            (
                np.array([-40000.0, -32768.0, 0.0, 32767.0, 40000.0]),
                np.int16,
                np.array([-32768, -32768, 0, 32767, 32767], dtype=np.int16),
            ),
        ],
    )
    def test_quantize_clipping_behavior(self, array, dtype, expected):
        params = make_params(array, dtype)
        result = quantize_array(array, params)
        np.testing.assert_array_equal(result, expected)


class TestDequantizeArray:
    @pytest.mark.parametrize(
        "array,dtype,qn,scale,offset,expected",
        [
            # qmode_0: (output + offset) / (qn * scale)
            # qn=1, scale=1, offset=0: values pass through
            (
                np.array([0, 1, 2, 3, 4], dtype=np.int8),
                np.int8,
                1.0,
                1.0,
                0,
                np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            ),
            (
                np.array([0, 1, 2, 3, 4], dtype=np.uint8),
                np.uint8,
                1.0,
                1.0,
                0,
                np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            ),
            (
                np.array([0, 1, 2, 3, 4], dtype=np.int16),
                np.int16,
                1.0,
                1.0,
                0,
                np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            ),
            # qn=2, scale=1, offset=0: (x + 0) / (2 * 1) = x / 2
            (
                np.array([0, 2, 4, 6, 8], dtype=np.uint8),
                np.uint8,
                2.0,
                1.0,
                0,
                np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            ),
            # qn=1, scale=2, offset=0: (x + 0) / (1 * 2) = x / 2
            (
                np.array([0, 2, 4, 6, 8], dtype=np.uint8),
                np.uint8,
                1.0,
                2.0,
                0,
                np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            ),
            # qn=1, scale=1, offset=5: (x + 5) / 1 = x + 5
            (
                np.array([0, 1, 2, 3, 4], dtype=np.uint8),
                np.uint8,
                1.0,
                1.0,
                5,
                np.array([5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32),
            ),
            # negative values with signed dtype
            (
                np.array([-4, -2, 0, 2, 4], dtype=np.int8),
                np.int8,
                1.0,
                1.0,
                0,
                np.array([-4.0, -2.0, 0.0, 2.0, 4.0], dtype=np.float32),
            ),
        ],
    )
    def test_dequantize_default_mode(self, array, dtype, qn, scale, offset, expected):
        params = make_params(array, dtype, qn=qn, scale=scale, offset=offset)
        result = dequantize_array(array, params)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, expected)

    @pytest.mark.parametrize(
        "array,dtype,qn,scale,offset,expected",
        [
            # qmode_9: (output - offset) / (qn * scale)
            # qn=1, scale=1, offset=0: values pass through
            (
                np.array([0, 1, 2, 3, 4], dtype=np.int8),
                np.int8,
                1.0,
                1.0,
                0,
                np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            ),
            (
                np.array([0, 1, 2, 3, 4], dtype=np.uint8),
                np.uint8,
                1.0,
                1.0,
                0,
                np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            ),
            # qn=1, offset=5: (x - 5) / 1 = x - 5
            (
                np.array([5, 6, 7, 8, 9], dtype=np.uint8),
                np.uint8,
                1.0,
                1.0,
                5,
                np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            ),
            # qn=0.5, scale=1, offset=0: (x - 0) / (0.5 * 1) = x * 2
            (
                np.array([0, 2, 4, 6, 8], dtype=np.uint8),
                np.uint8,
                0.5,
                1.0,
                0,
                np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            ),
        ],
    )
    def test_dequantize_qmode_9(self, array, dtype, qn, scale, offset, expected):
        params = make_params(
            array,
            dtype,
            qn=qn,
            scale=scale,
            offset=offset,
            qmode=ValidQmodes.QMODE_9.value,
        )
        result = dequantize_array(array, params)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, expected)


class TestQuantizeDequantizeRoundtrip:
    def test_roundtrip_qmode_0_integer_values(self):
        # integer inputs survive roundtrip exactly with scale=1, qn=1
        original = np.array([0.0, 1.0, 2.0, 100.0, 255.0])
        params = make_params(original, np.uint8)
        quantized = quantize_array(original, params)
        dequantized = dequantize_array(quantized, params)
        np.testing.assert_array_equal(dequantized, original)

    def test_roundtrip_qmode_0_with_scale(self):
        # qn=2, scale=1: quantize multiplies by 2, dequantize divides by 2
        original = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        params = make_params(original, np.uint8, qn=2.0, scale=1.0)
        quantized = quantize_array(original, params)
        dequantized = dequantize_array(quantized, params)
        np.testing.assert_array_almost_equal(dequantized, original, decimal=0)

    def test_roundtrip_qmode_9_integer_values(self):
        # integer inputs survive roundtrip exactly with qn=1, offset=0
        original = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        params = make_params(original, np.uint8, qmode=ValidQmodes.QMODE_9.value)
        quantized = quantize_array(original, params)
        dequantized = dequantize_array(quantized, params)
        np.testing.assert_array_equal(dequantized, original)

    def test_roundtrip_qmode_9_with_offset(self):
        # qmode_9 quantize adds offset, dequantize subtracts it — should cancel out
        original = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        params = make_params(
            original, np.uint8, qn=1.0, offset=5, qmode=ValidQmodes.QMODE_9.value
        )
        quantized = quantize_array(original, params)
        dequantized = dequantize_array(quantized, params)
        np.testing.assert_array_equal(dequantized, original)

    def test_roundtrip_signed_int16(self):
        original = np.array([10.0, 20.0, 30.0])
        params = make_params(original, np.int16, is_signed=True)
        quantized = quantize_array(original, params)
        dequantized = dequantize_array(quantized, params)
        np.testing.assert_array_equal(dequantized.astype(int), original.astype(int))

    def test_roundtrip_signed_negatives(self):
        original = np.array([-50.0, -25.0, 0.0, 25.0, 50.0])
        params = make_params(original, np.int8, scale=1.0, is_signed=True)
        quantized = quantize_array(original, params)
        dequantized = dequantize_array(quantized, params)
        np.testing.assert_array_equal(dequantized, original)
