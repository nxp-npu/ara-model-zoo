#!/usr/bin/env bash
# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

set -euo pipefail


# ─── Validate environment ─────────────────────────────────────────────────────

if [[ -z "${MODEL_PIPELINE_ROOT:-}" ]]; then
  error "MODEL_PIPELINE_ROOT is unset. Please export it before running this script."
  return 1
fi

if [[ -z "${ARA_NOUS_ROOT:-}" ]]; then
  error "ARA_NOUS_ROOT is unset. Please export it before running this script."
  return 1
fi

if [[ -z "${OUTPUT_DIR:-}" ]]; then
  error "OUTPUT_DIR is unset. Please export it before running this script."
  return 1
fi

source "${ARA_NOUS_ROOT}/core/shell/logger.sh"


# ─── Constants ───────────────────────────────────────────────────────────────
# Pre-quantized assets produced by the quantization (sparrow) flow. The optimized
# compile flow consumes these directly instead of exporting from timm.
#
# NOTE: update OPTIMIZED_SRC_DIR to the published /auto/share location once the
# quantized tiny_vit is promoted there. It currently points at the sparrow_eng
# workdir that produced the quantized model + encodings.

OPTIMIZED_SRC_DIR="/auto/share/sw/common/modelzoo_eng/optimized_models/classification/Tiny_vit"

SRC_ONNX="${OPTIMIZED_SRC_DIR}/quantized_model.onnx"
SRC_ENCODINGS="${OPTIMIZED_SRC_DIR}/quantization_encodings.json"

ASSETS_DIR="${OUTPUT_DIR}/compiled_model"

DST_ONNX="${ASSETS_DIR}/quantized_model.onnx"
DST_ENCODINGS="${ASSETS_DIR}/quantization_encodings.json"


# ─── Validate source assets ──────────────────────────────────────────────────

if [[ ! -f "${SRC_ONNX}" ]]; then
  error "Quantized ONNX not found at ${SRC_ONNX}"
  return 1
fi

if [[ ! -f "${SRC_ENCODINGS}" ]]; then
  error "Quantization encodings not found at ${SRC_ENCODINGS}"
  return 1
fi


# ─── Copy optimized assets into compiled_model ───────────────────────────────

mkdir -p "${ASSETS_DIR}"

info "Copying quantized ONNX to: ${DST_ONNX}"
cp -f "${SRC_ONNX}" "${DST_ONNX}"

info "Copying quantization encodings to: ${DST_ENCODINGS}"
cp -f "${SRC_ENCODINGS}" "${DST_ENCODINGS}"

info "Optimized assets staged in: ${ASSETS_DIR}"
