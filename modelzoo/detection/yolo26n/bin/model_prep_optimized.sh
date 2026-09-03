#!/usr/bin/env bash
# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.
# ─────────────────────────────────────────────────────────────────────────────
# model_prep_optmized.sh — transfer the optimized onnx and encodings.json file
#
# ARA_NOUS_ROOT - path to the ara-model-zoo repo root
# MODEL_PIPELINE_ROOT - path to the yolo26n directory
#
# Artifacts produced:
#   compiled_model/quantized_model.onnx  — ONNX file
#   compiled_model/quantization_encodings.json - encodings file
#
# ./bin/model_prep_optimized.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Resolve paths from script location ───────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PIPELINE_ROOT="${MODEL_PIPELINE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ARA_NOUS_ROOT="${ARA_NOUS_ROOT:-$(cd "${SCRIPT_DIR}/../../../../.." && pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_PIPELINE_ROOT}/output}"

source "${ARA_NOUS_ROOT}/core/shell/logger.sh"

# ── Paths ────────────────────────────────────────────────────────────────────
ASSETS_DIR="${OUTPUT_DIR}/compiled_model"
ONNX_OUTPUT="${ASSETS_DIR}/quantized_model.onnx"
QUANT_ENCODINGS="${ASSETS_DIR}/quantization_encodings.json"

# ── Create output directory ──────────────────────────────────────────────────
mkdir -p "${ASSETS_DIR}"

# ── Copy files to output directory ──────────────────────────────────────────────────
cp /auto/share/sw/common/modelzoo_eng/optimized_models/detection/yolo26n/quantized_model.onnx "${ONNX_OUTPUT}"
cp /auto/share/sw/common/modelzoo_eng/optimized_models/detection/yolo26n/quantization_encodings.json "${QUANT_ENCODINGS}"

info "Saved ONNX model to ${ONNX_OUTPUT}"
info "Saved encodings file to ${QUANT_ENCODINGS}"
