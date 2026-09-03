#!/usr/bin/env bash
# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.
# ─────────────────────────────────────────────────────────────────────────────
# model_prep.sh — Download and export YOLO26n to ONNX via Ultralytics
#
# Called by model_compile.sh. Expects env vars from the caller:
#   ARA_NOUS_ROOT, MODEL_PIPELINE_ROOT, OUTPUT_DIR
#   ARA_NOUS_ROOT - path to the ara-model-zoo repo root
#   MODEL_PIPELINE_ROOT - path to the yolo26n directory
#   OUTPUT_DIR - {MODEL_PIPELINE_ROOT}/output
#
# Artifacts produced:
#   compiled_model/model.onnx  — Exported ONNX (opset 18, simplified)
#   compiled_model/model.pt    — PyTorch state dict
#
# ./bin/model_prep.sh (Set all the paths before that )
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Validate environment ─────────────────────────────────────────────────────
if [[ -z "${ARA_NOUS_ROOT:-}" ]]; then
  echo "ARA_NOUS_ROOT is unset." >&2; return 1
fi
source "${ARA_NOUS_ROOT}/core/shell/logger.sh"

if [[ -z "${MODEL_PIPELINE_ROOT:-}" ]]; then
  error "MODEL_PIPELINE_ROOT is unset."; return 1
fi
if [[ -z "${OUTPUT_DIR:-}" ]]; then
  error "OUTPUT_DIR is unset."; return 1
fi

# ── Paths ────────────────────────────────────────────────────────────────────
ASSETS_DIR="${OUTPUT_DIR}/compiled_model"
VENV_DIR="${OUTPUT_DIR}/yolo26app"
WORK_DIR="${OUTPUT_DIR}/yolo26_export"
ONNX_OUTPUT="${ASSETS_DIR}/model.onnx"
PT_OUTPUT="${ASSETS_DIR}/model.pt"

# ── Cleanup on exit ──────────────────────────────────────────────────────────
cleanup() {
  deactivate 2>/dev/null || true
  rm -rf "${WORK_DIR}" "${VENV_DIR}"
}
trap cleanup EXIT

# ── Create output directory ──────────────────────────────────────────────────
mkdir -p "${ASSETS_DIR}"

# ── Set up isolated virtual environment ──────────────────────────────────────
info "Creating isolated virtual environment..."
unset PYTHONPATH
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

# ── Install dependencies ─────────────────────────────────────────────────────
pip install --upgrade pip setuptools wheel
pip install ultralytics

# ── Download weights and export to ONNX ──────────────────────────────────────
info "Exporting YOLO26n to ONNX..."
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

python3 - "${PT_OUTPUT}" <<'HEREDOC'
import sys
import torch
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
torch.save(model.model.state_dict(), sys.argv[1])
model.export(format="onnx", simplify=True, opset=18)
HEREDOC

# ── Validate export ──────────────────────────────────────────────────────────
if [[ ! -f "${WORK_DIR}/yolo26n.onnx" ]]; then
  error "ONNX export failed: ${WORK_DIR}/yolo26n.onnx was not created."
  return 1
fi

# ── Move artifact to final location ──────────────────────────────────────────
mv "${WORK_DIR}/yolo26n.onnx" "${ONNX_OUTPUT}"
info "ONNX model saved to ${ONNX_OUTPUT}"

# ── Teardown ─────────────────────────────────────────────────────────────────
trap - EXIT
cleanup
