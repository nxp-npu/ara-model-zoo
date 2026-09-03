#!/usr/bin/env bash

# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.
set -euo pipefail

if [[ -z "${ARA_NOUS_ROOT:-}" ]]; then
  echo "ARA_NOUS_ROOT is unset. Please export it before running this script." >&2
  return 1
fi

source "${ARA_NOUS_ROOT}/core/shell/logger.sh"

if [[ -z "${MODEL_PIPELINE_ROOT:-}" ]]; then
  error "MODEL_PIPELINE_ROOT is unset. Please export it before running this script."
  return 1
fi

if [[ -z "${OUTPUT_DIR:-}" ]]; then
  error "OUTPUT_DIR is unset. Please export it before running this script."
  return 1
fi

MODEL_URL="https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_2.5g.onnx"
CLASSIFICATION_NODES=("439" "459" "479")
BOX_NODES=("441" "461" "481")
KEYPOINT_NODES=("442" "462" "482")

ASSETS_DIR="${OUTPUT_DIR}/compiled_model"
WORK_DIR="${OUTPUT_DIR}/model_prep"
VENV_DIR="${OUTPUT_DIR}/scrfdapp"
RAW_ONNX="${WORK_DIR}/scrfd_2_5g.raw.onnx"
SIMPLIFIED_ONNX="${WORK_DIR}/scrfd_2_5g.sim.onnx"
ONNX_OUTPUT="${ASSETS_DIR}/model.onnx"
EXPORT_BATCH_SIZE="${batch_size:-1}"

if ! [[ "${EXPORT_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  error "batch_size must be a positive integer, got '${EXPORT_BATCH_SIZE}'"
  return 1
fi

cleanup() {
  deactivate 2>/dev/null || true
  rm -rf "${WORK_DIR}" "${VENV_DIR}"
}

trap cleanup EXIT

mkdir -p "${ASSETS_DIR}" "${WORK_DIR}"
rm -rf "${VENV_DIR}"

info "Creating virtual environment for SCRFD-2.5G export..."
unset PYTHONPATH
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install onnx onnxruntime onnxsim

info "Downloading SCRFD-2.5G ONNX model..."
wget --no-verbose -O "${RAW_ONNX}" "${MODEL_URL}"

info "Simplifying SCRFD-2.5G ONNX with fixed batch size ${EXPORT_BATCH_SIZE}..."
python -m onnxsim \
  "${RAW_ONNX}" \
  "${SIMPLIFIED_ONNX}" \
  --overwrite-input-shape "${EXPORT_BATCH_SIZE},3,640,640"

info "Pruning SCRFD-2.5G ONNX outputs..."
python "${MODEL_PIPELINE_ROOT}/bin/prune_scrfd_onnx.py" \
  --input-model "${SIMPLIFIED_ONNX}" \
  --output-model "${ONNX_OUTPUT}" \
  --cls-nodes "${CLASSIFICATION_NODES[@]}" \
  --bbox-nodes "${BOX_NODES[@]}" \
  --kps-nodes "${KEYPOINT_NODES[@]}"

MODEL_PREP_BATCH_SIZE="${EXPORT_BATCH_SIZE}" MODEL_PREP_ONNX_PATH="${ONNX_OUTPUT}" python <<'PY'
import os

import onnx

onnx_path = os.environ["MODEL_PREP_ONNX_PATH"]
batch_size = int(os.environ["MODEL_PREP_BATCH_SIZE"])

graph = onnx.load(onnx_path).graph
input_dims = [
    dim.dim_value if dim.HasField("dim_value") else dim.dim_param
    for dim in graph.input[0].type.tensor_type.shape.dim
]
expected_dims = [batch_size, 3, 640, 640]
if input_dims != expected_dims:
    raise RuntimeError(
        f"Unexpected ONNX input shape {input_dims}; expected {expected_dims}"
    )
PY

info "Saved SCRFD-2.5G ONNX model to ${ONNX_OUTPUT}"

trap - EXIT
cleanup
