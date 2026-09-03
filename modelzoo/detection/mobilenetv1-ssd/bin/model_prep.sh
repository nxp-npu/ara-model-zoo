#!/bin/bash
#
# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.
#

set -euo pipefail


# ─── Validate environment ─────────────────────────────────────────────────────

if [[ -z "${ARA_NOUS_ROOT:-}" ]]; then
  echo "$0: ARA_NOUS_ROOT is unset. Please export it before running this script." >&2
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

BATCH_SIZE="${BATCH_SIZE:-1}"


# ─── Resolve paths ────────────────────────────────────────────────────────────

mkdir -p "${OUTPUT_DIR}"


# ─── Constants ────────────────────────────────────────────────────────────────

MODEL="mobilenetv1-ssd"
IMAGE_SIZE=300
MODEL_ID="ssd_mobilenet_v1_coco_2018_01_28"
MODEL_URL="http://download.tensorflow.org/models/object_detection/${MODEL_ID}.tar.gz"

VENV_DIR="${OUTPUT_DIR}/${MODEL}_venv"
MODEL_DIR="${OUTPUT_DIR}/${MODEL}_model"


# ─── Cleanup helper ───────────────────────────────────────────────────────────

_model_prep_cleanup() {
  info "Cleaning up temporary MobileNetv1 SSD build directories..."
  deactivate 2>/dev/null || true
  rm -rf "${MODEL_DIR}"
  rm -rf "${VENV_DIR}"
}

trap _model_prep_cleanup EXIT


# ─── Create virtual environment ─────────────────────────────────────────────

info "Creating virtual environment..."
unset PYTHONPATH
virtualenv --python=python3.10 "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"


# ─── Install dependencies ─────────────────────────────────────────────────────

info "Installing dependencies..."
pip install "tf2onnx" "tensorflow" "onnx"


# ─── Download MobileNetv1 SSD checkpoint ─────────────────────────────────────

info "Downloading ${MODEL} checkpoint..."
mkdir -p "${MODEL_DIR}"
wget --content-disposition "${MODEL_URL}" -P "${OUTPUT_DIR}"
tar -xvf "${OUTPUT_DIR}/${MODEL_ID}.tar.gz" -C "${MODEL_DIR}"
info "Frozen graph at ${MODEL_DIR}/${MODEL_ID}/frozen_inference_graph.pb"
rm -rf "${OUTPUT_DIR}/${MODEL_ID}.tar.gz"


# ─── Export to ONNX ──────────────────────────────────────────────────────────

ASSETS_DIR="${OUTPUT_DIR}/compiled_model"
mkdir -p "${ASSETS_DIR}"

info "Converting frozen graph to ONNX (batch_size=${BATCH_SIZE})..."
python3 -m tf2onnx.convert \
  --graphdef "${MODEL_DIR}/${MODEL_ID}/frozen_inference_graph.pb" \
  --output "${ASSETS_DIR}/model.onnx" \
  --inputs "Preprocessor/sub:0[${BATCH_SIZE},300,300,3]" \
  --outputs "Postprocessor/ExpandDims_1:0,Postprocessor/Slice:0" \
  --inputs-as-nchw "Preprocessor/sub:0" \
  --opset 13

if [[ ! -f "${ASSETS_DIR}/model.onnx" ]]; then
  error "ONNX conversion failed: output file not found at ${ASSETS_DIR}/model.onnx"
  exit 1
fi

# ─── Rename tensor names ──────────────────────────────────────────────────────
info "Renaming ONNX tensor names..."
python3 - <<PYTHON
import onnx

ONNX_PATH = "${ASSETS_DIR}/model.onnx"
RENAMES = {
    "Preprocessor/sub:0":           "input",
    "Postprocessor/ExpandDims_1:0": "boxes",
    "Postprocessor/Slice:0":        "scores",
    "concat_1:0":                   "concat_1",
    "concat:0":                     "concat",
}

model = onnx.load(ONNX_PATH)
for item in list(model.graph.input) + list(model.graph.output):
    if item.name in RENAMES:
        item.name = RENAMES[item.name]
for node in model.graph.node:
    node.input[:]  = [RENAMES.get(x, x) for x in node.input]
    node.output[:] = [RENAMES.get(x, x) for x in node.output]
onnx.save(model, ONNX_PATH)
print("Tensor names renamed successfully.")
PYTHON

info "ONNX model saved to: ${ASSETS_DIR}/model.onnx"

# ─── Teardown ─────────────────────────────────────────────────────────────────
trap - EXIT          # clear the trap so it doesn't fire in the caller
_model_prep_cleanup  # run cleanup manually now that we're done

info "MobileNetv1 SSD model preparation complete."
