#!/bin/bash
#
# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.
#

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

BATCH_SIZE="${BATCH_SIZE:-1}"


# ─── Resolve paths ────────────────────────────────────────────────────────────

mkdir -p "${OUTPUT_DIR}"


# ─── Constants ────────────────────────────────────────────────────────────────

MODEL="efficientnet-lite3"
IMAGE_SIZE=280
CHECKPOINT_URL="https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/${MODEL}.tar.gz"
TPU_COMMIT="2758580eb3c3761b806e88127a7308fd016e5eda"

VENV_DIR="${OUTPUT_DIR}/efficientnet_lite3_venv"
MODEL_DIR="${OUTPUT_DIR}/efficientnet_lite3_model"


# ─── Cleanup helper ───────────────────────────────────────────────────────────

_model_prep_cleanup() {
  info "Cleaning up temporary EfficientNet build directories..."
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
pip install absl-py "numpy<2" "tensorflow==2.11.0" "tf2onnx" "onnx"


# ─── Clone TensorFlow TPU repository ─────────────────────────────────────────

info "Cloning TensorFlow TPU repository..."
mkdir -p "${MODEL_DIR}"
cd "${MODEL_DIR}"
git clone https://github.com/tensorflow/tpu
cd tpu

info "Checking out commit: ${TPU_COMMIT}"
git checkout "${TPU_COMMIT}"

cd models/official/efficientnet || exit 1
MODEL_PREP_DIRECTORY=$(pwd)
info "Working in: ${MODEL_PREP_DIRECTORY}"


# ─── Download EfficientNet-Lite3 checkpoint ───────────────────────────────────

info "Downloading EfficientNet-Lite3 checkpoint..."
wget "${CHECKPOINT_URL}"
tar -xvf "${MODEL}.tar.gz"


# ─── Export to TensorFlow SavedModel ─────────────────────────────────────────

info "Patching export_model.py placeholder batch dim to ${BATCH_SIZE}..."
sed -i "s/shape=(1, FLAGS.image_size, FLAGS.image_size, 3)/shape=(${BATCH_SIZE}, FLAGS.image_size, FLAGS.image_size, 3)/" export_model.py

info "Exporting EfficientNet-Lite3 to SavedModel format..."
python3 export_model.py \
  --model_name="${MODEL}" \
  --ckpt_dir="efficientnet-lite3" \
  --output_saved_model_dir=saved_model \
  --output_tflite="${MODEL}_float.tflite" \
  --quantize=False \
  --image_size=${IMAGE_SIZE}

if [[ ! -d "saved_model" ]]; then
  error "SavedModel export failed: output directory 'saved_model' not found."
  exit 1
fi

rm -rf "${OUTPUT_DIR}/saved_model"
mv saved_model "${OUTPUT_DIR}/saved_model"
info "SavedModel saved to: ${OUTPUT_DIR}/saved_model"


# ─── Export to ONNX ──────────────────────────────────────────────

ASSETS_DIR="${OUTPUT_DIR}/compiled_model"
mkdir -p "${ASSETS_DIR}"

info "Converting SavedModel to ONNX (batch_size=${BATCH_SIZE})..."
python3 -m tf2onnx.convert \
  --saved-model "${OUTPUT_DIR}/saved_model" \
  --output "${ASSETS_DIR}/model.onnx" \
  --opset 13 \
  --inputs-as-nchw input

if [[ ! -f "${ASSETS_DIR}/model.onnx" ]]; then
  error "ONNX conversion failed: output file not found at ${ASSETS_DIR}/model.onnx"
  exit 1
fi
info "ONNX model saved to: ${ASSETS_DIR}/model.onnx"


# ─── Teardown ─────────────────────────────────────────────────────────────────
trap - EXIT          # clear the trap so it doesn't fire in the caller
_model_prep_cleanup  # run cleanup manually now that we're done

info "EfficientNet-Lite3 model preparation complete."
