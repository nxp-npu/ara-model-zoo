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

export BATCH_SIZE="${BATCH_SIZE:-1}"
export MODEL_NAME="yolo11m"



ASSETS_DIR="${OUTPUT_DIR}/compiled_model"
YOLOV11_REPO="${OUTPUT_DIR}/yolov11"
VENV_DIR="${OUTPUT_DIR}/yolov11app"
ONNX_OUTPUT="${ASSETS_DIR}/model.onnx"
PT_OUTPUT="${ASSETS_DIR}/model.pt"
ULTRALYTICS_COMMIT="c0d1d27abebc6777c9ba51777f1eb33e47bd9940"
ULTRALYTICS_ARCHIVE="${OUTPUT_DIR}/ultralytics-${ULTRALYTICS_COMMIT}.tar.gz"

cleanup() {
  deactivate 2>/dev/null || true
  rm -rf "${YOLOV11_REPO}" "${VENV_DIR}" "${ULTRALYTICS_ARCHIVE}" "${OUTPUT_DIR}/ultralytics-${ULTRALYTICS_COMMIT}"
}
trap cleanup EXIT

mkdir -p "${ASSETS_DIR}"
rm -rf "${YOLOV11_REPO}" "${VENV_DIR}"

info "Creating virtual environment for YOLOv11 export..."
unset PYTHONPATH
virtualenv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

info "Downloading ultralytics source archive for commit ${ULTRALYTICS_COMMIT}..."
wget --no-verbose -O "${ULTRALYTICS_ARCHIVE}" "https://github.com/ultralytics/ultralytics/archive/${ULTRALYTICS_COMMIT}.tar.gz"
tar -xzf "${ULTRALYTICS_ARCHIVE}" -C "${OUTPUT_DIR}"
mv "${OUTPUT_DIR}/ultralytics-${ULTRALYTICS_COMMIT}" "${YOLOV11_REPO}"
cd "${YOLOV11_REPO}"

info "Installing dependencies..."
pip install numpy==1.26.4 "opencv-python==4.9.0.80"
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 torchvision==0.17.2
pip install onnxruntime==1.17.3
pip install onnxslim==0.1.36
pip install onnx==1.17.0
pip install -e . numpy==1.26.4

info "Exporting ${MODEL_NAME} to ONNX (batch_size=${BATCH_SIZE})..."
python3 <<'HEREDOC'
import os

from ultralytics import YOLO

model_onnx = str(os.environ["MODEL_NAME"]) + ".onnx"
model_pt = str(os.environ["MODEL_NAME"]) + ".pt"

batch_size = int(os.environ.get("BATCH_SIZE", "1"))
model = YOLO(model_pt)
model.export(format="onnx", simplify=True, opset=13, batch=batch_size)
HEREDOC

if [[ ! -f "${YOLOV11_REPO}/${MODEL_NAME}.onnx" ]]; then
  error "ONNX export failed: ${YOLOV11_REPO}/${MODEL_NAME}.onnx was not created"
  return 1
fi


if [[ -f "${YOLOV11_REPO}/${MODEL_NAME}.pt" ]]; then
  mv "${YOLOV11_REPO}/${MODEL_NAME}.pt" "${PT_OUTPUT}"
fi


mv "${YOLOV11_REPO}/${MODEL_NAME}.onnx" "${ONNX_OUTPUT}"
info "Saved ONNX model to ${ONNX_OUTPUT}"

trap - EXIT
cleanup
