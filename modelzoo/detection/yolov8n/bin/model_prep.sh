#!/usr/bin/env sh

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

ASSETS_DIR="${OUTPUT_DIR}/compiled_model"
YOLOV8_REPO="${OUTPUT_DIR}/yolov8"
VENV_DIR="${OUTPUT_DIR}/yolov8app"
PT_OUTPUT="${ASSETS_DIR}/model.pt"
ONNX_OUTPUT="${ASSETS_DIR}/model.onnx"
ULTRALYTICS_COMMIT="817aca355efe17178dabedf595965aedfdb45adc"
ULTRALYTICS_ARCHIVE="${OUTPUT_DIR}/ultralytics-${ULTRALYTICS_COMMIT}.tar.gz"

cleanup() {
  deactivate 2>/dev/null || true
  rm -rf "${YOLOV8_REPO}" "${VENV_DIR}" "${ULTRALYTICS_ARCHIVE}" "${OUTPUT_DIR}/ultralytics-${ULTRALYTICS_COMMIT}"
}

trap cleanup EXIT

mkdir -p "${ASSETS_DIR}"
rm -rf "${YOLOV8_REPO}" "${VENV_DIR}"

info "Creating virtual environment for YOLOv8 export..."
unset PYTHONPATH
virtualenv "${VENV_DIR}" -p python3
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install setuptools==68.2.2 wheel

info "Downloading ultralytics source archive for commit ${ULTRALYTICS_COMMIT}..."
wget --no-verbose -O "${ULTRALYTICS_ARCHIVE}" "https://github.com/ultralytics/ultralytics/archive/${ULTRALYTICS_COMMIT}.tar.gz"
tar -xzf "${ULTRALYTICS_ARCHIVE}" -C "${OUTPUT_DIR}"
mv "${OUTPUT_DIR}/ultralytics-${ULTRALYTICS_COMMIT}" "${YOLOV8_REPO}"
cd "${YOLOV8_REPO}"

python -m pip install -e . --no-build-isolation

info "Exporting yolov8n to ONNX..."
python3 <<HEREDOC
from ultralytics import YOLO
import torch


model = YOLO("yolov8n.pt")

# Extract state dict
state_dict = model.model.state_dict()

# Save state dict
torch.save(state_dict, "${PT_OUTPUT}")

model.export(format="onnx", simplify=True, opset=13)
HEREDOC

if [[ ! -f "${YOLOV8_REPO}/yolov8n.onnx" ]]; then
  error "ONNX export failed: ${YOLOV8_REPO}/yolov8n.onnx was not created"
  return 1
fi

mv "${YOLOV8_REPO}/yolov8n.onnx" "${ONNX_OUTPUT}"
info "Saved ONNX model to ${ONNX_OUTPUT}"

trap - EXIT
cleanup
