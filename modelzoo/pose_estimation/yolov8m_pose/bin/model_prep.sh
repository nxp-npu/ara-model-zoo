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
ONNX_OUTPUT="${ASSETS_DIR}/model.onnx"
PT_OUTPUT="${ASSETS_DIR}/model.pt"
ULTRALYTICS_COMMIT="a5735724c54a9f5bcb239c151fefbd1337d7123d"
ULTRALYTICS_ARCHIVE="${OUTPUT_DIR}/ultralytics-${ULTRALYTICS_COMMIT}.tar.gz"
EXPORT_BATCH_SIZE="${batch_size:-1}"

if ! [[ "${EXPORT_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  error "batch_size must be a positive integer, got '${EXPORT_BATCH_SIZE}'"
  return 1
fi

cleanup() {
  deactivate 2>/dev/null || true
  rm -rf \
    "${YOLOV8_REPO}" \
    "${VENV_DIR}" \
    "${ULTRALYTICS_ARCHIVE}" \
    "${OUTPUT_DIR}/ultralytics-${ULTRALYTICS_COMMIT}"
}

trap cleanup EXIT

mkdir -p "${ASSETS_DIR}"
rm -rf "${YOLOV8_REPO}" "${VENV_DIR}"

info "Creating virtual environment for YOLOv8 pose export..."
unset PYTHONPATH
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install setuptools==68.2.2 wheel

info "Downloading ultralytics source archive for commit ${ULTRALYTICS_COMMIT}..."
wget --no-verbose -O "${ULTRALYTICS_ARCHIVE}" "https://github.com/ultralytics/ultralytics/archive/${ULTRALYTICS_COMMIT}.tar.gz"
tar -xzf "${ULTRALYTICS_ARCHIVE}" -C "${OUTPUT_DIR}"
mv "${OUTPUT_DIR}/ultralytics-${ULTRALYTICS_COMMIT}" "${YOLOV8_REPO}"
cd "${YOLOV8_REPO}"

python -m pip install "numpy==1.26.4" "opencv-python==4.8.1.78"
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
python -m pip install onnx==1.20.1 onnxruntime==1.17.3 onnxsim-prebuilt==0.4.39.post2
python -m pip install onnxscript==0.1.0 onnxslim==0.1.36
python -m pip install -e . --no-build-isolation --no-deps
python -m pip install \
  matplotlib pillow pyyaml requests "scipy>=1.4.1,<1.14" tqdm pandas seaborn psutil py-cpuinfo "thop>=0.1.1"
python -m pip install --force-reinstall --no-deps "numpy==1.26.4" "opencv-python==4.8.1.78"

info "Exporting yolov8m_pose to ONNX (old modelzoo-style export)..."
python3 <<'HEREDOC'
from ultralytics import YOLO

model = YOLO("yolov8m-pose.pt")
model.export(format="onnx", simplify=True, opset=13)
HEREDOC

if [[ ! -f "${YOLOV8_REPO}/yolov8m-pose.onnx" ]]; then
  error "ONNX export failed: ${YOLOV8_REPO}/yolov8m-pose.onnx was not created"
  return 1
fi

mv "${YOLOV8_REPO}/yolov8m-pose.onnx" "${ONNX_OUTPUT}"
if [[ -f "${YOLOV8_REPO}/yolov8m-pose.pt" ]]; then
  mv "${YOLOV8_REPO}/yolov8m-pose.pt" "${PT_OUTPUT}"
fi
info "Saved ONNX model to ${ONNX_OUTPUT}"

trap - EXIT
cleanup
