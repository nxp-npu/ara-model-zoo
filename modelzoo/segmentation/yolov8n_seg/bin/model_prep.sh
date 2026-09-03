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

MODEL_NAME="yolov8n-seg"
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

python -m pip install opencv-python==4.8.1.78
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
python -m pip install onnx==1.20.1 onnxruntime==1.17.3 onnxsim-prebuilt==0.4.39.post2
python -m pip install onnxscript==0.1.0 onnxslim==0.1.36
python -m pip install -e . numpy==1.26.4 --no-build-isolation

info "Exporting yolov8n_pose to ONNX with fixed batch size ${EXPORT_BATCH_SIZE}..."

export MODEL_NAME
export EXPORT_BATCH_SIZE

python3 <<'HEREDOC'
import os
from pathlib import Path

import onnx
from ultralytics import YOLO

batch_size = int(os.environ["EXPORT_BATCH_SIZE"])
model_onnx = str(os.environ["MODEL_NAME"]) + ".onnx"
model_pt = str(os.environ["MODEL_NAME"]) + ".pt"
onnx_path = Path(model_onnx)
model = YOLO(model_pt)
model.export(
    format="onnx",
    imgsz=640,
    batch=batch_size,
    dynamic=False,
    simplify=True,
    opset=13,
)

graph = onnx.load(str(onnx_path)).graph
input_dims = [
    dim.dim_value if dim.HasField("dim_value") else dim.dim_param
    for dim in graph.input[0].type.tensor_type.shape.dim
]
expected_dims = [batch_size, 3, 640, 640]
if input_dims != expected_dims:
    raise RuntimeError(
        f"Unexpected ONNX input shape {input_dims}; expected {expected_dims}"
    )
HEREDOC

if [[ ! -f "${YOLOV8_REPO}/${MODEL_NAME}.onnx" ]]; then
  error "ONNX export failed: ${YOLOV8_REPO}/${MODEL_NAME}.onnx was not created"
  return 1
fi

mv "${YOLOV8_REPO}/${MODEL_NAME}.onnx" "${ONNX_OUTPUT}"
if [[ -f "${YOLOV8_REPO}/${MODEL_NAME}.pt" ]]; then
  mv "${YOLOV8_REPO}/${MODEL_NAME}.pt" "${PT_OUTPUT}"
fi
info "Saved ONNX model to ${ONNX_OUTPUT}"

trap - EXIT
cleanup
