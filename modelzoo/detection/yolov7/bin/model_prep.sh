#!/usr/bin/env bash
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


# ─── Constants ───────────────────────────────────────────────────────────────

YOLOV7="yolov7_virtual_env"
YOLOV7_REPO="yolov7_repo"
MODEL_FILE_NAME="yolov7"

ASSETS_DIR=${OUTPUT_DIR}/compiled_model

PT_OUTPUT_DIR="${ASSETS_DIR}/model.pt"
ONNX_OUTPUT_DIR="${ASSETS_DIR}/model.onnx"

ONNX_FILE="${MODEL_FILE_NAME}.onnx"
PT_FILE="${MODEL_FILE_NAME}.pt"

OUTPUT_FILE_NAME="model"


# ─── Resolve paths ───────────────────────────────────────────────────────────

mkdir -p "${OUTPUT_DIR}"


# ─── Cleanup helper ───────────────────────────────────────────────────────────

_model_prep_cleanup() {
  info "Cleaning up temporary yolov7 repository..."
  cd $OUTPUT_DIR
  deactivate 2>/dev/null || true
  rm -rf "${YOLOV7}"
  rm -rf "${YOLOV7_REPO}"

  trap - EXIT # Clear trap so caller's EXIT handler can run normally
}

trap _model_prep_cleanup EXIT


# ─── Set up virtual environment ───────────────────────────────────────────────
info " ----- "
info $OUTPUT_DIR
info " ----- "
cd $OUTPUT_DIR

info "Creating virtual environment..."
unset PYTHONPATH
virtualenv -p python3 ${YOLOV7}
source ${YOLOV7}/bin/activate


# ─── Install dependencies ─────────────────────────────────────────────────────

pip --version
pip install --upgrade pip  setuptools wheel

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip install onnx==1.16.1 onnxruntime==1.17.3 onnxscript==0.1.0 onnxsim-prebuilt==0.4.39.post2
pip install numpy==1.26.4 scipy==1.12.0 pandas==2.2.1
pip install tqdm==4.66.2 pyyaml==6.0.1 pillow==10.2.0
pip install protobuf==4.25.3 requests==2.31.0 matplotlib==3.8.3
pip install opencv-python-headless==4.9.0.80 seaborn==0.13.2


# ─── Export to ONNX ──────────────────────────────────────────────────────────

git clone https://github.com/WongKinYiu/yolov7.git $YOLOV7_REPO
cd $YOLOV7_REPO

wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
python3 export.py --weights yolov7.pt --grid \
  --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640


# ─── Simplify the ONNX graph ──────────────────────────────────────────────────────────
mv "${MODEL_FILE_NAME}.onnx" "${OUTPUT_DIR}/${MODEL_FILE_NAME}.onnx"
mv "${MODEL_FILE_NAME}.pt" "${OUTPUT_DIR}/${MODEL_FILE_NAME}.pt"

python3 -m onnxsim "${OUTPUT_DIR}/${ONNX_FILE}" "${OUTPUT_DIR}/${MODEL_FILE_NAME}_sim.onnx"


# ─── Move ONNX and Pytorch files to compiled_model fodler ──────────────────────────────────

mv "${OUTPUT_DIR}/${MODEL_FILE_NAME}_sim.onnx" "${ASSETS_DIR}/${OUTPUT_FILE_NAME}.onnx"
mv "${OUTPUT_DIR}/${PT_FILE}"   "${ASSETS_DIR}/${OUTPUT_FILE_NAME}.pt"

info "ONNX/Pytorch model saved to: ${ASSETS_DIR}"

# ─── Delete unnecessary files ──────────────────────────────────────────────────────────

rm "${OUTPUT_DIR}/${ONNX_FILE}"


# ─── Deactivate virtual Environment ─────────────────────────────────────────────────────
deactivate


# ─── Delete virtual Environment ─────────────────────────────────────────────────────
trap - EXIT          # clear the trap so it doesn't fire in the caller
_model_prep_cleanup  # run cleanup manually now that we're done
