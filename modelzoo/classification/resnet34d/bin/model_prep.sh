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

RESNET34D="resnet34d_virtual_env"
MODEL_FILE_NAME="resnet34d"

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
  info "Cleaning up temporary Resnet34d repository..."
  deactivate 2>/dev/null || true
  rm -rf "${RESNET34D}"

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
virtualenv -p python3 ${RESNET34D}
source ${RESNET34D}/bin/activate


# ─── Install dependencies ─────────────────────────────────────────────────────

pip --version
pip install --upgrade pip  setuptools wheel
python -m pip install numpy==1.26.4
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install timm==1.0.15
python -m pip install onnx==1.16.0 onnxscript==0.1.0 onnxsim-prebuilt==0.4.39.post2


# ─── Export to ONNX ──────────────────────────────────────────────────────────

python3 <<HEREDOC
import torch, timm
import torch.onnx as onnx

model = timm.create_model('resnet34d', pretrained=True)
model.eval()

# Save the torch version of the model
torch.save(model.state_dict(),"${OUTPUT_DIR}/${PT_FILE}")

# Save the ONNX version of the model
dummy_input = torch.randn(1, 3, 224, 224)
onnx.export(model,dummy_input, "${OUTPUT_DIR}/${ONNX_FILE}" , export_params=True, opset_version=13, do_constant_folding=True, input_names=["input"])

HEREDOC


# ─── Simplify the ONNX graph ──────────────────────────────────────────────────────────

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
