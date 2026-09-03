#!/usr/bin/env bash
# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

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

TINY_VIT_VENV="tiny_vit_virtual_env"
TINY_VIT_REPO="TinyViT"
MODEL_FILE_NAME="tiny_vit"

ASSETS_DIR=${OUTPUT_DIR}/compiled_model

ONNX_FILE="${MODEL_FILE_NAME}.onnx"
PT_FILE="${MODEL_FILE_NAME}.pt"

OUTPUT_FILE_NAME="model"

CHECKPOINT_URL="https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_11m_22kto1k_distill.pth"
CHECKPOINT_FILE="tiny_vit_11m_22kto1k_distill.pth"


# ─── Resolve paths ───────────────────────────────────────────────────────────

mkdir -p "${OUTPUT_DIR}"


# ─── Cleanup helper ───────────────────────────────────────────────────────────

_model_prep_cleanup() {
  info "Cleaning up temporary tiny_vit artifacts..."
  deactivate 2>/dev/null || true
  rm -rf "${TINY_VIT_VENV}" "${TINY_VIT_REPO}" "${CHECKPOINT_FILE}"

  trap - EXIT
}

trap _model_prep_cleanup EXIT


# ─── Set up virtual environment ───────────────────────────────────────────────
info " ----- "
info $OUTPUT_DIR
info " ----- "
cd $OUTPUT_DIR

info "Creating virtual environment..."
unset PYTHONPATH

info "Using interpreter for venv: $(python3 --version 2>&1)"

virtualenv -p python3 "${TINY_VIT_VENV}"
source "${TINY_VIT_VENV}/bin/activate"


# ─── Install dependencies ─────────────────────────────────────────────────────

pip --version
pip install --upgrade pip setuptools wheel
python -m pip install numpy==1.26.4
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
# timm is needed by TinyViT repo internals (DropPath, register_model) — not used for model creation
python -m pip install timm==1.0.15
python -m pip install onnx==1.16.0 onnxscript==0.1.0 onnxsim-prebuilt==0.4.39.post2


# ─── Clone source repo & download checkpoint ─────────────────────────────────

info "Cloning TinyViT source repo..."
git clone --depth 1 https://github.com/wkcn/TinyViT.git "${TINY_VIT_REPO}"

info "Downloading TinyViT-11M IN-22k-to-1k distill checkpoint..."
wget -q --show-progress "${CHECKPOINT_URL}" -O "${CHECKPOINT_FILE}"


# ─── Export to ONNX ──────────────────────────────────────────────────────────
# TinyViT-11M (ImageNet-22k pre-trained, distilled, fine-tuned to ImageNet-1k). 1000 classes.
# Model loaded directly from source repo, checkpoint from GitHub releases.

python3 <<HEREDOC
import sys, torch
import torch.onnx as onnx

sys.path.insert(0, "${TINY_VIT_REPO}")
from models.tiny_vit import TinyViT

# TinyViT-11M configuration (IN-22k → IN-1k distill, 224x224)
model = TinyViT(
    img_size=224,
    in_chans=3,
    num_classes=1000,
    embed_dims=[64, 128, 256, 448],
    depths=[2, 2, 6, 2],
    num_heads=[2, 4, 8, 14],
    window_sizes=[7, 7, 14, 7],
    mlp_ratio=4.,
)

ckpt = torch.load("${OUTPUT_DIR}/${CHECKPOINT_FILE}", map_location="cpu")
state = ckpt.get("model", ckpt)
model.load_state_dict(state)
model.eval()

# Save the torch version of the model
torch.save(model.state_dict(), "${OUTPUT_DIR}/${PT_FILE}")

# Save the ONNX version of the model
dummy_input = torch.randn(1, 3, 224, 224)
onnx.export(model, dummy_input, "${OUTPUT_DIR}/${ONNX_FILE}", export_params=True, opset_version=20, do_constant_folding=True, input_names=["input"], output_names=["output"])

HEREDOC


# ─── Simplify the ONNX graph ──────────────────────────────────────────────────────────

python3 -m onnxsim "${OUTPUT_DIR}/${ONNX_FILE}" "${OUTPUT_DIR}/${MODEL_FILE_NAME}_sim.onnx"


# ─── Move ONNX and Pytorch files to compiled_model folder ──────────────────────────────────

mv "${OUTPUT_DIR}/${MODEL_FILE_NAME}_sim.onnx" "${ASSETS_DIR}/${OUTPUT_FILE_NAME}.onnx"
mv "${OUTPUT_DIR}/${PT_FILE}"   "${ASSETS_DIR}/${OUTPUT_FILE_NAME}.pt"

info "ONNX/Pytorch model saved to: ${ASSETS_DIR}"

# ─── Delete unnecessary files ──────────────────────────────────────────────────────────

rm "${OUTPUT_DIR}/${ONNX_FILE}"

# ─── Deactivate virtual Environment ─────────────────────────────────────────────────────
deactivate


# ─── Delete virtual Environment ─────────────────────────────────────────────────────
trap - EXIT
_model_prep_cleanup
