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


# --------------------------------------------------
# Config
# --------------------------------------------------

VENV_DIR="${OUTPUT_DIR}/yoloxapp"
YOLOX_DIR="$OUTPUT_DIR/YOLOX"

MODEL_NAME="yolox-l"
CKPT_FILE="yolox_l.pth"


# ─── Cleanup helper ───────────────────────────────────────────────────────────

_model_prep_cleanup() {
  info "Cleaning up temporary YoloX build directories..."
  deactivate 2>/dev/null || true
  rm -rf "${YOLOX_DIR}" "${VENV_DIR}"
}

trap _model_prep_cleanup EXIT

# --------------------------------------------------
# Create virtual environment
# --------------------------------------------------

info "Creating virtual environment for YOLOX export..."
unset PYTHONPATH
virtualenv "${VENV_DIR}" --python=3.8
source "${VENV_DIR}/bin/activate"

# --------------------------------------------------
# Install PyTorch + dependencies
# --------------------------------------------------

pip install -U pip setuptools wheel

# PyTorch
pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cpu

pip install "numpy<2" onnx==1.12.0 loguru tabulate tqdm thop opencv-python-headless onnxsim

# --------------------------------------------------
# Clone YOLOX
# --------------------------------------------------

if [ ! -d "$YOLOX_DIR" ]; then
    git clone https://github.com/Megvii-BaseDetection/YOLOX.git "$YOLOX_DIR"
fi

cd "$YOLOX_DIR"
git checkout 0.3.0

# # Install YOLOX
# pip install -r requirements.txt
pip install -e . --no-deps


# --------------------------------------------------
# Download pretrained weights
# --------------------------------------------------

if [ ! -f "$CKPT_FILE" ]; then
    wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth
fi


# --------------------------------------------------
# Create output directory
# --------------------------------------------------

mkdir -p "$OUTPUT_DIR"

# --------------------------------------------------
# Export ONNX (dynamic batch enabled)
# --------------------------------------------------

python tools/export_onnx.py \
    -n $MODEL_NAME \
    -c $CKPT_FILE \
    --batch-size "$BATCH_SIZE" \
    --output-name "$OUTPUT_DIR/compiled_model/model.onnx"

export OUTPUT_DIR
export CKPT_FILE

python3 - <<'EOF'
import torch
import os

output_dir = os.environ.get("OUTPUT_DIR", ".")
pth_path = os.environ.get("CKPT_FILE", ".")
pt_path  = f"{output_dir}/compiled_model/model.pt"

checkpoint = torch.load(pth_path, map_location="cpu")

# YOLOX checkpoints are stored in checkpoint["model"]
if isinstance(checkpoint, dict) and "model" in checkpoint:
    state_dict = checkpoint["model"]
else:
    state_dict = checkpoint  # already a bare state_dict

torch.save(state_dict, pt_path)
print(f"Saved state_dict to {pt_path}")
EOF


info "YOLOX_large model file prepared at $OUTPUT_DIR/compiled_model/model.onnx"

# ─── Teardown ─────────────────────────────────────────────────────────────────
trap - EXIT          # clear the trap so it doesn't fire in the caller
_model_prep_cleanup  # run cleanup manually now that we're done
