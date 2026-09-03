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

# ─── Resolve paths ───────────────────────────────────────────────────────────


mkdir -p "${OUTPUT_DIR}"


# ─── Constants ────────────────────────────────────────────────────────────────


ASSETS_DIR=${OUTPUT_DIR}/compiled_model

YOLOV8_REPO="${OUTPUT_DIR}/yolov8"
PT_WEIGHTS="yolov8n-face.pt"

MODEL_GDRIVE_ID="1NnAvmXoXB5F5eFovm2LBWSs5yy6TKRUE"

PT_OUTPUT="${ASSETS_DIR}/model.pt"
ONNX_OUTPUT="${ASSETS_DIR}/model.onnx"

ONNX_FILE="${YOLOV8_REPO}/yolov8n-face.onnx"


# ─── Cleanup helper ───────────────────────────────────────────────────────────

_model_prep_cleanup() {
  info "Cleaning up temporary YOLOv8 repository..."
  deactivate 2>/dev/null || true
  rm -rf "${YOLOV8_REPO}"
}

trap _model_prep_cleanup EXIT

# ─── Clone YOLOv8 ────────────────────────────────────────────────────────────

info "Cloning YOLOv8 repository..."
rm -rf "${YOLOV8_REPO}"
git clone https://github.com/ultralytics/ultralytics "${YOLOV8_REPO}"
cd "${YOLOV8_REPO}"
git checkout c340f84ce9325de720fbd9ada6523a28fc432651

# ─── Set up virtual environment ───────────────────────────────────────────────

info "Creating virtual environment..."
unset PYTHONPATH
virtualenv yolov8app -p python3
source yolov8app/bin/activate

# ─── Install dependencies ─────────────────────────────────────────────────────

info "Installing dependencies..."

python -m pip install --upgrade pip
python -m pip install setuptools==68.2.2 wheel

ls
pip install -e . --no-build-isolation

pip install setuptools==68.2.2
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cpu
pip install onnx==1.20.1 onnxruntime==1.20.0 onnxsim-prebuilt==0.4.39.post2
pip install onnxscript==0.1.0 onnxslim==0.1.36

pip install gdown

# ─── Export to ONNX ──────────────────────────────────────────────────────────

gdown $MODEL_GDRIVE_ID

info "Exporting model to ONNX..."
python3 - <<HEREDOC
import sys
import torch

_original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = patched_torch_load


from ultralytics import YOLO

model = YOLO("${PT_WEIGHTS}")

# Extract state dict
state_dict = model.model.state_dict()

# Save state dict
torch.save(state_dict, "${PT_OUTPUT}")

model.export(format="onnx", simplify=True, opset=13, batch=1)
print("Export complete.", file=sys.stderr)
HEREDOC



if [[ ! -f "${ONNX_FILE}" ]]; then
  error "ONNX export failed: output file not found at ${ONNX_FILE}"
  return 1
fi

mv "${ONNX_FILE}" "${ONNX_OUTPUT}"
info "ONNX model saved to: ${ONNX_OUTPUT}"

# ─── Teardown ─────────────────────────────────────────────────────────────────

trap - EXIT          # clear the trap so it doesn't fire in the caller
_model_prep_cleanup  # run cleanup manually now that we're done
