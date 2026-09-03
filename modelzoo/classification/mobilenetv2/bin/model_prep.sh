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


ASSETS_DIR="${OUTPUT_DIR}"/compiled_model

MOBILENET_REPO="${OUTPUT_DIR}/mobilenet_model"

PT_OUTPUT="${ASSETS_DIR}/model.pt"
ONNX_OUTPUT="${ASSETS_DIR}/model.onnx"


# ─── Cleanup helper ───────────────────────────────────────────────────────────

_model_prep_cleanup() {
  info "Cleaning up temporary mobilenet files..."
  deactivate 2>/dev/null || true
  rm -rf "$OUTPUT_DIR"/mobilenet_v2_1.0_224.tgz  "$MOBILENET_REPO"
}

trap _model_prep_cleanup EXIT

mkdir -p "$MOBILENET_REPO"
cd "${MOBILENET_REPO}"

# ─── Set up virtual environment ───────────────────────────────────────────────

info "Creating virtual environment..."
unset PYTHONPATH
virtualenv mobilenetv2app -p python3
source mobilenetv2app/bin/activate

# ─── Install dependencies ─────────────────────────────────────────────────────

info "Installing dependencies..."
pip install tf2onnx
pip install tensorflow
pip install onnx

# ─── Download Mobilenet frozen graph ──────────────────────────────────────────
wget http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz -P "$OUTPUT_DIR"
tar -xvf "$OUTPUT_DIR"/mobilenet_v2_1.0_224.tgz  -C "$MOBILENET_REPO"

# ─── Export to ONNX ──────────────────────────────────────────────────────────

info "Exporting model to ONNX..."

python3 -m tf2onnx.convert \
  --graphdef "$MOBILENET_REPO"/mobilenet_v2_1.0_224_frozen.pb \
  --output "$ONNX_OUTPUT" \
  --inputs input:0[1,224,224,3] \
  --outputs MobilenetV2/Predictions/Softmax:0 \
  --inputs-as-nchw input:0 \
  --opset 13

python3 - <<PYTHON
import onnx

INPUT_OLD = "input:0"
INPUT_NEW = "input"

OUTPUT_OLD = "MobilenetV2/Predictions/Softmax:0"
OUTPUT_NEW = "MobilenetV2/Predictions/Softmax"

ONNX_PATH = "$ONNX_OUTPUT"

model = onnx.load(ONNX_PATH)

# Rename graph inputs
for inp in model.graph.input:
    if inp.name == INPUT_OLD:
        inp.name = INPUT_NEW

# Rename graph outputs
for out in model.graph.output:
    if out.name == OUTPUT_OLD:
        out.name = OUTPUT_NEW

# Rename internal references
for node in model.graph.node:
    node.input[:] = [
        INPUT_NEW if x == INPUT_OLD else x
        for x in node.input
    ]

    node.output[:] = [
        OUTPUT_NEW if x == OUTPUT_OLD else x
        for x in node.output
    ]

# Save in-place
onnx.save(model, ONNX_PATH)

print("ONNX input/output renamed successfully.")

PYTHON

info "ONNX model saved to: ${ONNX_OUTPUT}"

# ─── Teardown ─────────────────────────────────────────────────────────────────

trap - EXIT          # clear the trap so it doesn't fire in the caller
_model_prep_cleanup  # run cleanup manually now that we're done
