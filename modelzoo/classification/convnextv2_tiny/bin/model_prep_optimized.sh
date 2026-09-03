# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.
# -----------------------------------------------------------------------------
# Model Preparation Script (Optimized)
#

# Before running, export required environment variables:
#   ARA_NOUS_ROOT
#   MODEL_PIPELINE_ROOT
#
# Usage:
#   sh /Ara-model-zoo/modelzoo/classification/convnextv2_tiny/bin/model_prep_optimized.sh
# -----------------------------------------------------------------------------

set -euo pipefail


# export ARA_NOUS_ROOT=/auto/worka/sathyaveera.reddy/nxp_enclave/Ara-model-zoo
if [[ -z "${ARA_NOUS_ROOT:-}" ]]; then
  echo "ARA_NOUS_ROOT is unset." >&2; return 1
fi
source "${ARA_NOUS_ROOT}/core/shell/logger.sh"

# export MODEL_PIPELINE_ROOT=/auto/worka/sathyaveera.reddy/nxp_enclave/Ara-model-zoo/modelzoo/classification/convnextv2_tiny
if [[ -z "${MODEL_PIPELINE_ROOT:-}" ]]; then
  error "MODEL_PIPELINE_ROOT is unset."; return 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_PIPELINE_ROOT}/output}"
ASSETS_DIR="${OUTPUT_DIR}/compiled_model"
ONNX_OUTPUT="${ASSETS_DIR}/quantized_model.onnx"
QUANTIZED_ENCODINGS="${ASSETS_DIR}/quantization_encodings.json"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${ASSETS_DIR}"

cp /auto/share/sw/common/modelzoo_eng/optimized_models/classification/Convnext_v2/quantized_model.onnx "${ONNX_OUTPUT}"
cp /auto/share/sw/common/modelzoo_eng/optimized_models/classification/Convnext_v2/quantization_encodings.json "${QUANTIZED_ENCODINGS}"

info "Saved ONNX model to ${ONNX_OUTPUT}"
