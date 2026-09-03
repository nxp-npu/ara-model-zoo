#!/usr/bin/env bash
# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

# infer_float.sh — Inference runner for floating point model
# Usage: infer_float model=<n> [run=python|cpp] \
#                [images_folder=<path>]  [skip_proxy=true|false]
set -euo pipefail

# ---------------------------------------------------------------------------
# Source logger
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARA_NOUS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${ARA_NOUS_ROOT}/core/shell/logger.sh"


# ---------------------------------------------------------------------------
# Parse key=value arguments
# ---------------------------------------------------------------------------
for arg in "$@"; do
  case "$arg" in
    run=*)    run="${arg#*=}"   ;;
    model=*)  model="${arg#*=}" ;;
    images_folder=*) images_folder="${arg#*=}" ;;
    *) error "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done


# ---------------------------------------------------------------------------
# Validate required arguments and environment
# ---------------------------------------------------------------------------
if [[ -z "${model:-}" ]]; then
  error "model is undefined — please pass model=<n>" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Apply defaults
# ---------------------------------------------------------------------------
run="${run:-python}"

if [[ "$run" != "python" && "$run" != "cpp" ]]; then
  error "Invalid run type '$run'. Must be 'python' or 'cpp'." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Derive paths
# ---------------------------------------------------------------------------
MODEL_ZOO_ROOT="${ARA_NOUS_ROOT}/modelzoo"
MODEL_PIPELINE_ROOT="$(find "$MODEL_ZOO_ROOT" -mindepth 2 -maxdepth 2 -type d -name "$model" | head -n 1)"


if [[ ! -d "$MODEL_PIPELINE_ROOT" ]]; then
  error "Model directory not found: $MODEL_PIPELINE_ROOT" >&2
  exit 1
fi

RUN_YAML="${MODEL_PIPELINE_ROOT}/config/run.yaml"
if [[ ! -f "$RUN_YAML" ]]; then
  error "run.yaml not found: $RUN_YAML" >&2
  exit 1
fi

OUTPUT_DIR_REL="$(grep -E '^out' "$RUN_YAML" | cut -d ':' -f2 | tr -d ' "')"
OUTPUT_DIR="$(cd "${MODEL_PIPELINE_ROOT}/bin/${OUTPUT_DIR_REL}" && pwd)"

if [[ ! -d "$OUTPUT_DIR" ]]; then
  error "Output directory not found: $OUTPUT_DIR" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Start inference proxy
# ---------------------------------------------------------------------------

## Proxy not needed for floating point inference

# ---------------------------------------------------------------------------
# Copy precompiled model asset
# ---------------------------------------------------------------------------
ASSETS_DIR="${OUTPUT_DIR}/compiled_model"
ONNX_MODEL="${ASSETS_DIR}/model.onnx"

mkdir -p "$ASSETS_DIR"

if [[ ! -f "$ONNX_MODEL" ]]; then
  error "Onnx model not found: $ONNX_MODEL" >&2
  exit 1
fi


# ---------------------------------------------------------------------------
# Prepare image dataset
# ---------------------------------------------------------------------------
images_folder="${images_folder:-${ARA_NOUS_ROOT}/testimages}"

# ---------------------------------------------------------------------------
# Run inference application
# ---------------------------------------------------------------------------
info "Running floating point inference"

echo $RUN_YAML

if [[ "$run" == "cpp" ]]; then
  error "C++ implementation is not yet implemented. Please use run=python." >&2
  exit 1

elif [[ "$run" == "python" ]]; then
  python3 -u "${ARA_NOUS_ROOT}/core/python/scripts/run_infer.py" \
    --config "$RUN_YAML" \
    --runtime "onnx" \
    --images $images_folder
fi

info "Done"
