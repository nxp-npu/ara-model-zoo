#!/bin/bash

# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

set -euo pipefail

# ---------------------------------------------------------------------------
# Source logger and set Roots
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARA_NOUS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${ARA_NOUS_ROOT}/core/shell/logger.sh"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
data_path=""
model=""

for arg in "$@"; do
  case "$arg" in
    model=*)     model="${arg#*=}"     ;;
    data_path=*) data_path="${arg#*=}" ;;
    *)           error "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
if [[ -z "$model" ]]; then
  error "model is undefined — please pass model=<name>" >&2
  exit 1
fi

if [[ -z "$data_path" ]]; then
  error "data_path is undefined — please pass data_path=<path>" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Derive Paths
# ---------------------------------------------------------------------------
MODEL_ZOO_ROOT="${ARA_NOUS_ROOT}/modelzoo"
MODEL_PIPELINE_ROOT="$(find "$MODEL_ZOO_ROOT" -mindepth 2 -maxdepth 2 -type d -name "$model" | head -n 1)"

if [[ ! -d "$MODEL_PIPELINE_ROOT" ]]; then
  error "Model directory not found for: $model" >&2
  exit 1
fi

# Path to the specific config for this model
RUN_YAML="${MODEL_PIPELINE_ROOT}/config/run.yaml"

if [[ ! -f "$RUN_YAML" ]]; then
  error "run.yaml not found at ${RUN_YAML}" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Parse Output Directory
# ---------------------------------------------------------------------------
# Parse OUTPUT_DIR from run.yaml (lines starting with "out:")
OUTPUT_DIR_RAW="$(grep -E '^out' "${RUN_YAML}" | cut -d':' -f2 | tr -d ' "')"
if [[ -z "$OUTPUT_DIR_RAW" ]]; then
  error "Could not parse output directory from ${RUN_YAML}" >&2
  exit 1
fi

CONFIG_DIR="$(dirname "$(readlink -f "${RUN_YAML}")")"

# Resolve relative paths against config directory
if [[ "${OUTPUT_DIR_RAW}" != /* ]]; then
  OUTPUT_DIR="$(readlink -f "${CONFIG_DIR}/${OUTPUT_DIR_RAW}")"
else
  OUTPUT_DIR="${OUTPUT_DIR_RAW}"
fi

mkdir -p $OUTPUT_DIR

# ---------------------------------------------------------------------------
# Create folder to store artifacts
# ---------------------------------------------------------------------------
info "creating directory $OUTPUT_DIR/compiled_model"
mkdir -p $OUTPUT_DIR/compiled_model

# ---------------------------------------------------------------------------
# Model preparation
# ---------------------------------------------------------------------------
info "Running model_prep.sh for ${model}..."
source "${MODEL_PIPELINE_ROOT}/bin/model_prep.sh"
info "Model preparation complete."

status=$?
if [ $status -eq 0 ]; then
  info "successfully executed model preparation"
else
  error "failed to execute model preparation" >&2
  source ${MODEL_PIPELINE_ROOT}/bin/model_download.sh
fi

# Return to root to prevent MKL errors if the model_prep cloned directory was deleted
cd "$ARA_NOUS_ROOT" || {
    echo "Failed to change directory to: $ARA_NOUS_ROOT" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------
info "Starting quantization for model: $model"
info "Using dataset: $data_path"

python3 "${ARA_NOUS_ROOT}/core/python/scripts/run_quantization.py" \
    --config "$RUN_YAML" \
    --data_path "$data_path"
