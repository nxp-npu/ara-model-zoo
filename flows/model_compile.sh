#!/usr/bin/env bash
#
# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.
#
# model_compile.sh — preprocess, model-prep, and compile a face-detection model
# Usage: ./model_compile.sh model=<name> [run=<python|cpp>]

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve script and project roots (robust against symlinks and cd)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARA_NOUS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STARTING_DIR="$(pwd)"

# ---------------------------------------------------------------------------
# Source logger early so info/error are available throughout
# ---------------------------------------------------------------------------
# shellcheck source=../scripts/logger.sh
echo $ARA_NOUS_ROOT
echo $STARTING_DIR

source "${ARA_NOUS_ROOT}/core/shell/logger.sh"

# ---------------------------------------------------------------------------
# Validate environment
# ---------------------------------------------------------------------------
if [[ -z "${ARA_SDK_ROOT:-}" ]]; then
  error "ARA_SDK_ROOT is not set. Please export it before running this script." >&2
  exit 1
fi

python_path="$(command -v python3 || true)"
if [[ -z "$python_path" ]]; then
  error "No python3 interpreter found in PATH." >&2
  exit 1
fi
info "Using Python: ${python_path}"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
model=""
dataset_root=""
run="python"
optimized=false

for arg in "$@"; do
  case "$arg" in
    model=*) model="${arg#*=}" ;;
    dataset_root=*) dataset_root="${arg#*=}" ;;
    run=*)   run="${arg#*=}"   ;;
    --optimized)  optimized=true ;;
    *)
      error "Unknown argument: '${arg}'. Expected model=, dataset_root= or run=." >&2
      exit 1
      ;;
  esac
done

if [[ -z "$model" ]]; then
  error "Required argument 'model' is not set. Example: $0 model=my_model dataset_root=/datasets/coco2017" >&2
  exit 1
fi

if [[ -z "$dataset_root" ]]; then
  error "Required argument 'dataset_root' is not set. The calibration images listed in the model's calibration_image_set.txt are copied from it." >&2
  exit 1
fi

if [[ ! -d "$dataset_root" ]]; then
  error "Dataset root not found: ${dataset_root}" >&2
  exit 1
fi

if [[ "$run" != "python" && "$run" != "cpp" ]]; then
  error "Invalid run type '${run}'. Must be 'python' or 'cpp'." >&2
  exit 1
fi

info "model=${model}  dataset_root=${dataset_root}  run=${run}"

# ---------------------------------------------------------------------------
# Paths derived from model selection
# ---------------------------------------------------------------------------
MODEL_ZOO_ROOT="${ARA_NOUS_ROOT}/modelzoo"
MODEL_PIPELINE_ROOT="$(find "$MODEL_ZOO_ROOT" -mindepth 2 -maxdepth 2 -type d -name "$model" | head -n 1)"

if $optimized; then
    RUN_YAML="${MODEL_PIPELINE_ROOT}/config/quant_run.yaml"
else
    RUN_YAML="${MODEL_PIPELINE_ROOT}/config/run.yaml"
fi

ADJUST_FILE_ABS_PATH="${MODEL_PIPELINE_ROOT}/config/adjust.yaml"

if [[ ! -f "$RUN_YAML" ]]; then
  error "run.yaml not found at ${RUN_YAML}" >&2
  exit 1
fi


CONFIG_DIR="$(dirname "$(readlink -f "${RUN_YAML}")")"
CONFIG_NAME="dvrun.yaml"

# Parse OUTPUT_DIR from run.yaml (lines starting with "out:")
OUTPUT_DIR_RAW="$(grep -E '^out' "${RUN_YAML}" | cut -d':' -f2 | tr -d ' "')"
if [[ -z "$OUTPUT_DIR_RAW" ]]; then
  error "Could not parse output directory from ${RUN_YAML}" >&2
  exit 1
fi

# Resolve relative paths against config directory
if [[ "${OUTPUT_DIR_RAW}" != /* ]]; then
  OUTPUT_DIR="$(readlink -f "${CONFIG_DIR}/${OUTPUT_DIR_RAW}")"
else
  OUTPUT_DIR="${OUTPUT_DIR_RAW}"
fi

mkdir -p $OUTPUT_DIR

# ---------------------------------------------------------------------------
# Docker image handling
# ---------------------------------------------------------------------------
# TAR_PATH="${ARA_SDK_ROOT}/dvdocker/ara2.tar"
# info "Docker tar path: ${TAR_PATH}"

# if [[ ! -f "$TAR_PATH" ]]; then
#   error "Docker tar not found at ${TAR_PATH}" >&2
#   exit 1
# fi

# TAR_IMAGE_TAG="$(tar -xf "${TAR_PATH}" manifest.json -O \
#   | grep -o '"RepoTags":[^]]*' \
#   | sed 's/.*"\([^"]*\)".*/\1/')"

# if [[ -z "$TAR_IMAGE_TAG" ]]; then
#   error "Could not extract image tag from ${TAR_PATH}" >&2
#   exit 1
# fi
# info "Docker image in tar: ${TAR_IMAGE_TAG}"

# if docker image inspect "${TAR_IMAGE_TAG}" > /dev/null 2>&1; then
#   EXISTING_ID="$(docker images -q "${TAR_IMAGE_TAG}")"
#   info "Docker image already present (IMAGE ID: ${EXISTING_ID}). Skipping load."
# else
#   info "Loading Docker image from tar — this may take a while..."
#   docker load -i "${TAR_PATH}"
#   info "Docker image loaded successfully."
# fi

# ---------------------------------------------------------------------------
# Cleanup helper — runs on EXIT so intermediate state is always removed
# ---------------------------------------------------------------------------
cleanup() {
  local exit_code=$?
  if [[ $exit_code -ne 0 ]]; then
    info "Pipeline exited with status ${exit_code}. Cleaning up intermediate files..."
  fi

}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Stage calibration imgs from the dataset and preprocess them
# ---------------------------------------------------------------------------
info "Staging calibration imgs from ${dataset_root} and preprocessing (run=${run})..."
arch="$(uname -m)"

if [[ "$run" == "cpp" ]]; then
  error "C++ implementation is not available. Please use run=python." >&2
  exit 1
else
  "$python_path" -u "${ARA_NOUS_ROOT}/core/python/scripts/run_pre.py" \
    --config "${RUN_YAML}" \
    --dataset-root "${dataset_root}"
fi
info "Image preprocessing complete."

# ---------------------------------------------------------------------------
# Create folder to store artefacts
# ---------------------------------------------------------------------------
info "creating directory $OUTPUT_DIR/compiled_model"
mkdir -p $OUTPUT_DIR/compiled_model

# ---------------------------------------------------------------------------
# Model preparation
# ---------------------------------------------------------------------------
if $optimized; then
    info "Running model_prep_optimized.sh for ${model}..."
    source "${MODEL_PIPELINE_ROOT}/bin/model_prep_optimized.sh"
    info "Model preparation complete."
    status=$?
    if [ $status -eq 0 ]; then
        info "successfully executed model preparation"
    else
        error "failed to execute model preparation" >&2
    fi
else
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
fi


# Return to root to prevent MKL errors if the model_prep cloned directory was deleted
cd "$ARA_NOUS_ROOT" || {
    echo "Failed to change directory to: $ARA_NOUS_ROOT" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# Create Subgraphs
# ---------------------------------------------------------------------------
if $optimized; then
    model_file="${OUTPUT_DIR}/compiled_model/quantized_model.onnx"
else
    model_file="${OUTPUT_DIR}/compiled_model/model.onnx"
fi
if [ -e "${MODEL_PIPELINE_ROOT}/bin/extract_graph.py" ]; then
  info "Extracting subgraphs..."
  "$python_path" "${MODEL_PIPELINE_ROOT}/bin/extract_graph.py" \
    --path "$model_file" \
    --output-path "$OUTPUT_DIR/compiled_model" \
    --config "${RUN_YAML}"
  info "Subgraphs have been extracted"
else
  echo "extract_graph.py does not exist"
fi

# make sure we are still in the correct directory in case model prep tries to cd
cd "${STARTING_DIR}"

# ---------------------------------------------------------------------------
# Generate dvrun config
# ---------------------------------------------------------------------------
info "Generating dvrun config ..."
export PYTHONPATH="${PYTHONPATH:-}:${MODEL_PIPELINE_ROOT}/output"

"$python_path" -u "${ARA_NOUS_ROOT}/core/python/scripts/create_dvrun.py" \
  --config "${RUN_YAML}" \
  --name "$CONFIG_NAME"
info "dvrun config saved: ${OUTPUT_DIR}/${CONFIG_NAME}"


# ---------------------------------------------------------------------------
# Compile model
# ---------------------------------------------------------------------------
info "Compiling model ..."
info "interpreter being used is $python_path\n\n PYTHONPATH is $PYTHONPATH"
PYTHONPATH="${PYTHONPATH}" \
  "$python_path" "${ARA_SDK_ROOT}/dvrun" \
  --config "${OUTPUT_DIR}/${CONFIG_NAME}" \

info "Model compilation complete."

# ---------------------------------------------------------------------------
# Change ownership of output directory
# ---------------------------------------------------------------------------
# sudo chown -R $(whoami):$(whoami) "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Save compiled artefacts
# ---------------------------------------------------------------------------
cp "${OUTPUT_DIR}/assets/model.dvm" "${OUTPUT_DIR}/compiled_model/"

info "Model assets:           ${OUTPUT_DIR}/compiled_model"
