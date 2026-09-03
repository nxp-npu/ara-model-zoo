#!/usr/bin/env bash
#
# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.
#
# precompiled_model_download.sh — downloads a precompiled model and places it into correct folder structure
# Usage: ./precompiled_model_download.sh model=<name>


set -euo pipefail

while [ $# -gt 0 ]; do
  case "$1" in
    model=*) model="${1#*=}" ;;
  esac
  shift
done



if [ -z "$model" ]; then
  error "model is undefined, please select a model" >&2
  exit 1
fi


# ---------------------------------------------------------------------------
# Resolve script and project roots (robust against symlinks and cd)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARA_NOUS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Source logger early so info/error are available throughout
# ---------------------------------------------------------------------------
# shellcheck source=../scripts/logger.sh
source "${ARA_NOUS_ROOT}/core/shell/logger.sh"

MODEL_ZOO_ROOT="${ARA_NOUS_ROOT}/modelzoo"
MODEL_PIPELINE_ROOT="$(find "$MODEL_ZOO_ROOT" -mindepth 2 -maxdepth 2 -type d -name "$model" | head -n 1)"
RUN_YAML="${MODEL_PIPELINE_ROOT}/config/run.yaml"
CONFIG_DIR="$(dirname "$(readlink -f "${RUN_YAML}")")"
PRECOMPILED_SOURCE="$(grep -E "precompiled_model_link_path" "${RUN_YAML}" | sed 's/^[^:]*:[[:space:]]*//' | tr -d '"')"
info "precompiled_model_link_path: $PRECOMPILED_SOURCE"

if [ "$MODEL_PIPELINE_ROOT" == "" ]; then
  error "path to model pipeline root is undefined, please set variable MODEL_PIPELINE_ROOT" >&2
  exit 1
fi

info "model pipeline root: $MODEL_PIPELINE_ROOT"

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
info "OUTPUT_DIR is $OUTPUT_DIR"

if [[ -d "$OUTPUT_DIR/assets" ]]
then
  info "found existing assets, removing it"
  rm -rf $OUTPUT_DIR/assets
fi

if [[ -d "$OUTPUT_DIR/compiled_model" ]];
then
  info "found existing compiled_model. compiled_model will be replaced with downloaded version of compiled_model"
  rm -rf $OUTPUT_DIR/compiled_model
fi

mkdir -p "$OUTPUT_DIR/compiled_model"

# ---------------------------------------------------------------------------
# Determine source type and copy dvm to compiled_model
# ---------------------------------------------------------------------------
DVM_FILE=""

if [[ -e "$PRECOMPILED_SOURCE" ]]; then
  # Local path exists on filesystem
  if [[ -f "$PRECOMPILED_SOURCE" ]]; then
    if [[ "$PRECOMPILED_SOURCE" == *.dvm ]]; then
      info "local .dvm file found: $PRECOMPILED_SOURCE"
      DVM_FILE="$PRECOMPILED_SOURCE"
    else
      error "local file provided but does not have .dvm extension: $PRECOMPILED_SOURCE" >&2
      exit 1
    fi
  elif [[ -d "$PRECOMPILED_SOURCE" ]]; then
    DVM_COUNT=$(find "$PRECOMPILED_SOURCE" -maxdepth 1 -type f -name "*.dvm" | wc -l)
    if [[ "$DVM_COUNT" -eq 0 ]]; then
      error "no .dvm file found in directory: $PRECOMPILED_SOURCE" >&2
      exit 1
    elif [[ "$DVM_COUNT" -gt 1 ]]; then
      error "multiple .dvm files found in directory: $PRECOMPILED_SOURCE" >&2
      exit 1
    else
      DVM_FILE="$(find "$PRECOMPILED_SOURCE" -maxdepth 1 -type f -name "*.dvm")"
      info "found .dvm file in directory: $DVM_FILE"
    fi
  else
    error "unsupported local path type: $PRECOMPILED_SOURCE" >&2
    exit 1
  fi
else
  # Not a local path — treat as HuggingFace hub ID
  info "downloading from HuggingFace hub: $PRECOMPILED_SOURCE"

  HF_DOWNLOAD_DIR="$OUTPUT_DIR/hf_download"
  mkdir -p "$HF_DOWNLOAD_DIR"
  hf download "$PRECOMPILED_SOURCE" --local-dir "$HF_DOWNLOAD_DIR"

  DVM_COUNT=$(find "$HF_DOWNLOAD_DIR" -type f -name "*.dvm" | wc -l)
  if [[ "$DVM_COUNT" -eq 0 ]]; then
    error "no .dvm file found in HF download: $PRECOMPILED_SOURCE" >&2
    exit 1
  elif [[ "$DVM_COUNT" -gt 1 ]]; then
    error "multiple .dvm files found in HF download: $PRECOMPILED_SOURCE" >&2
    exit 1
  else
    DVM_FILE="$(find "$HF_DOWNLOAD_DIR" -type f -name "*.dvm")"
    info "found .dvm file from HF: $DVM_FILE"
  fi
fi

cp "$DVM_FILE" "$OUTPUT_DIR/compiled_model/model.dvm"
info "copied $DVM_FILE to $OUTPUT_DIR/compiled_model/model.dvm"

echo "Model file saved to ${OUTPUT_DIR}/compiled_model"

status=$?
if [ $status -eq 0 ]; then
  info "successfully executed precompiled download"
else
  error "failed to execute dvnc precompiled download" >&2
  exit $status
fi
