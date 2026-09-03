#!/usr/bin/env bash
#
# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.
# performance.sh — Performance for model on Ara Hardware
# Usage: performance.sh model=<name> [skip_proxy=true|false] [batch_size=<integer>] [iterations=<integer>]
#


set -euo pipefail

# ---------------------------------------------------------------------------
# Source logger
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARA_NOUS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${ARA_NOUS_ROOT}/core/shell/logger.sh"

# ---------------------------------------------------------------------------
# Parse key=value arguments
# ---------------------------------------------------------------------------
for arg in "$@"; do
  case "$arg" in
    model=*)         model="${arg#*=}"         ;;
    skip_proxy=*)    skip_proxy="${arg#*=}"    ;;
    batch_size=*)    batch_size="${arg#*=}"    ;;
    iterations=*)    iterations="${arg#*=}"    ;;
    *) error "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Apply defaults
# ---------------------------------------------------------------------------
skip_proxy="${skip_proxy:-false}"
batch_size="${batch_size:-1}"
iterations="${iterations:-10}"


# ---------------------------------------------------------------------------
# Validate required arguments and environment
# ---------------------------------------------------------------------------
if [ -z "$model" ]; then
  error "model is undefined, please select a model" >&2
  exit 1
fi

if [ "${DV_TGT_ROOT}" == "" ]; then
  error "path to Ara2 proxy root is undefined, please set variable DV_TGT_ROOT" >&2
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

NNAPP_CONFIG=${ARA_NOUS_ROOT}/nnapp_config.yaml
MODEL_DVM=${OUTPUT_DIR}/compiled_model/model.dvm
OUTPUT_PATH=${OUTPUT_DIR}/performance

mkdir -p ${OUTPUT_PATH}

# ---------------------------------------------------------------------------
# Patch proxy config (only when proxy will actually be used)
# ---------------------------------------------------------------------------
PROXY_CONFIG="${ARA_NOUS_ROOT}/proxy_config.yaml"
FIRMWARE_PATH="${DV_TGT_ROOT}/art/linux/x86/proxy/mcp0/"

if [[ "$skip_proxy" != "true" ]]; then
  if [[ ! -f "$PROXY_CONFIG" ]]; then
    error "proxy_config.yaml not found: $PROXY_CONFIG" >&2
    exit 1
  fi

  # Save original and register cleanup trap
  PROXY_CONFIG_BACKUP="$(mktemp)"
  cp "$PROXY_CONFIG" "$PROXY_CONFIG_BACKUP"

  restore_proxy_config() {
    if [[ -f "$PROXY_CONFIG_BACKUP" ]]; then
      mv "$PROXY_CONFIG_BACKUP" "$PROXY_CONFIG"
      info "Restored proxy_config.yaml to original state"
    fi
  }
  trap restore_proxy_config EXIT

  info "Patching dm_firmware_path in proxy_config.yaml → $FIRMWARE_PATH"
  sed -i "s|dm_firmware_path: .*|dm_firmware_path: \"${FIRMWARE_PATH}\"|g" "$PROXY_CONFIG"
fi


# ---------------------------------------------------------------------------
# Get the architecture
# ---------------------------------------------------------------------------
ARCH=`uname -m`


# ---------------------------------------------------------------------------
# Start inference proxy (unless skipped)
# ---------------------------------------------------------------------------
if [[ "$skip_proxy" != "true" ]]; then
  echo "Sudo access required to start Ara2 inference proxy"

  if [ "$ARCH" == "x86_64" ]; then
    PROXY_BIN="${DV_TGT_ROOT}/art/linux/x86/proxy/proxy_${ARCH}"
  else
    PROXY_BIN="${DV_TGT_ROOT}/art/linux/aarch64/proxy/proxy_${ARCH}"
  fi

  if [[ ! -x "$PROXY_BIN" ]]; then
    error "Proxy binary not found or not executable: $PROXY_BIN" >&2
    exit 1
  fi

  pid="$(pidof -s "proxy_${ARCH}" || true)"

  if [[ -n "$pid" ]]; then
    info "Proxy already running with PID $pid — stopping it first"
    sudo kill -9 "$pid" || true
    sleep 1
  fi

  info "Starting Ara2 inference proxy..."
  sudo rm -f /var/run/dvproxy.pid
  sudo rm -f /dev/shm/nnapp.shm
  sudo "${PROXY_BIN}" --config "${PROXY_CONFIG}" &

  info "Waiting for proxy to initialise..."
  sleep 5
fi

echo "Testing Performance on Hardware..."


# ---------------------------------------------------------------------------
# Parameters to be passed to nnapp
# ---------------------------------------------------------------------------
PARAM_NAME="0.name="$model
PARAM_PATH="0.path="$MODEL_DVM
PARAM_OUTPUT_PATH="0.output_path="$OUTPUT_PATH
PARAM_BATCH_SIZE="0.infer_bundle_size=$batch_size"
PARAM_ITERATIONS="0.iterations=$iterations"

PARAMS=$PARAM_NAME:$PARAM_PATH:$PARAM_OUTPUT_PATH:$PARAM_BATCH_SIZE:$PARAM_ITERATIONS

# ---------------------------------------------------------------------------
# Run Performance
# ---------------------------------------------------------------------------
if [ -f $NNAPP_CONFIG ]; then
if [ "$ARCH" == "aarch64" ]; then
    $DV_TGT_ROOT/art/linux/aarch64/nnapp/nnapp_aarch64 --config $NNAPP_CONFIG log --dump-path $OUTPUT_PATH mode infer async --override $PARAMS
  else
    $DV_TGT_ROOT/art/linux/x86/nnapp/nnapp_x86_64 --config $NNAPP_CONFIG log --dump-path $OUTPUT_PATH mode infer async --override $PARAMS
  fi
  status=$?
  if [ $status -eq 0 ]; then
    info "Successfully executed Ara-2 model performance computation"
    info "configuration for Ara-2 model performance computation is saved at $OUTPUT_PATH"
  else
    error "Failed to execute Ara-2 model performance computation" >&2
    exit $status
  fi
else
  error "Path to model file does not exist - $NNAPP_CONFIG" >&2
  exit 1
fi
