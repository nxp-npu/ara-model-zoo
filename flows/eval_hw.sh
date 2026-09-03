#!/usr/bin/env bash
# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

# eval_hw.sh — Evaluation runner for compiled hardware model
# Usage: eval_hw.sh model=<name> dataset_root=<path> [gt_dir=<path>] [run=python|cpp] [limit=<n>] [skip_proxy=true|false]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARA_NOUS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${ARA_NOUS_ROOT}/core/shell/logger.sh"

for arg in "$@"; do
  case "$arg" in
    run=*) run="${arg#*=}" ;;
    model=*) model="${arg#*=}" ;;
    dataset_root=*) dataset_root="${arg#*=}" ;;
    gt_dir=*) gt_dir="${arg#*=}" ;;
    limit=*) limit="${arg#*=}" ;;
    skip_proxy=*) skip_proxy="${arg#*=}" ;;
    *) error "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

if [[ -z "${model:-}" ]]; then
  error "model is undefined — please pass model=<name>" >&2
  exit 1
fi

if [[ -z "${dataset_root:-}" ]]; then
  error "dataset_root is undefined — please pass dataset_root=<path>" >&2
  exit 1
fi

if [[ -z "${DV_TGT_ROOT:-}" ]]; then
  error "DV_TGT_ROOT is undefined — please set the path to the Ara2 proxy root" >&2
  exit 1
fi

skip_proxy="${skip_proxy:-false}"
run="${run:-python}"

if [[ "$run" != "python" && "$run" != "cpp" ]]; then
  error "Invalid run type '$run'. Must be 'python' or 'cpp'." >&2
  exit 1
fi

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

CONFIG_DIR="$(dirname "$(readlink -f "${RUN_YAML}")")"
OUTPUT_DIR_REL="$(grep -E '^out' "$RUN_YAML" | cut -d ':' -f2 | tr -d ' "')"
if [[ "${OUTPUT_DIR_REL}" != /* ]]; then
  OUTPUT_DIR="$(readlink -f "${CONFIG_DIR}/${OUTPUT_DIR_REL}")"
else
  OUTPUT_DIR="${OUTPUT_DIR_REL}"
fi

if [[ ! -d "${dataset_root}" ]]; then
  error "Dataset root not found: ${dataset_root}" >&2
  exit 1
fi

if [[ -n "${gt_dir:-}" && ! -d "${gt_dir}" ]]; then
  error "Ground-truth directory not found: ${gt_dir}" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

ASSETS_DIR="${OUTPUT_DIR}/compiled_model"
DVM_MODEL="${ASSETS_DIR}/model.dvm"

if [[ ! -f "$DVM_MODEL" ]]; then
  error "Compiled model not found: $DVM_MODEL" >&2
  exit 1
fi

PROXY_CONFIG="${ARA_NOUS_ROOT}/proxy_config.yaml"
FIRMWARE_PATH="${DV_TGT_ROOT}/art/linux/x86/proxy/mcp0/"

if [[ "$skip_proxy" != "true" ]]; then
  if [[ ! -f "$PROXY_CONFIG" ]]; then
    error "proxy_config.yaml not found: $PROXY_CONFIG" >&2
    exit 1
  fi

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

if [[ "$skip_proxy" != "true" ]]; then
  echo "Sudo access required to start Ara2 inference proxy"
  arch="$(uname -m)"
  PROXY_BIN="${DV_TGT_ROOT}/art/linux/x86/proxy/proxy_${arch}"

  if [[ ! -x "$PROXY_BIN" ]]; then
    error "Proxy binary not found or not executable: $PROXY_BIN" >&2
    exit 1
  fi

  pid="$(pidof -s "proxy_${arch}" || true)"

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

info "Running evaluation on hardware (arch: $(uname -m))..."

if [[ "$run" == "cpp" ]]; then
  error "C++ implementation is not yet implemented. Please use run=python." >&2
  exit 1
fi

export PYTHONPATH="${ARA_NOUS_ROOT}:${PYTHONPATH:-}"

cmd=(
  python3 -u "${ARA_NOUS_ROOT}/core/python/scripts/run_evalation.py"
  --config "$RUN_YAML"
  --runtime "ara"
  --dataset-root "$dataset_root"
)

if [[ -n "${gt_dir:-}" ]]; then
  cmd+=(--gt-dir "$gt_dir")
fi

if [[ -n "${limit:-}" ]]; then
  cmd+=(--limit "$limit")
fi

"${cmd[@]}"

info "Done."
