#!/usr/bin/env sh

# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

# ── Color codes (disabled automatically if not a terminal) ──────────────────
if [ -t 2 ]; then
  _RESET='\033[0m'
  _BOLD='\033[1m'
  _DIM='\033[2m'
  _CYAN='\033[0;36m'
  _GREEN='\033[0;32m'
  _YELLOW='\033[0;33m'
  _RED='\033[0;31m'
  _MAGENTA='\033[0;35m'
else
  _RESET='' _BOLD='' _DIM='' _CYAN='' _GREEN='' _YELLOW='' _RED='' _MAGENTA=''
fi

# ── Internal timestamp ───────────────────────────────────────────────────────
_ts() { date +"%Y-%m-%d %H:%M:%S"; }

# ── Log levels ───────────────────────────────────────────────────────────────
log()     { printf "${_DIM}%s${_RESET}  ${_BOLD}${_CYAN}  INFO${_RESET}  %s\n"  "$(_ts)" "$*" >&2; }
info()    { printf "${_DIM}%s${_RESET}  ${_BOLD}${_GREEN}  INFO${_RESET}  %s\n"  "$(_ts)" "$*" >&2; }
success() { printf "${_DIM}%s${_RESET}  ${_BOLD}${_GREEN}    OK${_RESET}  %s\n"  "$(_ts)" "$*" >&2; }
warn()    { printf "${_DIM}%s${_RESET}  ${_BOLD}${_YELLOW}  WARN${_RESET}  %s\n"  "$(_ts)" "$*" >&2; }
error()   { printf "${_DIM}%s${_RESET}  ${_BOLD}${_RED} ERROR${_RESET}  %s\n"  "$(_ts)" "$*" >&2; }
fatal()   { printf "${_DIM}%s${_RESET}  ${_BOLD}${_MAGENTA} FATAL${_RESET}  %s\n" "$(_ts)" "$*" >&2; exit 1; }
