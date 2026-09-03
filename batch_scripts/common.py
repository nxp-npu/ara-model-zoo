# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

"""
common.py — shared utilities for batch scripts (logging, formatting, shell
runner, model list helpers, etc.).
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RESET = "\033[0m"


def _supports_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def info(msg: str) -> None:
    prefix = f"{CYAN}[INFO]{RESET} " if _supports_color() else "[INFO] "
    print(f"{prefix}{msg}", flush=True)


def success(msg: str) -> None:
    prefix = f"{GREEN}[PASS]{RESET} " if _supports_color() else "[PASS] "
    print(f"{prefix}{msg}", flush=True)


def warn(msg: str) -> None:
    prefix = f"{YELLOW}[WARN]{RESET} " if _supports_color() else "[WARN] "
    print(f"{prefix}{msg}", file=sys.stderr, flush=True)


def error(msg: str) -> None:
    prefix = f"{RED}[ERROR]{RESET} " if _supports_color() else "[ERROR] "
    print(f"{prefix}{msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    return f"{m}m{s:02d}s"


def divider(char: str = "-", width: int = 48) -> str:
    return char * width


# ---------------------------------------------------------------------------
# Model list helpers
# ---------------------------------------------------------------------------
def load_models_from_file(path: Path) -> list[str]:
    """Read model names from a file, skipping comments and blank lines."""
    models = []
    with path.open() as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.split("#")[0].strip()
            if not line:
                continue
            if " " in line:
                warn(f"{path}:{lineno}: model name contains spaces, skipping: '{line}'")
                continue
            models.append(line)
    return models


def build_model_list(
    model_names: list[str],
    file_path: Path | None,
) -> list[str]:
    """Merge explicit model *names* and a *file_path* into a deduplicated ordered list."""
    seen: dict[str, None] = {}

    for m in model_names:
        if m not in seen:
            seen[m] = None

    if file_path:
        if not file_path.is_file():
            error(f"Model list file not found: {file_path}")
            sys.exit(1)
        for m in load_models_from_file(file_path):
            if m not in seen:
                seen[m] = None

    return list(seen)


def map_dataset_roots(path: Path) -> dict[str, Path]:
    """Read paths from JSON file into a map"""
    if not path.is_file():
        error(f"Config file not found: {path}")
        sys.exit(1)
    try:
        with path.open() as f:
            config = json.load(f)
    except json.JSONDecodeError as exc:
        error(f"Failed to parse config file {path}: {exc}")
        sys.exit(1)

    dataset_roots: dict[str, Path] = {}
    for name, settings in config.items():
        if "dataset_root" not in settings:
            error(f"Model {name} is missing required 'dataset_root' in config.")
            sys.exit(1)
        dataset_roots[name] = Path(settings["dataset_root"])

    return dataset_roots


# ---------------------------------------------------------------------------
# Shell runner
# ---------------------------------------------------------------------------
def run_shell(
    cmd: list[str],
    *,
    log_path: Path,
    dry_run: bool,
) -> tuple[int, float]:
    """
    Execute *cmd* via ``bash``, tee-ing stdout and stderr to both the
    terminal and *log_path*.

    Returns ``(exit_code, elapsed_seconds)``.
    """
    if dry_run:
        info(f"[dry-run] Would execute: {' '.join(cmd)}")
        info(f"[dry-run] Log: {log_path}")
        return 0, 0.0

    start = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        process.wait()

    elapsed = time.monotonic() - start
    return process.returncode, elapsed


# ---------------------------------------------------------------------------
# Log directory resolution
# ---------------------------------------------------------------------------
def resolve_log_dir(
    caller_file: str,
    subfolder: str,
    *,
    override: Path | None = None,
) -> Path:
    """
    Return *override* if given, otherwise build the default path::

        ../logs/<YYYYMMDD_HHMMSS>/<subfolder>/

    *caller_file* should be ``__file__`` from the calling script.
    """
    if override is not None:
        return override
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(caller_file).parent.parent / "logs" / timestamp / subfolder


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------
def print_summary(
    label: str,
    passed: list[str],
    failed: list[str],
    *,
    elapsed_total: float,
) -> None:
    """Print a standard passed/failed summary block."""
    batch_elapsed = fmt_duration(elapsed_total)

    info("")
    info(divider("="))
    info(f"Batch {label} complete  ({batch_elapsed} total)")
    info(f"  Passed : {len(passed)}")
    info(f"  Failed : {len(failed)}")
    info(divider("="))

    if passed:
        info("Passed:")
        for m in passed:
            success(f"  ✓ {m}")

    if failed:
        info("Failed:")
        for m in failed:
            error(f"  ✗ {m}")
