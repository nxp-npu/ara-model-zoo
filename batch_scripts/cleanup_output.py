# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

#!/usr/bin/env python3
"""
cleanup_output.py — delete extra folders from model output directories to conserve space.

This script traverses the modelzoo directory and, for each model's output folder,
removes everything EXCEPT the following items that must be preserved:

    assets/          (directory)
    compiled_model/  (directory)
    evaluation/      (directory)
    performance/     (directory)
    dvrun.yaml       (file)
    run.yaml         (file)
    version           (file)

Usage:
    python cleanup_output.py
    python cleanup_output.py --modelzoo /path/to/modelzoo
    python cleanup_output.py --dry-run
    python cleanup_output.py --model yolov8n
    python cleanup_output.py --task object_detection
"""

import argparse
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RESET = "\033[0m"
BOLD = "\033[1m"


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
# Constants — items to KEEP inside each output folder
# ---------------------------------------------------------------------------
KEEP_ITEMS = {
    "assets",
    "compiled_model",
    "evaluation",
    "performance",
    "dvrun.yaml",
    "run.yaml",
    "version",
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cleanup_output.py",
        description="Delete extra folders/files from model output directories "
        "to conserve disk space.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python cleanup_output.py
  python cleanup_output.py --dry-run
  python cleanup_output.py --modelzoo ./modelzoo
  python cleanup_output.py --model yolov8n
  python cleanup_output.py --task object_detection
  python cleanup_output.py --model yolov8n --task object_detection
        """,
    )

    parser.add_argument(
        "--modelzoo",
        type=Path,
        metavar="PATH",
        default=Path(__file__).parent.parent / "modelzoo",
        help="Path to the modelzoo directory (default: modelzoo/ next to this script).",
    )
    parser.add_argument(
        "--model",
        type=str,
        metavar="MODEL",
        default=None,
        help="Only clean the output directory for a specific model.",
    )
    parser.add_argument(
        "--task",
        type=str,
        metavar="TASK",
        default=None,
        help="Only clean output directories under a specific task subfolder.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted without actually deleting anything.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def find_output_dirs(
    modelzoo: Path,
    target_model: str | None,
    target_task: str | None,
) -> list[Path]:
    """
    Find all 'output' directories inside modelzoo/<task>/<model>/output.

    Returns a list of Paths to output directories.
    """
    if not modelzoo.is_dir():
        error(f"modelzoo directory not found: {modelzoo}")
        sys.exit(1)

    output_dirs: list[Path] = []

    # Determine which tasks to scan
    if target_task:
        task_dirs = [modelzoo / target_task]
        if not task_dirs[0].is_dir():
            error(f"Task directory not found: {task_dirs[0]}")
            sys.exit(1)
    else:
        task_dirs = sorted(p for p in modelzoo.iterdir() if p.is_dir())

    for task_dir in task_dirs:
        # Determine which models to scan within each task
        if target_model:
            model_dirs = [task_dir / target_model]
            if not model_dirs[0].is_dir():
                # Model not in this task — skip silently when searching all tasks
                if target_task:
                    error(f"Model directory not found: {model_dirs[0]}")
                    sys.exit(1)
                continue
        else:
            model_dirs = sorted(p for p in task_dir.iterdir() if p.is_dir())

        for model_dir in model_dirs:
            output_dir = model_dir / "output"
            if output_dir.is_dir():
                output_dirs.append(output_dir)

    return output_dirs


def cleanup_output_dir(output_dir: Path, dry_run: bool) -> tuple[int, int]:
    """
    Remove all items inside *output_dir* that are not in KEEP_ITEMS.

    Returns (deleted_count, total_freed_bytes).
    """
    deleted_count = 0
    total_freed_bytes = 0

    for item in sorted(output_dir.iterdir()):
        if item.name in KEEP_ITEMS:
            continue

        # Calculate size before deleting
        size = item.stat().st_size if item.is_file() else _dir_size(item)

        if dry_run:
            item_type = "file" if item.is_file() else "dir "
            info(f"  [dry-run] Would delete {item_type}: {item}  ({_fmt_size(size)})")
        else:
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
                info(f"  Deleted: {item}  ({_fmt_size(size)})")
            except Exception as exc:
                error(f"  Failed to delete {item}: {exc}")
                continue

        deleted_count += 1
        total_freed_bytes += size

    return deleted_count, total_freed_bytes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _dir_size(path: Path) -> int:
    """Calculate total size of a directory recursively."""
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    except Exception:
        pass
    return total


def _fmt_size(num_bytes: int) -> str:
    """Human-readable size string."""
    size: float = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def divider(char: str = "=", width: int = 56) -> str:
    return char * width


# ---------------------------------------------------------------------------
# Public API — callable from other scripts
# ---------------------------------------------------------------------------
def run_cleanup(
    modelzoo: Path | None = None,
    model: str | None = None,
    task: str | None = None,
    dry_run: bool = False,
) -> None:
    """
    Delete extra folders/files from model output directories.

    Can be called programmatically from other scripts (e.g. batch_compile.py).
    """
    if modelzoo is None:
        modelzoo = Path(__file__).parent.parent / "modelzoo"

    # ── Header ──────────────────────────────────────────────────────────
    info(divider("="))
    info("Cleanup output directories — removing extra folders/files")
    info(f"  Modelzoo : {modelzoo}")
    if model:
        info(f"  Model    : {model}")
    if task:
        info(f"  Task     : {task}")
    if dry_run:
        info("  *** DRY RUN — nothing will be deleted ***")
    info(f"  Preserved items: {', '.join(sorted(KEEP_ITEMS))}")
    info(divider("="))

    # ── Find output dirs ────────────────────────────────────────────────
    output_dirs = find_output_dirs(modelzoo, model, task)

    if not output_dirs:
        warn("No output directories found.")
        return

    info(
        f"Found {len(output_dirs)} output director{'y' if len(output_dirs) == 1 else 'ies'}."
    )

    # ── Clean each output dir ───────────────────────────────────────────
    grand_total_deleted = 0
    grand_total_bytes = 0

    for output_dir in output_dirs:
        # Derive the task/model path components for display
        model_name = output_dir.parent.name
        task_name = output_dir.parent.parent.name
        label = f"{task_name}/{model_name}"

        info("")
        info(divider("-"))
        info(f"Processing: {label}")
        info(divider("-"))

        deleted, freed = cleanup_output_dir(output_dir, dry_run)
        grand_total_deleted += deleted
        grand_total_bytes += freed

        if deleted == 0:
            info(f"  Nothing to clean in {label}.")

    # ── Summary ─────────────────────────────────────────────────────────
    info("")
    info(divider("="))
    if dry_run:
        info(
            f"[dry-run] Would delete {grand_total_deleted} items "
            f"({_fmt_size(grand_total_bytes)} freed)."
        )
    else:
        info(
            f"Cleanup complete: {grand_total_deleted} items deleted "
            f"({_fmt_size(grand_total_bytes)} freed)."
        )
    info(divider("="))


# ---------------------------------------------------------------------------
# Main (CLI entry point)
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    run_cleanup(
        modelzoo=args.modelzoo,
        model=args.model,
        task=args.task,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
