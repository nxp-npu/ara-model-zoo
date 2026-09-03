# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

#!/usr/bin/env python3
"""
batch_eval_hw.py — run hardware evaluation for multiple models using eval_hw.sh

By default reads eval_config.json from the same directory as this script.
Pass --config to use a different file.

eval_config.json format:
    {
        "yolov8n":  { "dataset_root": "/data/coco" },
        "resnet50": { "dataset_root": "/data/imagenet" },
        "face_det": { "dataset_root": "/data/faces" }
    }

Usage:
    python batch_eval_hw.py
    python batch_eval_hw.py --config /path/to/my_config.json
    python batch_eval_hw.py --config my_config.json --dry-run
"""

import argparse
import json
import sys
import time
from pathlib import Path

from cleanup_output import run_cleanup
from common import (
    divider,
    error,
    fmt_duration,
    info,
    print_summary,
    resolve_log_dir,
    run_shell,
    success,
)


# ---------------------------------------------------------------------------
# Data class for a resolved model entry
# ---------------------------------------------------------------------------
class ModelEntry:
    def __init__(self, name: str, dataset_root: Path):
        self.name = name
        self.dataset_root = dataset_root


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="batch_eval_hw.py",
        description="Run hardware evaluation for multiple models using eval_hw.sh.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python batch_eval_hw.py
  python batch_eval_hw.py --config /path/to/my_config.json
  python batch_eval_hw.py --config my_config.json --limit 100 --skip-proxy
  python batch_eval_hw.py --dry-run

eval_config.json format:
  {
      "yolov8n":  { "dataset_root": "/data/coco" },
      "resnet50": { "dataset_root": "/data/imagenet" },
      "face_det": { "dataset_root": "/data/faces" }
  }
        """,
    )

    parser.add_argument(
        "--config",
        type=Path,
        metavar="PATH",
        default=Path(__file__).parent.parent / "eval_config.json",
        help="JSON config file mapping model names to their datasets. "
        "(default: eval_config.json next to this script)",
    )
    parser.add_argument(
        "--run",
        choices=["python", "cpp"],
        default="python",
        help="Evaluation backend passed to eval_hw.sh (default: python).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        default=None,
        help="Limit evaluation to the first N samples per model.",
    )
    parser.add_argument(
        "--skip-proxy",
        action="store_true",
        help="Pass skip_proxy=true to eval_hw.sh — skips proxy start/stop.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        metavar="DIR",
        default=None,
        help="Directory to write per-model log files. "
        "(default: logs/<timestamp>/eval_hw/)",
    )
    parser.add_argument(
        "--eval-script",
        type=Path,
        metavar="PATH",
        default=Path(__file__).parent.parent / "flows" / "eval_hw.sh",
        help="Path to eval_hw.sh.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be run without executing them.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="After evaluation, delete extra folders/files from each model's "
        "output directory, keeping only assets, compiled_model, evaluation, "
        "performance, dvrun.yaml, run.yaml, and version.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config(path: Path) -> dict:
    if not path.is_file():
        error(f"Config file not found: {path}")
        sys.exit(1)
    try:
        with path.open() as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        error(f"Failed to parse config file {path}: {exc}")
        sys.exit(1)


def build_model_entries(config: dict) -> list[ModelEntry]:
    """Parse the flat config dict into a list of ModelEntry objects."""
    entries = []
    for name, settings in config.items():
        if "dataset_root" not in settings:
            error(f"Model '{name}' is missing required 'dataset_root' in config.")
            sys.exit(1)
        entries.append(
            ModelEntry(
                name=name,
                dataset_root=Path(settings["dataset_root"]),
            )
        )
    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    args.log_dir = resolve_log_dir(__file__, "eval_hw", override=args.log_dir)

    # Validate eval script
    if not args.dry_run and not args.eval_script.is_file():
        error(f"eval_hw.sh not found at: {args.eval_script}")
        error("Use --eval-script to specify its location.")
        sys.exit(1)

    # Load config and build model entries
    config = load_config(args.config)
    model_entries = build_model_entries(config)

    if not model_entries:
        error(f"No models found in config: {args.config}")
        sys.exit(1)

    total = len(model_entries)

    # Validate all paths up front before running anything
    if not args.dry_run:
        all_valid = True
        for entry in model_entries:
            if not entry.dataset_root.is_dir():
                error(f"[{entry.name}] dataset_root not found: {entry.dataset_root}")
                all_valid = False
        if not all_valid:
            sys.exit(1)

    # Header
    info(divider("="))
    info("Batch eval_hw starting")
    info(f"  Config    : {args.config}")
    info(f"  Models    : {total}")
    info(f"  Backend   : {args.run}")
    info(f"  Limit     : {args.limit or '(all samples)'}")
    info(f"  Skip proxy: {args.skip_proxy}")
    info(f"  Log dir   : {args.log_dir}")
    if args.dry_run:
        info("  *** DRY RUN — no commands will be executed ***")
    if args.cleanup:
        info("  Cleanup : enabled (will run after evaluation)")
    info(divider("-"))
    info("  Model dataset assignments:")
    for entry in model_entries:
        info(f"    {entry.name:<20}  dataset={entry.dataset_root}")
    info(divider("="))

    # Eval loop
    passed: list[str] = []
    failed: list[str] = []
    batch_start = time.monotonic()

    for i, entry in enumerate(model_entries, start=1):
        info("")
        info(divider())
        info(f"[{i}/{total}] Starting : {entry.name}")
        info(f"             Dataset  : {entry.dataset_root}")
        info(divider())

        log_path = args.log_dir / f"{entry.name}.log"
        cmd = [
            "bash",
            str(args.eval_script),
            f"model={entry.name}",
            f"dataset_root={entry.dataset_root}",
            f"run={args.run}",
            f"skip_proxy={'true' if args.skip_proxy else 'false'}",
        ]
        if args.limit is not None:
            cmd.append(f"limit={args.limit}")

        try:
            exit_code, elapsed = run_shell(cmd, log_path=log_path, dry_run=args.dry_run)
            ok = exit_code == 0
        except Exception as exc:
            error(f"[{i}/{total}] FAILED: {entry.name}  (unexpected error: {exc})")
            failed.append(entry.name)
            info("Continuing with remaining models...")
            continue

        duration = fmt_duration(elapsed)

        if ok:
            success(f"[{i}/{total}] PASSED: {entry.name}  ({duration})")
            info(f"  Log: {log_path}")
            passed.append(entry.name)
        else:
            error(f"[{i}/{total}] FAILED: {entry.name}  ({duration})")
            info(f"  Log: {log_path}")
            failed.append(entry.name)
            info("Continuing with remaining models...")

    # Summary
    print_summary(
        "eval_hw", passed, failed, elapsed_total=time.monotonic() - batch_start
    )

    # ── Cleanup (if requested) ──────────────────────────────────────────
    if args.cleanup:
        run_cleanup(
            modelzoo=Path(__file__).parent.parent / "modelzoo",
            dry_run=args.dry_run,
        )

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
