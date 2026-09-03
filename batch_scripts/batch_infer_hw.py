# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

#!/usr/bin/env python3
"""
batch_infer_hw.py — run hardware inference for multiple models using infer_hw.sh

Usage:
    python batch_infer_hw.py --models model1 model2 model3
    python batch_infer_hw.py --file models.txt
    python batch_infer_hw.py --models yolov8n resnet50 --skip-proxy
"""

import argparse
import sys
import time
from pathlib import Path

from common import (
    build_model_list,
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
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="batch_infer_hw.py",
        description="Run hardware inference for multiple models using infer_hw.sh.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python batch_infer_hw.py --models yolov8n resnet50
  python batch_infer_hw.py --file models.txt
  python batch_infer_hw.py --models yolov8n --run cpp --skip-proxy
  python batch_infer_hw.py --file models.txt --dry-run

models.txt format (one model per line, # for comments):
  yolov8n
  resnet50
  # face_det   <- skipped
        """,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        default=[],
        help="One or more model names to run inference for.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        metavar="PATH",
        help="Text file with one model name per line (# for comments).",
    )
    parser.add_argument(
        "--run",
        choices=["python", "cpp"],
        default="python",
        help="Inference backend passed to infer_hw.sh (default: python).",
    )
    parser.add_argument(
        "--images-folder",
        type=Path,
        metavar="PATH",
        default=None,
        help="Path to the images folder. (default: testimages/ under the project root).",
    )
    parser.add_argument(
        "--skip-proxy",
        action="store_true",
        help="Pass skip_proxy=true to infer_hw.sh — skips proxy start/stop.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        metavar="DIR",
        default=None,
        help="Directory to write per-model log files. "
        "(default: logs/<timestamp>/infer_hw/)",
    )
    parser.add_argument(
        "--infer-script",
        type=Path,
        metavar="PATH",
        default=Path(__file__).parent.parent / "flows" / "infer_hw.sh",
        help="Path to infer_hw.sh.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be run without executing them.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    args.log_dir = resolve_log_dir(__file__, "infer_hw", override=args.log_dir)

    # Validate infer script
    if not args.dry_run and not args.infer_script.is_file():
        error(f"infer_hw.sh not found at: {args.infer_script}")
        error("Use --infer-script to specify its location.")
        sys.exit(1)

    # Build model list
    models = build_model_list(args.models, args.file)
    if not models:
        error("No models specified. Use --models or --file.")
        sys.exit(1)

    total = len(models)

    # Header
    info(divider("="))
    info("Batch infer_hw starting")
    info(f"  Models      : {total}")
    info(f"  Backend     : {args.run}")
    info(f"  Skip proxy  : {args.skip_proxy}")
    info(f"  Log dir     : {args.log_dir}")
    if args.images_folder:
        info(f"  Images folder: {args.images_folder}")
    if args.dry_run:
        info("  *** DRY RUN — no commands will be executed ***")
    info(divider("="))

    # Run loop
    passed: list[str] = []
    failed: list[str] = []
    batch_start = time.monotonic()

    for i, model in enumerate(models, start=1):
        info("")
        info(divider())
        info(f"[{i}/{total}] Starting: {model}")
        info(divider())

        log_path = args.log_dir / f"{model}.log"
        cmd = [
            "bash",
            str(args.infer_script),
            f"model={model}",
            f"run={args.run}",
            f"skip_proxy={'true' if args.skip_proxy else 'false'}",
        ]
        if args.images_folder is not None:
            cmd.append(f"images_folder={args.images_folder}")

        try:
            exit_code, elapsed = run_shell(cmd, log_path=log_path, dry_run=args.dry_run)
            ok = exit_code == 0
        except Exception as exc:
            error(f"[{i}/{total}] FAILED: {model}  (unexpected error: {exc})")
            failed.append(model)
            info("Continuing with remaining models...")
            continue

        duration = fmt_duration(elapsed)

        if ok:
            success(f"[{i}/{total}] PASSED: {model}  ({duration})")
            info(f"  Log: {log_path}")
            passed.append(model)
        else:
            error(f"[{i}/{total}] FAILED: {model}  ({duration})")
            info(f"  Log: {log_path}")
            failed.append(model)
            info("Continuing with remaining models...")

    # Summary
    print_summary(
        "infer_hw", passed, failed, elapsed_total=time.monotonic() - batch_start
    )

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
