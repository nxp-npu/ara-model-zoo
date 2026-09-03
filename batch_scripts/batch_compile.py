# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

#!/usr/bin/env python3
"""
batch_compile.py — compile multiple models sequentially using model_compile.sh

Every model is compiled against a datset, its calibration images are copied out of it.
We can either pass one dataset root for the whole batch or we can pass them per model.

Usage:
    python batch_compile.py --models model1 model2 model3 --dataset-root /datasets/coco2017
    python batch_compile.py --file models.txt --dataset-config eval_config.json
    python batch_compile.py --models face_det --file extra.txt --run python --cleanup
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from cleanup_output import run_cleanup
from common import (
    build_model_list,
    divider,
    error,
    fmt_duration,
    info,
    print_summary,
    resolve_log_dir,
    success,
    warn,
    map_dataset_roots,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="batch_compile.py",
        description="Compile multiple models sequentially using model_compile.sh.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python batch_compile.py --models face_det pose_net --dataset-root /datasets/coco2017
  python batch_compile.py --file models.txt --dataset-config eval_config.json
  python batch_compile.py --file models.txt --dataset-root /datasets/coco2017 --dry-run

models.txt format (one model per line, # for comments):
  face_detection_v2
  object_tracker
  # pose_estimator   <- skipped
        """,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        default=[],
        help="One or more model names to compile.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        metavar="PATH",
        help="Path to a text file with one model name per line. "
        "Lines starting with # are ignored.",
    )
    parser.add_argument(
        "--run",
        choices=["python", "cpp"],
        default="python",
        help="Compilation backend passed to model_compile.sh (default: python).",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Abort the batch immediately if any model fails.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        metavar="DIR",
        default=None,
        help="Directory to write per-model log files. "
        "(default: logs/<timestamp>/compile/)",
    )
    parser.add_argument(
        "--compile-script",
        type=Path,
        metavar="PATH",
        default=Path(__file__).parent.parent / "flows" / "model_compile.sh",
        help="Path to model_compile.sh.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be run without executing them.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="After compilation, delete extra folders/files from each model's "
        "output directory, keeping only assets, compiled_model, evaluation, "
        "performance, dvrun.yaml, run.yaml, and version.",
    )
    dataset = parser.add_mutually_exclusive_group(required=True)
    dataset.add_argument(
        "--dataset-config",
        type=Path,
        metavar="PATH",
        help="JSON file mapping models to their dataset roots, for batches that handle several datasets",
    )
    dataset.add_argument(
        "--dataset-root",
        type=Path,
        metavar="PATH",
        help="Dataset root used for every model in the batch, calibration images are copied out of it.",
    )

    return parser.parse_args()


def resolve_dataset_roots(
    models: list[str],
    args: argparse.Namespace,
) -> dict[str, Path]:
    """Map every model in the batch to its corresponding calibration dataset"""
    if args.dataset_root is not None:
        return {model: args.dataset_root for model in models}

    models_mapping = map_dataset_roots(args.dataset_config)
    missing_list = [model for model in models if model not in models_mapping]
    if missing_list:
        error(
            f"No dataset_root in {args.dataset_config} for: {', '.join(missing_list)}"
        )
        sys.exit(1)

    return {model: models_mapping[model] for model in models}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    args.log_dir = resolve_log_dir(__file__, "compile", override=args.log_dir)

    # ── Validate compile script ─────────────────────────────────────────
    if not args.dry_run and not args.compile_script.is_file():
        error(f"model_compile.sh not found at: {args.compile_script}")
        error("Use --compile-script to specify its location.")
        sys.exit(1)

    # ── Build model list ────────────────────────────────────────────────
    models = build_model_list(args.models, args.file)
    if not models:
        error("No models specified. Use --models or --file.")
        sys.exit(1)

    total = len(models)

    dataset_roots = resolve_dataset_roots(models, args)
    if not args.dry_run:
        missing = sorted(
            {root for root in dataset_roots.values() if not root.is_dir()},
        )
        for root in missing:
            error(f"dataset_root not found: {root}")
        if missing:
            sys.exit(1)

    # ── Header ──────────────────────────────────────────────────────────
    info(divider("="))
    info("Batch compile starting")
    info(f"  Models  : {total}")
    info(f"  Backend : {args.run}")
    info(f"  Log dir : {args.log_dir}")
    if args.stop_on_failure:
        info("  Mode   : stop on first failure")
    if args.dry_run:
        info("  *** DRY RUN — no commands will be executed ***")
    if args.cleanup:
        info("  Cleanup : enabled (will run after compilation)")
    if args.dataset_root is not None:
        info(f"  Dataset : {args.dataset_root}")
    else:
        info(divider("-"))
        info("  Model dataset assignments:")
        for model in models:
            info(f"   {model} dataset={dataset_roots[model]}")
    info(divider("="))

    # ── Compile loop ────────────────────────────────────────────────────
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
            str(args.compile_script),
            f"model={model}",
            f"dataset_root={dataset_roots[model]}",
            f"run={args.run}",
        ]

        if args.dry_run:
            info(f"[dry-run] Would execute: {' '.join(cmd)}")
            info(f"[dry-run] Log: {log_path}")
            ok, elapsed = True, 0.0
        else:
            try:
                exit_code, elapsed = _run_compile(cmd, log_path)
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

    # ── Summary ─────────────────────────────────────────────────────────
    skipped = total - len(passed) - len(failed)
    print_summary(
        "compile", passed, failed, elapsed_total=time.monotonic() - batch_start
    )
    if skipped:
        warn(f"  Skipped: {skipped} (stopped early)")

    # ── Cleanup (if requested) ──────────────────────────────────────────
    if args.cleanup:
        run_cleanup(
            modelzoo=Path(__file__).parent.parent / "modelzoo",
            dry_run=args.dry_run,
        )

    sys.exit(1 if failed else 0)


def _run_compile(cmd: list[str], log_path: Path) -> tuple[int, float]:
    """Run the compile command. Returns (exit_code, elapsed_seconds)."""
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


if __name__ == "__main__":
    main()
