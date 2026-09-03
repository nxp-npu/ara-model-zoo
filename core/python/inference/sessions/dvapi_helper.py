# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import importlib.util
import os
import platform
import sys
from pathlib import Path

sdk_root = os.environ.get("DV_TGT_ROOT")
if not sdk_root:
    raise EnvironmentError(
        "`DV_TGT_ROOT` environment variable is not set. Please set it to the SDK root directory."
    )

machine_architecture = platform.machine().lower()


if machine_architecture in ["x86_64", "amd64"]:
    arch_dir = "x86"
elif machine_architecture in ["aarch64", "arm64"]:
    arch_dir = "aarch64"
else:
    raise RuntimeError(
        f"Unsupported architecture: {machine_architecture}. Only x86 and aarch64 are supported."
    )

dvapi_path = Path(sdk_root) / "art" / "linux" / arch_dir / "include" / "dvapi.py"

if not dvapi_path.exists():
    raise FileNotFoundError(f"dvapi.py not found at {dvapi_path}")

spec = importlib.util.spec_from_file_location("dvapi", dvapi_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load module spec from {dvapi_path}")

dvapi_module = importlib.util.module_from_spec(spec)
sys.modules["dvapi"] = dvapi_module
sys.modules["core.python.inference.sessions.dvapi"] = dvapi_module
spec.loader.exec_module(dvapi_module)

import dvapi  # ty: ignore[unresolved-import] # noqa: E402

__all__ = ["dvapi"]
