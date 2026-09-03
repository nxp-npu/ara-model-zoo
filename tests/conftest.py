# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

"""
Global pytest configuration.

Provides shared test environment setup required by the test suite.

This module prepares lightweight runtime dependencies expected by the
production code (such as environment variables, temporary filesystem
artifacts, or mocked runtime resources) so unit tests remain isolated
from external installations and machine-specific configuration.
"""

from pathlib import Path
import os
import platform

# ---------------------------------------------------------------------
# Fake optional runtime installation
# ---------------------------------------------------------------------

FAKE_RUNTIME_ROOT = Path("/tmp/fake_runtime")

os.environ.setdefault(
    "DV_TGT_ROOT",
    str(FAKE_RUNTIME_ROOT),
)

# ---------------------------------------------------------------------
# Create the expected directory hierarchy
# ---------------------------------------------------------------------

machine_architecture = platform.machine().lower()

if machine_architecture in ["x86_64", "amd64"]:
    arch_dir = "x86"
elif machine_architecture in ["aarch64", "arm64"]:
    arch_dir = "aarch64"
else:
    raise RuntimeError(
        f"Unsupported architecture: {machine_architecture}. Only x86 and aarch64 are supported."
    )

RUNTIME_MODULE = FAKE_RUNTIME_ROOT / "art" / "linux" / arch_dir / "include" / "dvapi.py"

RUNTIME_MODULE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

RUNTIME_MODULE.touch(
    exist_ok=True,
)
