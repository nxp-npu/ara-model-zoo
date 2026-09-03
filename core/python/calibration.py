#
# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.
#
from __future__ import annotations

import os
from pathlib import Path
from core.python.logger import logger
from shutil import copyfile, rmtree
from typing import Sequence


def read_image_list(path: Path) -> list[str]:
    """
    Read image file namess from path, one per line
    """

    names: list[str] = []
    seen: set[str] = set()  # repeated names are kept only once

    for line in path.read_text().splitlines():
        name = line.split("#")[0].strip()  # ignore comments
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)

    if not names:
        raise ValueError(f"Calibration image list is empty: {path}")

    return names


def _index_by_name(dataset_path: Path) -> dict[str, Path]:
    """
    Map every file below dataset_path to its path
    """

    index: dict[str, Path] = {}

    for dpath, _, fnames in os.walk(dataset_path):
        for fname in fnames:
            index.setdefault(fname, Path(dpath) / fname)

    return index


def _locate(names: Sequence[str], dataset_path: Path) -> dict[str, Path]:
    """
    For each name, resolve it to a file below dataset_path
    """
    located = {name: dataset_path / name for name in names}
    unresolved = [name for name, path in located.items() if not path.is_file()]

    # handle unresolved
    if unresolved:
        idx = _index_by_name(dataset_path)
        # update if located
        located.update({name: idx[name] for name in unresolved if name in idx})

    for name in sorted(name for name, path in located.items() if not path.is_file()):
        logger.warning(
            f"Calibration image {name} not found under {dataset_path} so it won't be used"
        )
        del located[name]

    return located


def stage_calib_images(
    names: Sequence[str],
    dataset_path: Path,
    destination: Path,
) -> list[str]:
    """
    Copy the images (names) from dataset_path into a clean destination
    """
    located = _locate(names, dataset_path)

    # only abort if none of the listed images can be found
    if not located:
        raise FileNotFoundError(
            f"None of the {len(names)} listed calibration images were found under {dataset_path}"
        )

    if destination.exists():
        logger.info(f"Removing existing calibration image directory: {destination}")
        rmtree(destination)
    destination.mkdir(parents=True)

    for name, source in located.items():
        copyfile(source, destination / Path(name).name)

    logger.info(
        f"Staged {len(located)} of {len(names)} calibration images from {dataset_path} to {destination}"
    )

    return sorted(Path(name).name for name in located)
