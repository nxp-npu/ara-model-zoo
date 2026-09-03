#!/usr/bin/env bash
# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

source core/shell/logger.sh

# No pre-compiled / hosted tiny_vit artifact is published yet. This fallback
# exists so the compile flow has something to call if model_prep.sh fails; once
# the model is uploaded, replace the body with the gdown/download commands
# (see resnet50v1/bin/model_download.sh for the pattern).

error "No hosted tiny_vit model available to download. Run model_prep.sh to export it from timm."
exit 1
