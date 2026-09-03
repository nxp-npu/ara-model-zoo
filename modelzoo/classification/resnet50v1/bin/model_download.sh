#!/usr/bin/env bash

# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

source core/shell/logger.sh


info "Downloading onnx version of the model"
gdown https://drive.google.com/uc?id=1opQPPKkQPJ8Gzr16zgDiZrbXMQkEO5fs --output $OUTPUT_DIR/compiled_model/model.onnx

info "Downloading pytorch version of the model"
gdown https://drive.google.com/uc?id=1AAexJfEv-W4di4zOBaEOzZ6IN0DO9I6V --output $OUTPUT_DIR/compiled_model/model.pt

info "Downloaded onnx and pytorch version of the model in $OUTPUT_DIR/compiled_model"
