#!/usr/bin/env bash

# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

source core/shell/logger.sh


info "Downloading onnx version of the model"
gdown https://drive.google.com/uc?id=18xxmRBN-nlOhCuUUqtQ9IT0SHzexsVXv --output $OUTPUT_DIR/compiled_model/model.onnx

info "Downloading pytorch version of the model"
gdown https://drive.google.com/uc?id=1wXw9D6EuNacXMI7BCng-UuD0B-6Z8BIU --output $OUTPUT_DIR/compiled_model/model.pt

info "Downloaded onnx and pytorch version of the model in $OUTPUT_DIR/compiled_model"
