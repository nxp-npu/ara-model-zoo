#!/usr/bin/env sh

# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

source core/shell/logger.sh

mkdir $OUTPUT_DIR/../compiled

info "Downloading onnx version of the model"
gdown https://drive.google.com/uc?id=1Gl_rxwmvID3a9JBvV3egMJ4-ZGB9IrR- --output $OUTPUT_DIR/../compiled/yolov8n_face.onnx

info "Downloading pytorch version of the model"
gdown https://drive.google.com/uc?id=1F-rDfv4NbMNQ4m8nl159Ed0Jj3KqT-ll --output $OUTPUT_DIR/../compiled/yolov8n_face.pt

info "Downloaded onnx and pytorch version of the model in $OUTPUT_DIR/compiled_model"

cp $OUTPUT_DIR/../compiled/yolov8n_face.onnx $OUTPUT_DIR/compiled_model/model.onnx

cp $OUTPUT_DIR/../compiled/yolov8n_face.pt $OUTPUT_DIR/compiled_model/model.pt

# Remove the compiled folder
rm -rf $OUTPUT_DIR/../compiled