#!/usr/bin/env sh

# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

source core/shell/logger.sh

mkdir $OUTPUT_DIR/compiled

info "Downloading onnx version of the model"
wget -O $OUTPUT_DIR/compiled/yolox_tiny.onnx https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx 

info "Downloading pytorch version of the model"
wget -O $OUTPUT_DIR/compiled/yolox_tiny.pth https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth 

info "Downloaded onnx and pytorch version of the model in $OUTPUT_DIR/compiled_model"

cp $OUTPUT_DIR/compiled/yolox_tiny.onnx $OUTPUT_DIR/compiled_model/model.onnx

cp $OUTPUT_DIR/compiled/yolox_tiny.pth $OUTPUT_DIR/compiled_model/model.pth

# Remove the compiled folder
rm -rf $OUTPUT_DIR/compiled