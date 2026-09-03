# YOLO26n

## Model Overview

YOLO26n is the nano-scale variant of Ultralytics' YOLO26, a unified real-time object detection model family released in September 2025. It is designed for edge and low-power deployment, targeting hardware like dedicated NPUs, mobile CPUs, and embedded accelerators.

| Attribute | Value |
|---|---|
| Task | Object Detection (80-class COCO) |
| Input Resolution | 640 x 640 x 3 |
| Parameters (fused) | 2.4M |
| FLOPs | 5.4B |

We use YOLO26n specifically for the object detection task on COCO (80 classes).

## Model Source

This model is taken from the [Ultralytics](https://github.com/ultralytics/ultralytics) library. The ONNX graph is exported from the official pretrained weights.

| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection     |
| Source framework  | Onnx\*        |

## Key Features

YOLO26 introduces four coordinated improvements over its predecessors (YOLO11, YOLOv8):

1. **NMS-Free End-to-End Inference**: Uses a dual-head architecture with one-to-one label assignment during training, so each ground truth object is assigned exactly one prediction. This eliminates Non-Maximum Suppression at inference, removing a sequential, data-dependent post-processing step that is difficult to accelerate on NPU/TPU hardware.

2. **DFL-Free Detection Head**: Completely removes Distribution Focal Loss (DFL), which previous versions used for bounding box regression. This makes the detection head lighter (fewer output channels) and simplifies export across hardware formats (ONNX, TensorRT, TFLite, CoreML, OpenVINO). The regression range is now unconstrained rather than discretized.

3. **MuSGD Optimizer**: A hybrid Muon-SGD optimizer adapted from large language model training. It applies Muon-style orthogonalized gradient updates (via Newton-Schulz iterations) to weight matrices (parameters with ndim >= 2) while using standard SGD for biases and 1D parameters. This improves training stability and convergence.

4. **Progressive Loss + STAL**: Progressive Loss gradually shifts supervision toward the inference-time one-to-one head during training, improving alignment between training and deployment. STAL (Small-Target-Aware Label Assignment) guarantees positive label coverage for small objects, addressing a long-standing weakness in YOLO detectors.

## Accuracy

| Metric | FP32   | Int8  |
| ------ | ------ | ------ |
| mAP 50-95  | 40.1 | 38.2 |

## Model Parameters

### Pre-Process Stage

| Parameter | Value |
| :-------- | :------- |
| `Model input shape` | `[1, 3, 640, 640]` |
| `Color conversion` | `BGR -> RGB` |
| `Resize` | `Centered letterbox (maintain aspect ratio)` |
| `Pad value` | `114` |
| `Normalization` | `/255.0` |
| `Channel order` | `HWC -> CHW` |

### Post-Process Stage

| Parameter | Value |
| :-------- | :------- |
| `Model output shape` | `[1, 300, 6]` |
| `Decoding` | `dist2bbox (distance-to-bounding-box conversion)` |
| `Selection` | `Top-300 detections by confidence` |
| `Coordinate correction` | `Subtract pad offset, divide by scale factor` |
| `Classes` | `80 (COCO)` |

## Dataset Used for Checking Accuracy

|      |     |
| ------ | ------ |
| Dataset Name | COCO val2017 |
| Classes  | 80 |
| Validation images | 5,000 (used to calculate accuracy numbers) |
| Image format | JPEG |
| Metric | mAP 50-95 (pycocotools COCOeval) |

## Setup

* Ara SDK is required.
* Follow the steps in the main [README](../../README.md) to update config file with license key.

### Python

Install Python requirements:

```bash
cd ara-model-zoo
pip install -e .
```

### Environment Variables

```bash
export ARA_SDK_ROOT=<Path to SDK>
```

For hardware flows:

```bash
export DV_TGT_ROOT=<Path to SDK>
```

## Reproducibility

### Model Compile

```bash
./flows/model_compile.sh model=yolo26n
```

* Output model and assets will be stored in the `modelzoo/detection/yolo26n/output/assets` folder
* the model.dvm file will also be present in the `modelzoo/detection/yolo26n/output/compiled_model` folder

### Performance Checker

```bash
./flows/performance.sh model=yolo26n
```

Look for `HW IPS` for inferences per second in output.

### Inference Application

Runs inference on provided images to display detection results. If no images are provided, a default test image will be used.

```bash
./flows/infer_hw.sh model=yolo26n images_folder=<path to images>
```

* Output images saved in output/postprocessed_output_visualized folder

### Accuracy Checker

Note: Accuracy check can take several minutes to hours depending on dataset size and host machine. To run accuracy on the complete COCO val2017 dataset, download from [COCO](https://cocodataset.org/).

```bash
./flows/eval_hw.sh model=yolo26n dataset_root=<containing COCO val2017 images folder root path>
```

* Accuracy metrics (mAP 50-95) printed to terminal
