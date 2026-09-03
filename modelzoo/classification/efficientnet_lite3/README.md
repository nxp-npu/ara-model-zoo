# EfficientNet-Lite3 (onnx)

## Use Case and High-Level Description

EfficientNet-Lite3 is an image classification model pre-trained on the ImageNet dataset. It is a member of the EfficientNet-Lite family, which are variants of EfficientNet designed to run efficiently on edge and mobile hardware without relying on operations that are unfriendly to fixed-point quantization (such as squeeze-and-excitation blocks). The model is described in the paper ["EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"](https://arxiv.org/abs/1905.11946) and the Lite variants are detailed in the [Google TPU repository](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite).

The model input is a blob that consists of a single image of `1, 3, 280, 280` in NCHW format.

The model output is a standard image classifier producing scores for 1000 classes matching those in the ImageNet database.

## Model Source

The model is taken from the [Google TensorFlow TPU repository](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite). A TensorFlow checkpoint is exported to SavedModel format using the `export_model.py` script included in that repository, then converted to ONNX with NCHW input layout using [tf2onnx](https://github.com/onnx/tensorflow-onnx).

## Specification

| Metric           | Value          |
|------------------|----------------|
| Type             | Classification |
| Source framework | TensorFlow / ONNX\* |
| Input shape      | `[1, 3, 280, 280]` |
| Output shape     | `[1, 1000]` |
| Parameters       | ~8.2M |

## Accuracy

The FP32 reference value is taken from the [official EfficientNet-Lite repository](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite).

| Metric | FP32  | Int8 |
|--------|-------|------|
| Top-1  | 79.8  | 79.16|

## Dataset Used for Checking Accuracy

Original dataset: [ImageNet](https://image-net.org/download.php)

|                    |                                                         |
|--------------------|---------------------------------------------------------|
| Dataset Name       | imagenet-1k                                             |
| Classes            | 1000                                                    |
| Validation images  | 50,000 used to calculate accuracy numbers               |
| Image format       | JPEG                                                    |




## Model Parameters

### Pre-Process Stage

| Parameter          | Value              |
|:-------------------|:-------------------|
| Model input shape  | `[1, 3, 280, 280]` |
| Model output shape | `[1, 1000]`        |
| Center crop        | 87.5% of the shorter side at native resolution |
| Resize to          | `280 x 280`        |
| Interpolation      | `BILINEAR`         |
| Mean               | `{127.0, 127.0, 127.0}` |
| Scale              | `{0.0078125, 0.0078125, 0.0078125}` (i.e. 1/128) |

The preprocessing follows the official TensorFlow evaluation pipeline exactly: the centre crop is taken at the original image resolution before any resizing. This order is important, resizing first and then cropping produces lower accuracy.

### Post-Process Stage

| Parameter              | Value    |
|:-----------------------|:---------|
| `classifier_activation`| softmax  |
| `classes`              | 1000     |


## Setup

- Ara SDK `ara2-sdk-r3.0` is required.
- For model compilation Python 3.10 is required to be installed in the local machine.

### Python

Install python requirements

```bash
cd ara-model-zoo
pip3 install -e .
```

### Environment Variables

```bash
export ARA_SDK_ROOT=<Path to SDK>
export DV_TGT_ROOT=<Path to SDK>
```


### Pre-compiled model download

Ara-model-zoo included precompiled models to run on hardware. These steps can be used to run the precompiled model to check accuracy and run inference.

- Download pre-compiled model

```bash
cd ara-model-zoo
./flows/precompiled_model_download.sh model=efficientnet_lite3
```

- Precompiled models are compiled using Ara SDK.
- Output model and assets will be stored in the `modelzoo/classification/efficientnet_lite3/output/compiled_model` folder

### Model Compile

```bash
cd ara-model-zoo
./flows/model_compile.sh model=efficientnet_lite3 dataset_root=<path to imagenet directory>
```

Output model is stored in `modelzoo/classification/efficientnet_lite3/output/compiled_model/`

#### Performance Checker

```bash
cd ara-model-zoo
./flows/performance.sh model=efficientnet_lite3
```

Look for `HW IPS` in the output for throughput.

#### Inference Application

```bash
cd ara-model-zoo
./flows/infer_hw.sh model=efficientnet_lite3 images_folder=./testimages
./flows/infer_float.sh model=efficientnet_lite3 images_folder=./testimages
```

Output images are saved in the `output/postprocessed_images_viz_default` folder.

#### Accuracy Checker

```bash
cd ara-model-zoo
./flows/eval_hw.sh model=efficientnet_lite3 dataset_root=/media/data/datasets/imagenet
./flows/eval_float.sh model=efficientnet_lite3 dataset_root=<path to imagenet directory>
```
