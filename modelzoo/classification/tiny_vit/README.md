# TinyViT-11M-224 (onnx)

## Use Case and High-Level Description

TinyViT-11M-224 is a compact Vision Transformer image classification model
pre-trained on ImageNet-21k and distilled + fine-tuned on ImageNet-1k. This is
the `tiny_vit_11m_22kto1k_distill` checkpoint from the original
[TinyViT](https://github.com/wkcn/TinyViT) repository (equivalent to timm's
`tiny_vit_11m_224.dist_in22k_ft_in1k`).

The model input is a blob that consists of a single image of `1, 3, 224, 224`.

The model output is a typical object classifier for the 1000 different
classifications matching those in the ImageNet database.

## Model source

The model is built from the original [TinyViT](https://github.com/wkcn/TinyViT)
source repository, and the weights are the `tiny_vit_11m_22kto1k_distill.pth`
checkpoint published in the
[TinyViT-model-zoo](https://github.com/wkcn/TinyViT-model-zoo/releases/tag/checkpoints)
releases. `model_prep.sh` clones the repo for the `TinyViT` architecture,
downloads that checkpoint, loads the weights, and exports to ONNX.

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| Source framework  | Onnx\*        |

## Accuracy

| Metric | FP32   |
| ------ | ------ |
| Top 1  | 83.19  |

## Dataset used for checking accuracy

Original Dataset [ImageNet](https://image-net.org/download.php)

|      |     |
| ------ | ------ |
| Dataset Name | imagenet-1k |
| Classes  | 1000 |
| Validation  images | 50,000 Used to Calculate Accuracy Numbers |
| Image format | JPEG |

[Click for more info on imagenet-1k dataset](https://huggingface.co/datasets/imagenet-1k). You can download a subset of the complete dataset from there with a quick sign-up. The official website for the complete dataset is [ImageNet](https://image-net.org/download.php), which requires approval.

Tags file is a text file where each line contains a validation image name and the corresponding class integer.
Class Integer is expected in range of 0-999 for this model.

## Model Parameters

#### Pre-Process Stage

| Parameter | Value    |
| :-------- | :------- |
| `Model input shape` | `[1 3 224 224]` |
| `Model output shape` | `[1 1000]`|
| `Resize` | `256` |
| `Interpolation` | `BILINEAR` |
| `Center Crop`      | `224` |
| `Mean`      | `{123.68, 116.28, 103.53}` |
| `Scale`      | `{0.017125, 0.017507, 0.017429}` |

> Note: `Mean`/`Scale` are the standard ImageNet normalization
> (mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`) expressed in this
> repo's 0-255 convention `(x - mean) * scale`, i.e. `mean = 255 * IMAGENET_MEAN`
> and `scale = 1 / (255 * IMAGENET_STD)`. The resize is applied with torch/PIL
> `BILINEAR` in `preprocess.py`; the remaining ops run through `PreprocessingBuilder`.

#### Post-Process Stage

| Parameter | Value    |
| :-------- | :------- |
|  `classifier_activation` |`softmax`|
|  `classes`|   `1000`|

## Setup
* Ara SDK `ara2-sdk-r3.0` is required.
* Follow [these](../../README.md#License) steps to update config file with license_key.

### Python
Install python requirements
```bash
cd ara-model-zoo/
pip3 install -r requirements.txt
```

## Steps to prepare model
Ara Modelzoo offers two methods to generate the model file:
* Pre-compiled model download, where a pre-compiled model can be downloaded from the Ara repository.
* Manually compiled model, where the model is imported in its original framework and then compiled using Ara SDK.

Usage of Ara modelzoo requires access and download of Ara SDK. If you do not have Ara SDK, please contact our [support](https://support.nxp.com/). For details on host requirements, please refer to SDK Documentation.

### Environment Variables
```bash
export ARA_SDK_ROOT=<Path to SDK>
```

### Manual Model Preparation and Compile

#### Model Compile

```bash
cd ara-model-zoo/
./flows/model_compile.sh model=tiny_vit
```
* `model_prep.sh` clones the [TinyViT](https://github.com/wkcn/TinyViT) repo, downloads the `tiny_vit_11m_22kto1k_distill.pth` checkpoint from [TinyViT-model-zoo](https://github.com/wkcn/TinyViT-model-zoo/releases/tag/checkpoints), loads the weights into the `TinyViT` architecture, and exports the float ONNX (opset 20 + onnxsim, input/output node names `input`/`output`) into `output/compiled_model/model.onnx`.

#### Model Compile (optimized / quantized)
Stage the pre-quantized ONNX + encodings first, then compile with `--optimized`:

```bash
cd ara-model-zoo/
source modelzoo/classification/tiny_vit/bin/model_prep_optimized.sh   # stages quantized_model.onnx + encodings
./flows/model_compile.sh model=tiny_vit --optimized
```
* Output model and assets will be stored in the `modelzoo/classification/tiny_vit/output/assets` folder.

## Model Evaluation flows
Note that for the following flows the latest model that has been compiled or downloaded will be used.

#### Inference Application
Floating-point inference:
```bash
./flows/infer_float.sh model=tiny_vit images_folder=<path to images to run inference on>
```
Hardware inference:
```bash
./flows/infer_hw.sh model=tiny_vit images_folder=<path to images to run inference on>
```
#### Accuracy checker
Floating-point evaluation:
```bash
./flows/eval_float.sh model=tiny_vit dataset_root=<path to imagenet> limit=<num images>
```
Hardware evaluation:
```bash
./flows/eval_hw.sh model=tiny_vit dataset_root=<path for eval dataset on hardware>
```
* Accuracy metrics printed to terminal.
