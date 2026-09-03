# Resnet-50v1 (onnx)

## Use Case and High-Level Description

ResNet 50 is image classification model pre-trained on ImageNet dataset. This
is PyTorch\* implementation based on architecture described in paper ["Deep Residual
Learning for Image Recognition"](https://arxiv.org/abs/1512.03385) in TorchVision
package (see [here](https://github.com/pytorch/vision)).

The model input is a blob that consists of a single image of `1, 3, 224, 224`.

The model output is typical object classifier for the 1000 different classifications
matching with those in the ImageNet database.

## Model source

This model is being taken directly from [torchvision](https://github.com/pytorch/vision) library.

## Specification

More Information specific to this Particular model can be found here [resnet50 IMAGENET1K_V1](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|

## Accuracy

The FP32 Value is taken from Pytorch Documentation for [resnet50 IMAGENET1K_V1](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)

| Metric | FP32   | Int8  |
| ------ | ------ | ------ |
| Top 1  | 76.13| 75.97




## Dataset used for checking accuracy

Original Dataset [ImageNet](https://image-net.org/download.php)

|      |     |
| ------ | ------ |
| Dataset Name | imagenet-1k |
| Classes  | 1000 |
| Validation  images | 50,000 Used to Calculate Accuracy Numbers |
| Image format | JPEG |

[Click for more info on imagenet-1k dataset](https://huggingface.co/datasets/imagenet-1k), You can download a subset of complete dataset from here by a quick sign-up. Official website for Complete dataset is [ImageNet](https://image-net.org/download.php), requires approval.

Tags file is a text file where each line contains validation image name and corresponding class integer.
Class Integer is expected in range of 0-999 for this model.

## Model Parameters

#### Pre-Process Stage

| Parameter | Value    |
| :-------- | :------- |
| `Model input shape` | `[1 3 224 224]` |
| `Model output shape` | `[1 1000 1 1 ]`|
| `Resize` | `256` |
| `Interpolation` | `LINEAR` |
| `Center Crop`      | `224` |
| `Mean`      | `{123.68, 116.28, 103.53}` |
| `Scale`      | `{0.017125, 0.017507, 0.017429}` |


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

### Pre-compiled model download
Ara modelzoo included precompiled models to run on simulator and hardware. These steps can be used to run the precompiled model to check performance, accuracy and run sample application

* Download pre-compiled model
```bash
cd ara-model-zoo
./flows/precompiled_model_download.sh model=resnet50v1
```
* Precompiled models are compiled using Ara SDK.
* Output model and assets will be stored in the `modelzoo/classification/resnet50v1/output/assets` folder

#### Model Compile

```bash
cd ara-model-zoo
./flows/model_compile.sh model=resnet50v1 dataset_root=<path to imagenet directory>
```
* Output model and assets will be stored in the `modelzoo/classification/resnet50v1/output/assets` folder

## Model Evaluation flows
Note that for the following flows the latest model that has been compiled or downloaded will be used.

#### Performance checker

```bash
cd ara-model-zoo
./flows/performance.sh model=resnet50v1
```

Look for `HW IPS` for ips in output.

#### Inference Application
Runs inference application on images provided to display inference results, if no images are provided a default test image will be used

_Python execution_
```bash
cd ara-model-zoo
./flows/infer_float.sh model=resnet50v1 images_folder=<path to images to run inference on>
```

**Example:**

_Python example_
```bash
cd ara-model-zoo
./flows/infer_hw.sh images_folder=../testimages model=resnet50v1
```

* Output images saved in postprocessed_images_viz_default folder


#### Accuracy checker
* Note: Accuracy check can take several minutes to hours depending on dataset size and host machine. In order to run accuracy on complete dataset for this model, please download from [ImageNet](https://image-net.org/download.php) or refer [link](#dataset-used-for-checking-accuracy)

*Run with Python*
```bash
cd ara-model-zoo
./flows/eval_hw.sh model=resnet50v1 dataset_folder=<folder containing image dataset>
```


**Example:**

*Python example*
```bash
cd ara-model-zoo
./flows/eval_hw.sh model=resnet50v1 dataset_root=/media/data/datasets/imagenet/imagenet
```


* Accuracy metrics printed to terminal
