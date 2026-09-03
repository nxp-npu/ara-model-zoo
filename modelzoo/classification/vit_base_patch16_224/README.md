# ViT-Base-Patch16-224(onnx)

## Use Case and High-Level Description

ViT-Base-Patch16-224 is a Vision Transformer image classification model
pre-trained on ImageNet-21k and fine-tuned on ImageNet-1k. This is the PyTorch\*
implementation available in the Hugging Face model repository. You can find the
original model source [here](https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k)

The model input is a blob that consists of a single image of `1, 3, 224, 224`.

The model output is typical object classifier for the 1000 different classifications
matching with those in the ImageNet database.

## Model source

This model is being taken directly from [Hugging Face](https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k) library.

## Specification

More Information specific to this Particular model can be found here [vit_base_patch16_224](https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k)

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| Source framework  | Onnx\*     |

## Accuracy

The FP32 Value is taken from the model card for [vit_base_patch16_224](https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k)

| Metric | FP32   | Int8  |
| ------ | ------ | ------ |
| Top 1  | 85.10| 85.06




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
| `Model output shape` | `[1 1000]`|
| `Resize` | `248` |
| `Interpolation` | `BICUBIC` |
| `Center Crop`      | `224` |
| `Mean`      | `{127.5, 127.5, 127.5}` |
| `Scale`      | `{0.00784313725, 0.00784313725, 0.00784313725}` |


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

## Steps to prepare model
Ara Modelzoo offers two methods to generate the model file:
* Pre-compiled model download, where a pre-compiled can be downloaded from the Ara repository.
* Manually compiled model, where the model is imported in its original framework and then compiled using Ara SDK.

Usage of Ara modelzoo requires access and download of Ara SDK. If you do not have Ara SDK, please contact our [support](https://support.nxp.com/) . For details on host requirements, please refer to SDK Documentation.

### Environment Variables
```bash
export ARA_SDK_ROOT=<Path to SDK>
```

### Pre-compiled model download
Ara modelzoo included precompiled models to run on simulator and hardware. These steps can be used to run the precompiled model to check performance, accuracy and run sample application

* Download pre-compiled model
```bash
./flows/precompiled_model_download.sh model=vit_base_patch16_224
```
* Precompiled models are compiled using Ara SDK.
* Output model and assets will be stored in the `modelzoo/classification/vit_base_patch16_224/output/assets` folder

### Manual Model Preparation and Compile

#### Environment Variables
```bash
export ARA_SDK_ROOT=<Path to SDK>
```

#### Model Compile

```bash
./flows/model_compile.sh model=vit_base_patch16_224
```
* Output model and assets will be stored in the `modelzoo/classification/vit_base_patch16_224/output/assets` folder

## Model Evaluation flows
Note that for the following flows the latest model that has been compiled or downloaded will be used.


### Ara Hardware
These flows run on Ara hardware which needs to be procured separately from our SDK. Please refer SDK documentation on setting up Ara hardware. Copy or use git clone to get Ara modelzoo repository on the host. Copy Ara SDK to host as well.

#### Environment variables
This sets environment variables required to run the tools

```bash
# Download Ara SDK and Copy to Host
export DV_TGT_ROOT=<Path to SDK>
```

#### Performance checker

```bash
./flows/performance.sh model=vit_base_patch16_224
```

Look for `HW IPS` for ips in output.

#### Inference Application
Runs inference application on images provided to display inference results, if no images are provided a default test image will be used

Floating-point inference:
```bash
./flows/infer_float.sh model=vit_base_patch16_224 images_folder=<path to images to run inference on>
```
Hardware inference:
```bash
./flows/infer_hw.sh model=vit_base_patch16_224 images_folder=<path to images to run inference on>
```
* Output images saved in postprocessed_images_viz_default folder


#### Accuracy checker
* Note: Accuracy check can take several minutes to hours depending on dataset size and host machine. In order to run accuracy on complete dataset for this model, please download from [ImageNet](https://image-net.org/download.php) or refer [link](#dataset-used-for-checking-accuracy)

Floating-point evaluation:
```bash
./flows/eval_float.sh model=vit_base_patch16_224 dataset_root=<path to imagenet> limit=<num images>
```
Hardware evaluation:
```bash
./flows/eval_hw.sh model=vit_base_patch16_224 dataset_root=<path for eval dataset on hardware>
```
