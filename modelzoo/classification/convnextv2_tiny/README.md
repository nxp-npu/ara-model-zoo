# ConvNeXt V2 Tiny (onnx)

## Use Case and High-Level Description

ConvNeXt V2 Tiny is image classification model pre-trained on ImageNet dataset. This
is PyTorch\* implementation based on architecture described in paper ["ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders"](https://arxiv.org/pdf/2301.00808) in timm
package (see [here](https://huggingface.co/docs/timm/en/index)).

The model input is a blob that consists of a single image of `1, 3, 224, 224`.

The model output is typical object classifier for the 1000 different classifications
matching with those in the ImageNet database.

## Model source

This model is being taken directly from [timm](https://huggingface.co/docs/timm/en/index) library.

## Specification


| Model Type        | Metric        |
|-------------------|---------------|
| Classification    | Accuracy      |      |

## Accuracy

The FP32 Value is taken from Pytorch Documentation for [convnextv2_tiny](https://huggingface.co/timm/convnextv2_tiny.fcmae)

| Metric | FP32   | Int8  |
| ------ | ------ | ------ |
| Top 1  | 83.9 %  | 81.7 %



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
| `Interpolation` | `BICUBIC` |
| `Center Crop`      | `224` |
| `Mean`      | `{0.485, 0.456, 0.406}` |
| `Std`      | `{0.229, 0.224, 0.225}` |


#### Post-Process Stage

| Parameter | Value    |
| :-------- | :------- |
|  `classifier_activation` |`softmax`|
|  `classes`|   `1000`|


## Setup
* Ara SDK `ara2-sdk-r3.0` is required.

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

#### Model Compile

```bash
./flows/model_compile.sh model=convnextv2_tiny dataset_root=<path to imagenet directory>

```
* Output model and assets will be stored in the `modelzoo/classification/convnextv2_tiny/output/compiled_model` folder


#### Performance checker

```bash
./flows/performance.sh model=convnextv2_tiny
```

Look for `HW IPS` for ips in output.

#### Inference Application
Runs inference application on images provided to display inference results, if no images are provided a default test image will be used

Floating-point inference:
```bash
./flows/infer_float.sh model=convnextv2_tiny images_folder=<path to images to run inference on>
```
Hardware inference:
```bash
./flows/infer_hw.sh model=convnextv2_tiny images_folder=<path to images to run inference on>
```

* Output images saved in postprocessed_images_viz_default folder


#### Accuracy checker
* Note: Accuracy check can take several minutes to hours depending on dataset size and host machine. In order to run accuracy on complete dataset for this model, please download from [ImageNet](https://image-net.org/download.php) or refer [link](#dataset-used-for-checking-accuracy)

Floating-point evaluation:
```bash
./flows/eval_float.sh model=tiny_vit dataset_root=<path to imagenet> limit=<num images>
```
Hardware evaluation:
```bash
./flows/eval_hw.sh model=tiny_vit dataset_root=<path for eval dataset on hardware>
```

* Accuracy metrics printed to terminal
