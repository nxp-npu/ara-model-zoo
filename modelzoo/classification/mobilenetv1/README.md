# MobileNet v1(TensorFlow)

## Use Case and High-Level Description

MobileNet v1 is an image classification model which has been pre-trained on the ImageNet dataset. This is the tensorflow implementation of this model.

The model input is a blob that consists of a single image of `1, 3, 224, 224`.

The model output is typical object classifier for the 1000 different classifications
matching with those in the ImageNet database.

## Model source

The model is taken from [this table](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md) for `MobileNet_v1_1.0_224`

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| Source framework  | Tensorflow  |

## Accuracy

| Metric | FP32 | Int8 |
| ------ | ------   | ------  |
| Top 1  |  71.004  |  69.286  |

## Dataset used for checking accuracy

Original Dataset [ImageNet](https://image-net.org/download.php)

|      |     |
| ------ | ------ |
| Dataset Name | imagenet-1k |
| Classes  | 1001 |
| Validation  images | 50,000 Used to Calculate Accuracy Numbers |
| Image format | JPEG |

[Click for more info on imagenet-1k dataset](https://huggingface.co/datasets/imagenet-1k), You can download a subset of complete dataset from here by a quick sign-up. Official website for Complete dataset is [ImageNet](https://image-net.org/download.php), requires approval.


## Model Parameters

#### Pre-Process Stage
| Parameter | Value    |
| :-------- | :------- |
| `Model input shape` | `[1 3 224 224]` |
| `Model output shape` | `[1 1001 ]`|
| `Resize` | `224` |
| `Interpolation` | `INTER-LINEAR` |
| `Mean`      | `{127.5, 127.5, 127.5}` |
| `Scale`      | `{0.007843137, 0.007843137, 0.007843137}` |
| `center crop fraction`| `{0.875}` |


#### Post-Process Stage
| Parameter | Value    |
| :-------- | :------- |
|  `classifier_activation` |`softmax`|
|  `classes`|   `1001`|


## Setup

- Ara SDK `ara2-sdk-r3.0` is required.

### Python

Install python requirements

```bash
cd ara-model-zoo
pip3 install -e .
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

#### Pre-compiled model download

Ara-model-zoo included precompiled models. These steps can be used to run the precompiled model to check accuracy and run inference.

- Download pre-compiled model

```bash
./flows/precompiled_model_download.sh model=mobilenetv1
```

- Precompiled models are compiled using Ara SDK.
- Output model and assets will be stored in the `modelzoo/classification/mobilenetv1/output/assets` and `modelzoo/classification/mobilenetv1/output/compiled_model` folder

#### Model Compile

```bash
./flows/model_compile.sh model=mobilenetv1 dataset_root=<folder containing calibration image dataset>
```
* Output model and assets will be stored in the `modelzoo/classification/mobilenetv1/output/assets` folder


#### Performance checker

```bash
./flows/performance.sh model=mobilenetv1
```

Look for `HW IPS` for ips in output.

#### Inference Application
Runs inference application on images provided to display inference results, if no images are provided a default test image will be used

##### FP32 Inference

```bash
./flows/infer_float.sh model=mobilenetv1 images_folder=<path to images to run inference on>
```

##### HW Inference

```bash
./flows/infer_hw.sh model=mobilenetv1 images_folder=<path to images to run inference on>
```

* Output images saved in postprocessed_images_viz_default folder


#### Accuracy checker
* Note: Accuracy check can take several minutes to hours depending on dataset size and host machine. In order to run accuracy on complete dataset for this model, please download from [ImageNet](https://image-net.org/download.php) or refer [link](#dataset-used-for-checking-accuracy)

##### FP32 Evaluation

```bash
./flows/eval_float.sh model=mobilenetv1 dataset_root=<folder containing image dataset>
```

##### HW Evaluation

```bash
./flows/eval_hw.sh model=mobilenetv1 dataset_root=<folder containing image dataset>
```

* Accuracy metrics printed to terminal
