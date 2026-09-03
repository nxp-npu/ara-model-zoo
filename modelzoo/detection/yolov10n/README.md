# YOLOv10n

## Use Case and High-Level Description

YOLOv10n is an object detection model developed by Ultralytics and a part of the YOLO (You Only Look Once) family. The model is pretrained on the COCO 2017 dataset.

YOLOv10n is the smallest model in the YOLOv10 family. It is designed for environments where computational resources are limited, such as on mobile devices or edge devices. While it may not offer the same level of accuracy as the larger models, it is faster and requires less computational resources, making it a good choice for real-time object detection tasks in resource-constrained environments.

The model input is a blob that consists of a single image of `1, 3, 640, 640`.


## Model source

The model repository can be found [here](https://github.com/ultralytics/ultralytics)

## Specification

| Metric            | Value        |
|-------------------|--------------|
| Type              | Detection    |
| Source framework  | ONNX\*       |

## Accuracy

The FP32 accuracy value is taken from [here](https://docs.ultralytics.com/models/yolov10#performance)

| Metric | FP32 | Int8 |
| ------ | ---- | ---- |
| mAP    | 38.5 | 37.01|

Accuracy metrics reflect IOU threshold of 0.7 and NMS threshold of 0.001.


## Data set used for training and checking accuracy

COCO dataset download [page](https://cocodataset.org/#download)

Download validation images [val2017](http://images.cocodataset.org/zips/val2017.zip)

Download annotations [instances_val2017.json](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

## Model Parameters

#### Pre-Process Stage
| Parameter | Value |
| :-------- | :---- |
| `Model input shape` | `[3, 640, 640]` |
| `Model output shape` | `[1, 4, 8400]` `[1, 80, 8400]` |
| `Interpolation` | `Linear Interpolation` |
| `Padding value` | `114 (pad maintaining the aspect ratio)` |
| `Mean` | `{0, 0, 0}` |
| `Scale` | `{0.0039, 0.0039, 0.0039}` |

#### Post-Process Stage
| Parameter | Value |
| :-------- | :---- |
| `IOU Threshold` | `0.70` |
| `NMS Score Threshold` | `0.25` |
| `Labels` | `COCO 80 classes` |


## Setup

- Ara SDK `ara2-sdk-r3.0` is required.

### Python

Install python requirements

```bash
cd ara-model-zoo
pip3 install -e .
```

## Steps to prepare model

Ara Ara-model-zoo offers two methods to generate the model file:

- Pre-compiled model download, where a pre-compiled model can be downloaded from the Ara repository.
- Manually compiled model, where the model is imported in its original framework and then compiled using Ara SDK.

Usage of Ara-model-zoo requires access and download of Ara SDK. If you do not have Ara SDK, please contact our [support](https://support.nxp.com/). For details on host requirements, please refer to SDK Documentation.

### Environment Variables

```bash
export ARA_SDK_ROOT=<Path to SDK>
export DV_TGT_ROOT=<Path to SDK>
```

### Pre-compiled model download

Ara-model-zoo includes precompiled models to run on hardware. These steps can be used to run the precompiled model to check accuracy and run inference.

- Download pre-compiled model

```bash
cd ara-model-zoo
./flows/precompiled_model_download.sh model=yolov10n
```

- Precompiled models are compiled using Ara SDK.
- Output model and assets will be stored in the `modelzoo/detection/yolov10n/output/assets` folder

### Manual Model Preparation and Compile

#### Environment Variables

```bash
export ARA_SDK_ROOT=<Path to SDK>
export DV_TGT_ROOT=<Path to SDK>
```

#### Model Compile

```bash
cd ara-model-zoo
./flows/model_compile.sh model=yolov10n dataset_root=<path to val2017>
```

- Output model and assets will be stored in the `modelzoo/detection/yolov10n/output/compiled_model` folder

## Model Evaluation flows

Note that for the following flows the latest model that has been compiled or downloaded will be used.

### Ara Hardware

These flows run on Ara hardware which needs to be procured separately from our SDK. Please refer SDK documentation on setting up Ara hardware. Copy or use git clone to get Ara-model-zoo repository on the host. Copy Ara SDK to host as well.



#### Inference Application

Runs inference application on images provided to display inference results, if no images are provided a default test image will be used

_Python execution_

```bash
cd ara-model-zoo
./flows/infer_float.sh images_folder=<path_to_images_folder> model=yolov10n
./flows/infer_hw.sh images_folder=<path_to_images_folder> model=yolov10n
```

**Example:**

_Python example_

```bash
cd ara-model-zoo
./flows/infer_float.sh images_folder=./testimages model=yolov10n
./flows/infer_hw.sh images_folder=./testimages model=yolov10n
```

- Output images saved in `modelzoo/detection/yolov10n/output/postprocessed_output_visualized`

#### Accuracy checker



_Run with Python_

```bash
cd ara-model-zoo
./flows/eval_float.sh model=yolov10n dataset_root=<path to val2017>
./flows/eval_hw.sh model=yolov10n dataset_root=<path to val2017>
```

**Example:**

_Python example_

```bash
cd ara-model-zoo
./flows/eval_float.sh model=yolov10n dataset_root=/media/data/datasets/coco2017
./flows/eval_hw.sh model=yolov10n dataset_root=/media/data/datasets/coco2017
```

- Accuracy metrics printed to terminal

## Legal Information
The original model is distributed under GNU GPLv3.
Link: https://github.com/ultralytics/ultralytics/blob/main/LICENSE
