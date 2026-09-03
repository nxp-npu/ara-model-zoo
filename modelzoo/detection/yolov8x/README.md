# YOLOv8x

## Use Case and High-Level Description

YOLOv8x is an object detection model developed by ultralytics and a part of YOLO(You Only Look Once) family.  The model is pretrained with Coco2017 dataset

YOLOv8x is the largest model in the YOLOv8 family. It's designed for environments where computational resources are limited, such as on mobile devices or edge devices. While it may not offer the same level of accuracy as the larger models, it's faster and requires less computational resources, making it a good choice for real-time object detection tasks in resource-constrained environments.

The model input is a blob that consists of a single image of `1, 3, 640, 640`.

The model outputs are two blobs with the shape of `[1, 4, 8400], [1, 80, 8400]`. The first one are the bounding box coordinates, and the second one are the class probabilities of COCO dataset.

## Model source

Model is being saved from ultralytics library.
Ultralytics repository can be found [here](https://github.com/ultralytics/ultralytics)

## Specification

| Metric           | Value          |
| ---------------- | -------------- |
| Type             | Detection      |
| Source framework | ONNX\*         |

## Accuracy

The FP32 Accuracy value is taken from [here](https://github.com/ultralytics/ultralytics#models)

| Metric       | FP32  | Int8  |
| ------------ | ----- | ----- |
| mAP          | 53.9% | 49.98% |

Accuracy metrics reflect IOU threshold of 0.7 and NMS threshold of 0.001, during evaluation.

## Dataset used for checking accuracy

Coco datasets download [page](https://cocodataset.org/#download)

Download validation images [val2017](http://images.cocodataset.org/zips/val2017.zip)

Download tags [instances_val2017.json](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

## Model Parameters

#### Pre-Process Stage

| Parameter            | Value                                                 |
| :------------------- | :---------------------------------------------------- |
| `Model input shape`  | `[3,640,640]`                                         |
| `Model output shape` | `[1, 4, 8400], [1, 80, 8400]`                         |
| `Interpolation`      | `Linear Interpolation`                                |
| `Padding value`      | `114 (pad maintaining the aspect ratio)`              |
| `Mean`               | `{0, 0, 0}`                                           |
| `Scale`              | `{0.0039, 0.0039, 0.0039}`                            |

#### Post-Process Stage

| Parameter             | Value   |
| :-------------------- | :------ |
| `IOU Threshold`       | `0.70`  |
| `NMS Score Threshold` | `0.25`  |
| `label`               | `COCO 80 Classes` |

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
export DV_TGT_ROOT=<Path to SDK>
```

### Pre-compiled model download

Ara modelzoo included precompiled models to run on hardware. These steps can be used to run the precompiled model to check accuracy and run inference.

- Download pre-compiled model

```bash
cd ara-model-zoo
./flows/precompiled_model_download.sh model=yolov8x
```

- Precompiled models are compiled using Ara SDK.
- Output model and assets will be stored in the `modelzoo/face_detection/yolov8x/output/compiled_model` folder

#### Performance checker

```bash
./flows/performance.sh model=yolov8m
```
Look for `HW IPS` for ips in output.

### Manual Model Preparation and Compile

#### Model Compile

```bash
cd ara-model-zoo
./flows/model_compile.sh model=yolov8x dataset_root=<path to val2017>
```

- Output model and assets will be stored in the `modelzoo/face_detection/yolov8x/output/compiled_model` folder

#### Inference Application

Runs inference application on images provided to display inference results, if no images are provided a default test image will be used

```bash
cd ara-model-zoo
./flows/infer_float.sh images_folder=../images model=yolov8x
./flows/infer_hw.sh images_folder=../images model=yolov8x
```

- Output images saved in `modelzoo/face_detection/yolov8x/output/postprocessed_output_visualized`

#### Accuracy checker

- Note: Accuracy check can take several minutes to hours depending on dataset size and host machine. In order to run accuracy on complete dataset for this model, please download dataset from [COCO 2017](https://cocodataset.org/#download) and ground truth tags from [Ground Truth](http://images.cocodataset.org/annotations/annotations_trainval2017.zip).

_Run with Python_

```bash
cd ara-model-zoo
./flows/run_eval_float.sh dataset_root=<folder containing image dataset> model=yolov8x
./flows/run_eval_hw.sh dataset_root=<folder containing image dataset> model=yolov8x
```

- **Note** Only the dataset folder argument is required for COCO 2017 evaluation, the tags/ground truth should be contained inside the dataset_folder

- Accuracy metrics printed to terminal
