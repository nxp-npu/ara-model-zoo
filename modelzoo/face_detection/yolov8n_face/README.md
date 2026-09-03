# YOLOv8n-Face

## Use Case and High-Level Description

YOLOv8n-Face is a face detection model that was developed by modifying Yolov8's architecture. This is a modified version of the Yolov8n model which detects faces along with landmarks. The model is pretrained with WiderFace dataset

YOLOv8n is the smallest model in the YOLOv8 family. It's designed for environments where computational resources are limited, such as on mobile devices or edge devices. While it may not offer the same level of accuracy as the larger models, it's faster and requires less computational resources, making it a good choice for real-time object detection tasks in resource-constrained environments.

The model input is a blob that consists of a single image of `1, 3, 640, 640`.

The model output is a typical face detector for detecting faces in images, with bounding boxes, confidence scores, and five facial landmark keypoints corresponding to the eyes, nose, and mouth corners. The model is trained on the WIDERFace dataset.

## Model source

The model repository can be found [here](https://github.com/ultralytics/ultralytics)

## Specification

| Metric           | Value          |
| ---------------- | -------------- |
| Type             | Face Detection |
| Source framework | PyTorch\*      |

## Accuracy

The FP32 Accuracy value is taken from [here](https://github.com/derronqi/yolov8-face)

| Metric       | FP32  | Int8  |
| ------------ | ----- | ----- |
| mAP - Easy   | 94.6% | 93.6% |
| mAP - Medium | 92.4% | 91.1% |
| mAP - Hard   | 78.4% | 74.9% |

Accuracy metrics reflect IOU threshold of 0.7, top_k 500, and and EVALUATION_NMS_SCORE_THRESHOLD = 0.001.

## Data set used for checking accuracy

Original Dataset [WIDERFace](http://shuoyang1213.me/WIDERFACE/)

|                   |                                           |
| ----------------- | ----------------------------------------- |
| Dataset Name      | WIDERFace                                 |
| Classes           | 1 (Face)                                  |
| Validation images | 3,226                                     |
| Image format      | JPEG                                      |

[Click for more info on WIDERFace dataset](http://shuoyang1213.me/WIDERFACE/)

WIDERFace is a face detection benchmark dataset containing 32,203 images and 393,703 labeled faces, with significant variation in scale, pose, and occlusion. The dataset is divided into training, validation, and testing subsets using a 40%/10%/50% split.

The model is trained for face detection and predicts a bounding box, confidence score, and five facial landmark keypoints for each detected face. The five keypoints correspond to the left eye, right eye, nose, left mouth corner, and right mouth corner.

Tags: [Ground Truth](https://github.com/biubug6/Pytorch_Retinaface/tree/master/widerface_evaluate/ground_truth) (Link: https://github.com/biubug6/Pytorch_Retinaface/tree/master/widerface_evaluate/ground_truth)

## Model Parameters

#### Pre-Process Stage

| Parameter            | Value                                                 |
| :------------------- | :---------------------------------------------------- |
| `Model input shape`  | `[3,640,640]`                                         |
| `Model output shape` | `[1, 4, 8400], [1, 8400], [5, 2, 8400], [5, 1, 8400]` |
| `Interpolation`      | `Linear Interpolation`                                |
| `Padding value`      | `114 (pad maintaining the aspect ratio)`              |
| `Mean`               | `{0, 0, 0}`                                           |
| `Scale`              | `{0.0039, 0.0039, 0.0039}`                            |

#### Post-Process Stage

| Parameter             | Value   |
| :-------------------- | :------ |
| `IOU Threshold`       | `0.70`  |
| `NMS Score Threshold` | `0.25`  |
| `EVALUATION_NMS_SCORE_THRESHOLD` | `0.001`  |
| `label`               | `Faces` |

## Setup
* Ara SDK `ara2-sdk-r3.0` is required.
* Follow [these](../../README.md#License) steps to update config file with license_key.

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

#### Pre-compiled model download

Ara-model-zoo included precompiled models. These steps can be used to run the precompiled model to check accuracy and run inference.

- Download pre-compiled model

```bash
./flows/precompiled_model_download.sh model=yolov8n_face
```

- Precompiled models are compiled using Ara SDK.
- Output model and assets will be stored in the `modelzoo/face_detection/yolov8n_face/output/assets` and `modelzoo/face_detection/yolov8n_face/output/compiled_model` folder

#### Model Compile

```bash
./flows/model_compile.sh model=yolov8n_face dataset_root=<folder containing calibration image dataset>
```
* Output model and assets will be stored in the `modelzoo/face_detection/yolov8n_face/output/assets` folder


#### Performance checker

```bash
./flows/performance.sh model=yolov8n_face
```

Look for `HW IPS` for ips in output.

#### Inference Application
Runs inference application on images provided to display inference results, if no images are provided a default test image will be used

##### FP32 Inference

```bash
./flows/infer_float.sh model=yolov8n_face images_folder=<path to images to run inference on>
```

##### HW Inference

```bash
./flows/infer_hw.sh model=yolov8n_face images_folder=<path to images to run inference on>
```

* Output images saved in postprocessed_images_viz_default folder


#### Accuracy checker
* Note: Accuracy check can take several minutes to hours depending on dataset size and host machine. In order to run accuracy on complete dataset for this model, please refer to [WiderFace Dataset Page](http://shuoyang1213.me/WIDERFACE/) and for [Ground Truths](https://github.com/biubug6/Pytorch_Retinaface/tree/master/widerface_evaluate/ground_truth)

##### FP32 Evaluation

```bash
./flows/eval_float.sh model=yolov8n_face dataset_root=<folder containing image dataset>
```

##### HW Evaluation

```bash
./flows/eval_hw.sh model=yolov8n_face dataset_root=<folder containing image dataset>
```

* Accuracy metrics printed to terminal
