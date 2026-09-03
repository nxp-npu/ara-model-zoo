# YOLOv11s
## Use Case and High-Level Description

YOLOv11s is an object detection model developed by ultralytics and a part of YOLO(You Only Look Once) family.  The model is pretrained with Coco2017 dataset

YOLOv11s is the small variant in the YOLOv11 family, sitting just above the nano model in size. It strikes a better balance between accuracy and speed than YOLOv11n, making it a good choice when you need noticeably better detection quality but still want to run on relatively modest hardware. It's well suited for applications like lightweight edge devices, drones, or embedded systems where some additional compute is available but resources are still constrained.

The model input is a blob that consists of a single image of `1, 3, 640, 640`.

The model output is a typical object detector, predicting bounding boxes, class labels, and confidence scores for 80 object categories matching those in the COCO dataset.


## Model source

Model is being saved from ultralytics library.
Ultralytics repository can be found [here](https://github.com/ultralytics/ultralytics)

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection|
| Source framework  | Onnx\*     |

## Accuracy

The FP32 Accuracy value is taken from [here](https://github.com/ultralytics/ultralytics#models)

| Metric | FP32  | Int8  |
| ------ | ----- | -------- |
| mAP    | 47% | 45.16%  |

Accuracy metrics reflect IOU threshold of 0.7 and NMS threshold of 0.001.


## Data set used for training and checking accuracy

Dataset Folder: [COCO 2017](https://cocodataset.org/#download)

GroundTruth: [instances_val2017.json](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

## Model Parameters

#### Pre-Process Stage
| Parameter | Value    |
| :-------- | :------- |
| `Model input shape` | `[3,640,640]` |
| `Model output shape` | `[1, 4, 8400], [1, 80, 8400]` |
| `Interpolation` | `Linear Interpolation` |
| `Padding value` | `114 (pad maintaining the aspect ratio)` |
| `Mean`      | `{0, 0, 0}` |
| `Scale`      | `{0.0039, 0.0039, 0.0039}` |

#### Post-Process Stage
| Parameter | Value    |
| :-------- | :------- |
| `IOU Threshold`      | `0.70` |
| `NMS Score Threshold`      | `0.25` |
| `label`      | `COCO 80 classes` |

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

Ara-model-zoo included precompiled models to run on hardware. These steps can be used to run the precompiled model to check accuracy and run inference.

- Download pre-compiled model

```bash
cd ara-model-zoo
./flows/precompiled_model_download.sh model=yolov11s
```

- Precompiled models are compiled using Ara SDK.
- Output model and assets will be stored in the `modelzoo/detection/yolov11s/output/assets` folder

#### Environment Variables

```bash
export ARA_SDK_ROOT=<Path to SDK>
export DV_TGT_ROOT=<Path to SDK>
```

#### Model Compile

```bash
cd ara-model-zoo
./flows/model_compile.sh model=yolov11s dataset_root=<path to val2017>
```

- Output model and assets will be stored in the `modelzoo/detection/yolov11s/output/compiled_model` folder


## Model Evaluation flows

Note that for the following flows the latest model that has been compiled or downloaded will be used.

#### Inference Application

Runs inference application on images provided to display inference results, if no images are provided a default test image will be used


```bash
cd ara-model-zoo
./flows/infer_float.sh images_folder=../images model=yolov11s
./flows/infer_hw.sh images_folder=../images model=yolov11s
```

- Output images saved in `modelzoo/detection/yolov11s/output/postprocessed_output_visualized`


#### Accuracy checker

- Note: Accuracy check can take several minutes to hours depending on dataset size and host machine. In order to run accuracy on complete dataset for this model, please download dataset from [COCO 2017](https://cocodataset.org/#download) and ground truth tags from [Ground Truth](http://images.cocodataset.org/annotations/annotations_trainval2017.zip).


```bash
cd ara-model-zoo
./flows/eval_float.sh dataset_root='../coco2017/' model=yolov11s
./flows/eval_hw.sh dataset_root='../coco2017/' model=yolov11s
```

- **Note** Only the dataset root argument is required for COCO 2017 evaluation, the tags/ground truth should be contained inside the dataset_root

- Accuracy metrics printed to terminal
