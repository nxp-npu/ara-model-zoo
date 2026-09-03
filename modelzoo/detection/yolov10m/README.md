# YOLOv10m

## Use Case and High-Level Description

YOLOv10m is an object detection model developed by Ultralytics and a part of the YOLO (You Only Look Once) family. The model is pretrained on the COCO 2017 dataset.

YOLOv10m is the medium-sized variant of the YOLOv10 family. It offers higher accuracy than the smaller nano and small variants while remaining reasonably efficient. It is suited for edge devices or systems with moderate compute capacity where detection quality is prioritized alongside acceptable inference speed.

The model input is a blob that consists of a single image of `1, 3, 640, 640`.

## Model Source

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
| mAP    | 51.1 | 49.68|

Accuracy metrics reflect IOU threshold of 0.7 and NMS threshold of 0.001.

## Dataset Used for Checking Accuracy

COCO dataset download [page](https://cocodataset.org/#download)

Download validation images [val2017](http://images.cocodataset.org/zips/val2017.zip)

Download annotations [instances_val2017.json](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

## Model Parameters

### Pre-Process Stage

| Parameter | Value |
| :-------- | :---- |
| `Model input shape` | `[3, 640, 640]` |
| `Model output shape` | `[1, 4, 8400]` `[1, 80, 8400]` |
| `Interpolation` | `Linear Interpolation` |
| `Padding value` | `114 (pad maintaining the aspect ratio)` |
| `Mean` | `{0, 0, 0}` |
| `Scale` | `{0.0039, 0.0039, 0.0039}` |

### Post-Process Stage

| Parameter | Value |
| :-------- | :---- |
| `IOU Threshold` | `0.70` |
| `NMS Score Threshold` | `0.25` |
| `Labels` | `COCO 80 classes` |


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
./flows/precompiled_model_download.sh model=yolov10m
```

- Precompiled models are compiled using Ara SDK.
- Output model and assets will be stored in the `modelzoo/detection/yolov10m/output/compiled_model` folder

### Model Compile

```bash
cd ara-model-zoo
./flows/model_compile.sh model=yolov10m dataset_root=<path to val2017>
```

Output model is stored in `modelzoo/detection/yolov10m/output/compiled_model/`

#### Performance Checker

```bash
cd ara-model-zoo
./flows/performance.sh model=yolov10m
```

Look for `HW IPS` in the output for throughput.

#### Inference Application

```bash
cd ara-model-zoo
./flows/infer_hw.sh model=yolov10m images_folder=./testimages
./flows/infer_float.sh model=yolov10m images_folder=./testimages
```

Output images are saved in the `output/postprocessed_images_viz_default` folder.

#### Accuracy Checker

```bash
cd ara-model-zoo
./flows/eval_hw.sh model=yolov10m dataset_root=/media/data/datasets/coco2017
./flows/eval_float.sh model=yolov10m dataset_root=<path to val2017>
```
