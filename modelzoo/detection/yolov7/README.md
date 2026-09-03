# YOLOv7

## Use Case and High-Level Description

YOLOv7 is an object detection model and is a part of YOLO(You Only Look Once) family.

The model input is a blob that consists of a single image of `1, 3, 640, 640`.

## Model source

The github source of the present model is
[Yolov7-onnx](https://github.com/WongKinYiu/yolov7)

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection|

## Accuracy

The FP32 Value is taken from the github source of the present model [Yolov7-onnx](https://github.com/WongKinYiu/yolov7)

| Metric | FP32   | Int8   |
| ------ | ------ | ------ |
| mAP | 51.4| 48.41|

**Note**:
Accuracy metrics reflect IOU threshold of 0.65 and NMS threshold of 0.001.


## Dataset used for checking accuracy

Coco datasets download [page](https://cocodataset.org/#download)

Download validation images [val2017](http://images.cocodataset.org/zips/val2017.zip)

Download tags [instances_val2017.json](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

## Model Parameters

#### Pre-Process Stage

| Parameter | Value    |
| :-------- | :------- |
| `Model input shape` | `[3,640,640]` |
| `Model output shape` | `[1, 3, 80, 80, 85] [1, 3, 40, 40, 85] [1, 3, 20, 20, 85]` |
| `Interpolation` | `LINEAR Interpolation` |
| `Padding value` | `114 (pad maintaining the aspect ratio)` |
| `Mean`      | `{0, 0, 0}` |
| `Scale`      | `{0.0039, 0.0039, 0.0039}` |


#### Post-Process Stage

| Parameter | Value    |
| :-------- | :------- |
| `IOU Threshold`      | `0.65` |
| `NMS Score Threshold`      | `0.35` |
| `label`      | `COCO 80 classes` |


## Setup
- Ara SDK `ara2-sdk-r3.0` is required.
- Follow [these](../../README.md#License) steps to update config file with license_key.

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

### Pre-compiled model download
Ara-model-zoo included precompiled models to run on hardware. These steps can be used to run the precompiled model to check accuracy and run inference.

- Download pre-compiled model
```bash
cd ara-model-zoo
./flows/precompiled_model_download.sh model=yolov7
```
- Precompiled models are compiled using Ara SDK.
- Output model and assets will be stored in the `modelzoo/detection/yolov7/output/assets` folder

#### Model Compile

```bash
cd ara-model-zoo
./flows/model_compile.sh model=yolov7 dataset_root=<path to val2017>
```
- Output model and assets will be stored in the `modelzoo/detection/yolov7/output/assets` folder

## Model Evaluation flows
Note that for the following flows the latest model that has been compiled or downloaded will be used.

#### Performance checker

```bash
cd ara-model-zoo
./flows/performance.sh model=yolov7
```

Look for `HW IPS` for ips in output.

#### Inference Application
Runs inference application on images provided to display inference results, if no images are provided a default test image will be used

_Python execution_
```bash
cd ara-model-zoo
./flows/infer_float.sh model=yolov7 images_folder=<path to images to run inference on>
./flows/infer_hw.sh model=yolov7 images_folder=<path to images to run inference on>
```

**Example:**

_Python example_
```bash
cd ara-model-zoo
./flows/infer_float.sh images_folder=../testimages model=yolov7
./flows/infer_hw.sh images_folder=../testimages model=yolov7
```

- Output images saved in `modelzoo/detection/yolov7/output/postprocessed_output_visualized`


#### Accuracy checker
- Note: Accuracy check can take several minutes to hours depending on dataset size and host machine. In order to run accuracy on complete dataset for this model, please download from [Coco](https://cocodataset.org/#download)

- For the Accuracy metrics please modify the nms_score_threshold in run.yaml to 0.001

*Run with Python*
```bash
cd ara-model-zoo
./flows/eval_float.sh model=yolov7 dataset_root=<folder containing image dataset>
./flows/eval_hw.sh model=yolov7 dataset_root=<folder containing image dataset>
```


**Example:**

*Python example*
```bash
cd ara-model-zoo
./flows/eval_hw.sh model=yolov7 dataset_root=/media/data/datasets/coco2017
```

- **Note** Only the dataset_root argument is required for COCO evaluation

- Accuracy metrics printed to terminal
