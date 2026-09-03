# YOLOv8x Pose
## Use Case and High-Level Description

YOLOv8x-pose is a human pose estimation model from the YOLOv8 family developed by Ultralytics. The model is pretrained on the COCO 2017 keypoints dataset and predicts person bounding boxes with 17 body keypoints.

YOLOv8x-pose is the extra-large variant. It targets the highest accuracy in the YOLOv8 pose family and is best suited for deployments where accuracy is prioritized over model size and compile time.

The model input is a blob that consists of a single image of `1, 3, 640, 640`.

## Model Source

The model is exported from the Ultralytics YOLOv8 pose weights.
The Ultralytics repository can be found [here](https://github.com/ultralytics/ultralytics).

## Specification

| Metric | Value |
| --- | --- |
| Type | Pose Estimation |
| Source framework | ONNX |

## Accuracy

The FP32 accuracy value is taken from the [Ultralytics YOLOv8 pose model table](https://github.com/ultralytics/ultralytics#models). The INT8 value is from Ara hardware evaluation on the COCO keypoints validation subset used by this modelzoo.

| Metric | FP32 | INT8 |
| --- | --- | --- |
| mAP@[IoU=0.50:0.95] | 69.9% | 64.80% |

Accuracy metrics use COCO keypoints AP. During evaluation, the postprocess confidence threshold is overridden to `0.001`; the NMS IOU threshold is `0.70`.

## Data Set Used for Training and Checking Accuracy

Dataset folder: [COCO 2017](https://cocodataset.org/#download)

Validation images: [val2017](http://images.cocodataset.org/zips/val2017.zip)

Ground truth: [person_keypoints_val2017.json](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

This repository filters COCO keypoints validation to the supported 2346-image pose evaluation subset automatically.

## Model Parameters

#### Pre-Process Stage

| Parameter | Value |
| :--- | :--- |
| `Model input shape` | `[3, 640, 640]` |
| `Model output shape` | `[1, 4, 8400], [1, 1, 8400], [1, 17, 2, 8400], [1, 17, 1, 8400]` |
| `Interpolation` | `Linear Interpolation` |
| `Padding value` | `114 (pad maintaining the aspect ratio)` |
| `Mean` | `{0, 0, 0}` |
| `Scale` | `{0.0039, 0.0039, 0.0039}` |

#### Post-Process Stage

| Parameter | Value |
| :--- | :--- |
| `IOU Threshold` | `0.70` |
| `NMS Score Threshold` | `0.25` for inference, `0.001` during evaluation |
| `Keypoints` | `17 COCO person keypoints` |

## Setup

- Ara SDK `ara2-sdk-r2.0` is required.
- Python >= 3.12 is required.

### Python

Install Python requirements from the repository root:

```bash
cd ara-model-zoo
pip install -e .
```

## Steps to Prepare Model

Ara Modelzoo offers two methods to generate the model file:

- Pre-compiled model download, where a pre-compiled model is downloaded from the Ara repository.
- Manual model compile, where the model is imported in its original framework and compiled using the Ara SDK.

Usage of Ara modelzoo requires access to the Ara SDK. If you do not have the Ara SDK, please contact [Ara support](https://support.nxp.com/).

### Environment Variables

```bash
export ARA_SDK_ROOT=<Path to SDK>
export DV_TGT_ROOT=<Path to SDK runtime>
```

### Pre-Compiled Model Download

```bash
cd ara-model-zoo
./flows/precompiled_model_download.sh model=yolov8x_pose
```

- Pre-compiled models are compiled using the Ara SDK.
- Output model and assets will be stored in `modelzoo/pose_estimation/yolov8x_pose/output/assets`.
- A copy of `model.dvm` will be stored in `modelzoo/pose_estimation/yolov8x_pose/output/compiled_model`.

### Manual Model Preparation and Compile

```bash
cd ara-model-zoo
./flows/model_compile.sh model=yolov8x_pose dataset_root=<path_to_coco_keypoints_data> run=python
```

- Output model and assets will be stored in `modelzoo/pose_estimation/yolov8x_pose/output/compiled_model`.

## Model Evaluation Flows

The following flows use the latest model that has been compiled or downloaded.

### Ara Hardware

These flows run on Ara hardware, which must be set up separately from the SDK. Please refer to the SDK documentation for hardware setup details.

#### Performance Checker

```bash
cd ara-model-zoo
./flows/performance.sh model=yolov8x_pose
```

Look for `HW IPS` in the output.

#### Inference Application

Runs inference on images and saves postprocessed outputs.

```bash
cd ara-model-zoo
./flows/infer_float.sh images_folder=<path_to_images_folder> model=yolov8x_pose
./flows/infer_hw.sh images_folder=<path_to_images_folder> model=yolov8x_pose
```

- Output images are saved under `modelzoo/pose_estimation/yolov8x_pose/output/postprocessed_output_visualized`.

#### Accuracy Checker

Accuracy check can take several minutes depending on dataset size and host machine. Download COCO validation images and person keypoints annotations before running evaluation.

```bash
cd ara-model-zoo
./flows/eval_hw.sh model=yolov8x_pose dataset_root=<path_to_coco_keypoints_data>
```

- `dataset_root` should contain `val2017` images and `person_keypoints_val2017.json`.
- Accuracy metrics are printed to the terminal and written under `modelzoo/pose_estimation/yolov8x_pose/output/evaluation`.
