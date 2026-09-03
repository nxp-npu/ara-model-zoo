# MobilenetV2 SSD (TensorFlow)

## Use Case and High-Level Description

Mobilenet V2 SSD is an object detection model pre-trained on COCO 2017 dataset. This is the TensorFlow implementation.

The model input is a blob that consists of a single image of `1, 3, 300, 300`.

## Model Source

The model is taken from [this table](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) ssd-checkpoints for `ssd_mobilenet_v2_coco`

## Specification

| Metric           | Value      |
| ---------------- | ---------- |
| Type             | Detection  |
| Source framework | TensorFlow |

## Accuracy

| Metric | FP32 | Int8  |
| ------ | ---- | ----- |
| mAP    | 26.25| 25.30 |

Accuracy metrics reflect IOU threshold of 0.7 and NMS threshold of 0.001.

## Dataset used for checking accuracy

COCO datasets download [page](https://cocodataset.org/#download)

Download validation images [val2017](http://images.cocodataset.org/zips/val2017.zip)

Download tags [instances_val2017.json](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

## Model Parameters

#### Pre-Process Stage
| Parameter              | Value                                     |
| :--------------------- | :---------------------------------------- |
| `Model input shape`    | `[1 3 300 300]`                           |
| `Model output shape`   | `[[1 1917 4], [1 1917 91]]`              |
| `Interpolation`        | `INTER-LINEAR`                            |
| `Mean`                 | `{127.5, 127.5, 127.5}`                  |
| `Scale`                | `{0.007843137, 0.007843137, 0.007843137}` |
| `center crop fraction` | `{0.875}`                                 |


#### Post-Process Stage
| Parameter             | Value             |
| :-------------------- | :---------------- |
| `IOU Threshold`       | `0.70`            |
| `NMS Score Threshold` | `0.25`            |
| `label`               | `COCO 80 classes` |


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
./flows/precompiled_model_download.sh model=mobilenetv2-ssd
```

- Precompiled models are compiled using Ara SDK.
- Output model and assets will be stored in the `modelzoo/detection/mobilenetv2-ssd/output/assets` folder

### Manual Model Preparation and Compile

#### Model Compile

```bash
cd ara-model-zoo
./flows/model_compile.sh model=mobilenetv2-ssd dataset_root=<path to val2017>
```

- Output model and assets will be stored in the `modelzoo/detection/mobilenetv2-ssd/output/compiled_model` folder

#### Performance checker

```bash
cd ara-model-zoo
./flows/performance.sh model=mobilenetv2-ssd
```

Look for `HW IPS` for ips in output.

#### Inference Application

Runs inference application on images provided to display inference results, if no images are provided a default test image will be used

```bash
cd ara-model-zoo
./flows/infer_float.sh images_folder=../images model=mobilenetv2-ssd
./flows/infer_hw.sh images_folder=../images model=mobilenetv2-ssd
```

- Output images saved in `modelzoo/detection/mobilenetv2-ssd/output/postprocessed_output_visualized`

#### Accuracy checker

```bash
cd ara-model-zoo
./flows/eval_hw.sh dataset_root='../coco2017/' model=mobilenetv2-ssd
./flows/eval_float.sh dataset_root='../coco2017/' model=mobilenetv2-ssd
```

- Accuracy metrics printed to terminal
