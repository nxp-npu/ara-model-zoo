# SCRFD-10G

## Use Case and High-Level Description

SCRFD-10G is the largest SCRFD face detector included here, with five facial landmarks and the same SCRFD feature-head layout as the smaller variants. It targets higher accuracy at a higher compute cost.

The model input is a blob that consists of a single image of `1, 3, 640, 640`.

The model output consists of nine tensors containing face classification scores, bounding-box regressions, and five facial landmarks at three feature-map scales.

## Model Source

The SCRFD model family is from InsightFace: https://github.com/deepinsight/insightface/tree/master/detection/scrfd

This modelzoo entry downloads the ONNX model from the `yakhyo/facial-analysis` release used by `bin/model_prep.sh`, simplifies it with fixed batch size, and prunes the graph to the SCRFD classification, box, and keypoint heads.

## Specification

| Model Type | Metric |
| --- | --- |
| Face Detection | WIDER Face AP |

## Accuracy

The FP32 and INT8 hardware accuracy values were reproduced in Ara-model-zoo on the WIDER Face validation dataset.

| Metric | FP32 | INT8 Hardware |
| --- | --- | --- |
| Easy Val AP | 95.43% | 95.43% |
| Medium Val AP | 93.94% | 93.91% |
| Hard Val AP | 82.50% | 82.34% |

Accuracy metrics reflect an IOU threshold of `0.45`, an NMS score threshold of `0.02`, and top K of `1000` during evaluation. Inference uses an IOU threshold of `0.4` and an NMS score threshold of `0.5`.

## Dataset used for checking accuracy

Original dataset: [WIDER FACE](http://shuoyang1213.me/WIDERFACE/)

| Parameter | Value |
| --- | --- |
| Dataset Name | WIDER FACE |
| Classes | Face |
| Validation images | 3,226 |
| Image format | JPEG |
| Accuracy annotation | WIDER Face validation ground truth |

Download the evaluation annotations from [Ground Truth](https://github.com/biubug6/Pytorch_Retinaface/tree/master/widerface_evaluate/ground_truth). Place the directory containing `wider_face_val.mat` under `<dataset_root>/ground_truth`, or provide it with `gt_dir=<path>`.

## Model Parameters

#### Pre-Process Stage

| Parameter | Value |
| :-------- | :------ |
| `Model input shape` | `[1, 3, 640, 640]` |
| `Model output shape` | `[1, 2, 20, 20], [1, 8, 20, 20], [1, 20, 20, 20], [1, 2, 40, 40], [1, 8, 40, 40], [1, 20, 40, 40], [1, 2, 80, 80], [1, 8, 80, 80], [1, 20, 80, 80]` |
| `Resize` | `640 x 640 (aspect ratio preserved)` |
| `Interpolation` | `Area interpolation` |
| `Padding value` | `0 (pad maintaining the aspect ratio)` |
| `Mean` | `{127.5, 127.5, 127.5}` |
| `Scale` | `{0.0078125, 0.0078125, 0.0078125}` |

#### Post-Process Stage

| Parameter | Value |
| :-------- | :------ |
| `IOU Threshold` | `0.4` |
| `NMS Score Threshold` | `0.5` |
| `Top K` | `1000` |
| `label` | `Faces` |

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

#### Pre-compiled Model Download

Download the precompiled DVM and ONNX model:

```bash
./flows/precompiled_model_download.sh model=scrfd_10g
```

* The DVM is stored as `modelzoo/face_detection/scrfd_10g/output/compiled_model/model.dvm`.

#### Model Compile

```bash
./flows/model_compile.sh model=scrfd_10g dataset_root=<path to WIDER FACE>
```

* The ONNX and DVM files are stored in `modelzoo/face_detection/scrfd_10g/output/compiled_model`; compilation assets are generated in `modelzoo/face_detection/scrfd_10g/output/assets`.

#### Performance checker

```bash
./flows/performance.sh model=scrfd_10g
```

Look for `HW IPS` in the output.

#### Inference Application

Runs inference on the supplied images and visualizes the detected faces. If no images are supplied, the repository's default `testimages` directory is used.

Floating-point inference:

```bash
./flows/infer_float.sh model=scrfd_10g images_folder=<path to images>
```

Hardware inference:

```bash
./flows/infer_hw.sh model=scrfd_10g images_folder=<path to images>
```

* Output images are saved in `modelzoo/face_detection/scrfd_10g/output/postprocessed_output_visualized`.

#### Accuracy checker

* Download the [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) validation dataset and pass its path as `dataset_root`. Accuracy checking can take several minutes.

Floating-point evaluation:

```bash
./flows/eval_float.sh model=scrfd_10g dataset_root=<path to WIDER FACE>
```

Hardware evaluation:

```bash
./flows/eval_hw.sh model=scrfd_10g dataset_root=<path to WIDER FACE>
```

* Accuracy metrics are printed to the terminal.
