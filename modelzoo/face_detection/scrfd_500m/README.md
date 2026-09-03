# SCRFD-500M

## Use Case and High-Level Description

SCRFD-500M is a lightweight SCRFD face detector with five facial landmarks. SCRFD is designed for efficient face detection by redistributing samples and computation across the detector.

The model input is a blob that consists of a single image of `1, 3, 640, 640`.

The model output consists of nine tensors containing face classification scores, bounding-box regressions, and five facial landmarks at three feature-map scales.

## Model Source

The SCRFD model family is from InsightFace: https://github.com/deepinsight/insightface/tree/master/detection/scrfd

This modelzoo entry downloads the ONNX model from the `yakhyo/face-reidentification` release used by `bin/model_prep.sh`, simplifies it with fixed batch size, and prunes the graph to the SCRFD classification, box, and keypoint heads.

## Specification

| Model Type | Metric |
| --- | --- |
| Face Detection | WIDER Face AP |

## Accuracy

The table reports Ara-model-zoo results on the WIDER Face validation dataset. The checked-in full evaluation log verifies the INT8 hardware values; a corresponding FP32 evaluation log is not included in this repository.

| Metric | FP32 | INT8 Hardware |
| --- | --- | --- |
| Easy Val AP | 91.04% | 90.02% |
| Medium Val AP | 88.40% | 87.66% |
| Hard Val AP | 68.99% | 67.91% |

Evaluation uses an NMS IoU threshold of `0.45`, an NMS score threshold of `0.02`, and top K of `1000`; WIDER Face AP matching uses an IoU threshold of `0.5`. Inference uses an NMS IoU threshold of `0.4` and an NMS score threshold of `0.5`.

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
| `Model output shape` | `[1, 2, 80, 80], [1, 8, 80, 80], [1, 20, 80, 80], [1, 2, 40, 40], [1, 8, 40, 40], [1, 20, 40, 40], [1, 2, 20, 20], [1, 8, 20, 20], [1, 20, 20, 20]` |
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

Set `precompiled_model_link_path` in `config/run.yaml` to a local `.dvm` file or a Hugging Face repository ID. The checked-in value is a placeholder. Then retrieve the precompiled DVM:

```bash
./flows/precompiled_model_download.sh model=scrfd_500m
```

* The DVM is stored as `modelzoo/face_detection/scrfd_500m/output/compiled_model/model.dvm`.
* This flow does not download the ONNX model; use the model compilation flow below to prepare it.

#### Model Compile

```bash
./flows/model_compile.sh model=scrfd_500m dataset_root=<path to WIDER FACE>
```

* The ONNX and DVM files are stored in `modelzoo/face_detection/scrfd_500m/output/compiled_model`; compilation assets are generated in `modelzoo/face_detection/scrfd_500m/output/assets`.

#### Performance checker

```bash
./flows/performance.sh model=scrfd_500m
```

The flow requires a compiled DVM and uses `sudo` to start the hardware proxy unless `skip_proxy=true`. Results are written to `modelzoo/face_detection/scrfd_500m/output/performance`.

#### Inference Application

Runs inference on the supplied images and visualizes the detected faces. If no images are supplied, the repository's default `testimages` directory is used.

Floating-point inference:

```bash
./flows/infer_float.sh model=scrfd_500m images_folder=<path to images>
```

Hardware inference:

```bash
./flows/infer_hw.sh model=scrfd_500m images_folder=<path to images>
```

* Output images are saved in `modelzoo/face_detection/scrfd_500m/output/postprocessed_output_visualized`.

#### Accuracy checker

* Download the [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) validation dataset and pass its path as `dataset_root`. Accuracy checking can take several minutes.

Floating-point evaluation:

```bash
./flows/eval_float.sh model=scrfd_500m dataset_root=<path to WIDER FACE>
```

Hardware evaluation:

```bash
./flows/eval_hw.sh model=scrfd_500m dataset_root=<path to WIDER FACE>
```

* Accuracy metrics are printed to the terminal.
