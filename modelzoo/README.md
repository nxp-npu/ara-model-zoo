# Model Zoo

Task-specific benchmark pipelines live under this directory.
Each model folder bundles everything needed to preprocess data, quantize, compile, run NXP inference, and evaluate accuracy for a single model.

## Available Tasks

| Task             | Models                                                                                                         |
| ---------------- | -------------------------------------------------------------------------------------------------------------- |
| Face Detection   | `yolov8n_face`                                                                                                 |
| Detection        | `yolov8n`, `yolov10n`, `yolov10s`, `yolov10m`, `yolox_t`, `yolox_s`, `yolox_m`, `yolox_l`, `yolox_x`        |
| Classification   | `efficientnet_lite0`, `efficientnet_lite1`, `efficientnet_lite2`, `efficientnet_lite3`, `efficientnet_lite4`, `mobilenetv1`, `mobilenetv2`, `resnet18`, `resnet18d`, `resnet34`, `resnet34d`, `resnet50v1`, `resnet50v2`, `resnet50d`, `resnet101v1`, `resnet101v2`, `resnet152v1`, `resnet152v2` |
| Segmentation     | (scaffolded)                                                                                                   |

---

## Running a Model

### Typical order of execution

```bash
# 1. Compile the ONNX model to a .dvm artifact
bash flows/model_compile.sh model=yolov8n_face dataset_root=/datasets/widerface

# 2. Check that float inference output looks correct
bash flows/infer_float.sh model=yolov8n_face images_folder=/data/test_images

# 3. Check that hardware inference matches the float baseline
bash flows/infer_hw.sh model=yolov8n_face images_folder=/data/test_images

# 4. Measure float accuracy on the full dataset
bash flows/eval_float.sh model=yolov8n_face dataset_root=/datasets/widerface

# 5. Measure hardware accuracy and compare to the float baseline
bash flows/eval_hw.sh model=yolov8n_face dataset_root=/datasets/widerface

# 6. Measure hardware throughput and latency
bash flows/performance.sh model=yolov8n_face batch_size=1 iterations=50
```

### Using a precompiled model

If you have a precompiled `.dvm` available, skip step 1 and download the artifact instead:

```bash
bash flows/precompiled_model_download.sh model=yolov8n_face
```

Then continue from step 2.

---

## Folder Layout

Every model directory follows the same convention:

```
modelzoo/<task>/<model>/
    bin/
        model_prep.py             downloads the model from its original source repository
        model_download.sh         downloads an already-stored ONNX or PyTorch model file
        extract_graph.py          creates subgraphs from the ONNX file used in postprocessing
    config/
        run.yaml                  master configuration consumed by every flow stage
        calibration_image_set.txt image list used during quantization, copied from the dataset at compile time
    preprocess/py/
        preprocess.py             defines a Preprocessor class
    postprocess/py/
        postprocess.py            defines a Postprocessor class
    output/                       generated artifacts: quantized tensors, compiled model, overlays, reports
```

---

## Understanding `config/run.yaml`

Each model has a single `run.yaml` that drives all flow stages. Key sections:

| Section            | Controls                                                                                       |
| ------------------ | ---------------------------------------------------------------------------------------------- |
| `network`          | ONNX/DVM asset paths, input dimensions, image cache locations, label metadata.                |
| `dvconvert`,`dvnc`,`dvsim`|  NXP compiler, quantizer, and simulator knobs that are forwarded to the SDK tools.   |                                         |
| `inference`        | dvproxy socket paths, shared-memory settings, and runtime flags for hardware execution.       |
| `preprocess`       | Model-specific parameters passed to `preprocess.py` (resize hints, normalization values, crop settings, etc).     |
| `postprocess`      | Model-specific Parameters passed to `postprocess.py` (score thresholds, NMS settings, overlay toggles, etc).     |
| `calibration_data` | Where the calibration images are staged (`quantize`) and which ones to use (`quantize_list`).  |
| `out`              | Base output directory for all generated artifacts.                                             |

Enable or disable dumps, overlays, and accuracy reporting directly in the YAML without modifying code.

---

## Pipeline Stages

Each flow script in `flows/` maps to one stage. The table below shows which stage each script handles and what it reads/writes under `output/`.

| Stage               | Flow script              | Reads from `output/`          | Writes to `output/`           |
| ------------------- | ------------------------ | ----------------------------- | ----------------------------- |
| Preprocess          | part of `model_compile.sh` | calibration images staged from the dataset | `quant/`, `preprocessed_images_*/` |
| Compile             | `model_compile.sh`       | ONNX + calibration images     | `compiled_model/`, `assets/`  |
| Float inference     | `infer_float.sh`         | raw images, ONNX              | `postprocessed_output_*/`     |
| Hardware inference  | `infer_hw.sh`            | raw images, `.dvm`            | `postprocessed_output_*/`     |
| Float evaluation    | `eval_float.sh`          | dataset, ONNX                 | `evaluation/`                 |
| Hardware evaluation | `eval_hw.sh`             | dataset, `.dvm`               | `evaluation/`                 |
| Performance         | `performance.sh`         | `.dvm`                        | `performance/`                |
| Download precompiled| `precompiled_model_download.sh` | `run.yaml` (for model ID) | `assets/`                |

For the full parameter reference of each flow script, see [flows/README.md](../flows/README.md).

---

## Adding a New Model

1. Copy an existing model folder as a starting template (e.g., `face_detection/yolov8n_face`).
2. Update `config/run.yaml` with the new ONNX path, dataset locations, and processing parameters.
    List the calibration images in `config/calibration_image_set.txt`, one file name per line. they
   are copied out of the `dataset_root` passed to `model_compile.sh` when the model is compiled.
3. Implement `preprocess/py/preprocess.py` using helpers from `core/python/`.
4. Implement `postprocess/py/postprocess.py` using helpers from `core/python/`.
5. Run the pipeline stages in order to verify each step produces the expected output.
6. Register the model in the Available Tasks table above.
