# Flows

All flows live in `flows/` and accept `key=value` positional arguments.

> **Note:** The `run` parameter accepts `python` or `cpp`, but only `python` is currently supported. Passing `run=cpp` will exit with an error.

---

## `model_compile.sh`

Converts an ONNX model to a compiled `.dvm` artifact ready for Ara hardware inference.
Runs the full pipeline: ONNX preprocessing, `dvrun` conversion, and asset placement.

The calibration images the quantizer needs are copied out of `dataset_root`: the model's `config/calibration_image_set.txt` lists them
by file name and each is found either directly under `dataset_root` or by searching its subdirectories. So passing the dataset root or
the split folder itself both work.

A listed image that is not there is skipped with a warning but if none of
the listed images can be found the dataset path is wrong and compilation stops.

Requires `ARA_SDK_ROOT` to be set.

> **Note:** For SDK versions 3.0 or above, make sure to follow the steps to build
> the Docker image included in the SDK before running `model_compile.sh`.

> **Note for SDK versions older than 3.0:** the `model_compile` flow requires the
> compilation Docker image to be loaded in advance. Before running `model_compile.sh`,
> load the Docker image bundled with your SDK:
>
> ```bash
> docker load -i "$ARA_SDK_ROOT/dvdocker/ara2.tar"
> ```
>
> The `model_compile` flow will fail if this image has not been loaded manually for versions older than 3.0.

| Parameter | Required | Default  | Description                                        |
| --------- | -------- | -------- | -------------------------------------------------- |
| `model`   | yes      |          | Model name matching a directory under `modelzoo/`. |
| `dataset_root`   | yes      |          | Path to the dataset the calibration images are taken from. |
| `run`     | no       | `python` | Backend: `python` only (cpp not yet supported).    |

```bash
bash flows/model_compile.sh model=yolov8n dataset_root=/datasets/coco2017
```

---

## `infer_float.sh`

Runs inference with the original floating-point model on a folder of images.
Useful for verifying preprocessing and postprocessing before compiling.

Requires `DV_TGT_ROOT` to be set.

> **Note:** Running this float flow requires the development dependencies to be installed (e.g., `pip install -e .[dev]`). See the install section of the top-level [`README.md`](../README.md).

| Parameter       | Required | Default  | Description                                             |
| --------------- | -------- | -------- | ------------------------------------------------------- |
| `model`         | yes      |          | Model name matching a directory under `modelzoo/`.      |
| `run`           | no       | `python` | Backend: `python` only (cpp not yet supported).         |
| `images_folder` | no       |          | Path to input images. Uses config default when omitted. |

```bash
bash flows/infer_float.sh model=yolov8n
bash flows/infer_float.sh model=yolov8n images_folder=/data/test_images
```

---

## `infer_hw.sh`

Runs inference using the compiled `.dvm` on Ara hardware via `dvproxy`.
Use this to verify that hardware output is close to the float baseline.

Requires `DV_TGT_ROOT` to be set and a compiled `.dvm` in the model's `output/` directory.

| Parameter       | Required | Default  | Description                                                          |
| --------------- | -------- | -------- | -------------------------------------------------------------------- |
| `model`         | yes      |          | Model name matching a directory under `modelzoo/`.                   |
| `run`           | no       | `python` | Backend: `python` only (cpp not yet supported).                      |
| `images_folder` | no       |          | Path to input images. Uses config default when omitted.              |
| `skip_proxy`    | no       | `false`  | Set to `true` to skip dvproxy start/stop when it is already running. |

```bash
bash flows/infer_hw.sh model=yolov8n
bash flows/infer_hw.sh model=yolov8n images_folder=/data/test_images skip_proxy=true
```

---

## `eval_float.sh`

Evaluates accuracy of the floating-point model over a full dataset.
Outputs metrics (e.g., mAP, top-1) to `output/evaluation/`.

Requires `DV_TGT_ROOT` to be set.

> **Note:** Running this float flow requires the development dependencies to be installed (e.g., `pip install -e .[dev]`). See the install section of the top-level [`README.md`](../README.md).

| Parameter      | Required | Default  | Description                                                        |
| -------------- | -------- | -------- | ------------------------------------------------------------------ |
| `model`        | yes      |          | Model name matching a directory under `modelzoo/`.                 |
| `dataset_root` | yes      |          | Path to the evaluation dataset root (e.g., COCO, ImageNet).       |
| `run`          | no       | `python` | Backend: `python` only (cpp not yet supported).                    |
| `gt_dir`       | no       |          | Ground-truth annotation directory if separate from `dataset_root`. |
| `limit`        | no       |          | Cap evaluation at the first N samples.                             |

```bash
bash flows/eval_float.sh model=yolov8n dataset_root=/datasets/coco2017
bash flows/eval_float.sh model=yolov8n dataset_root=/datasets/coco2017 limit=100
```

---

## `eval_hw.sh`

Evaluates accuracy of the compiled hardware model over a full dataset.
Outputs metrics to `output/evaluation/`.

Requires `DV_TGT_ROOT` to be set and a compiled `.dvm` in the model's `output/` directory.

| Parameter      | Required | Default  | Description                                                          |
| -------------- | -------- | -------- | -------------------------------------------------------------------- |
| `model`        | yes      |          | Model name matching a directory under `modelzoo/`.                   |
| `dataset_root` | yes      |          | Path to the evaluation dataset root.                                 |
| `run`          | no       | `python` | Backend: `python` only (cpp not yet supported).                      |
| `gt_dir`       | no       |          | Ground-truth annotation directory if separate from `dataset_root`.   |
| `limit`        | no       |          | Cap evaluation at the first N samples.                               |
| `skip_proxy`   | no       | `false`  | Set to `true` to skip dvproxy start/stop when it is already running. |

```bash
bash flows/eval_hw.sh model=yolov8n dataset_root=/datasets/coco2017
bash flows/eval_hw.sh model=yolov8n dataset_root=/datasets/coco2017 limit=500 skip_proxy=true
```

---

## `performance.sh`

Measures inference throughput and latency on Ara hardware for a compiled model.
Results are written to `output/performance/`.

Requires `DV_TGT_ROOT` to be set and a compiled `.dvm` in the model's `output/` directory.

| Parameter    | Required | Default | Description                                                          |
| ------------ | -------- | ------- | -------------------------------------------------------------------- |
| `model`      | yes      |         | Model name matching a directory under `modelzoo/`.                   |
| `batch_size` | no       | `1`     | Number of inputs per inference call.                                 |
| `iterations` | no       | `10`    | Number of inference iterations to average over.                      |
| `skip_proxy` | no       | `false` | Set to `true` to skip dvproxy start/stop when it is already running. |

```bash
bash flows/performance.sh model=yolov8n
bash flows/performance.sh model=yolov8n batch_size=1 iterations=100
```

---

## `precompiled_model_download.sh`

Downloads a precompiled `.dvm` model artifact from Huging face hub.
The download URL is resolved from the `precompiled_model_link_path` field in the model's `config/run.yaml`.
Places the artifact into `output/assets/` so hardware flows can use it without a local compile step.

| Parameter | Required | Default | Description                                        |
| --------- | -------- | ------- | -------------------------------------------------- |
| `model`   | yes      |         | Model name matching a directory under `modelzoo/`. |

```bash
bash flows/precompiled_model_download.sh model=yolov8n
```
