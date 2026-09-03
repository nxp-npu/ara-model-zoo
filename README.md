# Ara-Model-zoo

Ara-Model-zoo is the evaluation and benchmarking harness for Ara's accelerated model zoo.
It standardizes how models are preprocessed, converted, executed on Ara hardware (via `dvrun`/`dvproxy`), and postprocessed so that accuracy and performance numbers are reproducible.

## Repository Layout

| Path                | Purpose                                                                                                                                    |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `core/`             | Shared Python/C++ libraries used by every pipeline (configuration loader, pre/post utilities, dvproxy app bindings). See `core/README.md`. |
| `modelzoo/`         | Task- and model-specific pipelines (detection, classification, segmentation). See `modelzoo/README.md`.                                    |
| `flows/`            | Shell-script entry points for each execution stage (compile, quantize, infer, evaluate, benchmark, download). See `flows/README.md`.                          |
| `batch_scripts/`    | Python wrappers that drive `flows/` scripts over a list of models. See `batch_scripts/README.md`.                                         |
| `env/`              | Optional environment helpers such as sample datasets or SDK config shells.                                                                 |
| `requirements.txt`  | Python dependencies needed to run the preprocessing, conversion, and post-processing stages.                                               |

Sub-module READMEs:

- [`core/README.md`](core/README.md)
- [`modelzoo/README.md`](modelzoo/README.md)
- [`flows/README.md`](flows/README.md)
- [`batch_scripts/README.md`](batch_scripts/README.md)

---

## Quick Start

### 1. Set up a virtual environment

Python >=3.12 is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
The project uses a `pyproject.toml` for dependency management. Choose the subsection that matches your hardware. Run the dev dependency and pre-commit hook commands inside the same subsection that matches your setup.

#### 2.1 CPU installation
If you have no NVIDIA GPU (or do not want the CUDA build of PyTorch), install the project in editable mode using the CPU-only index as an extra index:

```bash
pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu
```

Install the development dependencies (`pytest`, `ruff`, `pre-commit`, type checker, ...). You will need these dependencies to run the float flows (infer and eval), and they are required if you plan to develop on this project since they pull in the formatters, linters, and test tooling:

```bash
pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/cpu
```

Enable the pre-commit hooks:

```bash
pre-commit install
```

#### 2.2 GPU installation
Install the project in editable mode:

```bash
pip install -e .
```

Install the development dependencies (`pytest`, `ruff`, `pre-commit`, type checker, ...). You will need these dependencies to run the float flows (infer and eval), and they are required if you plan to develop on this project since they pull in the formatters, linters, and test tooling:

```bash
pip install -e .[dev]
```

Enable the pre-commit hooks:

```bash
pre-commit install
```

### 3. Set up the Ara SDK

```bash
export ARA_SDK_ROOT=/path/to/ara/sdk   # must contain dvrun + dvproxy binaries
export DV_TGT_ROOT=/path/to/ara/sdk       # must contain dvrun + dvproxy binaries
```

> **Note:** For SDK versions 3.0 or above, make sure to follow the steps to build
> the Docker image included in the SDK before running `model_compile.sh`.

> **Note for SDK versions older than 3.0:** The `model_compile` flow requires the
> compilation Docker image to be loaded in advance. Before running `model_compile.sh`,
> load the Docker image bundled with your SDK:
>
> ```bash
> docker load -i "$ARA_SDK_ROOT/dvdocker/ara2.tar"
> ```
>
> The `model_compile` flow will fail if this image has not been loaded manually for versions older than 3.0.

---

## Flow Reference

See [`flows/README.md`](flows/README.md) for the full parameter reference and usage examples for each flow script.

---

## Typical Workflow

Standard path from a fresh model to a verified hardware result:

```
1. model_compile.sh      convert ONNX to .dvm
2. infer_float.sh        verify float output looks correct on sample images
3. infer_hw.sh           verify hardware output matches float baseline
4. eval_float.sh         measure float accuracy on full dataset
5. eval_hw.sh            measure hardware accuracy and compare to float baseline
6. performance.sh        measure hardware throughput and latency
```

To run multiple models at once, see [`batch_scripts/README.md`](batch_scripts/README.md).

---

## Extending the Repository

- **New model**: copy an existing folder (e.g., `modelzoo/detection/yolov8n`), update `config/run.yaml` with the new ONNX path and dataset config, then implement custom `preprocess/preprocess.py` and `postprocess/postprocess.py`. See [`modelzoo/README.md`](modelzoo/README.md).
- **New task**: add a sibling directory under `modelzoo/` (e.g., `modelzoo/pose/`) and re-use the shared building blocks from `core/`.
- **Automation/CI**: integrate with the flows in `flows/` or with the batch scripts in `batch_scripts/`.

---

## Code Quality

Code quality is maintained through [Ruff](https://github.com/astral-sh/ruff) for linting and formatting. Ruff provides fast, modern Python linting and formatting that replaces tools like Flake8, isort, and Black.

```bash
ruff check .            # check for issues
ruff check --fix .      # auto-fix issues
ruff format .           # format code
```
