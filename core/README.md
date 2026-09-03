# Core

Shared libraries that power every pipeline in Arabench.
The modules here keep SDK interaction, configuration loading, and task utilities in one place so model folders can stay light-weight.

## Directory Guide

| Path | Description |
| --- | --- |
| `python/application/` | Runtime glue around Ara's `dvapi`. Provides the `NNApplication` class for launching dvproxy sessions, loading compiled `*.dvm` assets, managing shared memory, and dispatching inferences. |
| `python/preprocess/` | Canonical preprocessing primitives (`preprocessing_builder.py`) that model-specific scripts can import and compose. Covers resize/pad, normalization, layout conversions, and serialization helpers. |
| `python/postprocess/` | Accuracy computation utilities. Includes reusable detection blocks (`detection/boxes.py`, `detector.py`), ara dequantization helpers, and the `Postprocessor` base class. |
| `python/config.py` & `python/utils.py` | YAML parsing, dataset filtering, file I/O helpers, and pretty printers shared by preprocessors, postprocessors, and the application wrapper. |
| `python/args_parser.py` | Common CLI options for selecting run configs, modes, and devices. |
| `cpp/` | Placeholder for native extensions (e.g., accelerated preprocess or custom ops) that can be compiled and imported by the Python code when required. |

## Typical Usage

```python
from core.python import utils
from core.python.application.app_common import NNApplication

config = utils.load_config("config/run.yaml", mode="normal")
nnapp = NNApplication("config/run.yaml", mode="normal")
nnapp.init()
outputs = nnapp.infer(preprocessed_tensor)
```

Model-specific code usually follows this pattern:

1. **Configuration** - load YAML via `utils.load_config`, merge the selected `mode`, and resolve all relative paths with `utils.set_paths`.
2. **Preprocessing** - subclass or wrap helpers in `python/preprocess/*.py` to decode images and export Ara-ready tensors. Most models expose a `Preprocessor` class that orchestrates this logic.
3. **Application** - rely on `NNApplication` to talk to dvproxy or to run Ara inference loops.
4. **Post-processing** - extend `core/python/postprocess/postprocess.py` or import task-specific helpers (e.g., `postprocess/detection/boxes.py`) to compute accuracy, generate overlays, or dump tensors.

## Extending Core

- Place framework-agnostic building blocks here whenever multiple models need them.
- Re-export convenience functions via `core/python/__init__.py` to avoid deep import chains.
- Keep Ara SDK assumptions isolated in `python/application/` so upstream model code stays portable.

The [top-level README](../README.md) explains how this package fits into the global workflow, and the [model zoo documentation](../modelzoo/README.md) shows how these helpers are consumed by each model.

## IOTransformationEngine - Generic Pipeline Executor

**IOTransformationEngine** is a generic pipeline executor for both **pre-rocessing** and **post-processing** operations.
It orchestrates a sequence of transforms or prediction operations on dictionary-based data, ensuring that each step adheres to the `dict -> dict` contract defined by `Transform` and `PredictionOp`. The engine preserves execution order, validates configurations, and provides descriptive errors for reliable and predictable pipeline execution.

### Features:
- Validates pipeline configuration.
- Instantiates operations with provided parameters.
- Preserves execution order.
- Enforces dict -> dict contract at every step.
- Provides clear, actionable error messages.

### Responsibilities:
- Only handles pipeline execution and orchestration.
- Does not parse YAML, merge configs, or validate individual transform semantics.
- Works with any dataclass-based configuration.

### Configuration:
The transformation engine consumes a **Pydantic configuration model** containing pipeline sections.

Each pipeline section (e.g., ```preprocess```, ```postprocess```) is represented as a **flat list of operation configs**. Each operation config specifies the operation class and optional initialization parameters.

Each operation config follows the structure:

```python
class PreProcessConfig(BaseProcessConfig):
    _expected_base_class = Transform

class PostProcessConfig(BaseProcessConfig):
    _expected_base_class = PredictionOp
```

Where ```BaseProcessConfig``` defines:
```python
class BaseProcessConfig(BaseModel):
    operation: Type
    params: Dict[str, Any] = {}
```

**Example usage:**
```python
full_config = FullConfig(
    preprocess=[
        PreProcessConfig(
            operation=AspectRatioPreservableResize,
            params={"new_shape": [480, 640]},
        ),
        PreProcessConfig(operation=BGR2RGB),
        PreProcessConfig(operation=TransposeChannels),
    ],
    postprocess=[
        PostProcessConfig(operation=DecodeKeypoints),
        PostProcessConfig(operation=KeypointTransposeChannels),
    ],
)

from core.python import IOTransformationEngine

engine = IOTransformationEngine(
    config=full_config,
    section="preprocess"
)

output = engine.execute(input_data)
```

### Developer Notes:
- The engine ensures safe and predictable sequential execution.
- Both pre-processing and post-processing pipelines can use the same engine architecture.

## Pre-processing Pipeline

### Transform - Abstract Base Class for Transforms

Transform is the foundational building block for all preprocessing operations in the inference pipeline. It defines a strict dictionary-to-dictionary contract, ensuring consistent, predictable, and composable transformations.

#### Key Points:

**Contract:**
- Each transform must receive a dict and return a dict.
- Minimum required key: `"image"`.

**Extensibility:**
- New transforms can be added without modifying the pipeline. The dictionary acts as a mutable data carrier for evolving data (e.g., image, tensor, metadata).

#### Steps to Add a New Transform

1. **Subclass `Transform`**
   Create a new class that inherits from `Transform`.

2. **Implement `_apply()`**
   Write the transformation logic inside `_apply(data: dict) -> dict`.

3. **Operate on the `"image"` key**
   Access the image via `data["image"]`, modify it, and update the dictionary.

4. **Return the dictionary**
   Always return the updated `data` dictionary.

5. **Register it in the pipeline configuration**
   Add the transform class to the **model-specific preprocessing pipeline config** in `model-specific preprocess.py`.
   Since transformations are executed sequentially, place the new transform at the correct position in the sequence.
   The defined sequence will be strictly followed when applying the transformations.

**Example:**
```python
from core.python.preprocess import Transform

class BGR2RGB(Transform):
    def _apply(self, data):
        img = data["image"]
        data["image"] = img[..., ::-1]  # Convert BGR -> RGB
        return data
```

**Pipeline Compatibility:**
Transforms are designed to work sequentially:
```python
data = transform_1(data)
data = transform_2(data)
data = transform_3(data)
```

#### Guidelines for Developers:
- Keep transforms atomic (single responsibility).
- Do not replace the dictionary; update or insert keys.
- Include validation in `_apply()` if required, but rely on the engine for contract enforcement.
