# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import json
import os
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints


def _resolve_abs_paths(path: Path, config: BaseModel):
    """Resolve all relative paths in the config to absolute paths relative to path.

    Args:
        path: The base path for the model directory
        config: The configuration model to process
    """

    def _recursive_resolve_abs_paths(item: Any):
        """Recursively traverse and resolve Path objects to absolute paths."""
        if isinstance(item, Path):
            # Resolve the path relative to model_path
            abs_path = path / item
            resolved_path = abs_path.resolve()

            return resolved_path

        elif isinstance(item, dict):
            # Process dictionary values
            return {k: _recursive_resolve_abs_paths(v) for k, v in item.items()}

        elif isinstance(item, list):
            # Process list items
            return [_recursive_resolve_abs_paths(i) for i in item]

        elif isinstance(item, BaseModel):
            # Process Pydantic model fields
            for field_name, field_value in item:
                resolved_value = _recursive_resolve_abs_paths(field_value)
                setattr(item, field_name, resolved_value)
            return item

        else:
            # Return other types unchanged
            return item

    # Start the recursive resolution
    _recursive_resolve_abs_paths(config)


class BaseConfig(BaseModel):
    @classmethod
    def default(cls):
        return cls()


class NetworkImagesConfig(BaseConfig):
    quantize: Path = Field(
        ...,
        description="Path to folder containing images to be used for computing quantization parameters.",
    )
    verify: Path = Field(
        ...,
        description="Path to folder containing images to be used for any verification.",
    )


class NetworkConfig(BaseConfig):
    srcfw: Literal["caffe", "mxnet", "tensorflow", "pytorch", "onnx"] = Field(
        ..., description="Source Framework of Network."
    )

    inet: Path = Field(..., description="Path to file containing network definition.")
    iwt: Path = Field(..., description="Path to file containing weights.")

    dim: str = Field(
        ...,
        description="Dimensions of input, format for 3 dimensions is C,H,W and for 4 is C,D,H,W. In case of multiple, use : to separate.",
    )

    images: NetworkImagesConfig = Field(
        ...,
        description="Path to folder containing images to be used for computing quantization parameters."
        "Path to folder containing images to be used for any verification.",
    )

    tags: Path | None = Field(
        None, description="Path to file containing tags/classes for each image."
    )
    label: Path | None = Field(
        None,
        description='Path to file containing labels/names for each corresponding tag/class provided in "tags".',
    )
    pcfg: Path | None = Field(
        None,
        description="Path to file containing pipeline configuation for tensorflow based SSD models.",
    )


class DVConvertConfig(BaseConfig):
    inode: str = Field(
        ...,
        description="Names of the input nodes of the model, multiple inputs are to be : separated. If the name contains a colon, replace it with hash.",
    )
    onode: str = Field(
        ...,
        description="Names of the output nodes of the model, multiple outputs are to be : separated. If the name contains a colon, replace it with hash.",
    )

    data_format: str = Field(
        default="CHW",
        description="Data format for the model, can be one of: [CHW, HWC, CDHW, TNC, NTC].",
    )
    lname: str | None = Field(
        None,
        description="Label names for MXNet models if the model has label nodes. Use : to separate, if name contains colon, replace it with hash.",
    )
    lshape: str | None = Field(
        None, description="Label shapes for MXNet models. Use : to separate."
    )
    state_name: str | None = Field(
        None,
        description="State names for LSTM models. Use : to separate, if name contains colon, replace it with hash.",
    )
    state_shape: str | None = Field(
        None, description="State shapes for LSTM models. Use : to separate."
    )

    np: bool = Field(
        default=False,
        description="For MXNet models, if model is saved with numpy shapes.",
    )
    rsmax: bool = Field(
        default=False,
        description="On true, converter will retain softmax in segmentation networks, wherever softmax is input to argmax, else it is removed.",
    )
    ignore_mse: bool = Field(
        default=False,
        description="Force continue if mse comparison fails during conversion.",
    )
    tg: bool = Field(
        default=False, description="Apply network simplification if possible."
    )
    dynamic: bool = Field(
        default=False,
        description="Set to true if network has dynamic shape ROIs like maskrcnn/densepose.",
    )
    dynamic_input: bool = Field(
        default=False,
        description="Set to true if network has dynamic batch inputs like ROI boxes.",
    )
    no_final_perm_opt: bool = Field(default=False, description="???")

    adjust_file: Path | None = Field(None, description="Adjust file path.")

    dformat: None = Field(None, description="Data format of every input node.")

    encodings_json_file: Path | None = Field(None, description="Encodings json path.")


class DVNCConfig(BaseConfig):
    qmode: int = Field(
        ...,
        description="Quantization mode to be used. Each mode uses different methods for calculating quantization parameters.",
    )

    asymmetric: bool = Field(..., description="Enable offset.")

    adjust: Path | None = Field(
        None,
        description="Path to file containing Qn format for layers, which overwrites Qn format computed by quantizer.",
    )

    layeroutput: bool = Field(
        default=False,
        description="If true, compiler will generate metadata for simulator, which will dump per layer data.",
    )
    qonly: bool = Field(default=False, description="Run quantization only.")
    bc: bool = Field(
        default=False,
        description="Apply bias correction to improve quantization results.",
    )
    clampqlimits: bool = Field(
        default=False,
        description="Forces quantization parameters within hardware supported limits.",
    )
    use_tf_enhanced_encoding: bool = Field(
        default=False, description="Apply TFEnhanced method for scale and offset."
    )


class DVSimConfig(BaseConfig):
    emit: Path | None = Field(
        None,
        description="Path to file containing layer names per line. Leave empty for all layers.",
    )
    images: Path | None = Field(
        None,
        description="Path to folder containing images for simulate only mode. If left empty, image directory provided in network->verify will be used.",
    )


class AraProxyInterfaceType(str, Enum):
    """
    Enumeration of communication interface types for AraProxy sessions.

    Defines the different types of communication interfaces that can be used
    to connect to AraProxy inference sessions. The proxy must be running using
    the same connection type for successful connection.

    Values:
        IPV4: Use IPv4 network sockets for communication. Works on both Linux
              and Windows platforms.
        NAMED_PIPE: Use named pipes for communication. Primarily used on
                    Windows platforms.
        SOCKET: Use Unix domain sockets for communication. Primarily used on
                Linux platforms.

    Note:
        Default interface type for Linux is SOCKET.
        Default interface type for Windows is NAMED_PIPE.
        IPV4 can be used for both Linux and Windows.
    """

    IPV4 = "IPV4"
    NAMED_PIPE = "NAMED_PIPE"
    SOCKET = "SOCKET"


DEFAULT_INTERFACE_TYPE = (
    AraProxyInterfaceType.NAMED_PIPE
    if os.name == "nt"
    else AraProxyInterfaceType.SOCKET
    if os.name == "posix"
    else AraProxyInterfaceType.IPV4  # fallback for other platforms
)
"""Default communication interface type based on the operating system.

This constant determines the default interface type to use based on the
current operating system:
- Windows (nt): NAMED_PIPE
- Linux/Unix (posix): SOCKET
- Other platforms: IPV4 (fallback)
"""

DEFAULT_SOCKET_PATH = "/var/run/proxy.sock"


class InferenceConfig(BaseConfig):
    endpoint: int = Field(
        default=0, description="Endpoint Index to use for inferencing.", ge=0
    )
    interface: AraProxyInterfaceType = Field(
        default=DEFAULT_INTERFACE_TYPE,
        description="Interface type to use for inferencing on Ara Hardware",
    )
    socket: str = Field(
        default=DEFAULT_SOCKET_PATH,
        description="Path to socket file for proxy communication OR for IPV4, the IP:PORT string.",
    )


class InputShape(BaseModel):
    channels: int = Field(..., gt=0, description="Number of channels", alias="c")
    height: int = Field(..., gt=0, description="Height", alias="h")
    width: int = Field(..., gt=0, description="Width", alias="w")


class Mean(BaseModel):
    r: float = Field(..., ge=0, le=255, description="R channel mean in [0, 255]")
    g: float = Field(..., ge=0, le=255, description="G channel mean in [0, 255]")
    b: float = Field(..., ge=0, le=255, description="B channel mean in [0, 255]")


class Scale(BaseModel):
    r: float = Field(
        ...,
        ge=0,
        le=1,
        description="Normalization scale for R channel (pixel values scaled into [0, 1])",
    )
    g: float = Field(
        ...,
        ge=0,
        le=1,
        description="Normalization scale for G channel (pixel values scaled into [0, 1])",
    )
    b: float = Field(
        ...,
        ge=0,
        le=1,
        description="Normalization scale for B channel (pixel values scaled into [0, 1])",
    )


class InterpolationValues(str, Enum):
    LINEAR = "LINEAR"
    BILINEAR = "BILINEAR"
    INTER_LINEAR = "INTER_LINEAR"
    NEAREST = "NEAREST"
    INTER_NEAREST = "INTER_NEAREST"
    CUBIC = "CUBIC"
    BICUBIC = "BICUBIC"
    INTER_CUBIC = "INTER_CUBIC"
    AREA = "AREA"
    INTER_AREA = "INTER_AREA"
    LANCZOS = "LANCZOS"
    LANCZOS4 = "LANCZOS4"
    INTER_LANCZOS4 = "INTER_LANCZOS4"


class Resize(BaseModel):
    resize_size: int | None = Field(default=None, gt=0)
    aspect_ratio: bool = True
    scale: float = 1.0
    interpolation: str = InterpolationValues.LINEAR.value
    padding: int | None = Field(
        default=None,
        ge=0,
        le=255,
        description=(
            "Constant pixel value used for image padding (fill value in [0, 255]). "
            "If None, padding is disabled and aspect ratio is not preserved via padding."
        ),
    )


class CropShape(BaseModel):
    """
    Center crop configuration.

    Precedence rule (IMPORTANT):
    - If both (h, w) and fraction are provided → (h, w) takes precedence (fraction is ignored)
    - If (h, w) is provided → absolute crop is used
    - Else if fraction is provided → fractional crop is used
    """

    height: int | None = Field(
        default=None, description="Height of the image after croping", alias="h", gt=0
    )
    width: int | None = Field(
        default=None, description="Width of the image after croping", alias="w", gt=0
    )

    fraction: float | None = Field(
        default=None,
        description="Fraction of image size (0 < fraction <= 1)",
        gt=0,
        le=1,
    )


class PreprocessConfig(BaseModel):
    input_shape: InputShape
    mean: Mean
    scale: Scale
    resize: Resize | None = None
    centercrop: CropShape | None = None
    bgr_to_rgb: bool = True
    to_float: bool = True
    hwc_to_chw: bool = True


class ModelType(str, Enum):
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"
    FACE_DETECTION = "face_detection"
    POSE_ESTIMATION = "pose_estimation"


class PostProcessConfig(BaseModel):
    tags: Path | None = Field(None, description="Path to the tags file.")
    label: str | None = Field(None, description="Label string for postprocessing.")
    image_overlay: bool | None = Field(
        True, description="Whether to overlay results on the image."
    )
    dump_output: bool | None = Field(
        True, description="Whether to dump the output to disk."
    )
    iou_threshold: float | None = Field(
        0.5,
        description="IOU threshold for NMS or other overlap-based filtering.",
        ge=0.0,
        le=1.0,
    )
    nms_score_threshold: float | None = Field(
        0.25, description="Score threshold for NMS filtering.", ge=0.0, le=1.0
    )
    top_k: int | None = Field(
        100, description="Maximum number of detections to keep.", ge=1
    )
    print_accuracy_metrics: bool | None = Field(
        False, description="Whether to print accuracy metrics."
    )
    model_type: ModelType | None = Field(
        ModelType.DETECTION,
        description="Type of model (Detection, Segmentation, etc.).",
    )


class CalibrationDatasetConfig(BaseConfig):
    quantize: Path = Field(
        ...,
        description="Path to folder containing images to be used for computing quantization parameters.",
    )
    verify: Path | None = Field(
        None,
        description="Path to folder containing images to be used for verification of quantization results.",
    )
    quantize_list: Path = Field(
        ...,
        description="File containing a list of specific images to use for quantization, copied from the dataset passed to the compile flow.",
    )


class QuantizationSchemes(str, Enum):
    PTQ = "PTQ"
    QFT = "QFT"


class Flow(str, Enum):
    INFER_FLOAT = "infer_float"
    INFER_HW = "infer_hw"
    EVAL_FLOAT = "eval_float"
    EVAL_HW = "eval_hw"
    COMPILE = "compile"
    QUANTIZATION = "quantization"

    @property
    def required_keys(self) -> list[str]:
        common = ["modelname", "out", "preprocess"]
        _map = {
            Flow.INFER_FLOAT: common + ["postprocess"],
            Flow.INFER_HW: common + ["dvconvert", "inference", "postprocess"],
            Flow.EVAL_FLOAT: common + ["postprocess", "evaluation"],
            Flow.EVAL_HW: common
            + ["dvconvert", "inference", "postprocess", "evaluation"],
            Flow.COMPILE: common + ["network", "dvconvert", "dvnc", "calibration_data"],
            Flow.QUANTIZATION: common + ["postprocess", "evaluation", "quantization"],
        }
        return _map[self]


# Restrict device configuration to valid PyTorch device strings: "cpu", "cuda", or "cuda:<gpu_index>".
DeviceStr = Annotated[str, StringConstraints(pattern=r"^(cpu|cuda(:\d+)?)$")]


class QuantizationConfig(BaseConfig):
    scheme: QuantizationSchemes = Field(
        default=QuantizationSchemes.PTQ,
        description="The scheme to use for quantization, choose between PTQ or QFT.",
    )
    quantization_config: Path = Field(
        default=Path("./quantization_config.json"),
        description="Sparrow config.",
    )
    device: DeviceStr = Field(default="cuda" if torch.cuda.is_available() else "cpu")
    num_workers: int = Field(default=8, ge=0)
    val_batch_size: int = Field(
        default=16,
        ge=1,
        description="Batch size to use for validation.",
    )
    calibration_samples: int = Field(
        default=1000,
        ge=1,
        description="Number of samples to use for calibration/finetuning of PTQ/QFT respectively.",
    )
    calibration_batch_size: int = Field(
        default=8,
        ge=1,
        description="Calibration/finetuning batch size for PTQ/QFT.",
    )
    epochs: int = Field(
        default=15,
        ge=1,
        description="Number of epochs for finetuning.",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


class EvaluationConfig(BaseConfig):
    model_config = ConfigDict(extra="allow")

    type: str = Field(..., description="Evaluation dataset/metric type to run.")
    root: Path | None = Field(
        None, description="Optional root directory for the evaluation dataset."
    )
    images_dir: Path | None = Field(
        None, description="Optional explicit directory containing evaluation images."
    )
    annotation_file: Path | None = Field(
        None, description="Optional annotation file for the evaluation dataset."
    )
    gt_dir: Path | None = Field(
        None,
        description="Optional directory containing metric-specific ground-truth files.",
    )
    split: str = Field(default="val", description="Dataset split to evaluate.")
    prediction_dir: Path | None = Field(
        None, description="Optional directory to write raw prediction files."
    )
    iou_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="IoU threshold used by detection evaluators.",
    )


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    modelname: str = Field(..., description="Name of the model.")
    out: Path = Field(..., description="Output folder.")
    precompiled_model_link_path: Optional[str] = Field(
        default=None,
        description="HF-hub link or local path to the precompiled model.",
    )
    batch_size: int = Field(default=1, ge=1, description="Batch size for inference.")
    log_level: str = Field(default="info", description="What level to log at.")

    network: Optional[NetworkConfig] = Field(
        default=None,
        description="Defines the network and its framework details.",
    )

    dvconvert: Optional[DVConvertConfig] = Field(
        default=None,
        description="Defines the converters configuration.",
    )

    dvnc: Optional[DVNCConfig] = Field(
        default=None,
        description="Defines the compilers configuration.",
    )

    dvsim: Optional[DVSimConfig] = Field(
        default=None,
        description="Defines the simulators configuration.",
    )

    inference: Optional[InferenceConfig] = Field(
        default=None, description="Defines the configuration needed to run inference"
    )

    preprocess: Optional[PreprocessConfig] = Field(
        default=None, description="Contains preprocessing configs"
    )

    postprocess: Optional[PostProcessConfig] = Field(
        default=None, description="Postprocessing configuration as a single object."
    )

    calibration_data: Optional[CalibrationDatasetConfig] = Field(
        default=None,
        description="Configuration for calibration dataset used in quantization.",
    )
    quantization: Optional[QuantizationConfig] = Field(
        default=None,
        description="Defines hyperparameters for PTQ/QFT quantization using Sparrow.",
    )
    evaluation: Optional[EvaluationConfig] = Field(
        default=None,
        description="Optional evaluation configuration consumed by dedicated evaluation runners.",
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Config":
        if data is None:
            raise ValueError("from_dict() expects a dict, got None")
        return cls(**data)

    @staticmethod
    def validate_flow(config_data, flow: Flow | None = None):

        if flow is None:
            keys = set()
            for f in Flow:
                keys.update(f.required_keys)
            flow_label = ""
        else:
            if not isinstance(flow, Flow):
                raise ValueError(
                    f"Value of flow shall be one of: {', '.join(e.value for e in Flow)}"
                )
            keys = flow.required_keys
            flow_label = f" for {flow.value} flow"

        for key in keys:
            if key not in config_data or config_data[key] is None:
                raise ValueError(
                    f"{key} is not present in run.yaml and is required{flow_label}. {key} cannot be None"
                )

    @staticmethod
    def from_file(path: str | Path, flow: Flow | None = None) -> "Config":
        if isinstance(path, str):
            path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"No Config file found at {path}")

        ext = path.suffix.lower()

        with open(path, "r") as f:
            if ext == ".json":
                data = json.loads(f.read())
            elif ext in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            else:
                raise ValueError(
                    f"Unsupported file type: {ext}. Use .json, .yaml, .yml."
                )

        # Validate Flow specific Keys
        Config.validate_flow(data, flow)

        # Create Config object
        config = Config.from_dict(data)
        _resolve_abs_paths(path.parent, config)

        return config
