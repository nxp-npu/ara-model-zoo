# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TypeAlias

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from core.python.config.config import Config, Flow, _resolve_abs_paths

# =========================================================
# CONSTANTS
# =========================================================

PRODUCTION_YAML_PATH = "modelzoo/classification/mobilenetv1/config/run.yaml"

REQUIRED_KEYS = {
    "out",
    "modelname",
}

OPTIONAL_KEYS = {
    "dvnc",
    "network",
    "dvconvert",
    "inference",
    "preprocess",
    "postprocess",
    "calibration_data",
    "precompiled_model_link_path",
    "dvsim",
    "log_level",
    "evaluation",
    "quantization",
}

# =========================================================
# HELPERS
# =========================================================

WriterFixture = Callable[[Path, dict[str, Any]], Path]

JSONPrimitive: TypeAlias = str | int | float | bool | None
JSONType: TypeAlias = JSONPrimitive | list["JSONType"] | dict[str, "JSONType"]


def make_serializable(obj: Any) -> JSONType:
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, Enum):
        return obj.value

    if isinstance(obj, Mapping):
        return {k: make_serializable(v) for k, v in obj.items()}

    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return [make_serializable(v) for v in obj]

    return obj


def load_production_config_raw() -> dict[str, Any]:
    cfg = Config.from_file(PRODUCTION_YAML_PATH)
    return cfg.model_dump(by_alias=True)


def load_production_config_dict() -> JSONType:
    cfg = Config.from_file(PRODUCTION_YAML_PATH)
    return make_serializable(cfg.model_dump(by_alias=True))


def write_json(tmp_path: Path, data: dict[str, Any]) -> Path:
    file_path = tmp_path / "config.json"
    file_path.write_text(json.dumps(data))
    return file_path


def write_yaml(tmp_path: Path, data: dict[str, Any]) -> Path:
    file_path = tmp_path / "config.yml"

    file_path.write_text(yaml.safe_dump(data, sort_keys=False))

    return file_path


def set_nested_key(data: dict, dotted_path: str, value: object) -> None:
    parts = dotted_path.split(".")

    current = data

    for part in parts[:-1]:
        current = current[part]

    current[parts[-1]] = value


def is_pydantic_model(tp: Any) -> bool:
    return isinstance(tp, type) and issubclass(tp, BaseModel)


def strip_optional_fields(
    model_cls: type[BaseModel],
    data: dict[str, Any],
) -> dict[str, Any]:
    """
    Recursively removes optional fields using Pydantic alias-aware keys.
    Produces a minimal required-only configuration dict.
    """

    result: dict[str, Any] = {}

    for name, field in model_cls.model_fields.items():
        # resolve actual key used in dict/YAML/JSON
        key = field.alias or name

        # skip optional fields
        if not field.is_required():
            continue

        # IMPORTANT: check alias key, not field name
        if key not in data:
            continue

        value = data[key]
        field_type = field.annotation

        if is_pydantic_model(field_type):
            result[key] = strip_optional_fields(
                field_type,
                value,
            )
        else:
            result[key] = value

    return result


def assert_optional_fields_respected(
    model_cls: type[BaseModel],
    instance: BaseModel,
) -> None:
    """
    Recursively ensures:
    - required fields exist
    - optional fields are safely defaulted or None
    - nested models are validated

    NOTE:
    - torch.device is treated as a special-case runtime default (CPU)
    """

    for name, field in model_cls.model_fields.items():
        value = getattr(instance, name)
        field_type = field.annotation

        # -----------------------------
        # recursive nested models
        # -----------------------------
        if is_pydantic_model(field_type):
            assert_optional_fields_respected(field_type, value)
            continue

        # -----------------------------
        # only validate optional fields
        # -----------------------------
        if field.is_required():
            continue

        assert value == field.default


# =========================================================
# FIXTURES
# =========================================================


@pytest.fixture
def production_config_raw():
    return load_production_config_raw()


@pytest.fixture
def production_config_serializable():
    return load_production_config_dict()


# =========================================================
# TEST MODELS
# =========================================================


class DatasetConfig(BaseModel):
    annotation_file: Path
    metadata_file: Path | None = None
    dataset_name: str


class TrainingConfig(BaseModel):
    dataset: DatasetConfig
    checkpoints: list[Path]
    artifacts: dict[str, Path]
    output_dir: Path

    epochs: int
    enabled: bool


# =========================================================
# SCHEMA TESTS
# =========================================================


class TestConfigSchema:
    def test_production_yaml_creates_valid_config(self):
        cfg = Config.from_file(PRODUCTION_YAML_PATH)

        assert isinstance(cfg, Config)

    def test_required_fields_exist_in_production_config(
        self,
        production_config_raw,
    ):
        for key in REQUIRED_KEYS:
            assert key in production_config_raw, f"Missing required key: {key}"

    def test_optional_fields_behavior(
        self,
        production_config_raw,
    ):
        cfg_dict = deepcopy(production_config_raw)

        for key in OPTIONAL_KEYS:
            cfg_dict.pop(key, None)

        cfg = Config.from_dict(cfg_dict)

        assert isinstance(cfg, Config)

        for key in OPTIONAL_KEYS:
            assert hasattr(cfg, key)

    def test_recursive_optional_fields_behavior(self, production_config_raw):
        cfg_dict = deepcopy(production_config_raw)
        cfg_dict = strip_optional_fields(Config, cfg_dict)

        cfg = Config.from_dict(cfg_dict)

        assert isinstance(cfg, Config)
        assert_optional_fields_respected(Config, cfg)

    def test_unknown_keys_raise_validation_error(
        self,
        production_config_raw,
    ):
        cfg_dict = deepcopy(production_config_raw)

        cfg_dict["random_noise_key"] = 123
        cfg_dict["another_fake_key"] = "abc"

        with pytest.raises(ValidationError):
            Config.from_dict(cfg_dict)


# =========================================================
# VALIDATION TESTS
# =========================================================


class TestConfigValidation:
    @pytest.mark.parametrize(
        "missing_key",
        sorted(REQUIRED_KEYS),
    )
    def test_missing_required_fields_fail(
        self,
        missing_key,
        production_config_raw,
    ):
        cfg_dict = deepcopy(production_config_raw)

        cfg_dict.pop(missing_key, None)

        with pytest.raises(ValidationError):
            Config.from_dict(cfg_dict)

    def test_empty_dict_fails(self):
        with pytest.raises(ValidationError):
            Config.from_dict({})

    def test_none_input_fails(self):
        with pytest.raises(Exception):
            Config.from_dict(None)

    @pytest.mark.parametrize(
        ("field_path", "value"),
        [
            # ("dvnc.qmode", 100000),
            # ("log_level", "any log level"),
            ("preprocess.input_shape.c", -10),
            ("preprocess.mean.r", -10),
            ("quantization.device", "cuda1"),
            ("quantization.scheme", "PQT"),
            ("postprocess.model_type", "psoe_estimation"),
            ("network.srcfw", "onnnx"),
            ("dvnc.layeroutput", "Hello"),
            ("preprocess.resize.padding", 1000),
        ],
    )
    def test_invalid_field_values_raise_validation_error(
        self,
        production_config_raw,
        field_path,
        value,
    ) -> None:
        cfg_dict = deepcopy(production_config_raw)

        set_nested_key(cfg_dict, field_path, value)

        with pytest.raises(ValidationError):
            Config.from_dict(cfg_dict)


# =========================================================
# IO TESTS
# =========================================================


class TestConfigIO:
    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            Config.from_file("non_existent.yaml")

    @pytest.mark.parametrize(
        "writer",
        [
            write_json,
            write_yaml,
        ],
    )
    def test_supported_file_formats_load_correctly(
        self,
        tmp_path: Path,
        writer: WriterFixture,
        production_config_serializable,
    ):
        data = deepcopy(production_config_serializable)

        file_path = writer(
            tmp_path,
            data,
        )

        cfg = Config.from_file(str(file_path))
        raw_cfg = Config.from_file(PRODUCTION_YAML_PATH)

        assert isinstance(cfg, Config)

        assert cfg.model_dump(
            by_alias=True,
            exclude_defaults=True,
        ) == raw_cfg.model_dump(
            by_alias=True,
            exclude_defaults=True,
        )

    def test_round_trip_consistency(self):
        cfg = Config.from_file(
            PRODUCTION_YAML_PATH,
        )

        dumped = cfg.model_dump(
            by_alias=True,
        )

        recreated = Config.from_dict(
            dumped,
        )

        assert isinstance(
            recreated,
            Config,
        )

        assert recreated == cfg

    def test_unsupported_format_raises_value_error(
        self,
        tmp_path: Path,
    ):
        file_path = tmp_path / "config.txt"

        file_path.write_text("invalid content")

        with pytest.raises(ValueError):
            Config.from_file(str(file_path))


# =========================================================
# PATH RESOLUTION TESTS
# =========================================================


class TestPathNormalization:
    def test_converts_paths_recursively(self, tmp_path: Path):
        root_dir = tmp_path / "configs"

        cfg = TrainingConfig(
            dataset=DatasetConfig(
                annotation_file=Path("../annotations/train.json"),
                metadata_file=Path("./metadata.json"),
                dataset_name="imagenet",
            ),
            checkpoints=[
                Path("checkpoints/epoch_1.bin"),
                Path("../shared/epoch_2.bin"),
            ],
            artifacts={
                "weights": Path("./weights.bin"),
                "labels": Path("../labels.txt"),
            },
            output_dir=Path("/already/absolute/output"),
            epochs=100,
            enabled=True,
        )

        _resolve_abs_paths(root_dir, cfg)

        assert (
            cfg.dataset.annotation_file
            == (root_dir / "../annotations/train.json").resolve()
        )

        assert cfg.dataset.metadata_file == (root_dir / "metadata.json").resolve()

        assert cfg.checkpoints == [
            (root_dir / "checkpoints/epoch_1.bin").resolve(),
            (root_dir / "../shared/epoch_2.bin").resolve(),
        ]

        assert cfg.artifacts == {
            "weights": (root_dir / "weights.bin").resolve(),
            "labels": (root_dir / "../labels.txt").resolve(),
        }

        # Already absolute paths should remain unchanged
        assert cfg.output_dir == Path("/already/absolute/output")

        assert cfg.dataset.dataset_name == "imagenet"
        assert cfg.epochs == 100
        assert cfg.enabled is True


# =========================================================
# validate_flow() TESTS
# =========================================================
class TestConfigValidateFlow:
    @pytest.fixture
    def minimal_config(self, production_config_raw):
        """Fixture with only common keys"""

        minimal_config = {}

        common_configs = ["modelname", "out", "preprocess"]

        for key in production_config_raw:
            if key in common_configs:
                minimal_config[key] = production_config_raw[key]

        return minimal_config

    # ==================== Full Config Tests ====================
    def test_validate_flow_all_flows_with_full_config(self, production_config_raw):
        """
        Test that all flows pass validation with a complete configuration.
        This combines: infer_float, infer_hw, eval_float, eval_hw, compile, quantization
        """
        # List of all flows to test
        all_flows = list(Flow)

        # Test each flow with the full configuration
        for flow in all_flows:
            # Should not raise any exception
            Config.validate_flow(production_config_raw, flow)

    def test_validate_flow_infer_float_with_minimal_config(self, minimal_config):
        """Test infer_float flow with only required keys"""
        config = minimal_config.copy()
        config["postprocess"] = {"model_type": "detection"}
        Config.validate_flow(config, Flow.INFER_FLOAT)

    def test_validate_flow_flow_none_with_default_parameter(self):
        """Test function with flow=None validates all keys"""
        empty_config = {}
        with pytest.raises(ValueError) as exc_info:
            Config.validate_flow(empty_config)
        assert "is not present in run.yaml and is required" in str(exc_info.value)

    # ==================== Edge Cases ====================

    def test_validate_flow_empty_config_with_flow_raises_error(self):
        """Test empty config with a flow raises error for first required key"""
        empty_config = {}

        with pytest.raises(ValueError) as exc_info:
            Config.validate_flow(empty_config, Flow.INFER_FLOAT)
        assert "modelname is not present" in str(exc_info.value)

    def test_validate_flow_invalid_flow_key_raises_key_error(self):
        """Test invalid flow string raises ValueError"""
        config = {"modelname": "test", "out": "out", "preprocess": "preprocess"}

        with pytest.raises(ValueError):
            Config.validate_flow(config, "invalid_flow")  # ty: ignore[invalid-argument-type]

    def test_validate_flow_error_message_format(self, minimal_config):
        """Test error message contains all required information"""
        with pytest.raises(ValueError) as exc_info:
            Config.validate_flow(minimal_config, Flow.INFER_FLOAT)

        error_msg = str(exc_info.value)
        assert "postprocess is not present in run.yaml" in error_msg
        assert "required for infer_float flow" in error_msg
        assert "postprocess cannot be None" in error_msg

    # ==================== Parametrized Tests ====================

    @pytest.mark.parametrize(
        "flow, required_keys",
        [
            (Flow.INFER_FLOAT, ["postprocess"]),
            (Flow.INFER_HW, ["dvconvert", "inference", "postprocess"]),
            (Flow.EVAL_FLOAT, ["postprocess", "evaluation"]),
            (
                Flow.EVAL_HW,
                ["dvconvert", "inference", "postprocess", "evaluation"],
            ),
            (Flow.COMPILE, ["network", "dvconvert", "dvnc", "calibration_data"]),
            (Flow.QUANTIZATION, ["postprocess", "evaluation", "quantization"]),
        ],
    )
    def test_validate_flow_flow_requires_specific_keys(
        self, minimal_config, flow, required_keys
    ):
        """Parametrized test to verify each flow requires specific keys"""
        config = minimal_config.copy()

        # Test that missing specific keys raises error
        for key in required_keys:
            with pytest.raises(ValueError) as exc_info:
                Config.validate_flow(config, flow)
                assert key in str(exc_info)
                assert flow in str(exc_info)

    @pytest.mark.parametrize(
        "flow, required_keys",
        [
            (Flow.INFER_FLOAT, ["postprocess"]),
            (Flow.INFER_HW, ["dvconvert", "inference", "postprocess"]),
            (Flow.EVAL_FLOAT, ["postprocess", "evaluation"]),
            (
                Flow.EVAL_HW,
                ["dvconvert", "inference", "postprocess", "evaluation"],
            ),
            (Flow.COMPILE, ["network", "dvconvert", "dvnc", "calibration_data"]),
            (Flow.QUANTIZATION, ["postprocess", "evaluation", "quantization"]),
        ],
    )
    def test_validate_flow_missing_required_key_raises_error(
        self, production_config_raw, flow, required_keys
    ):
        """Verify each required key raises a ValueError when missing from config.

        Tests both common keys (modelname, out, preprocess) and flow-specific keys
        by removing one key at a time from a full configuration.
        """
        # Include common keys required by every flow
        all_required = set(required_keys) | {"modelname", "out", "preprocess"}

        for key in sorted(all_required):
            config = deepcopy(production_config_raw)
            del config[key]

            with pytest.raises(ValueError) as exc_info:
                Config.validate_flow(config, flow)

            error_msg = str(exc_info.value)
            assert key in error_msg
            assert flow.value in error_msg
            assert "is not present" in error_msg

    @pytest.mark.parametrize(
        "flow",
        list(Flow),
    )
    def test_validate_flow_common_keys_required_for_all_flows(
        self, flow, minimal_config
    ):
        """Test that all common keys are required for every flow"""
        # Test by deleting modelname
        config = deepcopy(minimal_config)
        del config["modelname"]

        with pytest.raises(ValueError) as exc_info:
            Config.validate_flow(config, flow)
        assert "modelname is not present" in str(exc_info.value)
        assert flow in str(exc_info.value)

        # Test by deleting out
        config = deepcopy(minimal_config)
        del config["out"]

        with pytest.raises(ValueError) as exc_info:
            Config.validate_flow(config, flow)
        assert "out is not present" in str(exc_info.value)
        assert flow in str(exc_info.value)

        # Test by deleting preprocess
        config = deepcopy(minimal_config)
        del config["preprocess"]

        with pytest.raises(ValueError) as exc_info:
            Config.validate_flow(config, flow)
        assert "preprocess is not present" in str(exc_info.value)
        assert flow in str(exc_info.value)

    @pytest.mark.parametrize("key_to_make_none", ["modelname", "out", "preprocess"])
    def test_validate_flow_none_common_keys_raise_error_all_flows(
        self, minimal_config, key_to_make_none
    ):
        """Test that None values for common keys raise errors for all flows"""
        for flow in list(Flow):
            config = minimal_config.copy()
            config[key_to_make_none] = None

            with pytest.raises(ValueError) as exc_info:
                Config.validate_flow(config, flow)
            assert f"{key_to_make_none} is not present" in str(exc_info.value)
            assert flow in str(exc_info.value)
