"""Contract tests for counting-domain settings.

计数域配置契约测试：结构校验不访问权重文件、YOLO class/alias/composite
配置严格校验、CountingSettings 确定性默认与交叉约束、无后端执行逻辑。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from agents.counting.settings import (
    AgentCountingSettings,
    CountingSettings,
    YoloCountingSettings,
    YoloDetectorSettings,
)


# ── CountingSettings / 计数配置 ────────────────────────────────────────────


def test_counting_settings_defaults() -> None:
    settings = CountingSettings()
    assert settings.tile_core_size == 896
    assert settings.halo_size == 128
    assert settings.sequential is True
    assert settings.concurrency == 1
    assert settings.max_recursive_depth == 2
    assert settings.prompt_version == "count-point-v4"
    assert settings.fallback_on_backend_unavailable is True
    assert settings.fallback_on_backend_error is True
    assert settings.verify_empty_detection is True
    assert settings.verify_empty_semantic is False
    assert settings.trust_empty_detection is False


def test_counting_settings_sequential_requires_single_concurrency() -> None:
    with pytest.raises(ValidationError, match="sequential counting"):
        CountingSettings(sequential=True, concurrency=4)


def test_counting_settings_recursive_depth_guard() -> None:
    with pytest.raises(ValidationError, match="max_recursive_depth"):
        CountingSettings(vrsbench_min_scan_depth=3)


def test_agent_counting_settings_default_backend() -> None:
    settings = AgentCountingSettings()
    assert settings.default_backend == "auto"
    assert AgentCountingSettings(default_backend="yolo_obb").default_backend == "yolo_obb"


# ── YoloDetectorSettings / YOLO 检测器配置 ─────────────────────────────────


def _detector(**overrides) -> YoloDetectorSettings:
    values = dict(
        name="det-a",
        weights=Path("/nonexistent/weights.pt"),  # never touched / 绝不访问
        model_id="m1",
        sha256="a" * 64,
        classes=["car", "truck"],
    )
    values.update(overrides)
    return YoloDetectorSettings(**values)


def test_detector_structural_validation_does_not_touch_weights() -> None:
    """Constructing a detector with a nonexistent weight path must succeed;
    validation is purely structural. 使用不存在的权重路径构造检测器必须
    成功——校验纯结构性的。"""
    detector = _detector()
    assert detector.name == "det-a"
    assert detector.enabled is False
    assert detector.sha256 == "a" * 64


def test_detector_normalizes_classes() -> None:
    detector = _detector(classes=[" Car ", "truck"])
    assert detector.classes == ["Car", "truck"]


def test_detector_rejects_uppercase_sha256() -> None:
    """The sha256 pattern is enforced at the field level; uppercase hex fails.
    sha256 模式在字段层强制；大写十六进制失败。"""
    with pytest.raises(ValidationError, match="pattern"):
        _detector(sha256="B" * 64)


def test_detector_rejects_duplicate_classes_after_normalization() -> None:
    with pytest.raises(ValidationError, match="unique after normalization"):
        _detector(classes=["car", "CAR"])


def test_detector_rejects_alias_to_unknown_class() -> None:
    with pytest.raises(ValidationError, match="unknown class"):
        _detector(aliases={"c": "plane"})


def test_detector_rejects_removed_composite_target_configuration() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        _detector(composite_targets={"all": []})


def test_detector_accepts_explicit_raw_model_alias() -> None:
    detector = _detector(aliases={"passenger-car": "car"})
    assert detector.aliases["passenger-car"] == "car"
    assert not hasattr(detector, "composite_targets")


@pytest.mark.parametrize("reserved", ["qwen_point", "vrsbench_qwen_count"])
def test_detector_rejects_reserved_names(reserved: str) -> None:
    with pytest.raises(ValidationError, match="reserved"):
        _detector(name=reserved)


def test_detector_rejects_bad_sha256() -> None:
    with pytest.raises(ValidationError):
        _detector(sha256="not-hex")


def test_detector_accepts_explicit_cpu_only_contract() -> None:
    detector = _detector(device="cpu", require_cuda=False, allow_cpu_fallback=False)
    assert detector.device == "cpu"
    assert detector.require_cuda is False
    assert detector.allow_cpu_fallback is False


def test_detector_rejects_cpu_mode_with_fallback_enabled() -> None:
    with pytest.raises(ValidationError, match="CPU-only mode"):
        _detector(device="cpu", require_cuda=False, allow_cpu_fallback=True)


def test_detector_keeps_generic_cuda_schema_capability() -> None:
    detector = _detector(device="0", require_cuda=True, allow_cpu_fallback=False)
    assert detector.device == "0"
    assert detector.require_cuda is True


# ── YoloCountingSettings / YOLO 计数配置 ───────────────────────────────────


def test_yolo_settings_do_not_embed_concrete_default_detector_inventory() -> None:
    settings = YoloCountingSettings()
    assert settings.enabled is False
    assert settings.detectors == []

    source = (
        Path(__file__).resolve().parents[3] / "agents" / "counting" / "settings.py"
    ).read_text(encoding="utf-8")
    assert "_default_yolo_detectors" not in source
    assert "detector_obb_csl_001" not in source
    assert "DOTA-v2.0" not in source


def test_yolo_counting_settings_requires_unique_detector_names() -> None:
    with pytest.raises(ValidationError, match="names must be unique"):
        YoloCountingSettings(detectors=[_detector(), _detector()])


def test_yolo_counting_settings_enabled_requires_enabled_detector() -> None:
    with pytest.raises(ValidationError, match="at least one enabled detector"):
        YoloCountingSettings(enabled=True, detectors=[_detector()])
    settings = YoloCountingSettings(
        enabled=True, detectors=[_detector(enabled=True)]
    )
    assert settings.enabled is True


# ── 边界 / boundaries ──────────────────────────────────────────────────────


def test_settings_modules_have_no_backend_execution() -> None:
    """Settings must not define backend selection or execution logic.
    配置不得定义后端选择或执行逻辑。"""
    source = (Path(__file__).resolve().parents[3] / "agents" / "counting" / "settings.py").read_text(
        encoding="utf-8"
    )
    assert "BackendSelector" not in source
    assert "spacers_agent" not in source
