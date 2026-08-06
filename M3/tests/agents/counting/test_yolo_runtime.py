"""YOLO planning tests without real weights or Ultralytics imports.
不使用真实权重或 Ultralytics 导入的 YOLO 计划测试。
"""

from pathlib import Path

import pytest

from spacers_agent.agents.counting.backends.qwen_point import QwenPointCountingBackend
from spacers_agent.agents.counting.backends.registry import BackendRegistry
from spacers_agent.agents.counting.backends.selector import BackendSelector
from spacers_agent.agents.counting.backends.vrsbench_qwen_count import VRSBenchQwenCountBackend
from spacers_agent.agents.counting.backends.yolo_obb import YoloOBBCountingBackend
from spacers_agent.agents.counting.backends.yolo_obb import _detector_duplicate_pairs, _is_clipped_border_fragment
from spacers_agent.schemas import CountTargetSpec, GlobalPointObservation, ImageRef, PointProvenance, UnifiedSample, YoloDetectorSettings, YoloCountingSettings


def _target(label: str) -> CountTargetSpec:
    return CountTargetSpec(canonical_label=label, inclusion_rule="count it", exclusion_rule="none")


def _sample(task: str = "counting", *, dataset: str = "fixture") -> UnifiedSample:
    return UnifiedSample(sample_id="sample", dataset=dataset, split="test", task=task, images=[ImageRef(image_id="image", path=Path("image.png"), role="image")], question="How many ships?")


def _registry() -> BackendRegistry:
    registry = BackendRegistry()
    registry.register(QwenPointCountingBackend(object(), system_prompt="", settings=type("S", (), {"counting": object(), "models": type("M", (), {"qwen": object()})()})()))
    detector = YoloDetectorSettings(name="yolo26s_dota_obb", enabled=True, weights=Path("missing.pt"), model_id="test-yolo", sha256="0" * 64, classes=["plane", "ship", "small vehicle", "large vehicle"], composite_targets={"vehicle": ["small vehicle", "large vehicle"]})
    registry.register(YoloOBBCountingBackend(detector))
    return registry


def test_enabled_yolo_requires_enabled_detector() -> None:
    with pytest.raises(ValueError, match="enabled detector"):
        YoloCountingSettings(enabled=True)


def test_yolo_schema_rejects_unknown_alias_target() -> None:
    with pytest.raises(ValueError, match="unknown class"):
        YoloDetectorSettings(name="detector", weights=Path("x.pt"), model_id="test", sha256="0" * 64, classes=["ship"], aliases={"boat": "plane"})


def test_auto_selects_yolo_for_ship_and_qwen_for_building() -> None:
    selector = BackendSelector(_registry(), default_backend="auto")
    assert selector.plan(_target("ship"), _sample()).primary_backend_name == "yolo26s_dota_obb"
    assert selector.plan(_target("building"), _sample()).primary_backend_name == "qwen_point"


def test_explicit_qwen_ignores_yolo() -> None:
    assert BackendSelector(_registry(), default_backend="qwen_point").plan(_target("ship"), _sample()).primary_backend_name == "qwen_point"


def test_vrsbench_quantity_uses_yolo_with_dedicated_qwen_fallback() -> None:
    registry = _registry()
    registry.register(VRSBenchQwenCountBackend(object(), settings=object(), prompts={"count_proposal": "", "count_localize": ""}))
    sample = _sample("general_vqa", dataset="VRSBench")
    plan = BackendSelector(registry).plan(_target("vehicle"), sample)
    assert plan.primary_backend_name == "yolo26s_dota_obb"
    assert plan.fallback_backend_names == ("vrsbench_qwen_count",)


def test_vrsbench_quantity_explicit_qwen_preserves_dedicated_backend() -> None:
    registry = _registry()
    registry.register(VRSBenchQwenCountBackend(object(), settings=object(), prompts={"count_proposal": "", "count_localize": ""}))
    sample = _sample("general_vqa", dataset="VRSBench")
    plan = BackendSelector(registry, default_backend="qwen_point").plan(_target("vehicle"), sample)
    assert plan.primary_backend_name == "vrsbench_qwen_count"


def _point(name: str, x: int, y: int, polygon: list[list[float]]) -> GlobalPointObservation:
    return GlobalPointObservation(
        global_id=name, target="small-vehicle", source_tile_id="r000_c000", local_id=name,
        local_x_norm=x, local_y_norm=y, local_radius_norm=10,
        global_x_px=round(x * 511 / 999), global_y_px=round(y * 511 / 999),
        global_x_norm=x, global_y_norm=y, radius_px=5.0, confidence=0.8,
        ownership_valid=True, near_core_boundary=y > 950, accepted=True,
        short_evidence="detector point",
        provenance=PointProvenance(
            source="yolo_obb_center", backend_name="yolo", model_id="model",
            source_class="small vehicle", detector_confidence=0.8,
            obb_polygon_local_px=polygon, obb_polygon_global_px=polygon,
            detector_task="obb", detector_source_dataset="DOTAv1", weights_sha256="0" * 64,
        ),
    )


def test_yolo_rejects_clipped_image_border_fragment() -> None:
    fragment = _point("fragment", 954, 991, [[476, 502], [499, 502], [499, 519], [476, 519]])
    interior = _point("interior", 895, 334, [[441, 149], [475, 149], [475, 193], [441, 193]])
    assert _is_clipped_border_fragment(fragment, 512, 512)
    assert not _is_clipped_border_fragment(interior, 512, 512)


def test_yolo_merges_same_tile_overlapping_obb_only_by_iou() -> None:
    detector = _registry().get("yolo26s_dota_obb")._detector
    first = _point("first", 500, 500, [[240, 240], [270, 240], [270, 270], [240, 270]])
    duplicate = _point("duplicate", 505, 505, [[242, 242], [272, 242], [272, 272], [242, 272]])
    adjacent = _point("adjacent", 510, 500, [[271, 240], [301, 240], [301, 270], [271, 270]])
    merged, unresolved = _detector_duplicate_pairs([first, duplicate, adjacent], [], detector)
    assert ("first", "duplicate") in merged
    assert all("adjacent" not in pair for pair in merged)
    assert unresolved == []
