"""Contract tests for the VRSBench ontology and task normalizer.

VRSBench 类别体系与任务规范化测试：subtype 分类、标准任务映射、
结构化提示字典、保守 fallback，以及与 Golden 规范化的对齐。
不调用模型，不返回 AgentName/backend 名。
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from data.adapters.vrsbench import (
    LARGE_VEHICLE_ALIASES,
    SMALL_VEHICLE_ALIASES,
    canonical_vehicle_class,
    classify_question_subtype,
    count_target_hint,
    normalize_task,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN = json.loads(
    (REPO_ROOT / "tests" / "fixtures" / "migration" / "vrsbench_normalization.json")
    .read_text(encoding="utf-8")
)


# ── Golden alignment / 与 Golden 对齐 ───────────────────────────────────────


def test_golden_questions_normalize_to_the_same_tasks() -> None:
    for record in GOLDEN:
        norm = normalize_task(record["question"])
        assert norm.normalized_task == record["normalized_task"], record["question"]
        assert norm.semantic_subtype == record["semantic_subtype"], record["question"]
        assert norm.reason_codes == record["reason_codes"], record["question"]
        assert norm.normalizer == "vrsbench_task_normalizer"
        assert norm.version == "1"
        assert norm.source_task == "vrsbench_vqa"
        assert norm.confidence == 1.0


# ── Subtype classification / subtype 分类 ───────────────────────────────────


def test_counting_subtype() -> None:
    assert classify_question_subtype("How many small vehicles are in the image?") == "counting"
    assert classify_question_subtype("What is the total number of buildings?") == "counting"
    assert classify_question_subtype("Count the ships.") == "counting"


def test_extreme_category_subtype() -> None:
    assert classify_question_subtype("What category is the topmost vehicle?") == "extreme_category"
    assert classify_question_subtype("What is the class of the bottom-most vehicle?") == "extreme_category"
    # topmost without a class request is not extreme_category. / 仅有 topmost 而无类别询问不算。
    assert classify_question_subtype("What is the topmost vehicle?") != "extreme_category"


def test_grid_position_subtype() -> None:
    assert classify_question_subtype("Where is the large vehicle located in the image?") == "grid_position"
    assert classify_question_subtype("What is the position of the building?") == "grid_position"
    assert classify_question_subtype("Where is the ship relative to the plane?") != "grid_position"


def test_orientation_and_arrangement_subtypes() -> None:
    assert classify_question_subtype("What is the orientation of the plane?") == "orientation"
    assert classify_question_subtype("What direction is the ship facing?") == "orientation"
    assert classify_question_subtype("What is the arrangement of the vehicles?") == "arrangement"
    assert classify_question_subtype("How are the buildings arranged?") == "arrangement"


def test_yes_no_subtypes() -> None:
    assert classify_question_subtype("Are there any small vehicles?") == "existence"
    assert classify_question_subtype("Is there a ship near the plane?") == "proximity"
    assert classify_question_subtype("Is the topmost vehicle a truck?") == "extreme_existence"


def test_color_and_category_subtypes() -> None:
    assert classify_question_subtype("What color is the building?") == "color"
    assert classify_question_subtype("What colour are the cars?") == "color"
    assert classify_question_subtype("What type of vehicle is it?") == "category"


def test_general_subtype_and_unknown_fallback() -> None:
    assert classify_question_subtype("Describe the scene.") == "general"
    assert classify_question_subtype("What is happening in this image?") == "general"
    assert classify_question_subtype("Totally unrelated text here.") == "general"


# ── Task mapping / 标准任务映射 ─────────────────────────────────────────────


def test_counting_questions_normalize_to_counting() -> None:
    norm = normalize_task("How many small vehicles are in the image?")
    assert norm.normalized_task == "counting"
    assert norm.reason_codes == ["quantity_question"]


def test_spatial_subtypes_normalize_to_spatial_relation() -> None:
    for question in (
        "What category is the topmost vehicle?",
        "Where is the large vehicle located in the image?",
        "What is the orientation of the plane?",
        "How are the vehicles arranged?",
    ):
        assert normalize_task(question).normalized_task == "spatial_relation", question


def test_other_subtypes_normalize_to_general_vqa() -> None:
    for question in (
        "Are there any small vehicles?",
        "Is the topmost vehicle a truck?",
        "Is there a ship near the plane?",
        "What color is the building?",
        "What type of vehicle is it?",
        "Describe the scene.",
    ):
        assert normalize_task(question).normalized_task == "general_vqa", question


def test_empty_question_falls_back_to_general_vqa() -> None:
    norm = normalize_task("   ")
    assert norm.normalized_task == "general_vqa"
    assert norm.confidence == 0.5
    assert norm.reason_codes == ["empty_question_fallback"]
    assert norm.semantic_subtype is None


# ── Structured hints / 结构化提示 ───────────────────────────────────────────


def test_spatial_query_is_structured_and_task_scoped() -> None:
    norm = normalize_task("What category is the topmost vehicle?")
    assert norm.spatial_query == {"operation": "extreme_category"}
    counting = normalize_task("How many small vehicles are in the image?")
    assert counting.spatial_query is None
    general = normalize_task("What color is the building?")
    assert general.spatial_query is None


def test_answer_constraints_for_yes_no_and_extreme() -> None:
    existence = normalize_task("Are there any small vehicles?")
    assert existence.answer_constraints == {
        "type": "closed_vocabulary", "values": ["yes", "no"], "closed": True,
    }
    extreme = normalize_task("What category is the topmost vehicle?")
    assert extreme.answer_constraints == {
        "type": "closed_vocabulary", "values": ["small-vehicle", "large-vehicle"], "closed": True,
    }
    general = normalize_task("What color is the building?")
    assert general.answer_constraints == {
        "type": "closed_vocabulary",
        "values": ["black", "blue", "brown", "gray", "green", "orange", "red", "white", "yellow"],
        "closed": True,
    }
    scene = normalize_task("Describe the scene.")
    assert scene.answer_constraints == {}


def test_spatial_closed_vocabularies() -> None:
    grid = normalize_task("Where is the large vehicle located in the image?")
    assert grid.answer_constraints["values"] == [
        "top-left", "top-middle", "top-right",
        "middle-left", "middle-middle", "middle-right",
        "bottom-left", "bottom-middle", "bottom-right",
    ]
    orientation = normalize_task("What is the orientation of the plane?")
    assert orientation.answer_constraints["values"] == ["north-south", "east-west"]
    arrangement = normalize_task("How are the vehicles arranged?")
    assert arrangement.answer_constraints["values"] == ["in rows", "clustered", "scattered"]


def test_count_target_hint_from_ontology() -> None:
    hint = normalize_task("How many small vehicles are in the image?").count_target_hint
    assert hint is not None
    assert hint["canonical_label"] == "small-vehicle"
    large = normalize_task("How many large vehicles are visible?").count_target_hint
    assert large is not None and large["canonical_label"] == "large-vehicle"
    generic = normalize_task("How many vehicles are visible?").count_target_hint
    assert generic is not None and generic["canonical_label"] == "vehicle"
    buildings = normalize_task("How many buildings are there?").count_target_hint
    assert buildings is None


# ── Vehicle ontology / 车辆类别体系 ─────────────────────────────────────────


def test_vehicle_alias_lists() -> None:
    assert "car" in SMALL_VEHICLE_ALIASES
    assert "truck" in LARGE_VEHICLE_ALIASES


def test_canonical_vehicle_class() -> None:
    assert canonical_vehicle_class("small vehicle") == "small-vehicle"
    assert canonical_vehicle_class("a large-vehicle") == "large-vehicle"
    assert canonical_vehicle_class("motorcycle") == "small-vehicle"
    assert canonical_vehicle_class("truck") == "large-vehicle"
    assert canonical_vehicle_class("building") is None


def test_count_target_hint_question_matching() -> None:
    assert count_target_hint("How many small vehicles are in the image?")["canonical_label"] == "small-vehicle"
    assert count_target_hint("Count the large-vehicles.")["canonical_label"] == "large-vehicle"
    assert count_target_hint("How many vehicles are visible?")["canonical_label"] == "vehicle"
    assert count_target_hint("How many buildings?") is None


# ── Frozen Golden / 冻结 Golden (B8) ────────────────────────────────────────


def test_frozen_task_normalization_golden() -> None:
    """16 representative questions locked against the legacy-derived golden.

    The legacy semantics (try_yolo @ ec962eb87c3ad0b8c1502efcbd08db0daec48868,
    vqa_geometry.vrsbench_question_subtype / execution_task_for_vrsbench) are
    frozen in tests/fixtures/migration/vrsbench_task_normalization_golden.json;
    the legacy modules are not importable from the new branch, so this golden
    is the long-term parity record.
    16 条代表性问题与源自 legacy 的冻结 Golden 对齐。legacy 语义冻结于
    vrsbench_task_normalization_golden.json（legacy 模块不可从新分支导入，
    以该 Golden 作为长期 parity 记录）。"""
    golden = json.loads(
        (REPO_ROOT / "tests" / "fixtures" / "migration" / "vrsbench_task_normalization_golden.json")
        .read_text(encoding="utf-8")
    )
    assert len(golden) == 16
    for record in golden:
        norm = normalize_task(record["question"])
        expected = record["expected"]
        assert norm.semantic_subtype == expected["semantic_subtype"], record["question"]
        assert norm.normalized_task == expected["normalized_task"], record["question"]
        assert norm.confidence == expected["confidence"], record["question"]
        assert norm.reason_codes == expected["reason_codes"], record["question"]
        assert norm.answer_constraints == expected["answer_constraints"], record["question"]
        assert norm.count_target_hint == expected["count_target_hint"], record["question"]


def test_golden_count_target_hints_are_complete() -> None:
    golden = json.loads(
        (REPO_ROOT / "tests" / "fixtures" / "migration" / "vrsbench_task_normalization_golden.json")
        .read_text(encoding="utf-8")
    )
    hints = {
        record["question"]: record["expected"]["count_target_hint"]
        for record in golden
        if record["expected"]["count_target_hint"] is not None
    }
    small = hints["How many small vehicles are in the image?"]
    assert small["canonical_label"] == "small-vehicle"
    assert "car" in small["aliases"] and "motorcycle" in small["aliases"]
    assert "truck" in small["exclusion_rule"]
    large = hints["How many large vehicles are visible?"]
    assert large["canonical_label"] == "large-vehicle"
    assert "car" in large["exclusion_rule"]
    all_vehicles = hints["How many vehicles are there?"]
    assert all_vehicles["canonical_label"] == "vehicle"
    assert "once" in all_vehicles["inclusion_rule"] or "once" in all_vehicles["exclusion_rule"]


def test_normalizer_never_imports_agents_or_backends() -> None:
    for relative in ("data/adapters/vrsbench/task_normalizer.py", "data/adapters/vrsbench/ontology.py"):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden = {"agents", "routing", "workflows", "evaluation", "reporting",
                     "application", "models", "spacers_agent", "eval"}
        tops = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                tops.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                tops.add(node.module.split(".")[0])
        assert not (tops & forbidden), f"{relative} imports forbidden: {tops & forbidden}"
        assert "AgentName" not in source and "Backend" not in source
