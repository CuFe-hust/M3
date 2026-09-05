"""Contract tests for the MME-RealWorld remote-sensing adapter.

MME-RealWorld 遥感子集适配器测试：RS 筛选、原始选项追加、答案格式校验、
allow_multiple/subtask metadata、非遥感记录排除、Registry 注册。
不调用模型；不把正确答案写进 question；不构造 system instruction。
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from PIL import Image

from data.adapters.base import DatasetProbeError
from data.adapters.mme_realworld import MMERealWorldAdapter
from data.registry import DatasetRegistry, register_default_adapters

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_image(path: Path, seed: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (seed, seed * 2, seed * 4)).save(path)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _rs_row(**overrides) -> dict:
    row = {
        "Question_id": "rs_1",
        "Subtask": "Remote Sensing",
        "Text": "How many ships are visible?",
        "Answer choices": ["0", "1", "2", "3", "4"],
        "Ground truth": "B",
        "image": "img_1.png",
    }
    row.update(overrides)
    return row


def _non_rs_row(**overrides) -> dict:
    row = {
        "Question_id": "oc_1",
        "Subtask": "Optical Character Recognition",
        "Text": "What does the sign say?",
        "Answer choices": ["A", "B", "C", "D"],
        "Ground truth": "A",
        "image": "img_1.png",
    }
    row.update(overrides)
    return row


def _build_root(tmp_path: Path) -> Path:
    root = tmp_path / "mme"
    _make_image(root / "img_1.png")
    _write_json(root / "MME_RealWorld.json", [_rs_row(), _non_rs_row()])
    return root


# ── RS 筛选 / remote-sensing filtering ─────────────────────────────────────


def test_non_remote_sensing_records_are_excluded(tmp_path: Path) -> None:
    root = _build_root(tmp_path)
    samples = list(MMERealWorldAdapter().iter_samples(root, "test", "multiple_choice_vqa"))
    assert len(samples) == 1
    assert samples[0].sample_id == "rs_1"


def test_subtask_underscore_and_question_id_matching(tmp_path: Path) -> None:
    root = tmp_path / "mme_match"
    _make_image(root / "img_1.png")
    _write_json(root / "MME_RealWorld.json", [
        _rs_row(Subtask="Remote_sensing"),
        _rs_row(Question_id="remote_sensing_42", Subtask="Other", Text="Q2",
                **{"Ground truth": "A"}, image="img_1.png"),
        _non_rs_row(),
    ])
    samples = list(MMERealWorldAdapter().iter_samples(root, "test", "multiple_choice_vqa"))
    assert len(samples) == 2
    assert {sample.sample_id for sample in samples} == {"rs_1", "remote_sensing_42"}


def test_probe_counts_only_remote_sensing_rows(tmp_path: Path) -> None:
    root = _build_root(tmp_path)
    probe = MMERealWorldAdapter().probe(root)
    assert probe.dataset == "MME-RealWorld"
    assert probe.sample_count == 1
    assert "Text" in probe.observed_fields


# ── 输出事实 / output facts ─────────────────────────────────────────────────


def test_question_and_choices_facts_are_preserved(tmp_path: Path) -> None:
    root = _build_root(tmp_path)
    sample = next(MMERealWorldAdapter().iter_samples(root, "test", "multiple_choice_vqa"))
    assert sample.task == "multiple_choice_vqa"
    assert sample.question == "How many ships are visible?\n0\n1\n2\n3\n4"
    assert sample.normalization is not None
    assert sample.normalization.choices == ["0", "1", "2", "3", "4"]
    assert sample.ground_truth is not None
    assert sample.ground_truth.answers == ["B"]
    assert sample.ground_truth.raw["choices"] == ["0", "1", "2", "3", "4"]
    # The answer key is not appended; only source question and choices appear.
    # 不追加答案键；question 只包含源问题和选项。
    assert sample.question.splitlines() == [
        "How many ships are visible?", "0", "1", "2", "3", "4"
    ]


def test_labeled_choices_are_appended_verbatim_in_source_order(tmp_path: Path) -> None:
    root = tmp_path / "mme_labeled"
    _make_image(root / "img_1.png")
    choices = ["(A) Yellow", "(B) Blue", "(C) Gray", "(D) White"]
    _write_json(
        root / "MME_RealWorld.json",
        [_rs_row(**{"Answer choices": choices, "Ground truth": "D"})],
    )

    sample = next(MMERealWorldAdapter().iter_samples(root, "test", "multiple_choice_vqa"))

    assert sample.question == "How many ships are visible?\n" + "\n".join(choices)
    assert sample.normalization is not None
    assert sample.normalization.choices == choices


def test_normalization_holds_choices_and_allow_multiple(tmp_path: Path) -> None:
    root = tmp_path / "mme_meta"
    _make_image(root / "img_1.png")
    _write_json(root / "MME_RealWorld.json", [
        _rs_row(Question_id="rs_a", allow_multiple=False),
        _rs_row(Question_id="rs_b", allow_multiple=True),
    ])
    samples = list(MMERealWorldAdapter().iter_samples(root, "test", "multiple_choice_vqa"))
    by_id = {sample.sample_id: sample for sample in samples}
    assert by_id["rs_a"].normalization is not None
    assert by_id["rs_b"].normalization is not None
    assert by_id["rs_a"].normalization.allow_multiple is False
    assert by_id["rs_b"].normalization.allow_multiple is True
    assert by_id["rs_a"].metadata["subtask"] == "remote sensing"


def test_answer_format_validation() -> None:
    adapter = MMERealWorldAdapter()
    for good in ("A", "B", "a", "C", "D"):
        adapter._validate_row(_rs_row(**{"Ground truth": good}), 0)
    adapter._validate_row(
        _rs_row(**{"Ground truth": "A, B"}, allow_multiple=True), 0
    )
    adapter._validate_row(
        _rs_row(**{"Ground truth": "A、B"}, allow_multiple=True), 0
    )
    for bad in ("", "AB", "1", "A-B", "A, B", "B and C", "A,A"):
        with pytest.raises(DatasetProbeError):
            adapter._validate_row(_rs_row(**{"Ground truth": bad}), 0)


def test_answer_outside_choice_count_fails() -> None:
    """Four choices allow A-D only; E must fail. / 4 个选项只允许 A–D；E 必须失败。"""
    adapter = MMERealWorldAdapter()
    four = _rs_row(**{"Answer choices": ["0", "1", "2", "3"]})
    adapter._validate_row(four, 0)  # answer B is valid / 答案 B 合法
    with pytest.raises(DatasetProbeError, match="invalid answer"):
        adapter._validate_row(_rs_row(**{"Answer choices": ["0", "1", "2", "3"], "Ground truth": "E"}), 0)


def test_missing_ground_truth_fails(tmp_path: Path) -> None:
    root = tmp_path / "mme_no_gt"
    _make_image(root / "img_1.png")
    row = _rs_row()
    del row["Ground truth"]
    _write_json(root / "MME_RealWorld.json", [row])
    with pytest.raises(DatasetProbeError, match="ground truth"):
        list(MMERealWorldAdapter().iter_samples(root, "test", "multiple_choice_vqa"))


def test_missing_image_fails(tmp_path: Path) -> None:
    root = tmp_path / "mme_missing_img"
    _write_json(root / "MME_RealWorld.json", [_rs_row()])
    with pytest.raises(DatasetProbeError, match="image is missing"):
        list(MMERealWorldAdapter().iter_samples(root, "test", "multiple_choice_vqa"))


def test_multiple_annotations_fail(tmp_path: Path) -> None:
    root = tmp_path / "mme_multi"
    _write_json(root / "a" / "MME_RealWorld.json", [_rs_row()])
    _write_json(root / "b" / "MME_RealWorld.json", [_rs_row()])
    with pytest.raises(DatasetProbeError, match="Expected exactly one"):
        MMERealWorldAdapter().probe(root)


def test_unsupported_task_fails(tmp_path: Path) -> None:
    root = _build_root(tmp_path)
    with pytest.raises(DatasetProbeError, match="not support"):
        list(MMERealWorldAdapter().iter_samples(root, "test", "general_vqa"))


def test_source_order_and_stable_ids(tmp_path: Path) -> None:
    root = tmp_path / "mme_order"
    _make_image(root / "img_1.png")
    _write_json(root / "MME_RealWorld.json", [
        _rs_row(Question_id="q1", Text="Q1"),
        _rs_row(Question_id="q2", Text="Q2"),
    ])
    first = list(MMERealWorldAdapter().iter_samples(root, "test", "multiple_choice_vqa"))
    second = list(MMERealWorldAdapter().iter_samples(root, "test", "multiple_choice_vqa"))
    assert [s.sample_id for s in first] == ["q1", "q2"]
    assert [s.sample_id for s in second] == [s.sample_id for s in first]


# ── Registry / 边界 ────────────────────────────────────────────────────────


def test_registry_registers_mme_realworld() -> None:
    registry = DatasetRegistry()
    register_default_adapters(registry)
    adapter = registry.get("MME-RealWorld")
    assert isinstance(adapter, MMERealWorldAdapter)
    assert registry.get("mme-realworld").name == "MME-RealWorld"
    assert registry.get("MME").name == "MME-RealWorld"
    assert "MME-RealWorld" in registry.names()


def test_adapter_never_builds_system_instructions(tmp_path: Path) -> None:
    source = (REPO_ROOT / "data" / "adapters" / "mme_realworld.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"agents", "routing", "workflows", "evaluation", "reporting",
                 "application", "models", "spacers_agent", "eval"}
    tops = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            tops.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            tops.add(node.module.split(".")[0])
    assert not (tops & forbidden), f"adapter imports forbidden: {tops & forbidden}"
    assert "prompt=" not in source and '"system"' not in source and "'system'" not in source
