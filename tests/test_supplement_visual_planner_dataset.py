"""Offline tests for structured visual-planner supplement compilation.
结构化 Visual Planner 补充编译器的离线测试。
"""

from __future__ import annotations

from pathlib import Path

from agents.evidence_catalog import EvidenceCatalog
from scripts.supplement_visual_planner_dataset import (
    _numeric_answer,
    _numeric_value,
    _valid_mc_answer,
    _ordered_categories,
    _question_evidence_categories,
    _CAPTION_QUESTIONS,
    _CHANGE_CAPTION_QUESTIONS,
    _CHANGE_QA_QUESTIONS,
    build_choices,
    format_multiple_choice,
    select_balanced_levir,
    validate_choices,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_vrs_categories_expand_generic_vehicle_in_catalog_order() -> None:
    categories = _ordered_categories(
        ("ship", "vehicle", "airplane"),
        ("plane", "small-vehicle", "large-vehicle", "ship"),
    )

    assert categories == ("plane", "small-vehicle", "large-vehicle", "ship")


def test_generic_caption_and_change_questions_never_enable_assistance() -> None:
    """Generic caption/change templates name no category, so assistance stays off.
    generic caption/change 模板不指明任何类别，因此 assistance 必须关闭。"""

    catalog = EvidenceCatalog.from_file(REPO_ROOT / "agents" / "evidence_catalog.json")
    questions = (*_CAPTION_QUESTIONS, *_CHANGE_CAPTION_QUESTIONS, *_CHANGE_QA_QUESTIONS)
    for question in questions:
        for task in ("caption", "change_caption", "change_qa"):
            categories = _question_evidence_categories(
                question,
                task=task,
                catalog=catalog,
                global_executable=catalog.leaf_categories,
            )
            assert categories == ()


def test_change_question_mentioning_category_can_enable_assistance() -> None:
    """A question that explicitly names a category may enable assistance.
    问题文本显式指明类别时才允许启用 assistance。"""

    catalog = EvidenceCatalog.from_file(REPO_ROOT / "agents" / "evidence_catalog.json")
    categories = _question_evidence_categories(
        "Did the buildings near the road change?",
        task="change_qa",
        catalog=catalog,
        global_executable=catalog.leaf_categories,
    )

    assert "building" in categories
    assert "road" in categories


def test_grounding_vehicle_scope_follows_referring_text() -> None:
    """Grounding categories must respect small/large scope in the referring text.
    Grounding 类别必须遵循 referring 文本中的 small/large 范围限定。"""

    catalog = EvidenceCatalog.from_file(REPO_ROOT / "agents" / "evidence_catalog.json")
    extract = lambda q: _question_evidence_categories(  # noqa: E731
        q, task="grounding", catalog=catalog, global_executable=catalog.leaf_categories
    )

    assert extract("The small vehicle is located in the top-right corner of the image") == (
        "small-vehicle",
    )
    assert extract("The large vehicle, which is a truck, positioned on the road") == (
        "large-vehicle",
        "road",
    )
    assert extract("The vehicle is located near the center of the image") == (
        "small-vehicle",
        "large-vehicle",
    )


def test_grounding_text_aliases_cover_synonym_expressions() -> None:
    """Synonym expressions map through explicit text aliases, not source obj_cls.
    同义表达通过显式文本 alias 映射，而不是依赖隐藏的源 obj_cls。"""

    catalog = EvidenceCatalog.from_file(REPO_ROOT / "agents" / "evidence_catalog.json")
    extract = lambda q: _question_evidence_categories(  # noqa: E731
        q, task="grounding", catalog=catalog, global_executable=catalog.leaf_categories
    )

    assert extract("The baseball field is located at the top-left of the image") == (
        "baseball-diamond",
    )
    assert extract("The track and field surrounded by green grass") == (
        "ground-track-field",
    )
    assert extract("The large truck parked on the middle-right side") == ("large-vehicle",)


def test_grounding_referring_without_category_closes_assistance() -> None:
    """A referring text with no inferable category must close assistance.
    referring 文本无可推导类别时关闭 assistance。"""

    catalog = EvidenceCatalog.from_file(REPO_ROOT / "agents" / "evidence_catalog.json")
    categories = _question_evidence_categories(
        "Located in the middle left of the image",
        task="grounding",
        catalog=catalog,
        global_executable=catalog.leaf_categories,
    )

    assert categories == ()


def test_multiple_choice_options_are_deterministic_and_closed() -> None:
    first = build_choices(
        "red",
        ("blue", "green", "red", "yellow"),
        identity="sample-1",
    )
    second = build_choices(
        "red",
        ("blue", "green", "red", "yellow"),
        identity="sample-1",
    )

    assert first == second
    assert first is not None and len(first) == 4 and "red" in first
    rendered = format_multiple_choice("What color is the plane?", first)
    assert "Choices: (A)" in rendered
    assert rendered.count("(") == 4


def test_binary_multiple_choice_uses_two_options() -> None:
    choices = build_choices(
        "Yes",
        ("No", "Yes", "Industrial area"),
        identity="binary",
    )

    assert choices is not None
    assert set(choices) == {"Yes", "No"}


def test_quantity_answer_shape_rejects_non_numeric_labels() -> None:
    assert _numeric_answer("Seven")
    assert _numeric_answer("twenty one")
    assert _numeric_answer("12")
    assert not _numeric_answer("fields")
    assert _numeric_value("6") == _numeric_value("Six") == 6


def test_numeric_choices_remove_semantically_duplicate_numbers() -> None:
    choices = build_choices(
        "6",
        ("Six", "2", "Two", "7", "Seven", "10"),
        identity="numeric-dedup",
    )

    assert choices is not None and len(choices) == 4
    assert len({_numeric_value(choice) for choice in choices}) == 4


def test_mc_answer_type_gate_keeps_only_compatible_closed_spaces() -> None:
    assert _valid_mc_answer("object quantity", "Seven")
    assert not _valid_mc_answer("object quantity", "fields")
    assert _valid_mc_answer("object existence", "Yes")
    assert not _valid_mc_answer("object existence", "A bridge")
    assert _valid_mc_answer("object color", "Brown, gray, and blue")
    assert not _valid_mc_answer("object position", "bottom-left")


def test_choice_validation_rejects_semantically_duplicate_numbers() -> None:
    import pytest

    validate_choices("object quantity", ("1", "Two", "3", "four"))
    with pytest.raises(ValueError, match="INCOMPATIBLE_QUANTITY_CHOICES"):
        validate_choices("object quantity", ("1", "one", "2", "3"))


def test_levir_selection_is_balanced_deterministic_and_excludes_test() -> None:
    rows = [
        {"imgid": index, "filename": f"{index}.png", "split": "train", "changeflag": index % 2}
        for index in range(12)
    ] + [
        {"imgid": 100 + index, "filename": f"t{index}.png", "split": "test", "changeflag": index % 2}
        for index in range(4)
    ]

    selected = select_balanced_levir(rows, split="train", quota=6)

    assert selected == select_balanced_levir(rows, split="train", quota=6)
    assert len(selected) == 6
    assert sum(row["changeflag"] == 0 for row in selected) == 3
    assert sum(row["changeflag"] == 1 for row in selected) == 3
    assert all(row["split"] == "train" for row in selected)
