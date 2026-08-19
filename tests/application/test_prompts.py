"""Contract tests for the versioned prompt catalog: bindings, missing-file
errors, cached texts, and snapshot paths.

版本化 Prompt 目录契约测试：绑定、缺失文件错误、缓存文本与快照路径。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from application.prompts import PromptCatalog, PromptNotFoundError

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_catalog_loads_all_bound_prompts() -> None:
    catalog = PromptCatalog(REPO_ROOT / "prompts")
    assert len(catalog.all_keys()) == 12
    for key in (
        "count_tile",
        "change",
        "general",
        "grounding",
        "caption",
        "seam",
        "visual_task_plan",
        "count_judge",
        "vqa_judge",
        "json_repair",
    ):
        assert catalog[key].strip()
        assert catalog.version(key)


def test_catalog_asset_and_versions() -> None:
    catalog = PromptCatalog(REPO_ROOT / "prompts")
    asset = catalog.asset("count_tile")
    assert asset.key == "count_tile"
    assert asset.version == "v4"
    assert asset.path.name == "count_tile_v4.md"
    assert catalog.version("count_tile") == "v4"
    assert catalog.version("general") == "v3"
    assert catalog.asset("general").path.name == "general_vqa_v3.md"
    assert catalog.version("visual_task_plan") == "v4"
    assert catalog.asset("visual_task_plan").path.name == "visual_task_plan_v4.md"
    assert catalog.version("vqa_judge") == "v2"
    assert catalog.version("seam") == "v2"
    assert catalog.version("change") == "v7"
    assert catalog.asset("change").path.name == "change_dual_path_v7.md"
    assert catalog.asset("seam").path.name == "seam_review_v2.md"
    assert catalog.asset("vqa_judge").path.name == "deepseek_vqa_judge_v2.md"
    assert (REPO_ROOT / "prompts" / "deepseek_vqa_judge_v1.md").is_file()


def test_catalog_snapshot_paths_stable_and_existing() -> None:
    catalog = PromptCatalog(REPO_ROOT / "prompts")
    paths = catalog.snapshot_paths()
    assert len(paths) == 11  # 12 keys, general_vqa_v3 shared by two keys
    assert all(path.is_file() for path in paths)
    assert catalog.snapshot_paths() == paths  # stable order / 稳定顺序


def test_vqa_judge_v2_declares_semantic_text_only_rules() -> None:
    prompt = PromptCatalog(REPO_ROOT / "prompts")["vqa_judge"].casefold()
    for required in (
        "meaning, not surface form",
        "question-sensitive",
        "number words and digits",
        "option label",
        "contradiction",
        "cannot inspect an image",
        "official reference answers are authoritative",
        "return json only",
    ):
        assert required in prompt


def test_change_prompt_v7_requires_factual_caption_not_class_label() -> None:
    prompt = PromptCatalog(REPO_ROOT / "prompts")["change"].casefold()
    for required in (
        "evidence-grounded semantic change analyst",
        "non-negotiable answer rule",
        "natural-language caption",
        "never use a standalone decision token",
        "must name at least one changed persistent entity",
        "no significant semantic change detected.",
    ):
        assert required in prompt
    for forbidden in (
        "balanced full-path semantic change analyst",
        "conservative full-path semantic change analyst",
    ):
        assert forbidden not in prompt


def test_change_prompt_v7_two_sided_confirmation_gate() -> None:
    prompt = PromptCatalog(REPO_ROOT / "prompts")["change"].casefold()
    for required in (
        "two-sided confirmation gate",
        "t1 state",
        "t2 state",
        "spatial correspondence",
        "persistence",
        "raw_full_t1",
        "raw_full_t2",
        "authoritative temporal pair",
        "image_manifest",
    ):
        assert required in prompt


def test_change_prompt_v7_mandatory_visual_scan() -> None:
    prompt = PromptCatalog(REPO_ROOT / "prompts")["change"].casefold()
    for required in (
        "mandatory visual scan",
        "3-by-3",
        "border and corner pass",
        "vegetation-extent pass",
        "artifact rejection pass",
        "new or removed buildings",
        "new/removed roads",
        "exact no-change sentence",
    ):
        assert required in prompt


def test_seam_review_v2_is_local_and_decision_only() -> None:
    prompt = PromptCatalog(REPO_ROOT / "prompts")["seam"].casefold()
    for required in (
        "local seam crop",
        "never count or rescan the full image",
        "same_instance",
        "different_instances",
        "uncertain",
        '{"decision":"same_instance"}',
    ):
        assert required in prompt
    for forbidden in ("canonical_point", "confidence", "short_reason"):
        assert forbidden not in prompt


def test_visual_task_plan_prompt_declares_visual_only_contract() -> None:
    """The active prompt accepts only images/raw text and emits v4 intent.
    active prompt 只接受图像/原始文本，并输出 v4 意图。"""
    prompt = PromptCatalog(REPO_ROOT / "prompts")["visual_task_plan"].casefold()
    for required in (
        "visual-task-plan-v4",
        "task",
        "needs_visual_assistance",
        "object_categories",
        "count_target",
        "region_request",
        "raw",
        "question",
        "ground truth",
        "image paths",
        "backend",
    ):
        assert required in prompt
    for forbidden in (
        "confidence",
        "probability",
        "certainty score",
        "uncertainty flag",
        "candidate task list",
    ):
        assert forbidden in prompt


def test_catalog_unknown_key_fails_stable() -> None:
    catalog = PromptCatalog(REPO_ROOT / "prompts")
    with pytest.raises(PromptNotFoundError, match="missing"):
        catalog["not_a_prompt"]
    with pytest.raises(PromptNotFoundError):
        catalog.asset("not_a_prompt")


def test_catalog_missing_file_fails_at_construction(tmp_path: Path) -> None:
    """A missing bound prompt file fails clearly at construction — never a
    silent empty text. 缺失的绑定 Prompt 文件在构造时明确失败——绝不静默给出
    空文本。"""
    prompts_root = tmp_path / "prompts"
    prompts_root.mkdir()
    (prompts_root / "caption_v1.md").write_text("caption", encoding="utf-8")
    with pytest.raises(PromptNotFoundError) as error:
        PromptCatalog(prompts_root)
    assert error.value.key == "count_tile"
    assert error.value.filename == "count_tile_v4.md"


def test_catalog_texts_are_cached_no_reread(tmp_path: Path) -> None:
    """Texts load once at construction; repeated access never re-reads the
    filesystem. 文本在构造时加载一次；重复访问绝不重读文件系统。"""
    prompts_root = tmp_path / "prompts"
    prompts_root.mkdir()
    for filename in (
        "count_tile_v4.md",
        "count_localize_v1.md",
        "target_parse_v1.md",
        "missing_point_review_v3.md",
            "change_dual_path_v7.md",
        "general_vqa_v3.md",
        "caption_v1.md",
        "seam_review_v2.md",
        "visual_task_plan_v4.md",
        "deepseek_judge_v1.md",
        "deepseek_vqa_judge_v2.md",
        "json_repair_v1.md",
    ):
        (prompts_root / filename).write_text(f"# {filename}\ncontent\n", encoding="utf-8")
    catalog = PromptCatalog(prompts_root)
    first = catalog["caption"]
    second = catalog["caption"]
    assert first == second
    # Deleting the file after construction must not affect cached access.
    # 构造后删除文件不得影响缓存访问。
    (prompts_root / "caption_v1.md").unlink()
    assert catalog["caption"] == first
