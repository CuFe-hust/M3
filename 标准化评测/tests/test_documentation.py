"""Documentation contract tests — Task 9 deliverables must contain required structure and keywords.

These tests fail before the documentation files are written (TDD Step 2),
then pass once the files are created with the required content.
"""

from __future__ import annotations

from pathlib import Path


# -------------------------------------------------------------------
# 测试说明.md contract
# -------------------------------------------------------------------

def test_manual_contains_all_required_workflows(project_root: Path) -> None:
    """测试说明.md must reference every canonical CLI invocation."""
    text = (project_root / "测试说明.md").read_text(encoding="utf-8")
    for required in ["doctor", "--mode smoke", "--mode full", "--resume", "rebuild-table", "prepare-report"]:
        assert required in text, f"测试说明.md missing required keyword: {required}"


def test_manual_contains_all_four_dataset_names(project_root: Path) -> None:
    """测试说明.md must name every fixed benchmark dataset."""
    text = (project_root / "测试说明.md").read_text(encoding="utf-8")
    for dataset in ["LEVIR-CC", "VRSBench", "XLRS-Bench", "MME-RealWorld-RS"]:
        assert dataset in text, f"测试说明.md missing dataset name: {dataset}"


def test_manual_contains_error_code_documentation(project_root: Path) -> None:
    """测试说明.md must explain the exit code meanings (0/1/2/3/4)."""
    text = (project_root / "测试说明.md").read_text(encoding="utf-8")
    for code in ["0", "1", "2", "3", "4"]:
        assert code in text, f"测试说明.md missing error code: {code}"


# -------------------------------------------------------------------
# prompts/生成评测对比报告提示词.md contract
# -------------------------------------------------------------------

def test_prompt_forbids_fabrication_and_requires_all_history(project_root: Path) -> None:
    """Prompt must forbid fabrication, require all compatible history, reference run_id and metric_id."""
    text = (project_root / "prompts" / "生成评测对比报告提示词.md").read_text(encoding="utf-8")
    assert "不得编造" in text, "Prompt must contain '不得编造'"
    assert "全部历史兼容版本" in text, "Prompt must contain '全部历史兼容版本'"
    assert "run_id" in text and "metric_id" in text, "Prompt must reference run_id and metric_id"


def test_prompt_requires_hypothesis_marking(project_root: Path) -> None:
    """Prompt must instruct the model to mark causal claims as assumptions (假设).

    Per design §18, reason analysis must be tagged as hypothesis with verification experiments.
    """
    text = (project_root / "prompts" / "生成评测对比报告提示词.md").read_text(encoding="utf-8")
    assert "假设" in text, "Prompt must require hypothesis marking (假设) for causal claims"


# -------------------------------------------------------------------
# README.md contract
# -------------------------------------------------------------------

def test_readme_contains_five_canonical_commands(project_root: Path) -> None:
    """README must present the five standard commands for quick start."""
    text = (project_root / "README.md").read_text(encoding="utf-8")
    for command in ["doctor", "run --mode smoke", "run --mode full", "rebuild-table", "prepare-report"]:
        assert command in text, f"README.md missing standard command: {command}"
