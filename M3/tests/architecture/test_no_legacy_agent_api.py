"""Guard the irreversible Agent-only cut-over. / 保护不可逆的仅 Agent 架构切换。"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = PROJECT_ROOT / "spacers_agent"
PROMPTS_ROOT = PROJECT_ROOT / "prompts"
REMOVED_MODULES = ("workflow.py", "counting.py", "experts.py")
FORBIDDEN_RUNTIME_TOKENS = (
    "ExpertResult",
    "ExpertAssignment",
    "ExpertName",
    "CountingExpert",
    "CountingExpertAnswer",
    "attach_qwen_budget",
    "normalize_agent_name",
    "LEGACY_AGENT_NAME_ALIASES",
    "EXPERT_TO_AGENT",
    "AGENT_TO_EXPERT",
    "expert_result.json",
    "counting_expert",
    "change_expert",
    "grounding_expert",
    "spatial_expert",
    "general_vqa_expert",
    "caption_expert",
)


def test_removed_legacy_modules_do_not_exist() -> None:
    """Removed modules have no import-compatible placeholders. / 已删除模块没有兼容占位符。"""
    assert all(not (RUNTIME_ROOT / module).exists() for module in REMOVED_MODULES)


def test_runtime_and_prompts_contain_no_legacy_agent_api() -> None:
    """Code and active prompt assets reject the old vocabulary. / 代码和活动提示词拒绝旧词汇。"""
    violations: list[str] = []
    for root in (RUNTIME_ROOT, PROMPTS_ROOT):
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".md"}:
                continue
            text = path.read_text(encoding="utf-8")
            for token in FORBIDDEN_RUNTIME_TOKENS:
                if token in text:
                    violations.append(f"{path.relative_to(PROJECT_ROOT)}: {token}")
    assert not violations, "Legacy Agent API references remain:\n" + "\n".join(violations)
