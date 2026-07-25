"""Freeze CLI parser contract — every subcommand must be constructable without network.
冻结 CLI 解析器契约 — 每个子命令必须可在不联网时构造。
"""

from __future__ import annotations

import pytest


# ── All expected subcommands / 所有预期子命令 ────────────────────────────

EXPECTED_COMMANDS = frozenset({
    "run-init",
    "health",
    "list-datasets",
    "smoke-qwen",
    "count-image",
    "run-dataset",
    "resume-run",
    "evaluate-run",
    "judge-vqa-run",
    "inspect-data",
    "render-count",
    "summarize-evaluations",
})


def _parser():
    """Build the CLI parser without side effects."""
    from spacers_agent.cli import build_parser
    return build_parser()


# ── Command existence / 命令存在性 ───────────────────────────────────────


def test_all_expected_commands_registered():
    """Every expected subcommand is registered."""
    parser = _parser()
    registered = set()
    for action in parser._actions:
        if hasattr(action, "choices") and action.choices is not None:
            registered.update(action.choices)
    missing = EXPECTED_COMMANDS - registered
    assert not missing, f"Missing CLI commands: {sorted(missing)}"


def test_cli_help_prints_without_error():
    """CLI --help prints without error."""
    parser = _parser()
    help_text = parser.format_help()
    assert "Remote-sensing" in help_text


# ── Subcommand argument contracts / 子命令参数契约 ────────────────────────


def test_run_dataset_required_args():
    """run-dataset requires --dataset and --root and --split."""
    parser = _parser()
    # Find the run-dataset subparser
    dataset_parser = _find_subparser(parser, "run-dataset")
    assert dataset_parser is not None, "run-dataset subcommand not found"

    # Check required args
    required = _required_args(dataset_parser)
    assert "--dataset" in required
    assert "--root" in required
    assert "--split" in required


def test_health_subcommands():
    """health accepts qwen and deepseek."""
    parser = _parser()
    health_parser = _find_subparser(parser, "health")
    assert health_parser is not None
    choices = _positional_choices(health_parser)
    assert "qwen" in choices
    assert "deepseek" in choices


def test_resume_run_arg_completeness():
    """resume-run has required --run-id."""
    parser = _parser()
    resume_parser = _find_subparser(parser, "resume-run")
    assert resume_parser is not None
    required = _required_args(resume_parser)
    assert "--run-id" in required


def test_count_image_has_required_args():
    """count-image has --image and --question."""
    # count_image uses a subparser factory from commands/; verify it exists
    parser = _parser()
    ci_parser = _find_subparser(parser, "count-image")
    assert ci_parser is not None


def test_evaluate_run_has_run_id():
    """evaluate-run has --run-id."""
    parser = _parser()
    eval_parser = _find_subparser(parser, "evaluate-run")
    assert eval_parser is not None
    assert "--run-id" in _required_args(eval_parser)


def test_judge_vqa_run_has_run_id():
    """judge-vqa-run has --run-id."""
    parser = _parser()
    judge_parser = _find_subparser(parser, "judge-vqa-run")
    assert judge_parser is not None
    assert "--run-id" in _required_args(judge_parser)


def test_inspect_data_has_required_args():
    """inspect-data has --root and --output."""
    parser = _parser()
    insp_parser = _find_subparser(parser, "inspect-data")
    assert insp_parser is not None
    required = _required_args(insp_parser)
    assert "--root" in required
    assert "--output" in required


def test_render_count_has_required_args():
    """render-count has --image, --result, --output."""
    parser = _parser()
    rc_parser = _find_subparser(parser, "render-count")
    assert rc_parser is not None
    required = _required_args(rc_parser)
    assert "--image" in required
    assert "--result" in required
    assert "--output" in required


def test_summarize_evaluations_has_required_args():
    """summarize-evaluations has --input and --output."""
    parser = _parser()
    se_parser = _find_subparser(parser, "summarize-evaluations")
    assert se_parser is not None
    required = _required_args(se_parser)
    assert "--input" in required
    assert "--output" in required


def test_run_init_optional_args():
    """run-init accepts --run-id, --dataset, --split, --sample-filter."""
    parser = _parser()
    ri_parser = _find_subparser(parser, "run-init")
    assert ri_parser is not None
    optional = {action.option_strings[0] for action in ri_parser._actions if action.option_strings}
    assert "--run-id" in optional
    assert "--dataset" in optional
    assert "--split" in optional
    assert "--sample-filter" in optional


# ── Helpers / 辅助函数 ──────────────────────────────────────────────────


def _find_subparser(parser, name):
    """Find a subparser by command name."""
    for action in parser._actions:
        if hasattr(action, "choices") and action.choices is not None:
            if name in action.choices:
                return action.choices[name]
    return None


def _required_args(subparser):
    """Return set of required argument flags."""
    required = set()
    for action in subparser._actions:
        if action.required and action.option_strings:
            required.add(action.option_strings[0])
    return required


def _positional_choices(subparser):
    """Return first positional's choices set."""
    for action in subparser._actions:
        if not action.option_strings and hasattr(action, "choices") and action.choices is not None:
            return set(action.choices)
    return set()
