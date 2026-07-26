"""Composition Root identity and Prompt binding tests. / 组合根身份与 Prompt 绑定测试。"""

from pathlib import Path

import pytest

from spacers_agent.bootstrap import assemble_runtime, build_dataset_runner
from spacers_agent.dataset_adapters import AdapterProbe
from spacers_agent.routing import CallBudget, TaskRouter
from spacers_agent.settings import AppSettings


class _FakeQwen:
    async def complete_json(self, **kwargs):
        raise AssertionError("bootstrap wiring must not call Qwen")


class _Adapter:
    name = "wiring-test"

    def probe(self, root: Path) -> AdapterProbe:
        return AdapterProbe(self.name, "1", root / "samples.json", (), 0)

    def iter_samples(self, root: Path, split: str, task: str):
        return iter(())


def test_bootstrap_wires_live_objects_without_dead_duplicates(tmp_path: Path) -> None:
    settings = AppSettings.model_validate(
        {"paths": {"dataset_root": tmp_path}, "runs": {"root": tmp_path / "runs"}}
    )
    runtime = assemble_runtime(settings, qwen_client=_FakeQwen())

    assert runtime.sample_runner.router is runtime.router
    assert runtime.sample_runner.agent_registry is runtime.agent_registry
    assert runtime.sample_runner.judge_service is runtime.judge_service
    assert runtime.sample_runner.artifact_writer is runtime.artifact_writer
    assert runtime.sample_runner.call_budget_factory is runtime.call_budget_factory
    assert runtime.judge_service.judge_client is runtime.judge_client

    dataset_runner = build_dataset_runner(
        runtime,
        adapter=_Adapter(),
        run_dir=tmp_path / "runs" / "run-1",
        settings=settings,
        judge_policy="all",
    )
    assert dataset_runner.sample_runner is runtime.sample_runner
    assert dataset_runner.artifact_writer is runtime.artifact_writer
    assert dataset_runner.judge_policy == "all"


def test_prompt_catalog_binds_active_files_and_request_versions() -> None:
    runtime = assemble_runtime(AppSettings(), qwen_client=_FakeQwen())
    expected = {
        "count": ("count_tile_v4.md", "count-point-v4"),
        "target": ("target_parse_v1.md", "target-parse-v1"),
        "seam": ("seam_verify_v1.md", "seam-verify-v1"),
        "zero_review": ("missing_point_review_v3.md", "missing-point-review-v3"),
        "vrsbench_proposal": ("general_vqa_v1.md", "general-vqa-v1-count-proposal"),
        "vrsbench_localizer": ("count_localize_v1.md", "count-localize-v1"),
        "change": ("change_v1.md", "change-expert-v1"),
        "spatial": ("spatial_v4.md", "spatial-v4"),
        "spatial_grid": ("spatial_v5.md", "spatial-v5"),
        "spatial_review": ("spatial_candidate_review_v2.md", "spatial-candidate-review-v2"),
        "spatial_grid_review": ("spatial_candidate_review_v3.md", "spatial-candidate-review-v3"),
        "general": ("general_vqa_v2.md", "general-vqa-v2"),
        "grounding": ("general_vqa_v2.md", "general-vqa-v2"),
        "caption": ("caption_v1.md", "caption-v1"),
        "count_judge": ("deepseek_judge_v1.md", "deepseek-judge-v1"),
        "vqa_judge": ("deepseek_vqa_judge_v1.md", "deepseek-vqa-judge-v1"),
    }

    for key, (filename, version) in expected.items():
        asset = runtime.prompt_catalog.asset(key)
        assert asset.path.name == filename
        assert asset.version == version
        assert asset.text == asset.path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_router_failure_uses_visible_reason_code(tmp_path: Path) -> None:
    sample = type(
        "UnknownSample",
        (),
        {
            "sample_id": "unknown-1",
            "dataset": "unknown",
            "task": "unknown",
            "question": "Route this request",
            "metadata": {},
        },
    )()
    decision = await TaskRouter(router_client=_FakeQwen(), router_prompt="route").route_sample(
        sample,
        budget=CallBudget(max_qwen_calls=1),
        artifact_dir=tmp_path,
    )

    assert decision.router_source == "rule_fallback"
    assert "rule_fallback" in decision.reason_codes
    assert "router_agent_failed_AssertionError" in decision.reason_codes
