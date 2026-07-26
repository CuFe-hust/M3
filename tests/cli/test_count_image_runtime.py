"""Regression coverage for the direct counting Runtime entry.
直接计数 Runtime 入口的回归覆盖。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from spacers_agent.commands import count_image
from spacers_agent.schemas import CountingResult
from spacers_agent.settings import AppSettings, RunSettings


@pytest.mark.asyncio
async def test_count_image_runs_a_canonical_sample_through_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """The direct command must delegate to the injected SampleRunner.
    直接命令必须委托给注入的 SampleRunner。
    """

    image_path = tmp_path / "sample.png"
    Image.new("RGB", (16, 12), "white").save(image_path)
    settings = AppSettings(runs=RunSettings(root=tmp_path))
    run_dir = tmp_path / "runtime-count"
    run_dir.mkdir()
    captured: dict[str, object] = {}
    result = CountingResult(
        sample_id="placeholder",
        target="building",
        question="How many buildings?",
        source_width=16,
        source_height=12,
        tile_count=1,
        final_count=0,
        status="completed",
    )

    class _Runner:
        async def run_one(self, sample, sample_dir, *, judge_policy: str):
            captured["sample"] = sample
            captured["sample_dir"] = sample_dir
            captured["judge_policy"] = judge_policy
            return SimpleNamespace(execution=SimpleNamespace(payload=result))

    runtime = SimpleNamespace(
        sample_runner=_Runner(),
        call_budget_factory=SimpleNamespace(default_qwen_calls=50, default_deepseek_calls=10),
    )
    monkeypatch.setattr(count_image, "qwen_client", lambda *_: object())
    def _assemble_runtime(passed_settings, **_):
        captured["settings"] = passed_settings
        return runtime

    monkeypatch.setattr(count_image, "assemble_runtime", _assemble_runtime)

    args = SimpleNamespace(
        image=image_path,
        question="How many buildings?",
        target_spec=None,
        run_id="runtime-count",
        evaluate=False,
        render=False,
        resume=False,
        force=False,
        no_seam_verify=False,
        max_qwen_calls=None,
        max_deepseek_calls=None,
    )

    assert await count_image._run(settings, args) == count_image.EXIT_OK
    sample = captured["sample"]
    assert sample.task == "counting"
    assert captured["judge_policy"] == "none"
