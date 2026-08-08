"""Contract tests for the offline HTML report renderer: escaping, no CDN /
Base64, deterministic output, and secret/path safety.

离线 HTML 报告渲染器契约测试：转义、无 CDN/Base64、确定性输出与密钥/路径
安全。直接构造持久化运行产物。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from agents.base import AgentExecution
from agents.schema import AgentResult
from data.schema import GroundTruth, ImageRef, UnifiedSample
from evaluation.records import EvaluationRecord, VQADeterministicMetrics
from reporting.builder import build_report
from reporting.html import build_html
from workflows.artifact_writer import ArtifactWriter
from workflows.run_store import RunStore
from workflows.schema import SampleRunStatus


def _storage_key(sample_id: str) -> str:
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:24]


def _build_escaping_run(tmp_path: Path) -> Path:
    store = RunStore(tmp_path / "runs", tmp_path)
    store.create_run(
        config_payload={"k": "v"},
        model_ids={"qwen": "q"},
        prompt_paths=[],
        run_id="html-run",
    )
    run_dir = tmp_path / "runs" / "html-run"
    writer = ArtifactWriter()
    sample = UnifiedSample(
        sample_id="evil-1",
        dataset="parity",
        split="test",
        task="general_vqa",
        images=[ImageRef(image_id="i0", path="img.png", role="image")],
        question='Is there a <script>alert(1)</script> road? & more "quotes"',
        ground_truth=GroundTruth(answers=["yes"]),
    )
    sample_dir = run_dir / "tasks" / "general_vqa" / "samples" / _storage_key("evil-1")
    writer.write_sample(sample_dir, sample)
    status = SampleRunStatus(
        sample_id="evil-1",
        task="general_vqa",
        state="succeeded",
        result_path=Path("agent_result.json"),
        updated_at="2026-01-01T00:00:00Z",
    )
    writer.write_final_status(sample_dir, status)
    writer.write_trace(
        sample_dir,
        {
            "resolved_task": "general_vqa",
            "execution_task": "general_vqa",
            "execution_agent": "general_vqa_agent",
            "fallback_used": False,
            "judge_status": "not_requested",
            "inference_seconds": 0.5,
        },
    )
    writer.write_evaluation(
        sample_dir,
        EvaluationRecord(
            sample_id="evil-1",
            task="general_vqa",
            deterministic_metrics=VQADeterministicMetrics(exact_match=True),
            judge_status="not_requested",
        ),
        filename="vqa_evaluation.json",
    )
    writer.write_execution(
        sample_dir,
        AgentExecution(
            agent_name="general_vqa_agent",
            payload=AgentResult(
                agent_name="general_vqa_agent",
                answer="yes <b>bold</b> & <i>italic</i>",
            ),
            result_filename="agent_result.json",
        ),
    )
    writer.append_prediction(
        run_dir,
        sample_id="evil-1",
        run_task="general_vqa",
        task="general_vqa",
        status=status,
        result_path=f"tasks/general_vqa/samples/{_storage_key('evil-1')}/agent_result.json",
    )
    return run_dir


def test_html_escapes_user_and_model_text(tmp_path: Path) -> None:
    document = build_html(build_report(_build_escaping_run(tmp_path)))
    assert "<script>alert(1)</script>" not in document
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in document
    assert "<b>bold</b>" not in document
    assert "&lt;b&gt;bold&lt;/b&gt;" in document
    assert "&quot;quotes&quot;" in document  # attribute-safe quoting / 属性安全引号


def test_html_is_offline_without_cdn_or_base64(tmp_path: Path) -> None:
    document = build_html(build_report(_build_escaping_run(tmp_path)))
    assert "http://" not in document
    assert "https://" not in document
    assert "cdn" not in document.casefold()
    assert "data:image" not in document
    assert "base64," not in document
    assert "sk-" not in document
    assert str(tmp_path) not in document  # machine paths never rendered


def test_html_deterministic(tmp_path: Path) -> None:
    report = build_report(_build_escaping_run(tmp_path))
    assert build_html(report) == build_html(report)


def test_html_contains_stable_fields_only(tmp_path: Path) -> None:
    document = build_html(build_report(_build_escaping_run(tmp_path)))
    assert "html-run" in document
    assert "evil-1" in document
    assert "general_vqa_agent" in document
    assert "exact_match=True" in document
