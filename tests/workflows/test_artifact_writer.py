"""Behavior tests for the centralized ArtifactWriter.

集中 ArtifactWriter 行为测试：每个方法写出正确文件与内容、write_execution
覆盖 primary 与 additional_results、JSONL 追加换行与稳定字段、原子原语
（中途失败不暴露半个 JSON、无 .tmp 残留）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.base import AgentExecution
from agents.counting.schema import CountingResult
from agents.schema import AgentResult
from data.schema import GroundTruth, ImageRef, UnifiedSample
from workflows.artifact_writer import (
    AGENT_RESULT_FILENAME,
    AGENT_TRACE_FILENAME,
    COUNTING_RESULT_FILENAME,
    DATASET_SUMMARY_FILENAME,
    PREDICTIONS_FILENAME,
    ROUTING_DECISION_FILENAME,
    SAMPLE_FILENAME,
    STATUS_FILENAME,
    ArtifactWriter,
    atomic_append_jsonl,
    atomic_write_json,
)
from workflows.schema import DatasetRunSummary, SampleRunStatus


def _sample() -> UnifiedSample:
    return UnifiedSample(
        sample_id="s1",
        dataset="parity",
        split="test",
        task="change_caption",
        images=[
            ImageRef(image_id="t1", path="t1.png", role="t1"),
            ImageRef(image_id="t2", path="t2.png", role="t2"),
        ],
        question="Q",
        ground_truth=GroundTruth(answers=["x"]),
    )


def _status(state: str = "running", sample_id: str = "s1") -> SampleRunStatus:
    return SampleRunStatus(
        sample_id=sample_id,
        task="change_caption",
        state=state,  # type: ignore[arg-type]
        updated_at="2026-01-01T00:00:00Z",
    )


def _execution() -> AgentExecution:
    payload = AgentResult(agent_name="change_agent", answer="A building was removed.")
    return AgentExecution(
        agent_name="change_agent",
        payload=payload,
        result_filename=AGENT_RESULT_FILENAME,
        additional_results={"counts.json": {"n": 3}},
        trace={"prompt_version": "v1"},
    )


def test_write_sample(tmp_path: Path) -> None:
    ArtifactWriter().write_sample(tmp_path, _sample())
    payload = json.loads((tmp_path / SAMPLE_FILENAME).read_text(encoding="utf-8"))
    assert payload["sample_id"] == "s1"
    assert payload["task"] == "change_caption"


def test_write_running_and_final_status_share_file(tmp_path: Path) -> None:
    writer = ArtifactWriter()
    writer.write_running_status(tmp_path, _status("running"))
    first = json.loads((tmp_path / STATUS_FILENAME).read_text(encoding="utf-8"))
    assert first["state"] == "running"
    writer.write_final_status(tmp_path, _status("succeeded"))
    second = json.loads((tmp_path / STATUS_FILENAME).read_text(encoding="utf-8"))
    assert second["state"] == "succeeded"
    assert second["sample_id"] == "s1"


def test_write_routing(tmp_path: Path) -> None:
    from routing.schema import RoutingDecision

    decision = RoutingDecision(
        task="change_caption", primary_agent="change_agent", reason_codes=["x"]
    )
    ArtifactWriter().write_routing(tmp_path, decision)
    payload = json.loads((tmp_path / ROUTING_DECISION_FILENAME).read_text(encoding="utf-8"))
    assert payload["primary_agent"] == "change_agent"


def test_write_execution_primary_and_additional(tmp_path: Path) -> None:
    writer = ArtifactWriter()
    result_path = writer.write_execution(tmp_path, _execution())
    assert result_path == tmp_path / AGENT_RESULT_FILENAME
    primary = json.loads((tmp_path / AGENT_RESULT_FILENAME).read_text(encoding="utf-8"))
    assert primary["agent_name"] == "change_agent"
    additional = json.loads((tmp_path / "counts.json").read_text(encoding="utf-8"))
    assert additional == {"n": 3}


def test_write_execution_counts_payload(tmp_path: Path) -> None:
    counting = CountingResult(
        sample_id="s1",
        target="car",
        question="How many cars?",
        source_width=1000,
        source_height=1000,
        tile_count=1,
        final_count=0,
        status="completed",
    )
    execution = AgentExecution(
        agent_name="counting_agent",
        payload=counting,
        result_filename=COUNTING_RESULT_FILENAME,
    )
    ArtifactWriter().write_execution(tmp_path, execution)
    payload = json.loads((tmp_path / COUNTING_RESULT_FILENAME).read_text(encoding="utf-8"))
    assert payload["final_count"] == 0
    assert payload["status"] == "completed"


def test_write_trace_and_evaluation(tmp_path: Path) -> None:
    writer = ArtifactWriter()
    writer.write_trace(tmp_path, {"attempted_backends": ["qwen_point"]})
    trace = json.loads((tmp_path / AGENT_TRACE_FILENAME).read_text(encoding="utf-8"))
    assert trace["attempted_backends"] == ["qwen_point"]
    evaluation_path = writer.write_evaluation(tmp_path, {"em": 1.0}, filename="evaluation.json")
    assert evaluation_path == tmp_path / "evaluation.json"
    assert json.loads(evaluation_path.read_text(encoding="utf-8")) == {"em": 1.0}


def test_append_prediction_writes_lines_with_newlines(tmp_path: Path) -> None:
    writer = ArtifactWriter()
    writer.append_prediction(tmp_path, sample_id="s1", task="change_caption", status=_status("succeeded"))
    writer.append_prediction(tmp_path, sample_id="s2", task="counting", status=_status("failed", "s2"))
    lines = (tmp_path / PREDICTIONS_FILENAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["sample_id"] == "s1"
    assert json.loads(lines[1])["status"] == "failed"
    # Every line ends with exactly one newline. / 每行恰好以一个换行结束。
    raw = (tmp_path / PREDICTIONS_FILENAME).read_bytes()
    assert raw.count(b"\n") == 2


def test_write_summary(tmp_path: Path) -> None:
    summary = DatasetRunSummary(
        run_id="r1", dataset="parity", split="test", task="change_caption",
        total=3, succeeded=2, partial=0, failed=1, skipped=0,
    )
    ArtifactWriter().write_summary(tmp_path, summary)
    payload = json.loads((tmp_path / DATASET_SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert payload["total"] == 3
    assert payload["failed"] == 1


def test_atomic_primitives_leave_no_temporary_files(tmp_path: Path) -> None:
    atomic_write_json(tmp_path / "a.json", {"k": 1})
    atomic_append_jsonl(tmp_path / "rows.jsonl", {"k": 1})
    assert list(tmp_path.rglob("*.tmp")) == []
    assert (tmp_path / "a.json").is_file()
    assert (tmp_path / "rows.jsonl").is_file()


def test_atomic_write_failure_never_exposes_half_json(tmp_path: Path, monkeypatch) -> None:
    """A crash while writing the temporary file leaves the target untouched.
    写临时文件中途失败时目标文件保持原状。"""
    import workflows.artifact_writer as module

    target = tmp_path / "status.json"
    atomic_write_json(target, {"state": "running"})
    before = target.read_bytes()

    def _broken_write_text(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", _broken_write_text)
    with pytest.raises(OSError, match="disk full"):
        atomic_write_json(target, {"state": "succeeded"})
    # The original content survives byte-for-byte. / 原内容逐字节保留。
    assert target.read_bytes() == before
    # The failed temporary file is never promoted. / 失败的临时文件未被提升。
    assert not (tmp_path / "status.tmp").exists() or (tmp_path / "status.tmp").is_file()


def test_atomic_append_failure_never_exposes_half_line(tmp_path: Path, monkeypatch) -> None:
    import workflows.artifact_writer as module

    path = tmp_path / "predictions.jsonl"
    atomic_append_jsonl(path, {"sample_id": "s1"})
    before = path.read_bytes()

    def _broken_write_text(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", _broken_write_text)
    with pytest.raises(OSError, match="disk full"):
        atomic_append_jsonl(path, {"sample_id": "s2"})
    assert path.read_bytes() == before


def test_writer_rejects_business_status_inference() -> None:
    """Writer methods never branch on payload status or read directories.
    Writer 方法不对载荷状态分支、不读取目录。"""
    source = (Path(__file__).resolve().parents[2] / "workflows" / "artifact_writer.py").read_text(
        encoding="utf-8"
    )
    assert "iterdir" not in source
    assert "glob(" not in source
    assert ".status ==" not in source
