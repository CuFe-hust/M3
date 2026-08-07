"""Golden artifact filename and stable-field contracts.

Golden 产物文件名与稳定字段契约：集中拥有的 9 个文件名常量、与 Agent
实际声明文件名一致（agent_result.json / counting_result.json）、
predictions.jsonl 稳定字段、write_execution 覆盖 primary 与
additional_results、Writer 不做业务判断。
"""

from __future__ import annotations

from pathlib import Path

from agents.base import AgentExecution
from agents.counting.schema import CountingResult
from agents.schema import AgentResult
from data.schema import UnifiedSample
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
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_all_golden_filenames_are_declared() -> None:
    assert SAMPLE_FILENAME == "sample.json"
    assert STATUS_FILENAME == "status.json"
    assert ROUTING_DECISION_FILENAME == "routing_decision.json"
    assert AGENT_RESULT_FILENAME == "agent_result.json"
    assert COUNTING_RESULT_FILENAME == "counting_result.json"
    assert AGENT_TRACE_FILENAME == "agent_trace.json"
    assert PREDICTIONS_FILENAME == "predictions.jsonl"
    assert DATASET_SUMMARY_FILENAME == "dataset_summary.json"
    assert len(
        {
            SAMPLE_FILENAME,
            STATUS_FILENAME,
            ROUTING_DECISION_FILENAME,
            AGENT_RESULT_FILENAME,
            COUNTING_RESULT_FILENAME,
            AGENT_TRACE_FILENAME,
            PREDICTIONS_FILENAME,
            DATASET_SUMMARY_FILENAME,
        }
    ) == 8


def test_agent_result_filename_matches_visual_agents() -> None:
    """Visual agents declare the golden agent_result.json filename. The
    literal lives in VisualAgentBase; change_agent declares it explicitly.
    视觉 Agent 声明 golden 的 agent_result.json 文件名。字面量位于
    VisualAgentBase；change_agent 显式声明。"""
    assert AGENT_RESULT_FILENAME == "agent_result.json"
    for agent_file in ("agents/visual_base.py", "agents/change/agent.py"):
        source = (REPO_ROOT / agent_file).read_text(encoding="utf-8")
        assert '"agent_result.json"' in source, agent_file


def test_counting_result_filename_matches_counting_agent() -> None:
    """The counting agent declares the golden counting_result.json filename.
    计数 Agent 声明 golden 的 counting_result.json 文件名。"""
    assert COUNTING_RESULT_FILENAME == "counting_result.json"
    source = (REPO_ROOT / "agents/counting/agent.py").read_text(encoding="utf-8")
    assert '"counting_result.json"' in source


def test_execution_writes_primary_and_additional_results(tmp_path: Path) -> None:
    writer = ArtifactWriter()
    payload = AgentResult(agent_name="change_agent", answer="x")
    execution = AgentExecution(
        agent_name="change_agent",
        payload=payload,
        result_filename=AGENT_RESULT_FILENAME,
        additional_results={"counts.json": {"n": 3}},
    )
    result_path = writer.write_execution(tmp_path, execution)
    assert result_path == tmp_path / AGENT_RESULT_FILENAME
    assert (tmp_path / AGENT_RESULT_FILENAME).is_file()
    assert (tmp_path / "counts.json").is_file()


def test_execution_writes_counting_payload_under_counting_filename(tmp_path: Path) -> None:
    writer = ArtifactWriter()
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
    writer.write_execution(tmp_path, execution)
    assert (tmp_path / COUNTING_RESULT_FILENAME).is_file()


def test_append_prediction_stable_fields_only(tmp_path: Path) -> None:
    """A prediction row carries exactly the four stable fields.
    预测行恰好携带四个稳定字段。"""
    import json

    from workflows.schema import SampleRunStatus

    writer = ArtifactWriter()
    status = SampleRunStatus(
        sample_id="s1",
        task="change_caption",
        state="succeeded",
        result_path=tmp_path / "agent_result.json",
        updated_at="2026-01-01T00:00:00Z",
    )
    writer.append_prediction(tmp_path, sample_id="s1", task="change_caption", status=status)
    line = (tmp_path / PREDICTIONS_FILENAME).read_text(encoding="utf-8").strip()
    row = json.loads(line)
    assert row == {
        "sample_id": "s1",
        "task": "change_caption",
        "status": "succeeded",
        "result_path": str(tmp_path / "agent_result.json"),
    }


def test_writer_never_makes_business_decisions() -> None:
    """The writer must not compute metrics or infer state from response
    directories; it only persists what it is given.
    Writer 不计算指标、不从模型响应目录推断状态；只持久化传入内容。"""
    source = (REPO_ROOT / "workflows" / "artifact_writer.py").read_text(encoding="utf-8")
    for token in ("metric", "accuracy", "score =", "glob(", "iterdir", "success", "failed"):
        assert token not in source.casefold(), token
    # No branch on status values. / 不对状态值做分支。
    assert "status.state ==" not in source
    assert "state ==" not in source


def test_all_writes_go_through_atomic_primitives() -> None:
    """Every JSON write must route through atomic_write_json or
    atomic_append_jsonl; direct file writes are forbidden.
    所有 JSON 写入必须经由 atomic_write_json / atomic_append_jsonl；
    禁止直接写文件。"""
    source = (REPO_ROOT / "workflows" / "artifact_writer.py").read_text(encoding="utf-8")
    for token in ('"a"', 'newline', ".write(", "open(", "w+", "wb"):
        assert token not in source, token
