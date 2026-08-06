"""ArtifactWriter tests. / ArtifactWriter 测试。"""

import json
from pathlib import Path

from spacers_agent.agents.base import AgentExecution
from spacers_agent.routing.schemas import RoutingDecision
from spacers_agent.schemas import DatasetRunSummary, AgentResult, ImageRef, SampleRunStatus, UnifiedSample
from spacers_agent.workflows.artifact_writer import ArtifactWriter


def test_artifact_writer_persists_declared_artifacts(tmp_path: Path) -> None:
    writer = ArtifactWriter()
    run_dir = tmp_path / "run"
    sample_dir = run_dir / "samples" / "sample-1"
    sample = UnifiedSample(
        sample_id="sample-1",
        dataset="test",
        split="test",
        task="general_vqa",
        images=[ImageRef(image_id="image-1", path=tmp_path / "image.png", role="image")],
        question="What is visible?",
    )
    running = SampleRunStatus(
        sample_id=sample.sample_id,
        task=sample.task,
        state="running",
        updated_at="2026-07-26T00:00:00+00:00",
    )
    routing = RoutingDecision(
        task="general_vqa",
        primary_agent="general_vqa_agent",
        requires_tiling=False,
        reason_codes=["task_general_vqa"],
        router_source="dataset_task",
    )
    execution = AgentExecution(
        agent_name="general_vqa_agent",
        payload=AgentResult(agent_name="general_vqa_agent", answer="yes", status="completed"),
        result_filename="agent_result.json",
    )
    final = running.model_copy(
        update={"state": "succeeded", "result_path": sample_dir / "agent_result.json"}
    )
    summary = DatasetRunSummary(
        run_id="run",
        dataset="test",
        split="test",
        task="general_vqa",
        total=1,
        succeeded=1,
        partial=0,
        failed=0,
        skipped=0,
    )

    writer.write_sample(sample_dir, sample)
    writer.write_running_status(sample_dir, running)
    writer.write_routing(sample_dir, routing)
    writer.write_execution(sample_dir, execution)
    writer.write_evaluation(sample_dir, {"judge_status": "not_requested"}, filename="vqa_evaluation.json")
    writer.write_trace(sample_dir, {"route": "test"})
    writer.write_final_status(sample_dir, final)
    writer.append_prediction(
        run_dir,
        sample_id=sample.sample_id,
        task=sample.task,
        status=final,
    )
    writer.write_summary(run_dir, summary)

    assert json.loads((sample_dir / "status.json").read_text(encoding="utf-8"))["state"] == "succeeded"
    assert json.loads((sample_dir / "routing_decision.json").read_text(encoding="utf-8"))["primary_agent"] == "general_vqa_agent"
    assert json.loads((sample_dir / "agent_result.json").read_text(encoding="utf-8"))["answer"] == "yes"
    prediction = json.loads((run_dir / "predictions.jsonl").read_text(encoding="utf-8"))
    assert prediction["status"] == "succeeded"
    assert json.loads((run_dir / "dataset_summary.json").read_text(encoding="utf-8"))["succeeded"] == 1
