import json

from PIL import Image

import main as main_module
from agents.langgraph_qwen import LangGraphQwenAgent
from data.schema import CanonicalPrediction, CanonicalSample


class FakeModel:
    def __init__(self, fail_ids: set[str] | None = None) -> None:
        self.fail_ids = fail_ids or set()
        self.called_ids: list[str] = []

    def predict(self, sample: CanonicalSample) -> CanonicalPrediction:
        self.called_ids.append(sample.id)
        if sample.id in self.fail_ids:
            raise RuntimeError(f"failure for {sample.id}")
        return CanonicalPrediction(
            id=sample.id,
            task_type=sample.task_type,
            text=f"prediction-{sample.id}",
            answer=f"prediction-{sample.id}",
        )


def _samples() -> list[CanonicalSample]:
    return [
        CanonicalSample(
            id=sample_id,
            task_type="vqa",
            images=[Image.new("RGB", (8, 8))],
            prompt="What is visible?",
            answers=["A"],
        )
        for sample_id in ("sample-1", "sample-2")
    ]


def test_inference_resume_skips_existing_predictions(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main_module, "load_samples", lambda *_: iter(_samples()))
    config = {"model": {"id": "fake-model"}}
    first_model = FakeModel()

    main_module._infer_target(
        "vrsbench_vqa", tmp_path, tmp_path, first_model, 1, False, False, config
    )

    resumed_model = FakeModel()
    main_module._infer_target(
        "vrsbench_vqa", tmp_path, tmp_path, resumed_model, None, False, True, config
    )

    result_path = tmp_path / "vrsbench_vqa.jsonl"
    records = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()]
    assert [record["sample"]["id"] for record in records] == ["sample-1", "sample-2"]
    assert resumed_model.called_ids == ["sample-2"]
    metadata = json.loads((tmp_path / "vrsbench_vqa.metadata.json").read_text(encoding="utf-8"))
    assert metadata["completed_samples"] == 2
    assert metadata["skipped_existing"] == 1


def test_inference_records_failure_and_continues(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main_module, "load_samples", lambda *_: iter(_samples()))
    model = FakeModel(fail_ids={"sample-1"})

    main_module._infer_target(
        "vrsbench_vqa", tmp_path, tmp_path, model, None, False, False, {"model": {}}
    )

    result = (tmp_path / "vrsbench_vqa.jsonl").read_text(encoding="utf-8")
    failures = (tmp_path / "vrsbench_vqa.failures.jsonl").read_text(encoding="utf-8")
    assert '"sample-2"' in result
    assert '"sample-1"' in failures
    metadata = json.loads((tmp_path / "vrsbench_vqa.metadata.json").read_text(encoding="utf-8"))
    assert metadata["completed_samples"] == 1
    assert metadata["failed_this_run"] == 1


def test_langgraph_agent_saves_inside_fixed_workflow() -> None:
    model = FakeModel()
    agent = LangGraphQwenAgent(model)
    sample = _samples()[0]
    saved: list[tuple[str, str, float]] = []

    prediction = agent.run(
        sample,
        lambda current_sample, current_prediction, elapsed: saved.append(
            (current_sample.id, current_prediction.text, elapsed)
        ),
    )

    assert prediction.text == "prediction-sample-1"
    assert saved[0][0:2] == ("sample-1", "prediction-sample-1")
    assert saved[0][2] >= 0.0
    sample.images[0].close()
