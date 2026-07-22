from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from spacers_agent.clients.base import RequestMeta, image_to_data_url
from spacers_agent.clients.mock import MockVisionClient
from spacers_agent.clients.qwen_transformers import QwenTransformersClient
from spacers_agent.dataset_adapters import get_adapter
from spacers_agent.evaluation import DeepSeekJudgeResult
from spacers_agent.schemas import ExpertResult
from spacers_agent.settings import AppSettings, PathSettings, QwenSettings, RunSettings
from spacers_agent.vqa_report import build_multiagent_vqa_report
from spacers_agent.workflow import DatasetRunner


class _FakeTensor:
    """Expose only tensor shape and slicing needed by the local client test.
    仅提供本地客户端测试所需的张量形状与切片行为。
    """

    def __init__(self, length: int) -> None:
        self.shape = (1, length)

    def __getitem__(self, item: slice) -> "_FakeTensor":
        start = int(item.start or 0)
        return _FakeTensor(max(0, self.shape[-1] - start))


class _FakeInputs(dict[str, Any]):
    def __init__(self) -> None:
        super().__init__(input_ids=_FakeTensor(3))
        self.input_ids = self["input_ids"]

    def to(self, _: object) -> "_FakeInputs":
        return self


class _FakeProcessor:
    def __init__(self, raw: str) -> None:
        self.raw = raw
        self.messages: list[dict[str, Any]] = []

    def apply_chat_template(self, messages: list[dict[str, Any]], **_: object) -> str:
        self.messages = messages
        return "rendered"

    def __call__(self, **_: object) -> _FakeInputs:
        return _FakeInputs()

    def batch_decode(self, *_: object, **__: object) -> list[str]:
        return [self.raw]


class _FakeModel:
    device = "cpu"

    def generate(self, **_: object) -> list[_FakeTensor]:
        return [_FakeTensor(7)]


class _Judge:
    judge_prompt = "vqa-judge-v1"

    async def judge(self, payload: dict[str, Any], *, request_meta: RequestMeta) -> DeepSeekJudgeResult:
        assert set(payload) == {"task", "question", "prediction", "ground_truth", "deterministic_metrics"}
        assert "image" not in json.dumps(payload).lower()
        return DeepSeekJudgeResult(
            judge_scope="text_and_structured_evidence_only",
            can_verify_visual_truth=False,
            semantic_correctness=1.0,
            answer_evidence_consistency=1.0,
            constraint_following=1.0,
            clarity=1.0,
            verdict="correct",
            concise_rationale="Equivalent to the reference answer.",
        )


class _CapturingClient:
    """Capture the General VQA system contract while returning a valid result.
    捕获 General VQA 系统契约并返回合法结果。
    """

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    async def complete_json(
        self,
        *,
        messages: list[dict[str, Any]],
        response_model: type[ExpertResult],
        request_meta: RequestMeta,
    ) -> ExpertResult:
        self.messages = messages
        return response_model(
            expert="general_vqa_expert",
            answer="Yes",
            boxes=[],
            evidence=[],
            status="completed",
        )


def _official_vrsbench(root: Path) -> None:
    Image.new("RGB", (8, 8)).save(root / "P0003_0002.png")
    (root / "VRSBench_EVAL_vqa.json").write_text(
        json.dumps(
            [
                {
                    "image_id": "P0003_0002.png",
                    "question": "Is a vehicle visible?",
                    "ground_truth": "Yes",
                    "question_id": 7,
                }
            ]
        ),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_transformers_client_runs_local_model_and_persists_trace(tmp_path: Path) -> None:
    raw = json.dumps(
        {
            "expert": "general_vqa_expert",
            "answer": "Yes",
            "boxes": [],
            "evidence": [],
            "status": "completed",
        }
    )
    processor = _FakeProcessor(raw)
    client = QwenTransformersClient(
        QwenSettings(backend="transformers", model="local", max_tokens=32),
        model=_FakeModel(),
        processor=processor,
    )
    image_url = image_to_data_url(_png_bytes(tmp_path))
    artifact_dir = tmp_path / "artifacts"
    result = await client.complete_json(
        messages=[
            {"role": "system", "content": "answer"},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}}]},
        ],
        response_model=ExpertResult,
        request_meta=RequestMeta(
            request_id="7:general_vqa_expert",
            request_hash="a" * 64,
            prompt_version="general-vqa-v1",
            artifact_dir=artifact_dir,
        ),
    )

    assert result.answer == "Yes"
    assert processor.messages[1]["content"][0]["type"] == "image"
    validation = json.loads((artifact_dir / "validation.json").read_text(encoding="utf-8"))
    assert validation["backend"] == "transformers"
    assert validation["response_metadata"]["token_usage"]["total_tokens"] == 7
    assert "base64" not in (artifact_dir / "request.json").read_text(encoding="utf-8")


def test_official_vrsbench_adapter_is_read_only_and_preserves_answer(tmp_path: Path) -> None:
    _official_vrsbench(tmp_path)
    adapter = get_adapter("VRSBench")

    probe = adapter.probe(tmp_path)
    sample = next(adapter.iter_samples(tmp_path, "validation", "general_vqa"))

    assert probe.version == "official-eval-v1"
    assert sample.sample_id == "7"
    assert sample.question == "Is a vehicle visible?"
    assert sample.ground_truth is not None and sample.ground_truth.answers == ["Yes"]
    assert not (tmp_path / "spacers_adapter.json").exists()


@pytest.mark.asyncio
async def test_vrsbench_runs_router_expert_judge_and_html_report(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    _official_vrsbench(dataset_root)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    client = MockVisionClient(
        {
            "7:general_vqa_expert": {
                "expert": "general_vqa_expert",
                "answer": "Yes",
                "boxes": [],
                "evidence": ["vehicle visible"],
                "status": "completed",
            }
        }
    )
    settings = AppSettings(
        paths=PathSettings(dataset_root=dataset_root),
        runs=RunSettings(root=tmp_path),
        models={"qwen": {"backend": "transformers", "model": "local-qwen"}},
    )
    prompts = {"count": "count", "target": "target", "change": "change", "spatial": "spatial", "general": "vqa", "seam": "seam"}
    summary = await DatasetRunner(
        settings,
        get_adapter("VRSBench"),
        run_dir=run_dir,
        client=client,
        prompts=prompts,
        judge_client=_Judge(),  # type: ignore[arg-type]
        judge_policy="all",
    ).run(split="validation", task="general_vqa", limit=1)

    assert summary.succeeded == 1
    route = json.loads((run_dir / "samples" / "7" / "routing_decision.json").read_text(encoding="utf-8"))
    assert route["experts"][0]["name"] == "general_vqa_expert"
    trace = json.loads((run_dir / "samples" / "7" / "agent_trace.json").read_text(encoding="utf-8"))
    assert trace["router_used"] is True
    assert "TaskRouter.route_known" in trace["route"]
    evaluation = json.loads((run_dir / "samples" / "7" / "vqa_evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["judge_score"] == 1

    report = build_multiagent_vqa_report(run_dir, qwen=settings.models.qwen)
    assert report is not None and report.is_file()
    html = report.read_text(encoding="utf-8")
    assert "spacers_agent.workflow.GeneralVQAExpert" in html
    assert "TaskRouter.route_known" in html


@pytest.mark.asyncio
async def test_general_vqa_contract_requires_empty_boxes(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    _official_vrsbench(dataset_root)
    sample = next(get_adapter("VRSBench").iter_samples(dataset_root, "validation", "general_vqa"))
    client = _CapturingClient()

    from spacers_agent.workflow import GeneralVQAExpert

    await GeneralVQAExpert(client, "answer", "local-qwen").run(sample, artifact_dir=tmp_path / "artifacts")

    system_prompt = str(client.messages[0]["content"])
    assert "boxes must always be []" in system_prompt
    assert '"boxes":[]' in system_prompt


def _png_bytes(root: Path) -> bytes:
    path = root / "input.png"
    Image.new("RGB", (4, 4)).save(path)
    return path.read_bytes()
