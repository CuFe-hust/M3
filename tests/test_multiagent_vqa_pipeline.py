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
from spacers_agent.evaluation import VQAAnswerJudgeResult
from spacers_agent.schemas import ExpertResult
from spacers_agent.settings import AppSettings, PathSettings, QwenSettings, RunSettings
from spacers_agent.vqa_report import _successful_vqa_sample_dirs, build_multiagent_vqa_report
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
    def __init__(self, raw: str | list[str]) -> None:
        self.raw = [raw] if isinstance(raw, str) else list(raw)
        self.messages: list[dict[str, Any]] = []
        self.message_history: list[list[dict[str, Any]]] = []

    def apply_chat_template(self, messages: list[dict[str, Any]], **_: object) -> str:
        self.messages = messages
        self.message_history.append(messages)
        return "rendered"

    def __call__(self, **_: object) -> _FakeInputs:
        return _FakeInputs()

    def batch_decode(self, *_: object, **__: object) -> list[str]:
        return [self.raw.pop(0)]


class _FakeModel:
    device = "cpu"

    def generate(self, **_: object) -> list[_FakeTensor]:
        return [_FakeTensor(7)]


class _Judge:
    judge_prompt = "vqa-judge-v1"

    async def judge_json(
        self,
        payload: dict[str, Any],
        *,
        response_model: type[VQAAnswerJudgeResult],
        request_meta: RequestMeta,
    ) -> VQAAnswerJudgeResult:
        assert set(payload) == {"task", "question", "prediction", "ground_truth", "deterministic_metrics"}
        assert "image" not in json.dumps(payload).lower()
        return response_model(
            score=1,
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


class _MessageMockClient(MockVisionClient):
    """Capture offline mock requests so review prompts remain auditable.
    捕获离线模拟请求，使复查提示保持可审计。
    """

    def __init__(self, responses: dict[str, dict[str, Any]]) -> None:
        super().__init__(responses)
        self.message_history: list[list[dict[str, Any]]] = []

    async def complete_json(
        self,
        *,
        messages: list[dict[str, Any]],
        response_model: type[ExpertResult],
        request_meta: RequestMeta,
    ) -> ExpertResult:
        self.message_history.append(messages)
        return await super().complete_json(
            messages=messages,
            response_model=response_model,
            request_meta=request_meta,
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
                    "type": "object existence",
                    "dataset": "RSBench",
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


@pytest.mark.asyncio
async def test_transformers_client_repairs_invalid_json_without_resending_image(tmp_path: Path) -> None:
    repaired = json.dumps(
        {
            "expert": "spatial_expert",
            "answer": "small-vehicle",
            "boxes": [[100, 100, 200, 200]],
            "evidence": [],
            "evidence_items": [
                {"label": "small-vehicle", "box": [100, 100, 200, 200], "confidence": 0.9}
            ],
            "status": "completed",
        }
    )
    processor = _FakeProcessor(['{"expert":"spatial_expert",', repaired])
    client = QwenTransformersClient(
        QwenSettings(backend="transformers", model="local", max_tokens=32),
        repair_prompt="Repair JSON without adding evidence.",
        model=_FakeModel(),
        processor=processor,
    )
    image_url = image_to_data_url(_png_bytes(tmp_path))
    artifact_dir = tmp_path / "repair"

    result = await client.complete_json(
        messages=[
            {"role": "system", "content": "answer"},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}}]},
        ],
        response_model=ExpertResult,
        request_meta=RequestMeta(
            request_id="repair",
            request_hash="b" * 64,
            prompt_version="spatial-v2",
            artifact_dir=artifact_dir,
        ),
    )

    assert result.answer == "small-vehicle"
    assert len(processor.message_history) == 2
    assert not any(
        item.get("type") == "image"
        for message in processor.message_history[1]
        for item in (message.get("content") if isinstance(message.get("content"), list) else [])
    )
    validation = json.loads((artifact_dir / "validation.json").read_text(encoding="utf-8"))
    assert validation["response_metadata"]["repair_used"] is True
    assert len(validation["response_metadata"]["attempt_errors"]) == 1


@pytest.mark.asyncio
async def test_transformers_client_prunes_only_incomplete_truncated_evidence_tail(tmp_path: Path) -> None:
    raw = (
        '{"expert":"spatial_expert","answer":"yes","boxes":[[10,10,20,20]],'
        '"evidence_items":['
        '{"label":"small-vehicle","box":[10,10,20,20],"confidence":0.9},'
        '{"label":"small-vehicle","box":[30,30,40,40],"confidence'
    )
    processor = _FakeProcessor(raw)
    client = QwenTransformersClient(
        QwenSettings(backend="transformers", model="local", max_tokens=32),
        repair_prompt="Repair JSON without adding evidence.",
        model=_FakeModel(),
        processor=processor,
    )
    artifact_dir = tmp_path / "truncated"

    result = await client.complete_json(
        messages=[{"role": "system", "content": "answer"}],
        response_model=ExpertResult,
        request_meta=RequestMeta(
            request_id="truncated",
            request_hash="c" * 64,
            prompt_version="spatial-v3",
            artifact_dir=artifact_dir,
        ),
    )

    assert len(processor.message_history) == 1
    assert len(result.evidence_items) == 1
    assert result.evidence_items[0].box == [10, 10, 20, 20]
    assert "truncated_json_incomplete_tail_pruned" in result.geometry["input_normalizations"]
    validation = json.loads((artifact_dir / "validation.json").read_text(encoding="utf-8"))
    assert validation["response_metadata"]["repair_used"] is False
    assert validation["response_metadata"]["local_recoveries"] == [
        "truncated_json_incomplete_tail_pruned"
    ]


def test_official_vrsbench_adapter_is_read_only_and_preserves_answer(tmp_path: Path) -> None:
    _official_vrsbench(tmp_path)
    adapter = get_adapter("VRSBench")

    probe = adapter.probe(tmp_path)
    sample = next(adapter.iter_samples(tmp_path, "validation", "general_vqa"))

    assert probe.version == "official-eval-v1"
    assert sample.sample_id == "7"
    assert sample.question == "Is a vehicle visible?"
    assert sample.ground_truth is not None and sample.ground_truth.answers == ["Yes"]
    assert sample.metadata["question_type"] == "object existence"
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
            "7:spatial_expert": {
                "expert": "spatial_expert",
                "answer": "Yes",
                "boxes": [[100, 100, 200, 200]],
                "evidence": ["vehicle visible"],
                "evidence_items": [
                    {"label": "small-vehicle", "box": [100, 100, 200, 200], "confidence": 0.9}
                ],
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
    assert route["experts"][0]["name"] == "spatial_expert"
    assert "vrsbench_type_object_existence" in route["reason_codes"]
    trace = json.loads((run_dir / "samples" / "7" / "agent_trace.json").read_text(encoding="utf-8"))
    assert trace["router_used"] is True
    assert "TaskRouter.route_vrsbench_vqa" in trace["route"]
    assert trace["prompt_version"] == "spatial-v5"
    evaluation = json.loads((run_dir / "samples" / "7" / "vqa_evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["judge_score"] == 1

    report = build_multiagent_vqa_report(run_dir, qwen=settings.models.qwen)
    assert report is not None and report.is_file()
    html = report.read_text(encoding="utf-8")
    assert "spacers_agent.workflow.SpatialExpert" in html
    assert "TaskRouter.route_vrsbench_vqa" in html
    assert "结构化视觉证据" in html


@pytest.mark.asyncio
async def test_vrsbench_quantity_uses_accepted_point_count(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    Image.new("RGB", (32, 32)).save(dataset_root / "quantity.png")
    (dataset_root / "VRSBench_EVAL_vqa.json").write_text(
        json.dumps(
            [
                {
                    "image_id": "quantity.png",
                    "question": "How many small vehicles are visible?",
                    "ground_truth": "1",
                    "question_id": 9,
                    "type": "object quantity",
                    "dataset": "RSBench",
                }
            ]
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    client = MockVisionClient(
        {
            "9:count-proposal": {
                "expert": "general_vqa_expert",
                "answer": "1",
                "boxes": [[400, 400, 600, 600]],
                "evidence": [],
                "status": "completed",
            },
        }
    )
    settings = AppSettings(
        paths=PathSettings(dataset_root=dataset_root),
        runs=RunSettings(root=tmp_path),
        models={"qwen": {"backend": "transformers", "model": "local-qwen"}},
    )
    prompts = {
        "count": "count",
        "count_proposal": "count proposal",
        "count_localize": "count localize",
        "target": "target",
        "change": "change",
        "spatial": "spatial",
        "general": "vqa",
        "seam": "seam",
    }

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
    result = json.loads((run_dir / "samples" / "9" / "expert_result.json").read_text(encoding="utf-8"))
    assert result["expert"] == "counting_expert"
    assert result["answer"] == "1"
    assert result["geometry"]["final_count"] == len(result["evidence_items"]) == 1
    point = result["evidence_items"][0]["point"]
    assert point[0] == point[1] and 0 <= point[0] <= 999


@pytest.mark.asyncio
async def test_vrsbench_quantity_localizes_missing_boxes_and_drops_border_fragment(tmp_path: Path) -> None:
    """Keep the independent count while converting verified boxes into accepted points.
    保留独立计数，并将核验框转换为接受点，同时排除图外边界残片。
    """

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    Image.new("RGB", (64, 64)).save(dataset_root / "quantity.png")
    (dataset_root / "VRSBench_EVAL_vqa.json").write_text(
        json.dumps(
            [
                {
                    "image_id": "quantity.png",
                    "question": "How many vehicles are visible?",
                    "ground_truth": "3",
                    "question_id": 4,
                    "type": "object quantity",
                    "dataset": "RSBench",
                }
            ]
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    client = MockVisionClient(
        {
            "4:count-proposal": {
                "expert": "general_vqa_expert",
                "answer": "3",
                "boxes": [],
                "evidence": [],
                "status": "completed",
            },
            "4:count-localizer": {
                "expert": "counting_localizer",
                "answer": "4",
                "boxes": [],
                "evidence": [],
                "evidence_items": [
                    {"label": "vehicle", "box": [600, 280, 700, 400], "confidence": 0.9},
                    {"label": "vehicle", "box": [600, 800, 700, 900], "confidence": 0.9},
                    {"label": "vehicle", "box": [20, 430, 90, 560], "confidence": 0.9},
                    {"label": "vehicle", "box": [680, 970, 740, 999], "confidence": 0.7},
                ],
                "status": "completed",
            },
        }
    )
    settings = AppSettings(
        paths=PathSettings(dataset_root=dataset_root),
        runs=RunSettings(root=tmp_path),
        models={"qwen": {"backend": "transformers", "model": "local-qwen"}},
    )
    prompts = {
        "count": "count",
        "count_proposal": "count proposal",
        "count_localize": "count localize",
        "target": "target",
        "change": "change",
        "spatial": "spatial",
        "general": "vqa",
        "seam": "seam",
    }

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
    result = json.loads((run_dir / "samples" / "4" / "expert_result.json").read_text(encoding="utf-8"))
    counting = json.loads((run_dir / "samples" / "4" / "counting_result.json").read_text(encoding="utf-8"))
    assert result["answer"] == "3"
    assert len(result["boxes"]) == len(result["evidence_items"]) == 3
    assert counting["final_count"] == len(counting["global_points"]) == 3
    assert result["geometry"]["localization_used"] is True
    assert result["geometry"]["localizer_answer"] == 4
    assert any(item["code"] == "COUNT_BORDER_OR_DUPLICATE_EVIDENCE_DROPPED" for item in counting["warnings"])


def test_count_header_recovery_and_border_fragment_filter_are_conservative() -> None:
    """Recover only a complete answer header and reject only centre-outside fragments.
    仅恢复完整答案头，并且只排除中心位于图外的边界残片。
    """

    from spacers_agent.workflow import _is_tiny_border_fragment, _recover_count_proposal_header

    assert _recover_count_proposal_header('{"answer":"3","boxes":[[1,2') == 3
    assert _recover_count_proposal_header('{"answer":"') is None
    assert _is_tiny_border_fragment([680, 970, 740, 999]) is True
    assert _is_tiny_border_fragment([0, 388, 100, 600]) is False


def test_count_deduplication_retains_adjacent_overlapping_vehicle_boxes() -> None:
    """Do not merge adjacent coarse boxes that represent separate vehicle centres.
    不合并代表不同车辆中心的相邻粗略框。
    """

    from spacers_agent.schemas import VisualEvidence
    from spacers_agent.workflow import _accepted_count_evidence

    evidence = [
        VisualEvidence(label="large-vehicle", box=[750, 585, 880, 935]),
        VisualEvidence(label="large-vehicle", box=[850, 585, 980, 935]),
        VisualEvidence(label="large-vehicle", box=[670, 640, 750, 935]),
        VisualEvidence(label="large-vehicle", box=[880, 585, 980, 935]),
    ]

    points, boxes, dropped = _accepted_count_evidence(evidence, "large-vehicle", "image.png")

    assert len(points) == len(boxes) == 4
    assert dropped == 0


@pytest.mark.asyncio
async def test_general_vqa_contract_retains_labeled_boxes(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    _official_vrsbench(dataset_root)
    sample = next(get_adapter("VRSBench").iter_samples(dataset_root, "validation", "general_vqa"))
    client = _CapturingClient()

    from spacers_agent.workflow import GeneralVQAExpert

    await GeneralVQAExpert(client, "answer", "local-qwen").run(sample, artifact_dir=tmp_path / "artifacts")

    system_prompt = str(client.messages[0]["content"])
    assert "retain relevant labeled boxes or points" in system_prompt
    assert "copy evidence boxes into boxes" in system_prompt


@pytest.mark.asyncio
async def test_spatial_expert_reviews_incomplete_extreme_candidates(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    _official_vrsbench(dataset_root)
    sample = next(get_adapter("VRSBench").iter_samples(dataset_root, "validation", "general_vqa"))
    sample = sample.model_copy(
        update={
            "question": "What object class is the top-most vehicle?",
            "metadata": {**sample.metadata, "question_type": "object category"},
        }
    )
    client = _MessageMockClient(
        {
            "7:spatial_expert": {
                "expert": "spatial_expert",
                "answer": "bus",
                "evidence_items": [
                    {"label": "large-vehicle", "box": [0, 400, 100, 600], "confidence": 0.8}
                ],
            },
            "7:spatial-candidate-review": {
                "expert": "spatial_expert",
                "answer": "small-vehicle",
                "evidence_items": [
                    {"label": "large-vehicle", "box": [0, 400, 100, 600], "confidence": 0.8},
                    {"label": "small-vehicle", "box": [600, 100, 680, 180], "confidence": 0.9},
                ],
            },
        }
    )

    from spacers_agent.workflow import SpatialExpert
    from spacers_agent.vqa_geometry import apply_vrsbench_geometry

    raw = await SpatialExpert(client, "spatial", "local-qwen", "review").run(
        sample,
        artifact_dir=tmp_path / "artifacts",
    )
    result = apply_vrsbench_geometry(sample.question, "object category", raw)

    assert len(raw.evidence_items) == 2
    assert raw.geometry["candidate_review_used"] is True
    assert result.answer == "small-vehicle"
    assert result.geometry["candidate_count"] == 2
    initial_payload = json.loads(client.message_history[0][1]["content"][-1]["text"])
    review_payload = json.loads(client.message_history[1][1]["content"][-1]["text"])
    assert initial_payload["semantic_subtype"] == "extreme_category"
    assert initial_payload["answer_vocabulary"] == ["small-vehicle", "large-vehicle"]
    assert review_payload["review_mode"] == "independent_candidate_enumeration"
    assert "first_pass_evidence" not in review_payload


@pytest.mark.asyncio
async def test_grid_position_uses_unanchored_target_box_without_candidate_review(tmp_path: Path) -> None:
    """Keep a valid physical target box and avoid answer-vocabulary anchoring.
    保留有效物理目标框，并避免答案词表造成锚定。
    """

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    _official_vrsbench(dataset_root)
    sample = next(get_adapter("VRSBench").iter_samples(dataset_root, "validation", "general_vqa"))
    sample = sample.model_copy(
        update={
            "question": "What is the position of the large vehicle in the image?",
            "metadata": {**sample.metadata, "question_type": "object position"},
        }
    )
    client = _MessageMockClient(
        {
            "7:spatial_expert": {
                "expert": "spatial_expert",
                "answer": "The bus is at the left edge.",
                "evidence_items": [
                    {"label": "large-vehicle", "box": [0, 384, 107, 600], "confidence": 0.9}
                ],
            }
        }
    )

    from spacers_agent.workflow import SpatialExpert

    result = await SpatialExpert(client, "spatial", "local-qwen", "review").run(
        sample,
        artifact_dir=tmp_path / "artifacts",
    )

    assert result.evidence_items[0].box == [0, 384, 107, 600]
    assert len(client.message_history) == 1
    payload = json.loads(client.message_history[0][1]["content"][-1]["text"])
    assert payload["semantic_subtype"] == "grid_position"
    assert payload["answer_vocabulary"] == []


@pytest.mark.asyncio
async def test_grid_position_review_replaces_corner_region_placeholder(tmp_path: Path) -> None:
    """Replace a corner-region placeholder with independently localized object evidence.
    用独立定位的物体证据替换角落区域占位框。
    """

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    _official_vrsbench(dataset_root)
    sample = next(get_adapter("VRSBench").iter_samples(dataset_root, "validation", "general_vqa"))
    sample = sample.model_copy(
        update={
            "question": "Where is the small vehicle located?",
            "metadata": {**sample.metadata, "question_type": "object position"},
        }
    )
    client = _MessageMockClient(
        {
            "7:spatial_expert": {
                "expert": "spatial_expert",
                "answer": "top-left",
                "evidence_items": [
                    {"label": "small-vehicle", "box": [0, 0, 100, 100], "confidence": 0.9}
                ],
            },
            "7:spatial-candidate-review": {
                "expert": "spatial_expert",
                "answer": "upper right",
                "evidence_items": [
                    {"label": "small-vehicle", "box": [840, 273, 925, 312], "confidence": 0.9}
                ],
            },
        }
    )

    from spacers_agent.vqa_geometry import apply_vrsbench_geometry
    from spacers_agent.workflow import SpatialExpert

    raw = await SpatialExpert(client, "spatial", "local-qwen", "review").run(
        sample,
        artifact_dir=tmp_path / "artifacts",
    )
    result = apply_vrsbench_geometry(sample.question, "object position", raw)

    assert [item.box for item in raw.evidence_items] == [[840, 273, 925, 312]]
    assert raw.geometry["candidate_review_replaced"] == 1
    assert result.answer == "top-right"
    assert result.geometry["candidate_review_replaced"] == 1
    review_payload = json.loads(client.message_history[1][1]["content"][-1]["text"])
    assert review_payload["answer_vocabulary"] == []


def test_spatial_evidence_merge_deduplicates_points_and_prefers_boxes() -> None:
    from spacers_agent.schemas import VisualEvidence
    from spacers_agent.workflow import _merge_visual_evidence

    first = [VisualEvidence(label="small-vehicle", point=[500, 500], confidence=0.7)]
    second = [
        VisualEvidence(label="car", point=[506, 503], confidence=0.9),
        VisualEvidence(label="small-vehicle", box=[480, 480, 530, 530], confidence=0.8),
    ]

    merged = _merge_visual_evidence(first, second)

    assert len(merged) == 1
    assert merged[0].box == [480, 480, 530, 530]


@pytest.mark.asyncio
async def test_resume_retries_failed_judge_without_reissuing_qwen(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    _official_vrsbench(dataset_root)
    run_dir = tmp_path / "run"
    sample_dir = run_dir / "samples" / "7"
    sample_dir.mkdir(parents=True)
    sample = next(get_adapter("VRSBench").iter_samples(dataset_root, "validation", "general_vqa"))
    (sample_dir / "sample.json").write_text(sample.model_dump_json(), encoding="utf-8")
    (sample_dir / "expert_result.json").write_text(
        ExpertResult(expert="general_vqa_expert", answer="Yes", status="completed").model_dump_json(),
        encoding="utf-8",
    )
    (sample_dir / "status.json").write_text(
        json.dumps(
            {
                "sample_id": "7",
                "task": "general_vqa",
                "state": "succeeded",
                "error_code": None,
                "error_message": None,
                "result_path": str(sample_dir / "expert_result.json"),
                "updated_at": "2026-07-22T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (sample_dir / "vqa_evaluation.json").write_text(
        json.dumps({"judge_status": "failed", "exact_match": True}), encoding="utf-8"
    )
    qwen = MockVisionClient({})
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
        client=qwen,
        prompts=prompts,
        judge_client=_Judge(),  # type: ignore[arg-type]
        judge_policy="all",
    ).run(split="validation", task="general_vqa", resume=True, limit=1)

    assert summary.succeeded == 1
    assert qwen.calls == []
    evaluation = json.loads((sample_dir / "vqa_evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["judge_status"] == "succeeded" and evaluation["judge_score"] == 1


@pytest.mark.asyncio
async def test_judge_vqa_run_reuses_persisted_qwen_result_and_rebuilds_report(tmp_path: Path) -> None:
    """Judge saved VQA answers and rebuild HTML without creating a Qwen client.
    评判已保存的 VQA 答案并重建 HTML，且不创建 Qwen 客户端。
    """

    from spacers_agent.cli import _judge_vqa_run

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    _official_vrsbench(dataset_root)
    sample = next(get_adapter("VRSBench").iter_samples(dataset_root, "validation", "general_vqa"))
    run_dir = tmp_path / "persisted-run"
    sample_dir = run_dir / "samples" / sample.sample_id
    sample_dir.mkdir(parents=True)
    (sample_dir / "sample.json").write_text(sample.model_dump_json(), encoding="utf-8")
    (sample_dir / "expert_result.json").write_text(
        ExpertResult(expert="general_vqa_expert", answer="Yes", status="completed").model_dump_json(),
        encoding="utf-8",
    )
    (sample_dir / "status.json").write_text(
        json.dumps({"state": "succeeded"}), encoding="utf-8"
    )
    (sample_dir / "vqa_evaluation.json").write_text(
        json.dumps({"judge_status": "not_requested", "exact_match": True}), encoding="utf-8"
    )
    (sample_dir / "agent_trace.json").write_text(
        json.dumps({"route": "GeneralVQAExpert.run", "inference_seconds": 1.0}), encoding="utf-8"
    )
    settings = AppSettings(
        paths=PathSettings(dataset_root=dataset_root),
        runs=RunSettings(root=tmp_path),
        models={"qwen": {"backend": "transformers", "model": "local-qwen"}},
    )

    exit_code = await _judge_vqa_run(
        settings,
        "persisted-run",
        judge_client=_Judge(),  # type: ignore[arg-type]
    )

    assert exit_code == 0
    evaluation = json.loads((sample_dir / "vqa_evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["judge_status"] == "succeeded" and evaluation["judge_score"] == 1
    trace = json.loads((sample_dir / "agent_trace.json").read_text(encoding="utf-8"))
    assert "DeepSeekJudgeClient.judge" in trace["route"]
    metrics = json.loads((run_dir / "vrsbench_vqa.metrics.json").read_text(encoding="utf-8"))
    assert metrics["deepseek_proxy"]["evaluated"] == 1
    assert (run_dir / "vrsbench_vqa.report" / "report.html").is_file()


def _png_bytes(root: Path) -> bytes:
    path = root / "input.png"
    Image.new("RGB", (4, 4)).save(path)
    return path.read_bytes()


def test_partial_vqa_artifacts_remain_visible_to_report(tmp_path: Path) -> None:
    sample_dir = tmp_path / "samples" / "partial-id"
    sample_dir.mkdir(parents=True)
    (sample_dir / "status.json").write_text(json.dumps({"state": "partial"}), encoding="utf-8")
    (sample_dir / "sample.json").write_text(json.dumps({"task": "general_vqa"}), encoding="utf-8")
    (sample_dir / "expert_result.json").write_text("{}", encoding="utf-8")

    assert _successful_vqa_sample_dirs(tmp_path) == [sample_dir]
