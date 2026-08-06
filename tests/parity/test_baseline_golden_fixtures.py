"""Golden fixture parity tests for the migration baseline.

These tests lock the observable behavior and persisted artifact shapes of the
try_yolo baseline (commit ec962eb87c3ad0b8c1502efcbd08db0daec48868) through
offline JSON fixtures under tests/fixtures/migration/. They never import
spacers_agent, eval, models, or any baseline runtime module; they only read the
frozen Golden files and assert stable fields plus file existence.

基线 Golden fixture 等价测试。只读 tests/fixtures/migration/ 下的冻结 JSON，
断言稳定字段与文件存在性，绝不导入旧运行时包。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "migration"

# Expected stable facts per scenario. / 每个场景的稳定预期。
CASES = {
    "vrsbench_vqa": {
        "dataset": "VRSBench",
        "task": "general_vqa",
        "state": "succeeded",
        "route": "general_vqa_agent",
        "result_file": "agent_result.json",
        "result_status": "completed",
        "images": ["image.png"],
        "has_evaluation": True,
    },
    "levir_cc": {
        "dataset": "LEVIR-CC",
        "task": "change_caption",
        "state": "succeeded",
        "route": "change_agent",
        "result_file": "agent_result.json",
        "result_status": "completed",
        "images": ["image_t1.png", "image_t2.png"],
        "has_evaluation": False,
    },
    "mme_realworld": {
        "dataset": "MME-RealWorld",
        "task": "general_vqa",
        "state": "succeeded",
        "route": "general_vqa_agent",
        "result_file": "agent_result.json",
        "result_status": "completed",
        "images": ["image.png"],
        "has_evaluation": True,
    },
    "xlrs": {
        "dataset": "XLRS-Bench-lite",
        "task": "general_vqa",
        "state": "succeeded",
        "route": "general_vqa_agent",
        "result_file": "agent_result.json",
        "result_status": "completed",
        "images": ["image.png"],
        "has_evaluation": True,
    },
    "counting": {
        "dataset": "parity",
        "task": "counting",
        "state": "succeeded",
        "route": "counting_agent",
        "result_file": "counting_result.json",
        "result_status": "completed",
        "images": ["image.png"],
        "has_evaluation": False,
    },
    "spatial": {
        "dataset": "parity",
        "task": "spatial_relation",
        "state": "succeeded",
        "route": "spatial_agent",
        "result_file": "agent_result.json",
        "result_status": "completed",
        "images": ["image.png"],
        "has_evaluation": False,
    },
    "change": {
        "dataset": "parity",
        "task": "change_caption",
        "state": "succeeded",
        "route": "change_agent",
        "result_file": "agent_result.json",
        "result_status": "completed",
        "images": ["image_t1.png", "image_t2.png"],
        "has_evaluation": False,
    },
    "counting_partial": {
        "dataset": "parity",
        "task": "counting",
        "state": "partial",
        "route": "counting_agent",
        "result_file": "counting_result.json",
        "result_status": "partial",
        "images": ["image.png"],
        "has_evaluation": False,
    },
    "failed": {
        "dataset": "parity",
        "task": "general_vqa",
        "state": "failed",
        "route": "general_vqa_agent",
        "result_file": None,
        "result_status": None,
        "images": ["image.png"],
        "has_evaluation": False,
    },
}

BASE_FILES = ("sample.json", "routing_decision.json", "status.json", "report_record.json")
ALLOWED_RESULT_FILES = ("agent_result.json", "counting_result.json")

# Volatile keys that the generator strips; their absence is itself a contract.
# 生成器必须剥离的易变字段；其缺失本身即是契约。
FORBIDDEN_KEYS = ("updated_at", "inference_seconds", "model_load_seconds")
# Absolute path fragments that must never appear. / 禁止出现的绝对路径片段。
FORBIDDEN_FRAGMENTS = ("C:", "/Users", "Desktop", "golden_ws", "<WS>")
UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)


def _load(name: str) -> dict:
    path = FIXTURE_ROOT / name / "sample.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _read(case: str, filename: str) -> dict:
    return json.loads((FIXTURE_ROOT / case / filename).read_text(encoding="utf-8"))


def _case_dirs() -> list[Path]:
    return sorted(
        p for p in FIXTURE_ROOT.iterdir()
        if p.is_dir() and (p / "sample.json").is_file()
    )


def _walk_json():
    for case_dir in FIXTURE_ROOT.iterdir():
        if not case_dir.is_dir():
            continue
        for path in sorted(case_dir.glob("*.json")):
            yield case_dir.name, path.name, json.loads(path.read_text(encoding="utf-8"))


def _iter_strings(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_strings(item)
    elif isinstance(value, str):
        yield value


@pytest.fixture(scope="module")
def fixture_root() -> Path:
    return FIXTURE_ROOT


@pytest.mark.parametrize("case", sorted(CASES))
def test_required_artifacts_exist(case: str, fixture_root: Path) -> None:
    """Every scenario ships its full artifact set. / 每个场景包含完整产物集。"""
    case_dir = fixture_root / case
    assert case_dir.is_dir(), f"missing fixture directory {case}"
    for filename in BASE_FILES:
        assert (case_dir / filename).is_file(), f"missing {filename} in {case}"
    result_file = CASES[case]["result_file"]
    if result_file is not None:
        assert (case_dir / result_file).is_file(), f"missing {result_file} in {case}"
    else:
        assert not any((case_dir / name).is_file() for name in ALLOWED_RESULT_FILES), (
            f"{case} must have no result file"
        )
    for image in CASES[case]["images"]:
        assert (case_dir / image).is_file(), f"missing image {image} in {case}"
        header = (case_dir / image).read_bytes()[:8]
        assert header == b"\x89PNG\r\n\x1a\n", f"{case}/{image} is not a PNG"
    has_evaluation = (case_dir / "vqa_evaluation.json").is_file()
    assert has_evaluation is CASES[case]["has_evaluation"], (
        f"evaluation presence mismatch in {case}"
    )


@pytest.mark.parametrize("case", sorted(CASES))
def test_sample_contract(case: str, fixture_root: Path) -> None:
    """Sample JSON locks dataset, task, split, and relative image refs.
    样本 JSON 锁定数据集、任务、划分与相对图片引用。"""
    sample = _read(case, "sample.json")
    expected = CASES[case]
    assert sample["sample_id"] == case or case == "counting_partial" and sample["sample_id"] == "counting_one_failed_tile" or case == "failed" and sample["sample_id"] == "primary_qwen_failure"
    assert sample["dataset"] == expected["dataset"]
    assert sample["task"] == expected["task"]
    assert sample["split"] == "validation"
    assert isinstance(sample["question"], str) and sample["question"]
    images = sample["images"]
    assert [Path(item["path"]).name for item in images] == expected["images"]
    for item in images:
        assert "/" not in item["path"].replace("\\", "/"), f"absolute image path in {case}"
        assert (fixture_root / case / item["path"]).is_file(), f"missing referenced image {item['path']}"
        assert item["role"] in {"image", "t1", "t2"}


@pytest.mark.parametrize("case", sorted(CASES))
def test_routing_contract(case: str, fixture_root: Path) -> None:
    """Routing decision locks the dispatched agent for the normalized task.
    路由决策锁定归一化任务对应的分派 Agent。"""
    routing = _read(case, "routing_decision.json")
    expected = CASES[case]
    assert routing["task"] == expected["task"]
    assert routing["primary_agent"] == expected["route"]
    assert routing["execution_mode"] == "single"
    # The router_source value is intentionally NOT frozen as a stable contract:
    # VRSBench semantic judgment moves from the runtime Router into the Adapter
    # (see docs/migration/GOLDEN_FIXTURES.md, "Intentional changes").
    # router_source 不锁定为最终契约：VRSBench 语义判断将从运行时 Router 前移到 Adapter。
    assert isinstance(routing["router_source"], str) and routing["router_source"]
    assert isinstance(routing["reason_codes"], list) and routing["reason_codes"]


@pytest.mark.parametrize("case", sorted(CASES))
def test_status_contract(case: str, fixture_root: Path) -> None:
    """Status JSON locks the sample state and run-relative result path.
    状态 JSON 锁定样本状态与相对运行目录的结果路径。"""
    status = _read(case, "status.json")
    expected = CASES[case]
    assert status["state"] == expected["state"]
    assert status["sample_id"] in {case, "counting_one_failed_tile", "primary_qwen_failure"}
    assert status["task"] == expected["task"]
    if expected["result_file"] is not None:
        assert status["result_path"] == (
            f"<RUN_ROOT>/samples/{case}/{expected['result_file']}"
        ), f"unexpected result_path in {case}"
    if case == "failed":
        assert status["error_code"] == "RuntimeError"
        assert status["error_message"] == "deterministic primary Qwen failure"
    else:
        assert status["error_code"] is None


@pytest.mark.parametrize("case", sorted(CASES))
def test_result_contract(case: str, fixture_root: Path) -> None:
    """Agent/Counting result locks the payload shape without type identity.
    结果 JSON 锁定载荷形状，不比较类型 identity。"""
    result_file = CASES[case]["result_file"]
    if result_file is None:
        return
    result = _read(case, result_file)
    assert result["status"] == CASES[case]["result_status"]
    if result_file == "agent_result.json":
        assert result["agent_name"] == CASES[case]["route"]
        assert isinstance(result["answer"], str)
        assert isinstance(result["boxes"], list)
        assert isinstance(result["evidence"], list)
    else:
        assert isinstance(result["final_count"], int) and result["final_count"] >= 0
        assert isinstance(result["target"], str) and result["target"]
        assert isinstance(result["succeeded_tiles"], (int, list))


@pytest.mark.parametrize("case", sorted(CASES))
def test_trace_contract(case: str, fixture_root: Path) -> None:
    """Trace JSON keeps the documented stable keys when present.
    轨迹 JSON 存在时保留文档化稳定键。"""
    trace_path = fixture_root / case / "agent_trace.json"
    if not trace_path.is_file():
        return
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    common = ("agent_class", "execution_mode", "execution_task", "fallback_agents",
              "fallback_used", "judge_status", "qwen_backend", "router_used",
              "routing_source", "task_type")
    for key in common:
        assert key in trace, f"missing trace key {key} in {case}"
    assert trace["execution_task"] == CASES[case]["task"]
    if CASES[case]["task"] == "counting":
        for key in ("backend", "primary_backend", "attempted_backends", "requested_backend_mode"):
            assert key in trace, f"missing counting trace key {key} in {case}"
    else:
        assert "prompt_version" in trace, f"missing trace key prompt_version in {case}"


@pytest.mark.parametrize("case", sorted(CASES))
def test_report_record_contract(case: str, fixture_root: Path) -> None:
    """Report record locks the reporting input shape derived from persisted files.
    报告记录锁定由持久化文件导出的报告输入形状。"""
    record = _read(case, "report_record.json")
    expected = CASES[case]
    for key in ("sample_id", "task", "dataset", "split", "state", "route",
                "agent", "result_file", "result", "trace", "evaluation", "errors"):
        assert key in record, f"missing report_record key {key} in {case}"
    assert record["state"] == expected["state"]
    assert record["task"] == expected["task"]
    assert record["route"] == expected["route"]
    assert record["result_file"] == expected["result_file"]
    assert record["errors"] == {
        "error_code": "RuntimeError" if case == "failed" else None,
        "error_message": "deterministic primary Qwen failure" if case == "failed" else None,
    }
    if expected["result_file"] is not None:
        assert record["result"] is not None
        assert record["result"]["status"] == expected["result_status"]
    else:
        assert record["result"] is None and record["agent"] is None


def test_no_volatile_or_absolute_values(fixture_root: Path) -> None:
    """No volatile keys, absolute paths, or random identifiers in any Golden JSON.
    任何 Golden JSON 不得包含易变键、绝对路径或随机标识符。"""
    seen_cases = set()
    for case, filename, payload in _walk_json():
        seen_cases.add(case)
        for key in FORBIDDEN_KEYS:
            assert key not in payload, f"{case}/{filename} contains volatile key {key}"
        for fragment in FORBIDDEN_FRAGMENTS:
            for text in _iter_strings(payload):
                assert fragment not in text, f"{case}/{filename} contains {fragment!r}"
        for text in _iter_strings(payload):
            assert not UUID_PATTERN.search(text), f"{case}/{filename} contains a UUID"
            assert not re.search(r"\.(png|jpg|jsonl?|yaml|txt|md|py)$", text) or "<" in text or "/" in text or "." not in text[:1], (
                f"{case}/{filename} may contain a machine path: {text!r}"
            )
    assert seen_cases == set(CASES), f"fixture coverage mismatch: {sorted(set(CASES) - seen_cases)}"


def test_states_cover_success_partial_failed(fixture_root: Path) -> None:
    """The fixture set must cover succeeded, partial, and failed states.
    fixture 集必须覆盖成功、部分、失败三种状态。"""
    states = {_read(case, "status.json")["state"] for case in CASES}
    assert {"succeeded", "partial", "failed"} <= states


# ── Adapter raw fixtures / 适配器原始场景 ───────────────────────────────────


def test_adapter_fixture_directories_are_complete() -> None:
    """Every dataset ships four raw scenarios and an expected line per scenario.
    每个数据集包含四个 raw 场景与每场景一条 expected 记录。"""
    for dataset in ("vrsbench", "levir_cc", "mme_realworld", "xlrs"):
        adapter_dir = FIXTURE_ROOT / "adapters" / dataset
        assert (adapter_dir / "expected_samples.jsonl").is_file(), dataset
        for scenario in ("success", "missing_image", "missing_field", "duplicate_candidates"):
            assert (adapter_dir / "raw" / scenario).is_dir(), f"{dataset}/{scenario}"
        lines = [
            json.loads(line)
            for line in (adapter_dir / "expected_samples.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert [line["case"] for line in lines] == [
            "success", "missing_image", "missing_field", "duplicate_candidates",
        ], dataset


def test_adapter_success_expected_fields_are_stable() -> None:
    """The reference adapter outputs are locked through stable fields.
    参考适配器输出通过稳定字段锁定。"""
    expected_by_dataset = {
        "vrsbench": ("VRSBench", "general_vqa", "validation"),
        "levir_cc": ("LEVIR-CC", "change_caption", "test"),
        "mme_realworld": ("MME-RealWorld", "general_vqa", "test"),
        "xlrs": ("XLRS-Bench-lite", "general_vqa", "test"),
    }
    for dataset, (name, task, split) in expected_by_dataset.items():
        lines = [
            json.loads(line)
            for line in (FIXTURE_ROOT / "adapters" / dataset / "expected_samples.jsonl")
            .read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        success = next(line for line in lines if line["case"] == "success")
        expected = success["expected"]
        assert expected["dataset"] == name, dataset
        assert expected["task"] == task, dataset
        assert expected["split"] == split, dataset
        assert expected["image_roles"], dataset
        assert expected["ground_truth"]["answers"], dataset


def test_adapter_failure_scenarios_record_failure_codes() -> None:
    """Missing images and missing fields must be recorded as failures.
    缺图与缺字段场景必须记录失败代码。"""
    for dataset in ("vrsbench", "levir_cc", "mme_realworld", "xlrs"):
        lines = [
            json.loads(line)
            for line in (FIXTURE_ROOT / "adapters" / dataset / "expected_samples.jsonl")
            .read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_case = {line["case"]: line for line in lines}
        assert by_case["missing_image"]["failure"] == "DatasetProbeError", dataset
        assert by_case["missing_field"]["failure"] == "DatasetProbeError", dataset
        if dataset == "vrsbench":
            assert by_case["duplicate_candidates"]["failure"] == "DatasetProbeError"
        else:
            assert by_case["duplicate_candidates"]["failure"] is None
            assert "decoy" in by_case["duplicate_candidates"]["behavior"], dataset


def test_adapter_fixtures_contain_no_absolute_paths() -> None:
    """Raw fixture JSONs must stay machine-independent. / raw 场景必须与机器无关。"""
    for dataset in ("vrsbench", "levir_cc", "mme_realworld", "xlrs"):
        for path in (FIXTURE_ROOT / "adapters" / dataset / "raw").rglob("*.json"):
            text = path.read_text(encoding="utf-8")
            assert "C:" not in text and "/Users" not in text, str(path)


# ── VRSBench task normalization golden / 规范化 Golden ──────────────────────


def test_vrsbench_normalization_golden_covers_all_three_tasks() -> None:
    """The six questions lock counting, spatial_relation, and general_vqa.
    六个问题锁定 counting / spatial_relation / general_vqa 三类任务。"""
    records = json.loads((FIXTURE_ROOT / "vrsbench_normalization.json").read_text(encoding="utf-8"))
    assert len(records) == 6
    assert {record["normalized_task"] for record in records} == {
        "counting", "spatial_relation", "general_vqa",
    }
    for record in records:
        assert record["source_task"] == "vrsbench_vqa"
        assert record["normalizer"] == "vrsbench_task_normalizer"
        assert record["version"] == "1"
        assert record["confidence"] == 1.0
        assert isinstance(record["reason_codes"], list) and record["reason_codes"]
        assert isinstance(record["answer_constraints"], dict)


def test_vrsbench_normalization_question_to_task_mapping() -> None:
    """The exact question-to-task mapping is part of the contract.
    问题到任务的精确映射属于契约。"""
    records = json.loads((FIXTURE_ROOT / "vrsbench_normalization.json").read_text(encoding="utf-8"))
    mapping = {record["question"]: record["normalized_task"] for record in records}
    assert mapping["How many small vehicles are in the image?"] == "counting"
    assert mapping["What category is the topmost vehicle?"] == "spatial_relation"
    assert mapping["Where is the large vehicle located in the image?"] == "spatial_relation"
    assert mapping["Are there any small vehicles?"] == "general_vqa"
    assert mapping["What color is the building?"] == "general_vqa"
    assert mapping["Describe the scene."] == "general_vqa"
