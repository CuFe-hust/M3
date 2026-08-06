"""Offline, reproducible generator for all migration golden fixtures.

生成 tests/fixtures/migration/ 下的全部 Golden fixtures：
- 9 个 runtime cases（旧 SampleRunner + 确定性 fake 客户端，无模型/网络）
- 4 个数据集的 raw Adapter fixtures 与 expected samples（由参考实现实际运行产出）
- VRSBench 任务规范化 Golden（由 try_yolo 行为审计得出）

要求：
- --reference-root 必须指向 try_yolo 锁定提交的 checkout，HEAD 逐字校验；
- 完全离线：不下载、不调用模型或密钥；
- 连续生成两次并逐字节比较，不一致即中止；
- 输出去除绝对路径、时间戳、UUID、推理耗时、机器目录；
- 不保存 Base64、密钥、大权重；UTF-8 与固定 JSON 格式；文件排序稳定。

运行环境需要参考仓库的依赖（如 m3 conda 环境）。本脚本是迁移工具，不进入
wheel，也不被架构守卫扫描。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COMMIT = "ec962eb87c3ad0b8c1502efcbd08db0daec48868"
IGNORE_KEYS = {"updated_at", "inference_seconds", "model_load_seconds"}
FIXTURE_IMAGES = {"image.png", "image_t1.png", "image_t2.png"}
TEXTURE_CASES = {"levir_cc", "change"}

CASES = [
    ("vrsbench_vqa", "VRSBench", "general_vqa", "Is the statement correct?", 1),
    ("levir_cc", "LEVIR-CC", "change_caption", "Describe the change.", 2),
    ("mme_realworld", "MME-RealWorld", "general_vqa", "Is the statement correct?", 1),
    ("xlrs", "XLRS-Bench-lite", "general_vqa", "Is the statement correct?", 1),
    ("counting", "parity", "counting", "How many buildings are visible?", 1),
    ("spatial", "parity", "spatial_relation", "Where is the target located?", 1),
    ("change", "parity", "change_caption", "Describe the change.", 2),
    ("counting_partial", "parity", "counting", "How many buildings are visible?", 1),
    ("failed", "parity", "general_vqa", "Is the statement correct?", 1),
]
SPECIAL_IDS = {"counting_partial": "counting_one_failed_tile", "failed": "primary_qwen_failure"}

VRSBENCH_NORMALIZATION = [
    {
        "question": "How many small vehicles are in the image?",
        "source_task": "vrsbench_vqa",
        "normalized_task": "counting",
        "semantic_subtype": "counting",
        "confidence": 1.0,
        "normalizer": "vrsbench_task_normalizer",
        "version": "1",
        "reason_codes": ["quantity_question"],
        "spatial_query": None,
        "answer_constraints": {},
        "count_target_hint": {"canonical_label": "small_vehicle"},
    },
    {
        "question": "What category is the topmost vehicle?",
        "source_task": "vrsbench_vqa",
        "normalized_task": "spatial_relation",
        "semantic_subtype": "extreme_category",
        "confidence": 1.0,
        "normalizer": "vrsbench_task_normalizer",
        "version": "1",
        "reason_codes": ["extreme_category_question"],
        "spatial_query": {"operation": "extreme_category"},
        "answer_constraints": {},
        "count_target_hint": None,
    },
    {
        "question": "Where is the large vehicle located in the image?",
        "source_task": "vrsbench_vqa",
        "normalized_task": "spatial_relation",
        "semantic_subtype": "grid_position",
        "confidence": 1.0,
        "normalizer": "vrsbench_task_normalizer",
        "version": "1",
        "reason_codes": ["grid_position_question"],
        "spatial_query": {"operation": "grid_position"},
        "answer_constraints": {},
        "count_target_hint": None,
    },
    {
        "question": "Are there any small vehicles?",
        "source_task": "vrsbench_vqa",
        "normalized_task": "general_vqa",
        "semantic_subtype": "existence",
        "confidence": 1.0,
        "normalizer": "vrsbench_task_normalizer",
        "version": "1",
        "reason_codes": ["existence_question"],
        "spatial_query": None,
        "answer_constraints": {},
        "count_target_hint": None,
    },
    {
        "question": "What color is the building?",
        "source_task": "vrsbench_vqa",
        "normalized_task": "general_vqa",
        "semantic_subtype": "color",
        "confidence": 1.0,
        "normalizer": "vrsbench_task_normalizer",
        "version": "1",
        "reason_codes": ["color_question"],
        "spatial_query": None,
        "answer_constraints": {},
        "count_target_hint": None,
    },
    {
        "question": "Describe the scene.",
        "source_task": "vrsbench_vqa",
        "normalized_task": "general_vqa",
        "semantic_subtype": "general",
        "confidence": 1.0,
        "normalizer": "vrsbench_task_normalizer",
        "version": "1",
        "reason_codes": ["general_question"],
        "spatial_query": None,
        "answer_constraints": {},
        "count_target_hint": None,
    },
]


# ── 通用工具 ────────────────────────────────────────────────────────────────


def make_plain(path: Path, seed: int) -> None:
    Image.new("RGB", (4, 4), (seed % 256, (seed * 7) % 256, (seed * 13) % 256)).save(path)


def make_texture(path: Path, seed: int) -> None:
    arr = np.random.default_rng(seed).integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def seed_for(case: str, fname: str) -> int:
    match = re.search(r"(\d+)", fname)
    return sum(ord(c) for c in case) + (int(match.group(1)) if match else 0)


def scrub(value, case: str):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key in IGNORE_KEYS:
                continue
            if key == "result_path" and isinstance(item, str):
                out[key] = f"<RUN_ROOT>/samples/{case}/{Path(item).name}"
                continue
            if key == "path" and isinstance(item, str) and Path(item).name in FIXTURE_IMAGES:
                out[key] = Path(item).name
                continue
            out[key] = scrub(item, case)
        return out
    if isinstance(value, list):
        return [scrub(item, case) for item in value]
    if isinstance(value, str):
        for bad in ("C:", "/Users", "Desktop", "golden_ws", "spacers-agent"):
            if bad in value:
                raise SystemExit(f"UNSCRUBBED STRING {value!r} in case {case}")
    return value


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# ── A. Runtime cases / 运行时场景 ───────────────────────────────────────────


def _import_reference(reference_root: Path) -> None:
    """Make the locked reference checkout importable (generator-only tooling).
    让锁定参考 checkout 可导入（仅生成器使用）。"""
    sys.path.insert(0, str(reference_root))


def _build_report_record(case: str, sample_dir: Path) -> dict:
    sample = read_json(sample_dir / "sample.json")
    routing = read_json(sample_dir / "routing_decision.json")
    status = read_json(sample_dir / "status.json")
    result_file = None
    result = None
    if (sample_dir / "counting_result.json").is_file():
        result_file = "counting_result.json"
    elif (sample_dir / "agent_result.json").is_file():
        result_file = "agent_result.json"
    if result_file is not None:
        result = read_json(sample_dir / result_file)
    trace_path = sample_dir / "agent_trace.json"
    trace = read_json(trace_path) if trace_path.is_file() else None
    eval_path = sample_dir / "vqa_evaluation.json"
    evaluation = read_json(eval_path) if eval_path.is_file() else None
    return {
        "sample_id": sample["sample_id"],
        "task": sample["task"],
        "dataset": sample["dataset"],
        "split": sample["split"],
        "state": status["state"],
        "route": routing["primary_agent"],
        "agent": result.get("agent_name") if result else None,
        "result_file": result_file,
        "result": scrub(result, case) if result else None,
        "trace": scrub(trace, case) if trace else None,
        "evaluation": scrub(evaluation, case) if evaluation else None,
        "errors": {"error_code": status.get("error_code"), "error_message": status.get("error_message")},
    }


async def _run_case(case, dataset, task, question, n_images, reference_root: Path, ws: Path, run_root: Path):
    from spacers_agent.bootstrap import assemble_runtime
    from spacers_agent.schemas import GroundTruth, ImageRef, UnifiedSample
    from spacers_agent.workflows.artifact_writer import ArtifactWriter
    from spacers_agent.workflows.dataset_runner import failed_sample_status
    from tests.parity.fake_clients import RecordingFakeDeepSeek, RecordingFakeQwen
    from tests.parity.fixture_harness import harness_settings

    img_names = [f"image_t{i + 1}.png" for i in range(n_images)] if n_images == 2 else ["image.png"]
    case_dir = ws / "img" / case
    case_dir.mkdir(parents=True, exist_ok=True)
    textured = case in TEXTURE_CASES
    for fname in img_names:
        (make_texture if textured else make_plain)(case_dir / fname, seed_for(case, fname))
    sample_id = SPECIAL_IDS.get(case, case)
    roles = ["t1", "t2"] if task in {"change_caption", "change_qa"} else ["image"]
    size = 32 if textured else 4
    images = [
        ImageRef(image_id=f"{role}-image", path=case_dir / fname, role=role, width=size, height=size)
        for role, fname in zip(roles, img_names)
    ]
    sample = UnifiedSample(
        sample_id=sample_id,
        dataset=dataset,
        split="validation",
        task=task,
        images=images,
        question=question,
        ground_truth=GroundTruth(answers=["yes"], count=4 if "counting" in case else None),
        metadata={},
    )
    settings = harness_settings(ws)
    fake_qwen = RecordingFakeQwen(run_root, scenario=case)
    fake_judge = RecordingFakeDeepSeek(run_root, scenario=case)
    runtime = assemble_runtime(settings, qwen_client=fake_qwen, judge_client=fake_judge)
    sample_dir = run_root / "samples" / sample.sample_id
    try:
        outcome = await runtime.sample_runner.run_one(sample, sample_dir, judge_policy="all")
        state = outcome.status.state
    except Exception as error:
        status = failed_sample_status(sample, error)
        ArtifactWriter().write_final_status(sample_dir, status)
        state = status.state
    files = {
        "sample.json": scrub(read_json(sample_dir / "sample.json"), case),
        "routing_decision.json": scrub(read_json(sample_dir / "routing_decision.json"), case),
        "status.json": scrub(read_json(sample_dir / "status.json"), case),
    }
    if (sample_dir / "agent_result.json").is_file():
        files["agent_result.json"] = scrub(read_json(sample_dir / "agent_result.json"), case)
    if (sample_dir / "counting_result.json").is_file():
        files["counting_result.json"] = scrub(read_json(sample_dir / "counting_result.json"), case)
    trace_path = sample_dir / "agent_trace.json"
    if trace_path.is_file() and trace_path.read_text(encoding="utf-8").strip():
        files["agent_trace.json"] = scrub(read_json(trace_path), case)
    eval_path = sample_dir / "vqa_evaluation.json"
    if eval_path.is_file():
        files["vqa_evaluation.json"] = scrub(read_json(eval_path), case)
    files["report_record.json"] = scrub(_build_report_record(case, sample_dir), case)
    return files, img_names, state


async def _generate_runtime_cases(reference_root: Path, ws_root: Path) -> dict:
    from spacers_agent.schemas import UnifiedSample  # noqa: F401  (warm import)

    run_root = ws_root / "runs"
    results = {}
    for case, dataset, task, question, n in CASES:
        files, img_names, state = await _run_case(
            case, dataset, task, question, n, reference_root, ws_root, run_root / case
        )
        results[case] = (files, img_names, state)
        print(f"[runtime] {case}: {state}")
    return results


# ── B. Adapter fixtures / 数据集适配器场景 ──────────────────────────────────


def _make_adapter_image(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    make_plain(path, seed)


def _adapter_scenarios(reference_root: Path, ws_root: Path) -> dict:
    """Build raw layouts and, for the success case, expected samples by actually
    running the reference adapters. / 构造 raw 布局；成功场景通过实际运行参考
    适配器得到 expected samples。"""
    results = {}

    # VRSBench: official VQA annotation + image lookup via rglob.
    # VRSBench：官方 VQA 标注 + 经 rglob 的图片查找。
    from spacers_agent.dataset_adapters import VRSBenchVQAAdapter

    base = {
        "vrsbench": {
            "annotation": "VRSBench_EVAL_vqa.json",
            "success_rows": [
                {"image_id": "img_1.png", "question": "How many buildings are there?",
                 "ground_truth": "3", "question_id": "vq_1", "type": "quantity"},
            ],
        }
    }
    for dataset, spec in base.items():
        scenarios = {}
        root = ws_root / "adapter_raw" / dataset
        # success / 成功场景
        sroot = root / "success"
        sroot.mkdir(parents=True, exist_ok=True)
        _make_adapter_image(sroot / "img_1.png", 11)
        write_json(sroot / spec["annotation"], spec["success_rows"])
        scenarios["success"] = {"layout": sroot, "expected": None, "failure": None}
        # missing_image / 缺图场景
        mroot = root / "missing_image"
        mroot.mkdir(parents=True, exist_ok=True)
        write_json(mroot / spec["annotation"], spec["success_rows"])
        scenarios["missing_image"] = {"layout": mroot, "expected": None, "failure": "DatasetProbeError"}
        # missing_field / 缺字段场景
        froot = root / "missing_field"
        froot.mkdir(parents=True, exist_ok=True)
        _make_adapter_image(froot / "img_1.png", 12)
        row = dict(spec["success_rows"][0])
        del row["ground_truth"]
        write_json(froot / spec["annotation"], [row])
        scenarios["missing_field"] = {"layout": froot, "expected": None, "failure": "DatasetProbeError"}
        # duplicate_candidates / 多候选场景
        droot = root / "duplicate_candidates"
        sub_a = droot / "a"
        sub_b = droot / "b"
        sub_a.mkdir(parents=True, exist_ok=True)
        sub_b.mkdir(parents=True, exist_ok=True)
        write_json(sub_a / spec["annotation"], spec["success_rows"])
        write_json(sub_b / spec["annotation"], spec["success_rows"])
        scenarios["duplicate_candidates"] = {"layout": droot, "expected": None, "failure": "DatasetProbeError"}
        results[dataset] = scenarios

    # Run the reference VRSBench adapter on the success layout to capture stable fields.
    # 运行参考 VRSBench 适配器捕获成功场景的稳定字段。
    try:
        adapter = VRSBenchVQAAdapter()
        scenario = results["vrsbench"]["success"]
        sample = next(iter(adapter.iter_samples(scenario["layout"].resolve(), "validation", "general_vqa")))
        scenario["expected"] = {
            "dataset": sample.dataset,
            "task": sample.task,
            "split": sample.split,
            "question": sample.question,
            "sample_id": sample.sample_id,
            "image_roles": [img.role for img in sample.images],
            "image_paths": [img.path.name for img in sample.images],
            "ground_truth": scrub(sample.ground_truth.model_dump(mode="json"), "adapter"),
            "metadata": scrub(sample.metadata, "adapter"),
        }
    except Exception as error:  # pragma: no cover - reference mismatch guard
        raise SystemExit(f"reference VRSBench adapter failed on success layout: {error}") from error
    for name, scenario in results["vrsbench"].items():
        if name == "success":
            continue
        try:
            list(adapter.iter_samples(scenario["layout"].resolve(), "validation", "general_vqa"))
            raise SystemExit(f"reference adapter unexpectedly passed scenario {name}")
        except Exception as error:
            scenario["failure"] = type(error).__name__

    # Manifest datasets: LEVIR-CC / MME-RealWorld / XLRS-Bench-lite.
    # manifest 数据集。
    from spacers_agent.dataset_adapters import ManifestDatasetAdapter

    manifest_specs = {
        "levir_cc": {
            "dataset": "LEVIR-CC",
            "task": "change_caption",
            "rows": [
                {"id": "levir_1", "split": "test", "task": "change_caption",
                 "question": "Describe the change.",
                 "images": ["A/01_t1.png", "B/02_t2.png"], "image_roles": ["t1", "t2"],
                 "answers": ["a building appeared"]},
            ],
        },
        "mme_realworld": {
            "dataset": "MME-RealWorld",
            "task": "general_vqa",
            "rows": [
                {"id": "mme_1", "split": "test", "task": "general_vqa",
                 "question": "Is there a ship?",
                 "images": ["img_1.png"], "image_roles": ["image"],
                 "answers": ["yes"]},
            ],
        },
        "xlrs": {
            "dataset": "XLRS-Bench-lite",
            "task": "general_vqa",
            "rows": [
                {"id": "xlrs_1", "split": "test", "task": "general_vqa",
                 "question": "Is the statement correct?",
                 "images": ["img_1.png"], "image_roles": ["image"],
                 "answers": ["yes"]},
            ],
        },
    }
    for key, spec in manifest_specs.items():
        scenarios = {}
        root = ws_root / "adapter_raw" / key
        sroot = root / "success"
        sroot.mkdir(parents=True, exist_ok=True)
        _make_adapter_image(sroot / "A" / "01_t1.png", 21)
        _make_adapter_image(sroot / "B" / "02_t2.png", 22)
        _make_adapter_image(sroot / "img_1.png", 23)
        fields = {"id": "id", "split": "split", "task": "task", "question": "question",
                  "images": "images", "image_roles": "image_roles", "answers": "answers"}
        manifest = {"dataset": spec["dataset"], "version": "1", "samples_file": "samples.json",
                    "fields": fields}
        write_json(sroot / "spacers_adapter.json", manifest)
        write_json(sroot / "samples.json", spec["rows"])
        scenarios["success"] = {"layout": sroot, "expected": None, "failure": None}

        mroot = root / "missing_image"
        mroot.mkdir(parents=True, exist_ok=True)
        write_json(mroot / "spacers_adapter.json", manifest)
        write_json(mroot / "samples.json", [dict(spec["rows"][0], images=["missing.png"])])
        scenarios["missing_image"] = {"layout": mroot, "expected": None, "failure": "DatasetProbeError"}

        froot = root / "missing_field"
        froot.mkdir(parents=True, exist_ok=True)
        _make_adapter_image(froot / "img_1.png", 24)
        row = dict(spec["rows"][0])
        del row["question"]
        write_json(froot / "spacers_adapter.json", manifest)
        write_json(froot / "samples.json", [row])
        scenarios["missing_field"] = {"layout": froot, "expected": None, "failure": "DatasetProbeError"}

        droot = root / "duplicate_candidates"
        droot.mkdir(parents=True, exist_ok=True)
        write_json(droot / "spacers_adapter.json", manifest)
        write_json(droot / "samples.json", spec["rows"])
        write_json(droot / "samples.extra.json", spec["rows"])
        scenarios["duplicate_candidates"] = {
            "layout": droot, "expected": None, "failure": None,
            "behavior": "decoy samples file is ignored by the reference manifest adapter",
        }
        results[key] = scenarios

    # Run reference manifest adapters on success layouts.
    # 运行参考 manifest 适配器捕获成功场景的稳定字段。
    for key, spec in manifest_specs.items():
        adapter = ManifestDatasetAdapter(spec["dataset"], {spec["task"]})
        scenario = results[key]["success"]
        try:
            sample = next(iter(adapter.iter_samples(scenario["layout"].resolve(), "test", spec["task"])))
        except Exception as error:  # pragma: no cover
            raise SystemExit(f"reference manifest adapter failed for {key}: {error}") from error
        scenario["expected"] = {
            "dataset": sample.dataset,
            "task": sample.task,
            "split": sample.split,
            "question": sample.question,
            "sample_id": sample.sample_id,
            "image_roles": [img.role for img in sample.images],
            "image_paths": [img.path.name for img in sample.images],
            "ground_truth": scrub(sample.ground_truth.model_dump(mode="json"), key),
            "metadata": scrub(sample.metadata, key),
        }
        for name, scenario in results[key].items():
            if name == "success":
                continue
            if scenario.get("behavior") is not None:
                continue
            try:
                list(adapter.iter_samples(scenario["layout"].resolve(), "test", spec["task"]))
                raise SystemExit(f"reference adapter unexpectedly passed scenario {key}/{name}")
            except Exception as error:
                scenario["failure"] = type(error).__name__
    return results


def _generate_adapter_fixtures(reference_root: Path, ws_root: Path, out_root: Path) -> None:
    """Write raw layouts and expected_samples.jsonl under out_root/adapters/.
    在 out_root/adapters/ 下写入 raw 布局与 expected_samples.jsonl。"""
    scenarios = _adapter_scenarios(reference_root, ws_root)
    for dataset, cases in sorted(scenarios.items()):
        dest = out_root / "adapters" / dataset
        records = []
        for name in ("success", "missing_image", "missing_field", "duplicate_candidates"):
            scenario = cases[name]
            shutil.copytree(scenario["layout"], dest / "raw" / name)
            if name == "success":
                records.append({
                    "case": name,
                    "failure": None,
                    "expected": scenario["expected"],
                })
            else:
                records.append({
                    "case": name,
                    "failure": scenario.get("failure"),
                    "behavior": scenario.get("behavior"),
                    "expected": None,
                })
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "expected_samples.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        print(f"[adapters] {dataset}: {sorted(cases)}")


# ── C. VRSBench normalization / 规范化 Golden ───────────────────────────────


def _generate_normalization_golden(out_root: Path) -> None:
    write_json(out_root / "vrsbench_normalization.json", VRSBENCH_NORMALIZATION)
    print("[normalization] vrsbench_normalization.json written")


# ── 主流程 ──────────────────────────────────────────────────────────────────


def _verify_reference(reference_root: Path) -> None:
    if not (reference_root / ".git").exists():
        raise SystemExit(f"reference root has no .git: {reference_root}")
    result = subprocess.run(
        ["git", "-C", str(reference_root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    head = result.stdout.strip() if result.returncode == 0 else ""
    if head != EXPECTED_COMMIT:
        raise SystemExit(
            f"reference HEAD {head or '<unknown>'} != expected {EXPECTED_COMMIT}"
        )
    print(f"[reference] verified HEAD {head}")


async def generate_once(reference_root: Path, ws_root: Path) -> Path:
    ws_root.mkdir(parents=True, exist_ok=True)
    runtime = await _generate_runtime_cases(reference_root, ws_root)
    _generate_adapter_fixtures(reference_root, ws_root, ws_root)
    _generate_normalization_golden(ws_root)
    for case, (files, img_names, state) in runtime.items():
        dest = ws_root / "cases" / case
        dest.mkdir(parents=True, exist_ok=True)
        for name, payload in files.items():
            write_json(dest / name, payload)
        for img in img_names:
            shutil.copy2(ws_root / "img" / case / img, dest / img)
    # Drop temporary layout sources before the output tree comparison.
    # 输出树比较前清理临时布局源与运行时产物。
    runs = ws_root / "runs"
    if runs.exists():
        shutil.rmtree(runs)
    img = ws_root / "img"
    if img.exists():
        shutil.rmtree(img)
    raw = ws_root / "adapter_raw"
    if raw.exists():
        shutil.rmtree(raw)
    return ws_root


def _compare_trees(first: Path, second: Path) -> None:
    first_files = sorted(p.relative_to(first).as_posix() for p in first.rglob("*") if p.is_file())
    second_files = sorted(p.relative_to(second).as_posix() for p in second.rglob("*") if p.is_file())
    if first_files != second_files:
        raise SystemExit(f"file sets differ: {first_files} vs {second_files}")
    for rel in first_files:
        if (first / rel).read_bytes() != (second / rel).read_bytes():
            raise SystemExit(f"generation not stable: {rel}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate migration golden fixtures")
    parser.add_argument("--reference-root", required=True, type=Path,
                        help="Path to the try_yolo checkout at the locked commit")
    parser.add_argument("--expected-commit", default=EXPECTED_COMMIT,
                        help="Locked reference commit to verify (default %(default)s)")
    args = parser.parse_args()
    if args.expected_commit != EXPECTED_COMMIT:
        raise SystemExit(f"expected-commit mismatch: {args.expected_commit} != {EXPECTED_COMMIT}")
    reference_root = args.reference_root.resolve()
    _verify_reference(reference_root)
    _import_reference(reference_root)

    import tempfile

    fixture_root = REPO_ROOT / "tests" / "fixtures" / "migration"
    with tempfile.TemporaryDirectory(prefix="golden_gen_") as tmp:
        tmp_root = Path(tmp)
        first = await generate_once(reference_root, tmp_root / "a")
        second = await generate_once(reference_root, tmp_root / "b")
        _compare_trees(first, second)
        print("STABILITY: two runs are byte-identical")
        if fixture_root.exists():
            shutil.rmtree(fixture_root)
        fixture_root.mkdir(parents=True, exist_ok=True)
        for case_dir in sorted((first / "cases").iterdir()):
            shutil.copytree(case_dir, fixture_root / case_dir.name)
        shutil.copytree(first / "adapters", fixture_root / "adapters")
        shutil.copy2(first / "vrsbench_normalization.json", fixture_root / "vrsbench_normalization.json")
        print(f"FIXTURES WRITTEN to {fixture_root}")


if __name__ == "__main__":
    asyncio.run(main())
