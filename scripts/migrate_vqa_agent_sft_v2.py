#!/usr/bin/env python3
"""Migrate VQA Agent SFT records to the production payload v2 contract.

将 VQA Agent SFT 记录迁移到生产 payload v2 契约。输入记录只读解析；每个
JSONL 与 manifest 均通过同目录临时文件原子替换。该转换只移动既有选择项事实，
绝不从答案或 Ground Truth 反推 choices。
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "data" / "2026-08-24_vqa-agent-io"
DATA_FILES = (
    "VRSBench/train.jsonl",
    "VRSBench/validation.jsonl",
    "DOTA/train.jsonl",
    "DOTA/validation.jsonl",
    "HRSCD/train.jsonl",
    "HRSCD/validation.jsonl",
    "MiniFrance/train.jsonl",
    "MiniFrance/validation.jsonl",
    "LRS-VQA-Supplement/train.jsonl",
    "LRS-VQA-Supplement/validation.jsonl",
)

sys.path.insert(0, str(REPO_ROOT))

from agents.general_vqa.agent import GeneralVQAAgent  # noqa: E402
from agents.schema import (  # noqa: E402
    GENERAL_VQA_AGENT_TASKS,
    AgentResult,
    VisualTaskPlan,
)
from data.adapters.vrsbench.adapter import VRSBenchAdapter  # noqa: E402
from data.schema import UnifiedSample  # noqa: E402


class MigrationError(ValueError):
    """Stable local migration failure. / 稳定的本地迁移失败。"""


def _atomic_json(path: Path, value: Any) -> None:
    """Write one JSON object atomically. / 原子写入一个 JSON 对象。"""
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _identity(record: dict[str, Any]) -> dict[str, Any]:
    """Capture fields that the migration is forbidden to change.
    捕获迁移禁止改变的字段。"""
    sample = record["input"]["agent_input"]["sample"]
    result = record["output"]["agent_result"]
    return {
        "sample_id": record["sample_id"],
        "sample_sample_id": sample["sample_id"],
        "question": sample["question"],
        "split": sample["split"],
        "images": [
            {
                "image_id": image["image_id"],
                "path": image["path"],
                "role": image["role"],
                "width": image.get("width"),
                "height": image.get("height"),
                "sha256": image.get("sha256"),
            }
            for image in sample["images"]
        ],
        "answer": result["answer"],
        "source_answer": record["supervision"]["source_answer"],
    }


def _canonicalize_sample(
    sample_data: dict[str, Any],
    old_payload: dict[str, Any],
    *,
    location: str,
) -> UnifiedSample:
    """Move existing MC facts into TaskNormalization, then validate.
    将既有多选事实移入 TaskNormalization 后执行校验。"""
    normalization = sample_data.get("normalization")
    if normalization is not None:
        normalization = dict(normalization)
        normalization.setdefault("choices", [])
        normalization.setdefault("allow_multiple", False)
        if sample_data.get("task") == "multiple_choice_vqa":
            choices = old_payload.get("choices")
            if not (
                isinstance(choices, list)
                and len(choices) >= 2
                and all(isinstance(choice, str) and choice.strip() for choice in choices)
            ):
                raise MigrationError(f"{location}: canonical choices missing")
            allow_multiple = old_payload.get("allow_multiple", False)
            if not isinstance(allow_multiple, bool):
                raise MigrationError(f"{location}: allow_multiple is not boolean")
            normalization["choices"] = list(choices)
            normalization["allow_multiple"] = allow_multiple
            # v2 has one canonical choice representation. Historical answer
            # constraints are not copied beside it. v2 只保留一份规范选项表达；
            # 不在旁边复制历史 answer constraints。
            normalization["answer_constraints"] = {}
        sample_data = {**sample_data, "normalization": normalization}
    try:
        return UnifiedSample.model_validate(sample_data)
    except Exception as exc:
        raise MigrationError(f"{location}: UnifiedSample invalid") from exc


def _migrate_record(record: dict[str, Any], *, location: str) -> dict[str, Any]:
    """Migrate and fully validate one top-level SFT record.
    迁移并完整校验一条顶层 SFT 记录。"""
    before = _identity(record)
    try:
        agent_input = record["input"]["agent_input"]
        old_payload = agent_input.get("base_user_payload", agent_input.get("user_payload"))
        if not isinstance(old_payload, dict):
            raise MigrationError(f"{location}: source user payload missing")
        sample = _canonicalize_sample(
            dict(agent_input["sample"]), old_payload, location=location
        )
        plan = VisualTaskPlan.model_validate(record["input"]["visual_task_plan"])
        result = AgentResult.model_validate(record["output"]["agent_result"])
    except MigrationError:
        raise
    except Exception as exc:
        raise MigrationError(f"{location}: record contract invalid") from exc
    if plan.task != sample.task:
        raise MigrationError(f"{location}: plan/sample task mismatch")
    base_user_payload = GeneralVQAAgent.build_user_payload(sample)
    migrated = {
        **record,
        "schema_version": "vqa-agent-sft-v2",
        "input": {
            **record["input"],
            "visual_task_plan": plan.model_dump(mode="json"),
            "agent_input": {
                "sample": sample.model_dump(mode="json"),
                "base_user_payload": base_user_payload,
            },
        },
        "output": {
            **record["output"],
            "agent_result": result.model_dump(mode="json"),
        },
    }
    if _identity(migrated) != before:
        raise MigrationError(f"{location}: immutable sample identity changed")
    if migrated["input"]["agent_input"]["base_user_payload"] != (
        GeneralVQAAgent.build_user_payload(sample)
    ):
        raise MigrationError(f"{location}: production payload mismatch")
    return migrated


def _migrate_file(path: Path) -> int:
    """Atomically rewrite one JSONL after validating every row.
    校验全部行后原子重写一个 JSONL。"""
    count = 0
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            with path.open("r", encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    migrated = _migrate_record(
                        record, location=f"{path}:{line_number}"
                    )
                    handle.write(json.dumps(migrated, ensure_ascii=False) + "\n")
                    count += 1
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)
    return count


def _generate_vrsbench_validation(root: Path) -> int:
    """Generate the missing validation split through the audited adapter.
    通过已审计适配器生成缺失的 validation split。"""
    dataset_root = root.parent / "VRSBench-full"
    output_path = root / "VRSBench" / "validation.jsonl"
    adapter = VRSBenchAdapter()
    count = 0
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=output_path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            for sample in adapter.iter_samples(
                dataset_root, split="validation", task="general_vqa"
            ):
                if sample.task not in GENERAL_VQA_AGENT_TASKS:
                    continue
                if sample.ground_truth is None or not sample.ground_truth.answers:
                    raise MigrationError(
                        f"VRSBench validation answer missing: {sample.sample_id}"
                    )
                answer = sample.ground_truth.answers[0]
                model_sample = sample.model_copy(update={"ground_truth": None})
                plan = VisualTaskPlan(
                    version="visual-task-plan-v5",
                    task=model_sample.task,
                    needs_visual_assistance=False,
                    reason_codes=["supervision.source_vqa"],
                )
                result = AgentResult(
                    agent_name="general_vqa_agent",
                    answer=answer,
                )
                record = {
                    "schema_version": "vqa-agent-sft-v2",
                    "sample_id": model_sample.sample_id,
                    "input": {
                        "visual_task_plan": plan.model_dump(mode="json"),
                        "agent_input": {
                            "sample": model_sample.model_dump(mode="json"),
                            "base_user_payload": GeneralVQAAgent.build_user_payload(
                                model_sample
                            ),
                        },
                    },
                    "output": {"agent_result": result.model_dump(mode="json")},
                    "supervision": {
                        "loss_scope": ["output.agent_result.answer"],
                        "evidence_supervised": False,
                        "source_answer": answer,
                    },
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, output_path)
    return count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _update_manifests(root: Path) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "vqa-agent-sft-v2"
    for split, relative in (
        ("train", "VRSBench/train.jsonl"),
        ("validation", "VRSBench/validation.jsonl"),
    ):
        path = root / relative
        task_counts: Counter[str] = Counter()
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    task_counts[
                        json.loads(line)["input"]["agent_input"]["sample"]["task"]
                    ] += 1
        split_record = manifest["splits"][split]
        # Retain raw-source audit totals separately from emitted-file facts.
        # 将原始来源审计总数与实际输出文件事实分开保留。
        source_audit = {
            key: value
            for key, value in split_record.items()
            if key == "non_vqa_source_task" or key == "unsupported_agent_task"
        }
        manifest["splits"][split] = {
            "file": relative,
            "accepted": sum(task_counts.values()),
            "tasks": dict(sorted(task_counts.items())),
            "sha256": _sha256(path),
            "source_audit": source_audit,
        }
    _atomic_json(manifest_path, manifest)

    large_path = root / "large-image-vqa.manifest.json"
    large = json.loads(large_path.read_text(encoding="utf-8"))
    large["schema_version"] = "vqa-agent-sft-v2"
    for dataset, dataset_record in large["datasets"].items():
        for split_record in dataset_record["splits"].values():
            relative = split_record["file"]
            path = root / relative
            split_record["accepted"] = sum(
                1 for line in path.open("r", encoding="utf-8") if line.strip()
            )
            split_record["sha256"] = _sha256(path)
    _atomic_json(large_path, large)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    counts: dict[str, int] = {}
    validation_path = root / "VRSBench" / "validation.jsonl"
    if not validation_path.is_file():
        counts["VRSBench/validation.jsonl:generated"] = (
            _generate_vrsbench_validation(root)
        )
    for relative in DATA_FILES:
        path = root / relative
        if not path.is_file():
            raise MigrationError(f"required data file missing: {relative}")
        counts[relative] = _migrate_file(path)
    _update_manifests(root)
    print(json.dumps({"schema_version": "vqa-agent-sft-v2", "counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
