from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPOSITORY_IMPORT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_IMPORT_ROOT))

from agents.evidence_catalog import load_evidence_catalog
from data.schema import ImageRef, SampleDraft
from models.entry import create_model
from models.settings import QwenSettings
from workflows.schema import VQA_ASSISTANCE_SCOPE
from workflows.visual_planner import VisualTaskPlanner


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def excluded_source_names(refined_root: Path) -> set[str]:
    excluded: set[str] = set()
    for split in ("train", "val"):
        path = refined_root / "datasets" / f"{split}.jsonl"
        for record in read_jsonl(path):
            provenance = record.get("provenance") or {}
            candidates = [
                provenance.get("source_image_id"),
                (provenance.get("source") or {}).get("image"),
            ]
            for candidate in candidates:
                if isinstance(candidate, str) and candidate:
                    source = Path(candidate)
                    excluded.add(source.name.casefold())
                    excluded.add(source.stem.casefold())
    return excluded


def select_records(
    lrs_root: Path,
    refined_root: Path,
    image_limit: int,
    questions_per_image: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    excluded = excluded_source_names(refined_root)
    selected_by_image: dict[str, list[dict[str, Any]]] = {}
    missing = 0
    overlap = 0
    per_image_limit_skipped = 0
    for record in read_jsonl(lrs_root / "LRS_VQA_merged.jsonl"):
        image_value = record.get("image")
        if not isinstance(image_value, str) or not image_value:
            continue
        relative = Path(image_value)
        key = relative.as_posix().casefold()
        source = lrs_root / relative
        if not source.is_file():
            missing += 1
            continue
        if relative.name.casefold() in excluded or relative.stem.casefold() in excluded:
            overlap += 1
            continue
        if key not in selected_by_image:
            if len(selected_by_image) >= image_limit:
                continue
            selected_by_image[key] = []
        image_records = selected_by_image[key]
        if len(image_records) >= questions_per_image:
            per_image_limit_skipped += 1
            continue
        image_records.append(record)
    selected = [
        record
        for image_records in selected_by_image.values()
        for record in image_records
    ]
    return selected, {
        "selected_images": len(selected_by_image),
        "selected_questions": len(selected),
        "image_limit": image_limit,
        "questions_per_image_limit": questions_per_image,
        "excluded_source_name_count": len(excluded),
        "overlap_skipped": overlap,
        "missing_skipped": missing,
        "per_image_limit_skipped": per_image_limit_skipped,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def run(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    lrs_root = repo / "data" / "LRS-VQA"
    refined_root = repo / "data" / "phase2-train-visualplanning-refined-v4"
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected, selection_summary = select_records(
        lrs_root,
        refined_root,
        args.limit,
        args.questions_per_image,
    )

    selection_path = output / "selection.jsonl"
    with selection_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(selected):
            image_path = lrs_root / record["image"]
            item = {
                "index": index,
                "question_id": record.get("question_id"),
                "image": record["image"],
                "image_sha256": sha256_file(image_path),
                "question": record.get("text", ""),
                "source_category": record.get("category"),
            }
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    (output / "selection_summary.json").write_text(
        json.dumps(selection_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    selected_type_counts = Counter(
        str(record.get("category", "unknown")) for record in selected
    )
    source_records = read_jsonl(lrs_root / "LRS_VQA_merged.jsonl")
    source_type_counts = Counter(
        str(record.get("category", "unknown")) for record in source_records
    )
    type_examples: dict[str, dict[str, Any]] = {}
    for record in source_records:
        category = str(record.get("category", "unknown"))
        type_examples.setdefault(
            category,
            {
                "question_id": record.get("question_id"),
                "source_image_field": record.get("image"),
                "source_question_field": record.get("text"),
                "source_ground_truth_field": record.get("ground_truth"),
                "source_hbox_field": record.get("hbox"),
                "source_rbox_field": record.get("rbox"),
            },
        )
    input_format_audit = {
        "source_question_type_counts": dict(sorted(source_type_counts.items())),
        "selected_question_type_counts": dict(sorted(selected_type_counts.items())),
        "source_examples_by_question_type": dict(sorted(type_examples.items())),
        "planner_sample_specific_input": {
            "destination": "VisualTaskPlanner.plan -> VisionLanguageClient.complete_json",
            "system": "system_prompt.txt plus planner_binding",
            "user_content_order": ["normalized_image_preview", "raw_question_text"],
            "not_sent_to_planner": [
                "category",
                "ground_truth",
                "hbox",
                "rbox",
                "dataset_name",
                "question_id",
            ],
        },
    }
    (output / "input_format_by_question_type.json").write_text(
        json.dumps(input_format_audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.select_only:
        print(json.dumps(selection_summary, sort_keys=True))
        return

    import torch
    from peft import PeftModel
    from scripts.finetune_qwen35_9b_visual_planner_lora import attach_roi_head

    adapter_path = Path(args.adapter).resolve()
    adapter_digest = sha256_file(adapter_path / "adapter_model.safetensors")
    settings = QwenSettings(
        model=str((repo / "models" / "Qwen3.5-9B").resolve()),
        cache_model_id="Qwen/Qwen3.5-9B:visual-planner-lora-final",
        revision=adapter_digest,
        max_tokens=args.max_tokens,
        dtype="bfloat16",
        device_map="auto",
        use_kernels=False,
        allow_download=False,
    )
    client = create_model("qwen3_5_transformers", settings=settings)
    hidden_size = int(client.model.config.text_config.hidden_size)
    attach_roi_head(client.model, hidden_size)
    client.model = PeftModel.from_pretrained(
        client.model,
        str(adapter_path),
        is_trainable=False,
        local_files_only=True,
    )
    client.model.eval()

    catalog = load_evidence_catalog(repo / "agents" / "evidence_catalog.json")
    planner = VisualTaskPlanner(
        client,
        system_prompt=(repo / "prompts" / "visual_task_plan_v5.md").read_text(encoding="utf-8"),
        prompt_version="v5",
        catalog=catalog,
        max_side=1080,
        roi_quantum=1024,
        vqa_assistance_scope=VQA_ASSISTANCE_SCOPE,
    )
    (output / "system_prompt.txt").write_text(planner.system_prompt, encoding="utf-8")
    manifest = {
        "repo_head": args.repo_head,
        "base_model_id": settings.cache_model_id,
        "adapter_sha256": adapter_digest,
        "adapter_path": str(adapter_path.relative_to(repo)),
        "selected": len(selected),
        "selected_images": selection_summary["selected_images"],
        "questions_per_image_limit": args.questions_per_image,
        "max_tokens": args.max_tokens,
        "dtype": "bfloat16",
        "device_map": "auto",
        "offline": True,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    results_path = output / "results.jsonl"
    succeeded = 0
    failed = 0
    with results_path.open("w", encoding="utf-8") as results:
        for index, record in enumerate(selected):
            sample_id = f"lrs-vqa-{record.get('question_id', index)}"
            sample_dir = output / "samples" / f"{index:03d}-{sample_id}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            relative_image = Path(record["image"])
            raw_input = {
                "sample_id": sample_id,
                "dataset": "LRS-VQA",
                "split": "test",
                "image": record["image"],
                "image_sha256": sha256_file(lrs_root / relative_image),
                "question": record.get("text", ""),
            }
            (sample_dir / "raw_input.json").write_text(
                json.dumps(raw_input, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            draft = SampleDraft(
                sample_id=sample_id,
                dataset="LRS-VQA",
                split="test",
                images=[ImageRef(image_id=relative_image.stem, path=relative_image, role="image")],
                question=record.get("text", ""),
                metadata={"question_id": str(record.get("question_id", ""))},
            )
            row: dict[str, Any] = {"index": index, **raw_input}
            try:
                plan = await planner.plan(
                    draft,
                    data_root=lrs_root,
                    artifact_dir=sample_dir,
                )
                row["status"] = "succeeded"
                row["parsed_output"] = plan.model_dump(mode="json")
                succeeded += 1
            except Exception as error:
                row["status"] = "failed"
                row["error_type"] = type(error).__name__
                failed += 1
            results.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            results.flush()
            print(json.dumps({"index": index, "sample_id": sample_id, "status": row["status"]}))
    summary = {"total": len(selected), "succeeded": succeeded, "failed": failed}
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="/home/lijia/M3")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--questions-per-image", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--repo-head", required=True)
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()
    if args.limit < 1 or args.questions_per_image < 1:
        parser.error("--limit and --questions-per-image must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
