"""Smoke test: process N LRS-VQA-Supplement large images with GPU memory monitoring."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import scripts.finetune_qwen35_9b_general_vqa_agent_lora as ft
from application.bootstrap import assemble_runtime
from application.settings import load_settings


def build_args(output_dir: str, count: int) -> SimpleNamespace:
    return SimpleNamespace(
        model_path="models/Qwen3.5-9B",
        base_model_id="Qwen/Qwen3.5-9B",
        config="configs/local.yaml",
        annotation_root="data/2026-08-24_vqa-agent-io",
        vrsbench_root="data/vrsbench",
        large_image_root="data/phase2-train-visualplanning-refined-v4",
        supplement_root="data/20260824-visual-planner-supplement",
        output_dir=output_dir,
        max_train_samples=count,
        max_eval_samples=0,
        prepare_only=True,
        max_seq_length=6144,
        lora_rank=32,
        lora_alpha=64,
        lora_dropout=0.05,
        learning_rate=1e-4,
        weight_decay=0.01,
        epochs=2.0,
        gradient_accumulation_steps=16,
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        logging_steps=10,
        eval_steps=100,
        save_steps=100,
        save_total_limit=3,
        max_steps=-1,
        seed=42,
        resume_from_checkpoint=None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--output-dir", default="outputs/finetune/qwen35-9b-general-vqa-agent-lora-lrs-1000-smoke")
    args = parser.parse_args(argv)

    ft.TRAIN_SOURCES = (("LRS-VQA-Supplement", "LRS-VQA-Supplement/train.jsonl"),)
    ft.VALIDATION_SOURCES = (("LRS-VQA-Supplement", "LRS-VQA-Supplement/validation.jsonl"),)

    run_args = build_args(args.output_dir, args.count)
    output_dir = Path(run_args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_sha256 = ft.sha256_file(Path(run_args.config).resolve())

    train_source = ft.load_source_records(run_args, split="train", max_samples=args.count)
    print("loaded LRS-VQA-Supplement train records:", len(train_source), flush=True)

    capture = ft.CaptureQwenClient()
    settings = load_settings(Path(run_args.config).resolve(), environ={})
    components = assemble_runtime(settings, project_root=REPO_ROOT, qwen_client=capture)

    records = asyncio.run(ft.prepare_records(
        run_args,
        train_source,
        split="train",
        capture=capture,
        components=components,
        output_dir=output_dir,
        config_sha256=config_sha256,
    ))
    print("prepared records:", len(records), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
