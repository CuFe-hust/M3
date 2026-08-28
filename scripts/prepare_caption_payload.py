#!/usr/bin/env python
"""prepare_caption_payload.py — convert existing caption annotations to the
standard caption user-payload form without touching adapters.

将既有 caption 数据集注解直接转换为标准 caption user-payload 形式（不改
adapter、不改原始注解文件，输出新的 JSONL）。评测脚本直接消费本脚本的
输出，不再经过 data adapter。

输入（只读，已有数据集）:
  XLRS:     <xlrs_root>/XLRS-Bench_caption_en/train/captions.json + train/images/
  VRSBench: <vrsbench_root>/VRSBench_EVAL_Cap.json + Images_val/
输出（新文件，不覆盖源数据）:
  <xlrs_root>/XLRS-Bench_caption_en/train/captions.user_payload.jsonl
  <vrsbench_root>/VRSBench_EVAL_Cap.user_payload.jsonl

每行统一 schema（caption user-payload 形式）:
  sample_id     稳定且跨数据集唯一（xlrs-<id> / vrsbench-<question_id>）
  dataset       XLRS-Bench | VRSBench
  split         train | validation
  image         相对数据集 root 的图片路径
  user_payload  {"task": "caption", "question": "Describe the image in detail."}
  references    全部参考 caption（list[str]）
  metadata      审计字段（source_index、image_id、原始注解来源）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CAPTION_QUESTION = "Describe the image in detail."
SCHEMA_VERSION = "caption-user-payload-v1"


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _prepare_xlrs(root: Path, out: Path, check_images: bool) -> dict:
    release = root / "XLRS-Bench_caption_en"
    annotations = release / "train" / "captions.json"
    rows = json.loads(annotations.read_text(encoding="utf-8"))
    prepared: list[dict] = []
    for index, row in enumerate(rows):
        image = str(row.get("image", "")).strip()
        answer = row.get("answer")
        texts = answer if isinstance(answer, list) else [answer]
        references = [str(t).strip() for t in texts if str(t).strip()]
        if not image or not references:
            raise ValueError(f"XLRS row {index}: missing image or caption")
        if check_images and not (release / image).is_file():
            raise FileNotFoundError(f"XLRS image missing: {release / image}")
        prepared.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": f"xlrs-{row.get('id', index)}",
                "dataset": "XLRS-Bench",
                "split": "train",
                "image": image,
                "user_payload": {
                    "task": "caption",
                    "question": CAPTION_QUESTION,
                },
                "references": references,
                "metadata": {
                    "source": "XLRS-Bench_caption_en",
                    "source_index": index,
                    "source_id": row.get("id"),
                },
            }
        )
    _atomic_write_jsonl(out, prepared)
    return {"dataset": "XLRS-Bench", "count": len(prepared), "output": str(out)}


def _prepare_vrsbench(root: Path, out: Path, check_images: bool) -> dict:
    annotations = root / "VRSBench_EVAL_Cap.json"
    rows = json.loads(annotations.read_text(encoding="utf-8"))
    prepared: list[dict] = []
    for index, row in enumerate(rows):
        image_id = str(row.get("image_id", "")).strip()
        ground_truth = row.get("ground_truth")
        references = [str(ground_truth).strip()] if ground_truth else []
        if not image_id or not references:
            raise ValueError(f"VRSBench row {index}: missing image_id or ground_truth")
        if check_images and not (root / "Images_val" / image_id).is_file():
            raise FileNotFoundError(
                f"VRSBench image missing: {root / 'Images_val' / image_id}"
            )
        prepared.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": f"vrsbench-{row.get('question_id', index)}",
                "dataset": "VRSBench",
                "split": "validation",
                "image": f"Images_val/{image_id}",
                "user_payload": {
                    "task": "caption",
                    "question": CAPTION_QUESTION,
                },
                "references": references,
                "metadata": {
                    "source": "VRSBench_EVAL_Cap",
                    "source_index": index,
                    "image_id": image_id,
                    "question_id": row.get("question_id"),
                    "source_dataset": row.get("dataset"),
                },
            }
        )
    _atomic_write_jsonl(out, prepared)
    return {"dataset": "VRSBench", "count": len(prepared), "output": str(out)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prepare_caption_payload.py",
        description="convert caption annotations to standard user-payload JSONL",
    )
    parser.add_argument("--xlrs-root", default=None, help="XLRS-Bench dataset root")
    parser.add_argument("--vrsbench-root", default=None, help="VRSBench dataset root")
    parser.add_argument("--check-images", action="store_true", help="verify image files exist")
    parser.add_argument("--preview", type=int, default=0, help="print first N rows")
    args = parser.parse_args(argv)

    summary: list[dict] = []
    if args.xlrs_root:
        root = Path(args.xlrs_root)
        out = root / "XLRS-Bench_caption_en" / "train" / "captions.user_payload.jsonl"
        summary.append(_prepare_xlrs(root, out, args.check_images))
    if args.vrsbench_root:
        root = Path(args.vrsbench_root)
        out = root / "VRSBench_EVAL_Cap.user_payload.jsonl"
        summary.append(_prepare_vrsbench(root, out, args.check_images))
    if not summary:
        parser.error("at least one of --xlrs-root / --vrsbench-root is required")
    print(
        json.dumps(
            {"status": "ok", "schema_version": SCHEMA_VERSION, "datasets": summary},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    if args.preview:
        for entry in summary:
            path = Path(entry["output"])
            print(f"\n=== preview {path} ===")
            with path.open(encoding="utf-8") as handle:
                for _ in range(args.preview):
                    line = handle.readline()
                    if not line:
                        break
                    print(json.dumps(json.loads(line), ensure_ascii=False, indent=1)[:1200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
