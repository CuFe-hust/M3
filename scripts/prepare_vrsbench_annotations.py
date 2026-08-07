"""Convert VRSBench per-image annotations into clean caption/VQA JSONL files.
将 VRSBench 逐图标注转换为干净的 caption/VQA JSONL 标注文件。

Reads Annotations_{split}/Annotations_{split}/*.json and writes
VRSBench_{split}_caption.jsonl plus VRSBench_{split}_vqa.jsonl at the output root.
All image paths in the generated records are relative to the VRSBench dataset root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Map original VRSBench QA types to normalized task names.
# 将原始 VRSBench QA 类型映射为规范化任务名。
QA_TASK_MAP = {
    "object existence": "object_existence",
    "object category": "object_classification",
    "scene type": "scene_classification",
    "rural or urban": "scene_classification",
}

# Fixed instruction used for every caption record.
# 所有 caption 记录使用的统一指令。
CAPTION_INSTRUCTION = "Could you describe the contents of this image for me?"


def build_parser() -> argparse.ArgumentParser:
    """Build the annotation conversion CLI. / 构建标注转换 CLI。"""
    parser = argparse.ArgumentParser(
        description="Generate clean VRSBench caption and VQA JSONL annotation files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="VRSBench dataset root containing Annotations_* and Images_* directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to --root.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val"),
        default=("train", "val"),
        help="Splits to convert.",
    )
    return parser


def convert_split(root: Path, split: str, out_root: Path) -> dict[str, object]:
    """Convert one split and return generation statistics.
    转换一个 split 并返回生成统计。
    """
    ann_dir = root / f"Annotations_{split}" / f"Annotations_{split}"
    img_dir = f"Images_{split}/Images_{split}"
    if not ann_dir.is_dir():
        raise SystemExit(f"Annotation directory not found: {ann_dir}")
    cap_path = out_root / f"VRSBench_{split}_caption.jsonl"
    vqa_path = out_root / f"VRSBench_{split}_vqa.jsonl"
    task_counts: dict[str, int] = {}
    stats: dict[str, object] = {
        "files": 0,
        "caption": 0,
        "no_caption": 0,
        "vqa": 0,
        "image_field_mismatches": [],
        "missing_images": [],
    }
    with cap_path.open("w", encoding="utf-8") as fc, vqa_path.open("w", encoding="utf-8") as fq:
        for annotation_file in sorted(ann_dir.glob("*.json")):
            stats["files"] += 1
            data = json.loads(annotation_file.read_text(encoding="utf-8"))
            # The annotation filename is authoritative for image_id because the
            # dataset pairs every image with its same-named annotation JSON.
            # 以标注文件名为准确定 image_id，因为数据集中图片与同名标注 JSON 一一对应。
            image_id = annotation_file.stem + ".png"
            original_image = data.get("image")
            image_mismatch = original_image is not None and original_image != image_id
            if image_mismatch:
                stats["image_field_mismatches"].append((annotation_file.name, original_image))
            image_relpath = f"{img_dir}/{image_id}"
            if not (root / image_relpath).is_file():
                stats["missing_images"].append(image_relpath)
            relative_annotation = annotation_file.relative_to(root).as_posix()
            if data.get("caption"):
                source: dict[str, object] = {
                    "annotation_file": relative_annotation,
                    "original_field": "caption",
                }
                if image_mismatch:
                    source["original_image"] = original_image
                record = {
                    "id": f"vrsbench/{split}/{image_id}/caption",
                    "dataset": "VRSBench",
                    "split": split,
                    "task": "caption",
                    "image_id": image_id,
                    "image": image_relpath,
                    "instruction": CAPTION_INSTRUCTION,
                    "caption": data["caption"],
                    "source": source,
                }
                fc.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                stats["caption"] += 1
            else:
                stats["no_caption"] += 1
            for pair in data.get("qa_pairs", []):
                original_type = pair.get("type")
                task = QA_TASK_MAP.get(original_type)
                if task is None:
                    continue
                question = pair.get("question")
                answer = pair.get("answer")
                ques_id = pair.get("ques_id")
                if not question or not answer:
                    raise RuntimeError(
                        f"Missing question or answer: {annotation_file.name} ques_id={ques_id}"
                    )
                source = {
                    "annotation_file": relative_annotation,
                    "original_type": original_type,
                    "ques_id": ques_id,
                }
                if image_mismatch:
                    source["original_image"] = original_image
                record = {
                    "id": f"vrsbench/{split}/{image_id}/{task}/{ques_id}",
                    "dataset": "VRSBench",
                    "split": split,
                    "task": task,
                    "image_id": image_id,
                    "image": image_relpath,
                    "question": question,
                    "answer": answer,
                    "source": source,
                }
                fq.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                stats["vqa"] += 1
                task_counts[task] = task_counts.get(task, 0) + 1
    stats["tasks"] = task_counts
    if stats["missing_images"]:
        raise SystemExit(
            f"Missing images ({len(stats['missing_images'])}): "
            + ", ".join(stats["missing_images"][:5])
        )
    return stats


def main() -> int:
    args = build_parser().parse_args()
    out_root = args.output_dir if args.output_dir is not None else args.root
    out_root.mkdir(parents=True, exist_ok=True)
    all_stats: dict[str, object] = {}
    for split in args.splits:
        all_stats[split] = convert_split(args.root, split, out_root)
    print(json.dumps(all_stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
