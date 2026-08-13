#!/usr/bin/env python3
"""Prepare the phase2-train training subset from raw VRSBench-full annotations.

从 VRSBench-full 原始逐图标注生成 phase2-train 训练子集：

- ``Annotations_train/*.json`` 按稳定 sha256 排序切分为 train / val（9:1）；
- 输出遵循原始数据集的组织方式：**一行 = 一张图片**，该图的 grounding
  （``objects``）与 VQA（``qa_pairs``）放在同一条记录内，不拆成不同 task 的记录；
- ``Annotations_val/*.json`` 作为 test 不做任何提取，仅输出原始文件清单。

输出目录：``data/phase2-train/VRSBench/``，含 3 个 JSONL + manifest.json + README.md。

设计约束：
- 切分与行序完全确定性（sha256 稳定排序，不依赖进程随机 hash）；
- 原始标注逐字保留（question / answer / referring_sentence / obj_coord 不改写，
  对象与问答字段名沿用原始数据集命名）；
- box_999 是派生视图（0..1 归一化 -> 0..999 整数 xyxy），仅当原始坐标合法时给出；
  越界（目标延伸到图像边缘）与退化框只保留原始坐标并在 manifest 中计数；
- 不含任何机器绝对路径；图片只引用不复制。
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "data" / "VRSBench-full"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "phase2-train" / "VRSBench"

# 官方 question type -> 任务名（与 VRSBench-mission-type 子集保持一致）。
# Official question type -> mission name, consistent with VRSBench-mission-type.
OFFICIAL_TYPE_TO_MISSION = {
    "object quantity": "counting",
    "object position": "object_position",
    "object existence": "object_existence",
    "object category": "object_category",
    "object color": "object_color",
    "scene type": "scene_type",
    "image": "image",
    "object shape": "object_shape",
    "object size": "object_size",
    "reasoning": "reasoning",
    "rural or urban": "rural_or_urban",
    "object direction": "object_direction",
}

TRAIN_RATIO = 0.9
DATASET = "VRSBench"


def stable_order(paths: list[Path]) -> list[Path]:
    """Sort annotation files by sha256(basename), tie-broken by basename.

    按 sha256(basename) 稳定排序；跨 Python 版本与平台可复现。
    """
    def key(p: Path) -> tuple[str, str]:
        name = p.name
        return hashlib.sha256(name.encode("utf-8")).hexdigest(), name

    return sorted(paths, key=key)


def derive_box_999(raw) -> list[int] | None:
    """Derive the 0..999 integer xyxy view from a normalized 0..1 obj_coord.

    Returns None when the raw box is not a strictly-ordered 4-number tuple fully
    inside [0, 1] -- such boxes (image-edge objects or degenerate geometry) keep
    only the raw coordinates and are counted in the manifest.
    """
    if (
        not isinstance(raw, list)
        or len(raw) != 4
        or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in raw)
    ):
        return None
    x1, y1, x2, y2 = raw
    if not all(0.0 <= v <= 1.0 for v in raw):
        return None
    if x1 >= x2 or y1 >= y2:
        return None
    return [round(v * 999) for v in raw]


def collect_annotation(path: Path) -> dict:
    """Load one per-image annotation and normalize it to a plain dict.

    Image identity follows the annotation file stem (mission-type convention);
    an annotation whose official ``image`` field points to a different image
    keeps that field under ``original_image`` for audit.
    """
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    image_id = path.stem + ".png"
    record = {
        "image_id": image_id,
        "objects": list(data.get("objects", [])),
        "qa_pairs": list(data.get("qa_pairs", [])),
    }
    official_image = data.get("image", "")
    if official_image and official_image != image_id:
        record["original_image"] = official_image
    return record


def build_record(annot: dict, split: str, ann_dir: str, img_dir: str, stats: dict) -> dict:
    """Build one per-image record: grounding objects + VQA pairs nested together.

    构造单条图片级记录：该图的 grounding 对象与 VQA 问答对放在同一记录内。
    对象/问答字段沿用原始数据集命名，额外追加派生字段（box_999 / box_valid / task）。
    """
    objects = []
    for obj in annot["objects"]:
        raw = obj.get("obj_coord")
        box_999 = derive_box_999(raw) if isinstance(raw, list) else None
        if box_999 is None:
            stats["box_999_excluded"] += 1
            if not all(math.isfinite(v) for v in raw) if isinstance(raw, list) else False:
                stats["non_finite"] += 1
            elif isinstance(raw, list) and all(0.0 <= v <= 1.0 for v in raw):
                # 在 [0,1] 内但退化（x1 >= x2 或 y1 >= y2）。
                stats["degenerate"] += 1
            else:
                # 越界：目标延伸到图像边缘，合法但无法直接派生 0..999 视图。
                stats["out_of_range"] += 1
        entry = dict(obj)  # 原始字段逐字保留，命名沿用原始数据集。
        entry["box_999"] = box_999
        entry["box_valid"] = box_999 is not None
        objects.append(entry)

    qa_pairs = []
    for qa in annot["qa_pairs"]:
        official_type = qa.get("type", "")
        mission = OFFICIAL_TYPE_TO_MISSION.get(official_type, "general_vqa")
        if mission == "general_vqa" and official_type:
            stats["unknown_qa_type"].append(official_type)
        entry = dict(qa)  # 原始问答逐字保留。
        entry["task"] = mission
        qa_pairs.append(entry)

    source = {"annotation_file": f"{ann_dir}/{annot['_name']}"}
    if "original_image" in annot:
        source["original_image"] = annot["original_image"]

    return {
        "id": f"vrsbench/{split}/{annot['image_id']}",
        "dataset": DATASET,
        "split": split,
        "image_id": annot["image_id"],
        "image": f"{img_dir}/{annot['image_id']}",
        "objects": objects,
        "qa_pairs": qa_pairs,
        "source": source,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write standard single-line JSONL: one JSON object per line, keys in
    logical (insertion) order.

    标准单行 JSONL：每行一条记录，字段保持逻辑顺序，训练代码可直接消费。
    """
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """Parse a single-line JSONL file back to records (read-back verification).

    按行解析单行 JSONL 用于读回验证。
    """
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT

    train_ann_dir = source / "Annotations_train"
    val_ann_dir = source / "Annotations_val"

    train_files = stable_order(train_ann_dir.glob("*.json"))
    val_files = stable_order(val_ann_dir.glob("*.json"))
    total = len(train_files)
    n_train = int(total * TRAIN_RATIO)
    train_split, val_split = train_files[:n_train], train_files[n_train:]

    stats = {
        "parse_errors": [],
        "box_999_excluded": 0,
        "degenerate": 0,
        "out_of_range": 0,
        "non_finite": 0,
        "missing_images": [],
        "original_image_anomalies": [],
        "unknown_qa_type": [],
    }

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    test_rows_out: list[dict] = []

    # (files, target rows, split label, annotation dir, image dir)
    groups = [
        (train_split, train_rows, "train", "Annotations_train", "Images_train"),
        (val_split, val_rows, "val", "Annotations_train", "Images_train"),
        (val_files, test_rows_out, "test", "Annotations_val", "Images_val"),
    ]
    for files, rows, split, ann_dir, img_dir in groups:
        for path in files:
            try:
                annot = collect_annotation(path)
            except (json.JSONDecodeError, OSError) as exc:
                stats["parse_errors"].append(f"{path.name}: {type(exc).__name__}")
                continue
            annot["_name"] = path.name
            if "original_image" in annot:
                stats["original_image_anomalies"].append(path.name)
            if not (source / img_dir / f"{path.stem}.png").exists():
                stats["missing_images"].append(f"{img_dir}/{path.stem}.png")
            if split != "test":
                rows.append(build_record(annot, split, ann_dir, img_dir, stats))
            else:
                source_field = {"annotation_file": f"{ann_dir}/{path.name}"}
                if "original_image" in annot:
                    source_field["original_image"] = annot["original_image"]
                rows.append(
                    {
                        "id": f"vrsbench/test/{annot['image_id']}",
                        "dataset": DATASET,
                        "split": "test",
                        "image_id": annot["image_id"],
                        "image": f"{img_dir}/{annot['image_id']}",
                        "annotation_file": f"{ann_dir}/{path.name}",
                        "processed": False,
                        "source": source_field,
                    }
                )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_files = {
        "VRSBench_train.jsonl": train_rows,
        "VRSBench_val.jsonl": val_rows,
        "VRSBench_test_raw.jsonl": test_rows_out,
    }
    for name, rows in out_files.items():
        write_jsonl(out_dir / name, rows)

    # ---- self checks ------------------------------------------------------
    # 读回已写文件，确认每行可解析且记录数一致。
    for name, rows in out_files.items():
        read_back = read_jsonl(out_dir / name)
        assert len(read_back) == len(rows), f"{name}: read-back count mismatch"

    def check_unique(rows: list[dict], label: str) -> int:
        ids = [r["id"] for r in rows]
        dup = len(ids) - len(set(ids))
        assert dup == 0, f"{label}: {dup} duplicate ids"
        return len(ids)

    train_ids = check_unique(train_rows, "train")
    val_ids = check_unique(val_rows, "val")
    test_ids = check_unique(test_rows_out, "test")

    train_imgs = {r["image_id"] for r in train_rows}
    val_imgs = {r["image_id"] for r in val_rows}
    assert train_imgs.isdisjoint(val_imgs), "train/val image leakage"
    assert train_imgs == {p.stem + ".png" for p in train_split}, "train image set not closed"
    assert val_imgs == {p.stem + ".png" for p in val_split}, "val image set not closed"
    assert {r["image_id"] for r in test_rows_out} == {p.stem + ".png" for p in val_files}, "test listing not closed"

    # ---- manifest ---------------------------------------------------------
    all_objects = [obj for r in train_rows + val_rows for obj in r["objects"]]
    all_qa = [qa for r in train_rows + val_rows for qa in r["qa_pairs"]]
    vqa_breakdown: dict[str, int] = {}
    for qa in all_qa:
        vqa_breakdown[qa["type"]] = vqa_breakdown.get(qa["type"], 0) + 1

    manifest = {
        "dataset": DATASET,
        "subset": "phase2-train",
        "version": "2",
        "description": "VRSBench-full 原始逐图标注的 phase2 训练子集：Annotations_train 按稳定 hash 9:1 切分为 train/val；每行一条图片级记录，grounding（objects）与 VQA（qa_pairs）在同一记录内（遵循原始数据集组织方式）；Annotations_val 作为 test 仅保留原始文件清单，不做提取。",
        "split_rule": {
            "method": "sha256(basename) stable ordering; first 90% train, remainder val",
            "ratio": f"{TRAIN_RATIO}:{1 - TRAIN_RATIO}",
            "annotations_train_total": total,
            "train_images": len(train_split),
            "val_images": len(val_split),
            "test_images": len(val_files),
            "note": "Deterministic across Python versions and platforms; independent of the phase1 split.",
        },
        "sources": {
            "train": {"root_relative": "../VRSBench-full", "annotation_dir": "../VRSBench-full/Annotations_train"},
            "val": {"root_relative": "../VRSBench-full", "annotation_dir": "../VRSBench-full/Annotations_train", "note": "val is a subset of Annotations_train, not the official Annotations_val"},
            "test": {"root_relative": "../VRSBench-full", "annotation_dir": "../VRSBench-full/Annotations_val", "note": "unprocessed raw listing"},
        },
        "format": {
            "layout": "standard single-line JSONL; one line = one image; grounding objects and VQA pairs nested in the same record (objects[] / qa_pairs[]), following the original per-image annotation organization",
            "fields": "object entries keep original field names plus derived box_999/box_valid; qa entries keep original fields plus mapped task",
        },
        "counts": {
            "VRSBench_train.jsonl": train_ids,
            "VRSBench_val.jsonl": val_ids,
            "VRSBench_test_raw.jsonl": test_ids,
            "grounding_objects": len(all_objects),
            "vqa_pairs": len(all_qa),
            "box_999_available": sum(1 for o in all_objects if o["box_valid"]),
            "vqa_by_official_type": vqa_breakdown,
        },
        "anomalies": {
            "parse_errors": stats["parse_errors"],
            "box_999_excluded_total": stats["box_999_excluded"],
            "out_of_range": stats["out_of_range"],
            "degenerate": stats["degenerate"],
            "non_finite": stats["non_finite"],
            "missing_images": stats["missing_images"],
            "original_image_anomalies": stats["original_image_anomalies"],
            "unknown_qa_types": sorted(set(stats["unknown_qa_type"])),
        },
        "checksums": {name: sha256_file(out_dir / name) for name in out_files},
        "files": {
            "train": "VRSBench_train.jsonl (per-image records: grounding + VQA)",
            "val": "VRSBench_val.jsonl (per-image records: grounding + VQA)",
            "test": "VRSBench_test_raw.jsonl (unprocessed annotation listing)",
        },
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")

    write_readme(out_dir, manifest)

    print(f"train images: {len(train_split)}  val images: {len(val_split)}  test images: {len(val_files)}")
    print(f"train rows: {train_ids}  val rows: {val_ids}  test rows: {test_ids}  (每行 = 一张图)")
    print(f"grounding objects: {len(all_objects)}  vqa pairs: {len(all_qa)}  box_999 available: {sum(1 for o in all_objects if o['box_valid'])}")
    print(f"missing images: {len(stats['missing_images'])}  box_999 excluded: {stats['box_999_excluded']} (oob {stats['out_of_range']} + degenerate {stats['degenerate']})")
    print(f"output: {out_dir}")
    return 0


def write_readme(out_dir: Path, manifest: dict) -> None:
    """Write the dataset README with counts interpolated from the run.

    生成数据集 README，统计数字从本次运行结果动态填入，保证整体可复现。
    """
    split = manifest["split_rule"]
    counts = manifest["counts"]
    anomalies = manifest["anomalies"]
    text = f"""# VRSBench（phase2-train）

从 VRSBench 官方全量数据 `../../VRSBench-full` 的最原始逐图标注生成的 phase2 训练子集：
grounding + VQA 混合，供 Qwen3-VL LoRA 阶段训练使用。

## 切分

`../../VRSBench-full/Annotations_train/*.json`（{split["annotations_train_total"]:,} 个文件）按**稳定 sha256(basename)
排序**切分 9:1：

```text
train: {split["train_images"]:,} 张（90%）
val:   {split["val_images"]:,} 张（10%）
test:  {split["test_images"]:,} 张（../../VRSBench-full/Annotations_val/*.json，官方评测集，仅清单，不做提取）
```

- 切分与 phase1（`data/vrsbench-phase1-train`）无关，两者 val 集合不同，但同为 9:1；
- 切分跨 Python 版本与平台可复现（不依赖进程随机 hash）；
- train / val / test 三集合两两不重叠。

## 文件

| 文件 | 说明 |
|---|---|
| `VRSBench_train.jsonl` | train：每行一条**图片级记录**（{counts["VRSBench_train.jsonl"]:,} 条） |
| `VRSBench_val.jsonl` | val：同上（{counts["VRSBench_val.jsonl"]:,} 条） |
| `VRSBench_test_raw.jsonl` | test：仅 `Annotations_val` 原始文件清单（`processed: false`），未提取任何标注（{counts["VRSBench_test_raw.jsonl"]:,} 条） |
| `manifest.json` | 切分规则、提取口径、统计与输出文件校验和 |
| `README.md` | 本说明 |

**格式**：标准单行 JSONL——**一行 = 一张图片**。该图的 grounding 对象
（`objects`）与 VQA 问答对（`qa_pairs`）放在同一条记录内（遵循原始数据集组织方式，
不拆成不同 task 的记录）。字段按逻辑顺序排列，训练代码可直接逐行解析。

## 记录结构

```json
{{
  "id": "vrsbench/train/<image>",
  "dataset": "VRSBench",
  "split": "train | val",
  "image_id": "<image>",
  "image": "Images_train/<image>",
  "objects": [
    {{
      "obj_id": 0,
      "referring_sentence": "The small vehicle with a white color is positioned near the top-middle of the image.",
      "obj_cls": "vehicle",
      "obj_coord": [0.38, 0.2, 0.45, 0.26],
      "box_999": [380, 200, 450, 260],
      "box_valid": true,
      ...
    }}
  ],
  "qa_pairs": [
    {{
      "ques_id": 1,
      "question": "What is the main structure in the center of the image?",
      "type": "scene type",
      "task": "scene_type",
      "answer": "expressway-toll-station"
    }}
  ],
  "source": {{"annotation_file": "Annotations_train/<image>.json"}}
}}
```

### `objects`（grounding）

来自原始标注的 `objects` 数组，**逐字保留**全部原始字段（`obj_id`、`referring_sentence`、
`obj_cls`、`obj_coord`、`obj_corner`、`is_unique`、`flag`、`obj_position`、
`obj_rel_position`、`obj_size`、`obj_rel_size`），并追加派生字段：

- `box_999`：0..1 归一化 `xyxy` → 0..999 整数 `xyxy`（四舍五入），仅供训练目标使用；
  与官方 grounding 评测指标不做等价性声明；
- `box_valid`：为 `false` 的记录只有原始 `obj_coord` 没有 `box_999` —— 原始框越界
  （目标延伸到图像边缘）或退化（`x1 >= x2` 或 `y1 >= y2`）。共
  {anomalies["box_999_excluded_total"]:,} 条（越界 {anomalies["out_of_range"]:,} +
  退化 {anomalies["degenerate"]:,}），原始坐标保留，未做钳制；如需钳制后进入训练，
  应在下游导出阶段显式处理并记录。

### `qa_pairs`（VQA）

来自原始标注的 `qa_pairs` 数组，全部 **12 类官方 question type** 全量提取，
question / answer / type 逐字保留；追加 `task` 为官方类型映射的任务名
（与 `VRSBench-mission-type` 子集一致）：

| type（官方） | task |
|---|---|
| object quantity | `counting` |
| object position | `object_position` |
| object existence | `object_existence` |
| object category | `object_category` |
| object color | `object_color` |
| scene type | `scene_type` |
| image | `image` |
| object shape | `object_shape` |
| object size | `object_size` |
| reasoning | `reasoning` |
| rural or urban | `rural_or_urban` |
| object direction | `object_direction` |

## Test 清单（`VRSBench_test_raw.jsonl`）

```json
{{
  "id": "vrsbench/test/<image>",
  "dataset": "VRSBench",
  "split": "test",
  "image_id": "<image>",
  "image": "Images_val/<image>",
  "annotation_file": "Annotations_val/<image>.json",
  "processed": false,
  "source": {{"annotation_file": "Annotations_val/<image>.json"}}
}}
```

test 不做任何提取（grounding / VQA 均不解析），仅冻结官方评测集的文件清单。

## 说明

- 两个标注文件（`02726_0000.json`、`P2708_0017.json`）的官方 `image` 字段指向
  其他图片；本子集按标注文件名定图像身份（与 mission-type 约定一致），
  官方字段保留在 `source.original_image` 供审计；
- 图片未复制；`image` 与 `annotation_file` 均为相对本数据集根目录
  （即 `data/phase2-train/VRSBench/`，解析时拼接到 `data/VRSBench-full/` 下）的路径，
  不含机器绝对路径；
- 提取为纯文本处理，未读取图片、未调用模型、未联网；
- 重新生成：`python3 scripts/prepare_vrsbench_phase2.py`（结果字节级可复现）。
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
