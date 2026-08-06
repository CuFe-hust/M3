"""Audited VRSBench adapter for the official caption/VQA/grounding releases.

经审计的 VRSBench 适配器：加载官方 caption/VQA/grounding 评测发布。
- 标注定位：直连根目录或全树唯一匹配；多候选显式报错。
- 字段验证：每任务必需字段，缺失显式失败；source row 保留到 GroundTruth.raw/metadata。
- VQA 每行调用任务规范化器，直接设置标准任务（不再全部 general_vqa）。
- 图片解析只使用受审计候选路径，不通过全盘 basename 字典静默覆盖重复文件。
只读源数据；不导入 routing / agents；不拼接任何 Agent prompt。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from data.adapters.base import AdapterProbe, DatasetProbeError, read_json_rows
from data.adapters.vrsbench.task_normalizer import normalize_task
from data.schema import GroundTruth, ImageRef, UnifiedSample, stable_sample_id

ADAPTER_VERSION = "official-eval-v1"
SUPPORTED_TASKS = frozenset({"general_vqa", "caption", "grounding"})
# Official release filenames per task. / 各任务的官方发布文件名。
ANNOTATION_FILES = {
    "general_vqa": "VRSBench_EVAL_vqa.json",
    "caption": "VRSBench_EVAL_Cap.json",
    "grounding": "VRSBench_EVAL_Det.json",
}
REQUIRED_FIELDS = {
    "general_vqa": ("image_id", "question", "ground_truth", "question_id", "type"),
    "caption": ("image_id", "caption"),
    "grounding": ("image_id", "objects"),
}


class VRSBenchAdapter:
    """Read-only adapter over the official VRSBench evaluation releases.
    只读加载官方 VRSBench 评测发布的适配器。"""

    name = "VRSBench"
    supported_tasks = SUPPORTED_TASKS

    def probe(self, root: Path) -> AdapterProbe:
        """Validate the selected official annotation before yielding samples.
        Only the VQA annotation is mandatory; caption/grounding releases are
        validated when present, and multiple candidates always fail.
        在产出样本前校验所选官方标注。VQA 标注必需；caption/grounding 发布
        存在时校验；多候选一律显式失败。"""
        vqa_matches = self._annotation_matches(root, "general_vqa")
        if len(vqa_matches) > 1:
            raise DatasetProbeError(
                f"Expected exactly one VRSBench_EVAL_vqa.json; observed {len(vqa_matches)} under {root}"
            )
        if not vqa_matches:
            raise DatasetProbeError(
                f"Official VRSBench VQA annotation is missing under {root}"
            )
        vqa_rows = read_json_rows(vqa_matches[0])
        self._validate_fields("general_vqa", vqa_rows, vqa_matches[0])
        for task in ("caption", "grounding"):
            matches = self._annotation_matches(root, task)
            if len(matches) > 1:
                raise DatasetProbeError(
                    f"Expected exactly one {ANNOTATION_FILES[task]}; observed {len(matches)} under {root}"
                )
            if matches:
                rows = read_json_rows(matches[0])
                self._validate_fields(task, rows, matches[0])
        observed = tuple(sorted({key for row in vqa_rows[:20] for key in row}))
        return AdapterProbe(
            dataset=self.name,
            version=ADAPTER_VERSION,
            sample_file=vqa_matches[0],
            observed_fields=observed,
            sample_count=len(vqa_rows),
        )

    def iter_samples(self, root: Path, split: str, task: str) -> Iterator[UnifiedSample]:
        """Yield schema-validated unified samples in source order.
        按源顺序产出具 schema 校验的统一样本。"""
        if split not in {"validation", "val"}:
            raise DatasetProbeError(
                f"Official VRSBench annotations support split='validation' only, got {split!r}"
            )
        if task not in SUPPORTED_TASKS:
            raise DatasetProbeError(
                f"VRSBench does not support task={task!r}; supported={sorted(SUPPORTED_TASKS)}"
            )
        annotation = self._annotation_path(root, task)
        rows = read_json_rows(annotation)
        self._validate_fields(task, rows, annotation)
        for index, row in enumerate(rows):
            image_id = str(row["image_id"])
            image_path = self._image_path(root, annotation.parent, image_id)
            if task == "general_vqa":
                yield self._vqa_sample(root, split, row, image_path, index)
            elif task == "caption":
                yield self._caption_sample(root, split, row, image_path, index)
            else:
                yield from self._grounding_samples(root, split, row, image_path, index)

    # ── per-task mapping / 分任务映射 ────────────────────────────────────────

    def _vqa_sample(
        self,
        root: Path,
        split: str,
        row: dict[str, Any],
        image_path: Path,
        index: int,
    ) -> UnifiedSample:
        image_id = str(row["image_id"])
        question = str(row["question"])
        question_id = str(row["question_id"])
        question_type = str(row["type"])
        normalization = normalize_task(question, question_type)
        metadata: dict[str, Any] = {
            "source": "VRSBench",
            "source_index": index,
            "question_id": question_id,
            "question_type": question_type,
            "source_dataset": row.get("dataset"),
            "adapter_version": ADAPTER_VERSION,
        }
        return UnifiedSample(
            sample_id=stable_sample_id(
                dataset=self.name,
                split=split,
                source_id=question_id,
                relative_image_paths=[image_path.relative_to(root)],
                question=question,
                source_index=index,
            ),
            dataset=self.name,
            split="validation",
            task=normalization.normalized_task,
            images=[ImageRef(image_id=image_id, path=image_path, role="image")],
            question=question,
            ground_truth=GroundTruth(
                answers=[str(row["ground_truth"])],
                raw={
                    "adapter_version": ADAPTER_VERSION,
                    "image_id": image_id,
                    "question_id": row["question_id"],
                    "question_type": question_type,
                    "source_dataset": row.get("dataset"),
                },
            ),
            metadata=metadata,
            normalization=normalization,
        )

    def _caption_sample(
        self,
        root: Path,
        split: str,
        row: dict[str, Any],
        image_path: Path,
        index: int,
    ) -> UnifiedSample:
        caption = str(row["caption"])
        return UnifiedSample(
            sample_id=stable_sample_id(
                dataset=self.name,
                split=split,
                source_id=None,
                relative_image_paths=[image_path.relative_to(root)],
                question="",
                source_index=index,
            ),
            dataset=self.name,
            split="validation",
            task="caption",
            images=[ImageRef(image_id=str(row["image_id"]), path=image_path, role="image")],
            question="",
            ground_truth=GroundTruth(
                answers=[caption],
                raw={
                    "adapter_version": ADAPTER_VERSION,
                    "image_id": str(row["image_id"]),
                    "source_row": dict(row),
                },
            ),
            metadata={
                "source": "VRSBench",
                "source_index": index,
                "adapter_version": ADAPTER_VERSION,
            },
        )

    def _grounding_samples(
        self,
        root: Path,
        split: str,
        row: dict[str, Any],
        image_path: Path,
        index: int,
    ) -> Iterator[UnifiedSample]:
        objects = row["objects"]
        if not isinstance(objects, list) or not objects:
            raise DatasetProbeError(f"VRSBench grounding row {index} has invalid objects field")
        for object_index, obj in enumerate(objects):
            if not isinstance(obj, dict) or "name" not in obj or "bbox" not in obj:
                raise DatasetProbeError(
                    f"VRSBench grounding row {index} object {object_index} misses name/bbox"
                )
            name = str(obj["name"])
            bbox = list(obj["bbox"])
            yield UnifiedSample(
                sample_id=stable_sample_id(
                    dataset=self.name,
                    split=split,
                    source_id=None,
                    relative_image_paths=[image_path.relative_to(root)],
                    question=f"Locate the {name}.",
                    source_index=index * 1000 + object_index,
                ),
                dataset=self.name,
                split="validation",
                task="grounding",
                images=[ImageRef(image_id=str(row["image_id"]), path=image_path, role="image")],
                question=f"Locate the {name}.",
                ground_truth=GroundTruth(
                    boxes=[bbox],
                    labels=[name],
                    raw={
                        "adapter_version": ADAPTER_VERSION,
                        "image_id": str(row["image_id"]),
                        "source_row": dict(row),
                    },
                ),
                metadata={
                    "source": "VRSBench",
                    "source_index": index,
                    "object_index": object_index,
                    "adapter_version": ADAPTER_VERSION,
                },
            )

    # ── layout resolution / 布局定位 ────────────────────────────────────────

    @staticmethod
    def _validate_fields(
        task: str,
        rows: list[dict[str, Any]],
        annotation: Path,
    ) -> None:
        required = REQUIRED_FIELDS[task]
        for index, row in enumerate(rows):
            missing = sorted(set(required) - set(row))
            if missing:
                raise DatasetProbeError(
                    f"Official VRSBench {task} row {index} in {annotation.name} misses fields: {missing}"
                )

    def _annotation_matches(self, root: Path, task: str) -> list[Path]:
        """All candidate annotation files for one task, deduplicated and sorted.
        某任务的全部候选标注文件（去重、排序）。"""
        filename = ANNOTATION_FILES[task]
        direct = root / filename
        matches = [direct] if direct.is_file() else []
        if root.is_dir():
            matches += [path for path in sorted(root.rglob(filename)) if path != direct]
        return matches

    def _annotation_path(self, root: Path, task: str) -> Path:
        matches = self._annotation_matches(root, task)
        if len(matches) != 1:
            raise DatasetProbeError(
                f"Expected exactly one {ANNOTATION_FILES[task]}; observed {len(matches)} under {root}"
            )
        return matches[0]

    @staticmethod
    def _image_path(root: Path, annotation_root: Path, image_id: str) -> Path:
        candidates = (
            root / image_id,
            annotation_root / image_id,
            root / "Images_val" / image_id,
            root / "Images_val" / "Images_val" / image_id,
            annotation_root / "Images_val" / "Images_val" / image_id,
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise DatasetProbeError(f"Official VRSBench image is missing: {image_id}")
