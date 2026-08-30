"""Audited VRSBench adapter for the official caption/VQA/grounding releases.

经审计的 VRSBench 适配器：加载官方 caption/VQA/grounding 评测发布。
- 任务独立标注发现（显式受审计文件名列表）；多候选显式失败；零记录失败；
- caption/VQA/grounding 字段兼容（多候选键）；VQA 每行调用任务规范化器；
- grounding 支持顶层单对象 / objects / refs 三种结构；4 值 xyxy、8 值 polygon；
- 图片解析只使用受审计候选基目录；0 候选失败、多候选不同文件失败；
- ImageRef.path 一律相对 dataset root；label_binding 明确。
只读源数据；不导入 routing / agents；不拼接任何 Agent prompt。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import re
from typing import Any

from data.adapters.base import AdapterProbe, DatasetProbeError, read_json_rows
from data.adapters.vrsbench.task_normalizer import normalize_task
from data.schema import GroundTruth, ImageRef, UnifiedSample, stable_sample_id

ADAPTER_VERSION = "official-eval-v1"
SUPPORTED_TASKS = frozenset({"general_vqa", "caption", "grounding"})
# Canonical question for every VRSBench caption sample, fixed at the adapter
# boundary so registry / dataset runtime / caption evaluation share one input.
# The source row is preserved in GroundTruth.raw["source_row"] for audit and
# can never override this value.
# 所有 VRSBench caption 样本的规范化固定问句，由 adapter 边界统一提供，
# 保证 registry / dataset runtime / caption 评测得到同一输入；源行仍保留在
# GroundTruth.raw["source_row"] 供审计，且不允许覆盖该固定值。
CAPTION_QUESTION = "Describe the image in detail."
# Official release filenames per task (audited explicit list). / 各任务官方发布文件名。
ANNOTATION_FILENAMES = {
    "general_vqa": ("VRSBench_EVAL_vqa.json",),
    "caption": ("VRSBench_EVAL_Cap.json",),
    # The repository's official release calls the referring-grounding file
    # ``VRSBench_EVAL_referring.json``; keep the audited Det name for existing
    # normalized fixtures and derived releases.
    # 官方发布将 referring grounding 文件命名为
    # ``VRSBench_EVAL_referring.json``；保留已审计的 Det 名称以兼容现有
    # 规范化 fixture 与派生发布。
    "grounding": ("VRSBench_EVAL_referring.json", "VRSBench_EVAL_Det.json"),
}
# Field groups; at least one key per group must be present in each row.
# 字段组；每行每组至少存在一个键。
_REQUIRED_FIELD_GROUPS = {
    "general_vqa": (
        ("image_id", "image", "image_path", "file_name", "filename"),
        ("question",),
        ("ground_truth", "answer"),
        ("question_id",),
        ("type", "question_type"),
    ),
    "caption": (
        ("image_id", "image", "image_path", "file_name", "filename"),
        ("caption", "text", "answer", "description", "ground_truth"),
    ),
    "grounding": (
        ("image_id", "image", "image_path", "file_name", "filename"),
    ),
}
_CAPTION_ANSWER_KEYS = ("caption", "text", "answer", "description", "ground_truth")
_CAPTION_TEXT_KEYS = ("raw", "caption", "text", "sentence")
_IMAGE_FIELD_KEYS = ("image_id", "image", "image_path", "file_name", "filename")
_GROUNDING_TEXT_KEYS = ("ref", "referring", "question", "text", "name")
_GROUNDING_BOX_KEYS = ("bbox", "box", "boxes", "polygon", "obj_corner", "ground_truth")
_GROUNDING_CLASS_KEYS = ("name", "label", "class", "category", "obj_cls")


def _first_value(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    value = _first_value(row, keys)
    return str(value).strip() if value is not None else None


def _parse_box(value: Any) -> list[float]:
    """Parse a 4-value xyxy or 8-value polygon box; nested boxes are split by
    the caller. 解析 4 值 xyxy 或 8 值 polygon 框；嵌套框由调用方拆分。"""
    if isinstance(value, str):
        try:
            value = [float(part) for part in value.replace("[", "").replace("]", "").split(",")]
        except ValueError as error:
            raise DatasetProbeError(f"unparseable box value: {value!r}") from error
    if not isinstance(value, (list, tuple)):
        raise DatasetProbeError(f"invalid box structure: {type(value).__name__}")
    box = [float(part) for part in value]
    if len(box) not in (4, 8):
        raise DatasetProbeError(f"box must have 4 or 8 coordinates, got {len(box)}")
    return box


def _parse_official_referring_box(value: Any) -> list[int]:
    """Parse official VRSBench 0-100 box into M3 0-999 xyxy."""
    if not isinstance(value, str):
        raise DatasetProbeError(
            "VRSBench official referring ground_truth must be a string"
        )
    values = [
        float(item)
        for item in re.findall(
            r"<\s*(-?(?:\d+(?:\.\d*)?|\.\d+))\s*>", value
        )
    ]
    if len(values) != 4 or any(item < 0 or item > 100 for item in values):
        raise DatasetProbeError(
            f"invalid VRSBench official referring ground_truth: {value!r}"
        )
    return [max(0, min(999, int(round(item * 999 / 100)))) for item in values]


def _normalize_vrsbench_split(split: str) -> str:
    """Canonical split: validation/val normalize to validation; others fail.
    split 规范化：validation/val 统一为 validation；其他失败。"""
    canonical = {"validation": "validation", "val": "validation"}.get(split.strip().lower())
    if canonical is None:
        raise DatasetProbeError(
            f"VRSBench supports split 'validation' (alias 'val') only, got {split!r}"
        )
    return canonical


def _image_id_for_row(row: dict[str, Any], source_image_ref: str, index: int) -> str:
    """Stable, non-empty, unique-in-sample image id for ImageRef.image_id.
    供 ImageRef.image_id 使用的稳定、非空、样本内唯一的图片 ID。"""
    explicit = row.get("image_id")
    if explicit not in (None, ""):
        return str(explicit)
    basename = Path(source_image_ref).name
    if basename:
        return basename
    return f"vrsbench-image-{index}"


class VRSBenchAdapter:
    """Read-only adapter over the official VRSBench evaluation releases.
    只读加载官方 VRSBench 评测发布的适配器。"""

    name = "VRSBench"
    supported_tasks = SUPPORTED_TASKS

    # ── probe / 探测 ────────────────────────────────────────────────────────

    def probe(self, root: Path, task: str | None = None) -> AdapterProbe:
        """Validate one task's release, or discover all available tasks.
        校验单个任务的发布，或发现当前 root 下全部可用任务。"""
        if task is not None:
            if task not in SUPPORTED_TASKS:
                raise DatasetProbeError(
                    f"VRSBench does not support task={task!r}; supported={sorted(SUPPORTED_TASKS)}"
                )
            return self._probe_task(root, task)
        available: list[str] = []
        for candidate in sorted(SUPPORTED_TASKS):
            matches = self._annotation_matches(root, candidate)
            if len(matches) > 1:
                raise DatasetProbeError(
                    f"Expected exactly one {ANNOTATION_FILENAMES[candidate][0]}; "
                    f"observed {len(matches)} under {root}"
                )
            if matches:
                available.append(candidate)
        if not available:
            raise DatasetProbeError(f"no VRSBench annotation found under {root}")
        probes = {name: self._probe_task(root, name) for name in available}
        primary = "general_vqa" if "general_vqa" in available else available[0]
        return AdapterProbe(
            dataset=self.name,
            version=ADAPTER_VERSION,
            sample_file=probes[primary].sample_file,
            observed_fields=probes[primary].observed_fields,
            sample_count=probes[primary].sample_count,
            available_tasks=tuple(available),
        )

    def _probe_task(self, root: Path, task: str) -> AdapterProbe:
        annotation = self._annotation_path(root, task)
        rows = read_json_rows(annotation)
        if not rows:
            raise DatasetProbeError(
                f"zero records in {ANNOTATION_FILENAMES[task][0]} under {root}"
            )
        self._validate_fields(task, rows, annotation)
        observed = tuple(sorted({key for row in rows[:20] for key in row}))
        return AdapterProbe(
            dataset=self.name,
            version=ADAPTER_VERSION,
            sample_file=annotation,
            observed_fields=observed,
            sample_count=len(rows),
            task=task,
            available_tasks=(task,),
        )

    # ── iter_samples / 样本迭代 ────────────────────────────────────────────

    def iter_samples(self, root: Path, split: str, task: str) -> Iterator[UnifiedSample]:
        """Yield schema-validated unified samples in source order.
        按源顺序产出具 schema 校验的统一样本。"""
        canonical_split = _normalize_vrsbench_split(split)
        if task not in SUPPORTED_TASKS:
            raise DatasetProbeError(
                f"VRSBench does not support task={task!r}; supported={sorted(SUPPORTED_TASKS)}"
            )
        annotation = self._annotation_path(root, task)
        rows = read_json_rows(annotation)
        if not rows:
            raise DatasetProbeError(
                f"zero records in {ANNOTATION_FILENAMES[task][0]} under {root}"
            )
        self._validate_fields(task, rows, annotation)
        for index, row in enumerate(rows):
            image_value = _first_value(row, _IMAGE_FIELD_KEYS)
            if image_value is None:
                raise DatasetProbeError(f"VRSBench {task} row {index} has no image field")
            source_image_ref = str(image_value)
            image_path = self._image_path(root, annotation.parent, source_image_ref)
            if task == "general_vqa":
                yield self._vqa_sample(root, canonical_split, row, image_path, index,
                                       source_image_ref=source_image_ref)
            elif task == "caption":
                yield self._caption_sample(root, canonical_split, row, image_path, index,
                                           source_image_ref=source_image_ref)
            else:
                yield from self._grounding_samples(
                    root,
                    canonical_split,
                    row,
                    image_path,
                    index,
                    source_image_ref=source_image_ref,
                    annotation=annotation,
                )

    # ── per-task mapping / 分任务映射 ───────────────────────────────────────

    def _vqa_sample(
        self,
        root: Path,
        split: str,
        row: dict[str, Any],
        image_path: Path,
        index: int,
        *,
        source_image_ref: str,
    ) -> UnifiedSample:
        image_id = _image_id_for_row(row, source_image_ref, index)
        question = str(row["question"])
        answer_value = _first_value(row, ("ground_truth", "answer"))
        question_id = str(row["question_id"]) if "question_id" in row else str(index)
        question_type = str(_first_value(row, ("type", "question_type")) or "")
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
            images=[ImageRef(image_id=image_id, path=image_path.relative_to(root), role="image")],
            question=question,
            ground_truth=GroundTruth(
                answers=[str(answer_value)],
                raw={
                    "adapter_version": ADAPTER_VERSION,
                    "image_id": image_id,
                    "question_id": question_id,
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
        *,
        source_image_ref: str,
    ) -> UnifiedSample:
        image_id = _image_id_for_row(row, source_image_ref, index)
        answer_value = _first_value(row, _CAPTION_ANSWER_KEYS)
        if answer_value is None:
            raise DatasetProbeError(f"VRSBench caption row {index} has no caption field")
        answers = _caption_texts(answer_value, index)
        # Both UnifiedSample.question and stable_sample_id(...) must use the
        # same canonical value so persisted sample content and logical identity
        # stay consistent. / UnifiedSample.question 与 stable_sample_id(...)
        # 必须使用同一固定问句，保证持久化样本内容与逻辑身份一致。
        question = CAPTION_QUESTION
        return UnifiedSample(
            sample_id=stable_sample_id(
                dataset=self.name,
                split=split,
                source_id=None,
                relative_image_paths=[image_path.relative_to(root)],
                question=question,
                source_index=index,
            ),
            dataset=self.name,
            split="validation",
            task="caption",
            images=[ImageRef(image_id=image_id, path=image_path.relative_to(root), role="image")],
            question=question,
            ground_truth=GroundTruth(
                answers=answers,
                raw={
                    "adapter_version": ADAPTER_VERSION,
                    "image_id": image_id,
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
        *,
        source_image_ref: str,
        annotation: Path,
    ) -> Iterator[UnifiedSample]:
        image_id = _image_id_for_row(row, source_image_ref, index)
        coordinate_frame = _grounding_coordinate_frame(annotation)
        if "objects" in row:
            items = row["objects"]
            text_keys = ("referring", "question", "text")
        elif "refs" in row:
            items = row["refs"]
            text_keys = ("ref", "referring", "question", "text")
        else:
            items = [row]
            text_keys = ("ref", "referring", "question", "text")
        if not isinstance(items, list) or not items:
            raise DatasetProbeError(f"VRSBench grounding row {index} has invalid object list")
        for object_index, obj in enumerate(items):
            if not isinstance(obj, dict):
                raise DatasetProbeError(
                    f"VRSBench grounding row {index} object {object_index} is not an object"
                )
            text = _first_text(obj, text_keys)
            label = _first_text(obj, _GROUNDING_CLASS_KEYS)
            question_id = row.get("question_id")
            base_question_id = str(
                row.get("question_id") or row.get("id") or index
            )
            if annotation.name == "VRSBench_EVAL_referring.json":
                official_box = _first_value(obj, ("ground_truth",))
                if official_box is None:
                    raise DatasetProbeError(
                        f"VRSBench referring row {index} object {object_index} has no ground_truth"
                    )
                boxes = [_parse_official_referring_box(official_box)]
            else:
                box_value = _first_value(obj, _GROUNDING_BOX_KEYS)
                if box_value is None:
                    raise DatasetProbeError(
                        f"VRSBench grounding row {index} object {object_index} has no box"
                    )
                # Nested boxes=[[...]] are split explicitly; each box is one sample.
                # 嵌套 boxes=[[...]] 显式拆分；每个框一条样本。
                if isinstance(box_value, list) and box_value and isinstance(box_value[0], (list, tuple)):
                    boxes = [_parse_box(item) for item in box_value]
                else:
                    boxes = [_parse_box(box_value)]
            for box_position, box in enumerate(boxes):
                question = text or (
                    f"Locate the {label}." if label else "Locate the target object."
                )
                object_source_id = (
                    f"{base_question_id}-obj-{object_index}-box-{box_position}"
                )
                yield UnifiedSample(
                    sample_id=stable_sample_id(
                        dataset=self.name,
                        split=split,
                        source_id=object_source_id,
                        relative_image_paths=[image_path.relative_to(root)],
                        question=question,
                        source_index=index,
                    ),
                    dataset=self.name,
                    split="validation",
                    task="grounding",
                    images=[
                        ImageRef(image_id=image_id, path=image_path.relative_to(root), role="image")
                    ],
                    question=question,
                    ground_truth=GroundTruth(
                        boxes=[box],
                        labels=[label] if label else [],
                        label_binding="boxes" if label else None,
                        coordinate_frame=coordinate_frame,
                        raw={
                            "adapter_version": ADAPTER_VERSION,
                            "image_id": image_id,
                            "question_id": question_id,
                            "object_index": object_index,
                            "box_source_field": _box_source_field(obj),
                            "coordinate_frame": coordinate_frame,
                            "source_row": dict(row),
                        },
                    ),
                    metadata={
                        "source": "VRSBench",
                        "source_index": index,
                        "object_index": object_index,
                        "adapter_version": ADAPTER_VERSION,
                        "coordinate_frame": coordinate_frame,
                    },
                )

    # ── layout resolution / 布局定位 ────────────────────────────────────────

    def _validate_fields(
        self,
        task: str,
        rows: list[dict[str, Any]],
        annotation: Path,
    ) -> None:
        for index, row in enumerate(rows):
            for group in _REQUIRED_FIELD_GROUPS[task]:
                if _first_value(row, group) is None:
                    raise DatasetProbeError(
                        f"Official VRSBench {task} row {index} in {annotation.name} "
                        f"misses one of fields: {group}"
                    )
            if task == "grounding":
                has_structure = (
                    "objects" in row or "refs" in row
                    or _first_value(row, _GROUNDING_BOX_KEYS) is not None
                )
                if not has_structure:
                    raise DatasetProbeError(
                        f"Official VRSBench grounding row {index} in {annotation.name} "
                        "has no objects/refs/box structure"
                    )

    def _annotation_matches(self, root: Path, task: str) -> list[Path]:
        """All candidate annotation files for one task, deduplicated and sorted.
        某任务的全部候选标注文件（去重、排序）。"""
        matches: list[Path] = []
        for filename in ANNOTATION_FILENAMES[task]:
            direct = root / filename
            if direct.is_file():
                matches.append(direct)
            if root.is_dir():
                matches += [path for path in sorted(root.rglob(filename)) if path != direct]
        return matches

    def _annotation_path(self, root: Path, task: str) -> Path:
        matches = self._annotation_matches(root, task)
        if len(matches) != 1:
            raise DatasetProbeError(
                f"Expected exactly one {ANNOTATION_FILENAMES[task][0]}; "
                f"observed {len(matches)} under {root}"
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
        existing = sorted({candidate.resolve() for candidate in candidates if candidate.is_file()})
        if not existing:
            raise DatasetProbeError(f"Official VRSBench image is missing: {image_id}")
        if len(existing) > 1:
            raise DatasetProbeError(
                f"Official VRSBench image is ambiguous: {image_id} matches "
                f"{len(existing)} different files"
            )
        return existing[0]


def _caption_texts(value: Any, index: int) -> list[str]:
    """Extract all non-empty reference captions, preserving source order.
    提取全部非空参考 caption，保留源顺序。"""
    values = value if isinstance(value, list) else [value]
    texts: list[str] = []
    for item in values:
        if isinstance(item, dict):
            text = _first_text(item, _CAPTION_TEXT_KEYS)
            if text is None:
                raise DatasetProbeError(
                    f"VRSBench caption row {index} has a dict caption without a text key"
                )
            texts.append(text)
        else:
            text = str(item).strip()
            if text:
                texts.append(text)
    if not texts:
        raise DatasetProbeError(f"VRSBench caption row {index} has no non-empty caption")
    return texts


def _box_source_field(obj: dict[str, Any]) -> str | None:
    for key in _GROUNDING_BOX_KEYS:
        if key in obj and obj[key] not in (None, ""):
            return key
    return None


def _grounding_coordinate_frame(annotation: Path) -> str:
    """Declare the audited coordinate frame for each official release.
    为每个官方发布显式声明经过审计的坐标系。

    ``obj_corner`` in the official referring release is a four-corner polygon
    normalized to [0, 1]; the legacy/derived Det release stores pixel
    coordinates. The adapter preserves the polygon and never converts it to
    xyxy while claiming metric equivalence.
    官方 referring 发布中的 ``obj_corner`` 是 [0, 1] 归一化四角 polygon；
    legacy/派生 Det 发布保存像素坐标。适配器保留 polygon，绝不把它转换成
    xyxy 后声称与官方指标等价。
    """

    if annotation.name == "VRSBench_EVAL_referring.json":
        return "normalized_0_999_top_left"
    return "source_pixels_top_left"
