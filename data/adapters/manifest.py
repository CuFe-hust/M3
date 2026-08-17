"""Self-contained manifest-driven draft adapter for the data layer.

数据层自包含的 manifest 驱动 draft 适配器。本模块位于 data 包，不导入
workflows/models（data 层零业务依赖、绝不写 run artifacts）：读取用户提供
的显式版本化映射清单（spacers_adapter.json）产出 SampleDraft 与
AdapterProbe，task 列可选，字段名绝不猜测、绝不调用模型；samples_file
必须位于 dataset root 内。run manifest 的写回不属于数据层（见
workflows.artifact_writer.write_dataset_probe）。
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from data.adapters.base import (
    AdapterProbe,
    DatasetProbeError,
    read_json_rows,
    resolve_dataset_relative_path,
    validate_manifest_mapping,
)
from data.schema import GroundTruth, ImageRef, SampleDraft


# ── Manifest-driven draft adapter / manifest 驱动的 draft 适配器 ────────────

# User-provided versioned mapping manifest, matching the baseline contract.
# 用户提供的版本化映射清单，与基线契约一致。
MANIFEST_FILENAME = "spacers_adapter.json"

# Semantic fields every draft manifest must map; "task" is optional.
# 每个 draft manifest 必须映射的语义字段；"task" 可选。
_DRAFT_REQUIRED_FIELDS = ("id", "split", "question", "images")


def load_manifest_mapping(
    root: Path,
    *,
    dataset: str,
    version: str = "1",
) -> tuple[Path, Mapping[str, str]]:
    """Load and validate the versioned adapter manifest; returns
    (samples_file, fields). Field names are never guessed — every semantic
    field must be declared explicitly. 加载并校验版本化适配器清单；返回
    （samples_file、fields）。字段名绝不猜测——每个语义字段必须显式声明。"""

    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise DatasetProbeError(
            f"{dataset} requires {MANIFEST_FILENAME} under the dataset root"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DatasetProbeError(
            f"Invalid {MANIFEST_FILENAME}: {type(exc).__name__}"
        ) from exc
    fields = validate_manifest_mapping(
        manifest,
        dataset=dataset,
        version=version,
        required_fields=_DRAFT_REQUIRED_FIELDS,
    )
    samples_file = resolve_dataset_relative_path(
        root, str(manifest["samples_file"]), field_name="samples_file"
    )
    return samples_file, fields


def iter_manifest_drafts(
    root: Path,
    *,
    dataset: str,
    version: str = "1",
    split: str,
) -> Iterator[SampleDraft]:
    """Yield SampleDraft rows from the explicit field mapping; the task column
    is optional — drafts without a task go through the visual planner and never
    pretend to be general_vqa. 从显式字段映射产出 SampleDraft；task 列可选——
    无 task 的 draft 走视觉规划器，绝不冒充 general_vqa。"""

    samples_file, fields = load_manifest_mapping(root, dataset=dataset, version=version)
    for index, row in enumerate(read_json_rows(samples_file)):
        if str(row[fields["split"]]) != split:
            continue
        yield _row_to_draft(
            row, fields, index, root=root, dataset=dataset, split=split
        )


def _row_to_draft(
    row: Mapping[str, Any],
    fields: Mapping[str, str],
    index: int,
    *,
    root: Path,
    dataset: str,
    split: str,
) -> SampleDraft:
    """Map one row through the declared fields into a SampleDraft; invalid
    rows fail with stable DatasetProbeError messages that never echo raw
    values. 按声明字段将一行映射为 SampleDraft；非法行以稳定且不回显原始值
    的 DatasetProbeError 失败。"""

    images_value = row[fields["images"]]
    if not isinstance(images_value, list) or not images_value:
        raise DatasetProbeError(f"Row {index} has an invalid images field")
    images: list[ImageRef] = []
    for image_index, relative in enumerate(images_value):
        try:
            image = ImageRef(
                image_id=f"{index}-{image_index}",
                path=str(relative),
                role="image",
            )
        except (ValueError, ValidationError):
            raise DatasetProbeError(
                f"Row {index} has an invalid image path at position {image_index}"
            ) from None
        if not (root / image.path).is_file():
            raise DatasetProbeError(f"Row {index} references a missing image file")
        images.append(image)
    task_value = row.get(fields["task"]) if "task" in fields else None
    explicit_task = None
    if task_value not in (None, ""):
        explicit_task = str(task_value)
    try:
        return SampleDraft(
            sample_id=str(row[fields["id"]]),
            dataset=dataset,
            split=split,
            images=images,
            question=str(row[fields["question"]]),
            explicit_task=explicit_task,  # type: ignore[arg-type]
            ground_truth=_draft_ground_truth(row, fields),
        )
    except (ValueError, ValidationError):
        raise DatasetProbeError(
            f"Row {index} does not satisfy the draft contract"
        ) from None


def _draft_ground_truth(
    row: Mapping[str, Any],
    fields: Mapping[str, str],
) -> GroundTruth | None:
    """Optional mapped count/answers become a ground truth; nothing is
    guessed when the manifest declares no such fields.
    可选的 count/answers 映射构成真值；manifest 未声明时绝不猜测。"""

    count = row.get(fields["count"]) if "count" in fields else None
    answers = row.get(fields["answers"]) if "answers" in fields else None
    if count in (None, "") and not answers:
        return None
    parsed_answers = list(answers) if isinstance(answers, list) else []
    return GroundTruth(
        count=int(count) if count not in (None, "") else None,
        answers=parsed_answers,
    )


class ManifestDraftAdapter:
    """DraftDatasetAdapter driven by an explicit user-provided mapping
    manifest (spacers_adapter.json); yields SampleDraft with an optional task
    column. Never imports workflows or models; never calls a model.
    由用户提供的显式映射清单（spacers_adapter.json）驱动的 DraftDatasetAdapter；
    产出 task 列可选的 SampleDraft。绝不 import workflows/models；绝不调用
    模型。"""

    manifest_name = MANIFEST_FILENAME

    def __init__(
        self,
        name: str,
        supported_tasks: set[str] | frozenset[str] | tuple[str, ...],
    ) -> None:
        self.name = name
        self.supported_tasks = supported_tasks

    def probe(self, root: Path, task: str | None = None) -> AdapterProbe:
        """Validate the explicit mapping instead of inferring field names.
        验证显式映射而非推测字段名。"""

        samples_file, fields = load_manifest_mapping(root, dataset=self.name)
        rows = read_json_rows(samples_file)
        observed = tuple(sorted({key for row in rows[:20] for key in row}))
        mapped = tuple(str(value) for value in fields.values())
        absent = sorted(set(mapped) - set(observed))
        if absent:
            raise DatasetProbeError(
                f"Mapped fields absent from sample rows: {absent}"
            )
        return AdapterProbe(
            dataset=self.name,
            version="1",
            sample_file=samples_file,
            observed_fields=observed,
            sample_count=len(rows),
            task=task,
            available_tasks=tuple(sorted(self.supported_tasks)),
        )

    def iter_drafts(self, root: Path, split: str) -> Iterator[SampleDraft]:
        """Yield drafts for one split from the declared mapping.
        从声明映射中产出单个 split 的 drafts。"""

        return iter_manifest_drafts(root, dataset=self.name, split=split)
