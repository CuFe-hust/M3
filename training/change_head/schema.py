"""JSONL records for reproducible ChangeHead training samples."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChangeTrainingRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    t1_path: Path
    t2_path: Path
    mask_path: Path
    split: Literal["train", "val", "test"]
    group_id: str = Field(min_length=1)
    tags: tuple[str, ...] = ()
    sample_weight: float = Field(default=1.0, gt=0.0)
    source_dataset: str = Field(min_length=1)

    @field_validator("t1_path", "t2_path", "mask_path")
    @classmethod
    def paths_must_be_relative(cls, value: Path) -> Path:
        if value.is_absolute() or any(part in {"", ".", ".."} for part in value.parts):
            raise ValueError("training paths must be relative to data_root")
        return value


def load_training_records(
    path: Path,
    *,
    allow_empty_changed: bool = False,
) -> list[ChangeTrainingRecord]:
    import json
    from PIL import Image
    import numpy as np

    records: list[ChangeTrainingRecord] = []
    seen_ids: set[str] = set()
    groups: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = ChangeTrainingRecord.model_validate(json.loads(line))
        except Exception as error:
            raise ValueError(f"invalid training record line {line_number}") from error
        if record.sample_id in seen_ids:
            raise ValueError("duplicate sample_id")
        prior_split = groups.setdefault(record.group_id, record.split)
        if prior_split != record.split:
            raise ValueError("group_id crosses splits")
        for field in ("t1_path", "t2_path", "mask_path"):
            if not (path.parent / getattr(record, field)).exists():
                # The caller normally resolves against data_root; existence is
                # intentionally checked by PreparedChangeTrainingDataset.
                pass
        mask_path = path.parent / record.mask_path
        if mask_path.is_file():
            mask = np.asarray(Image.open(mask_path).convert("L"))
            nonzero = bool(np.any(mask > 0))
            if "no_change" in record.tags and nonzero:
                raise ValueError("no_change record has a non-zero mask")
            if not allow_empty_changed and "no_change" not in record.tags and not nonzero:
                raise ValueError("changed record has an empty mask")
        seen_ids.add(record.sample_id)
        records.append(record)
    return records

