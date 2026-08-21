"""Training dataset that delegates pair preparation to production code."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from agents.change.preprocess import ChangePreparedPair, prepare_pair
from agents.change.settings import AgentChangeSettings
from data.schema import GroundTruth, ImageRef, UnifiedSample
from training.change_head.schema import ChangeTrainingRecord


@dataclass(frozen=True)
class PreparedChangeTrainingExample:
    record: ChangeTrainingRecord
    sample: UnifiedSample
    prepared: ChangePreparedPair
    target_mask: np.ndarray
    loss_valid_mask: np.ndarray


class PreparedChangeTrainingDataset:
    def __init__(
        self,
        records: list[ChangeTrainingRecord],
        *,
        data_root: Path,
        settings: AgentChangeSettings,
        artifact_root: Path,
    ) -> None:
        self.records = records
        self.data_root = data_root.resolve()
        self.settings = settings.model_copy(deep=True)
        self.settings.registration.save_artifacts = False
        self.settings.harmonization.save_artifacts = False
        self.artifact_root = artifact_root

    def __len__(self) -> int:
        return len(self.records)

    def _resolve(self, value: Path) -> Path:
        path = (self.data_root / value).resolve()
        if path == self.data_root or self.data_root not in path.parents:
            raise ValueError("training path escapes data_root")
        if not path.is_file():
            raise FileNotFoundError(path.name)
        return path

    def _sample(self, record: ChangeTrainingRecord) -> UnifiedSample:
        t1 = self._resolve(record.t1_path)
        t2 = self._resolve(record.t2_path)
        return UnifiedSample(
            sample_id=record.sample_id,
            dataset=record.source_dataset,
            split=record.split,
            task="change_caption",
            images=[
                ImageRef(image_id=f"{record.sample_id}:t1", path=t1.relative_to(self.data_root), role="t1"),
                ImageRef(image_id=f"{record.sample_id}:t2", path=t2.relative_to(self.data_root), role="t2"),
            ],
            question="Describe the change.",
            ground_truth=GroundTruth(answers=[]),
            metadata={"geometry_aligned": True, "mask_frame": "t1_reference"},
        )

    def __getitem__(self, index: int) -> PreparedChangeTrainingExample:
        record = self.records[index]
        sample = self._sample(record)
        artifact_dir = self.artifact_root / hashlib.sha256(record.sample_id.encode()).hexdigest()[:16]
        prepared = prepare_pair(
            sample,
            self.settings,
            artifact_dir,
            data_root=self.data_root,
        )
        mask = np.asarray(Image.open(self._resolve(record.mask_path)).convert("L")) > 0
        if mask.shape != prepared.raw_t1.shape[:2]:
            raise ValueError("training mask does not match t1 canvas")
        valid = np.asarray(prepared.registration_valid_mask, dtype=bool)
        return PreparedChangeTrainingExample(
            record=record,
            sample=sample,
            prepared=prepared,
            target_mask=mask.astype(np.float32),
            loss_valid_mask=valid,
        )

