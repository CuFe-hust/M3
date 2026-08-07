"""Read-only validation and normalization for temporal image pairs.

双时相图像对的只读校验与规范化。绝不原地修改输入图；不做任何数据集
分支（对齐状态完全由元数据与尺寸证据驱动）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from agents.change.schema import PairValidationReport
from agents.counting.schema import IssueRecord
from data.schema import UnifiedSample


@dataclass(frozen=True)
class ValidatedPair:
    """Decoded RGB images plus their audit report. / 解码后的 RGB 图像及其审计报告。"""

    t1: np.ndarray | None
    t2: np.ndarray | None
    report: PairValidationReport


class PairValidator:
    """Validate temporal roles, files, decoding, size, and alignment evidence.
    校验时相角色、文件、解码、尺寸与配准证据。"""

    def validate(self, sample: UnifiedSample, *, data_root: Path) -> ValidatedPair:
        warnings: list[IssueRecord] = []
        roles = [image.role for image in sample.images]
        roles_valid = len(sample.images) == 2 and roles == ["t1", "t2"]
        if not roles_valid:
            warnings.append(
                IssueRecord(
                    code="INVALID_TEMPORAL_ROLES",
                    message="Expected exactly ordered t1/t2 images.",
                )
            )
            return self._invalid(roles_valid, warnings)

        decoded: list[np.ndarray] = []
        sizes: list[list[int]] = []
        try:
            for ref in sample.images:
                path = _resolve_image(sample, ref.path, data_root)
                with Image.open(path) as source:
                    rgb = ImageOps.exif_transpose(source).convert("RGB")
                    sizes.append([rgb.width, rgb.height])
                    decoded.append(np.asarray(rgb, dtype=np.uint8).copy())
        except (OSError, ValueError) as error:
            warnings.append(
                IssueRecord(
                    code="IMAGE_DECODE_FAILED",
                    message=f"{type(error).__name__}",
                )
            )
            return self._invalid(roles_valid, warnings, sizes)

        same_size = decoded[0].shape == decoded[1].shape
        if not same_size:
            warnings.append(
                IssueRecord(
                    code="SIZE_MISMATCH_NO_POLICY",
                    message="Temporal images differ in size; no implicit stretching was applied.",
                )
            )

        metadata_aligned = bool(
            sample.metadata.get("geometry_aligned")
            or sample.metadata.get("registration_id")
        )
        if metadata_aligned and same_size:
            alignment = "metadata_aligned"
        elif same_size:
            alignment = "weakly_aligned"
            warnings.append(
                IssueRecord(
                    code="ALIGNMENT_ONLY_SIZE_MATCH",
                    message="No explicit registration metadata was supplied.",
                )
            )
        else:
            alignment = "unreliable"

        valid = roles_valid and same_size and alignment != "unreliable"
        report = PairValidationReport(
            valid=valid,
            temporal_roles_valid=roles_valid,
            same_size=same_size,
            alignment_status=alignment,
            original_sizes=sizes,
            warnings=warnings,
        )
        return ValidatedPair(decoded[0], decoded[1], report)

    @staticmethod
    def _invalid(
        roles_valid: bool,
        warnings: list[IssueRecord],
        sizes: list[list[int]] | None = None,
    ) -> ValidatedPair:
        report = PairValidationReport(
            valid=False,
            temporal_roles_valid=roles_valid,
            same_size=False,
            alignment_status="unreliable",
            original_sizes=sizes or [],
            warnings=warnings,
        )
        return ValidatedPair(None, None, report)


def _resolve_image(sample: UnifiedSample, path: Path, data_root: Path) -> Path:
    """Resolve one image path against data_root with escape protection.
    按 data_root 解析图像路径并防逃逸。"""
    root = data_root.resolve()
    candidate = (root / path).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("image path escapes data root")
    if not candidate.is_file():
        raise FileNotFoundError(path)
    return candidate
