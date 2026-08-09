"""Change preprocessing orchestration and sample-scoped artifacts.

变化预处理编排与样本级产物。组合 pair 校验、一致化与差异提议；只在
artifact_dir 内写入派生产物（绝不修改源图片）；不调用视觉模型；写盘失败
显式向上暴露。numpy 与 cv2 是可选依赖（[change] extra），仅在编排执行时
惰性加载，模块导入本身不触发。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from agents.change.difference_proposal import propose_changes, render_overlay
from agents.change.harmonizer import PairHarmonizer
from agents.change.pair_validator import PairValidator
from agents.change.schema import (
    ChangePreprocessResult,
    ChangeProposal,
    HarmonizationDecision,
    PairValidationReport,
)
from agents.change.settings import AgentChangeSettings
from agents.errors import OptionalDependencyMissingError
from data.schema import UnifiedSample


def _require_numpy():
    """Return the numpy module or a stable optional-dependency error.
    返回 numpy 模块，缺失时抛出稳定的可选依赖错误。"""
    try:
        import numpy as np
    except ImportError as error:
        raise OptionalDependencyMissingError(
            "change", dependency="numpy"
        ) from error
    return np


@dataclass(frozen=True)
class ChangePreparedPair:
    """In-memory pair state shared by legacy and V2 proposal paths.

    Large arrays deliberately stay outside Pydantic trace schemas. Invalid
    pairs use ``None`` images and an empty all-zero PIF mask; every valid pair
    owns raw arrays plus comparison arrays selected by the harmonization gate.
    """

    raw_t1: Any
    raw_t2: Any
    comparison_t1: Any
    comparison_t2: Any
    pif_mask: Any
    validation: PairValidationReport
    decision: HarmonizationDecision
    transform_summary: dict[str, object]


def prepare_pair(
    sample: UnifiedSample,
    settings: AgentChangeSettings,
    artifact_dir: Path,
    *,
    data_root: Path,
) -> ChangePreparedPair:
    """Validate and harmonize exactly once, then publish compact audit JSON."""

    np = _require_numpy()
    output = artifact_dir / "change_preprocess"
    output.mkdir(parents=True, exist_ok=True)
    validated = PairValidator().validate(sample, data_root=data_root)
    _write_json(
        output / "validation_report.json",
        validated.report.model_dump(mode="json"),
    )
    if not validated.report.valid or validated.t1 is None or validated.t2 is None:
        prepared = ChangePreparedPair(
            raw_t1=None,
            raw_t2=None,
            comparison_t1=None,
            comparison_t2=None,
            pif_mask=np.zeros((0, 0), dtype=np.uint8),
            validation=validated.report,
            decision=HarmonizationDecision(
                version=settings.harmonization.version,
                status="skipped",
                reason_codes=["SKIPPED_INVALID_PAIR", "RAW_FALLBACK_USED"],
                metrics=None,
                used_for_proposal=False,
            ),
            transform_summary={},
        )
        _publish_preparation_audit(prepared, output)
        return prepared

    raw1 = validated.t1.copy()
    raw2 = validated.t2.copy()
    transform_summary: dict[str, object] = {}
    if settings.harmonization.enabled:
        try:
            candidate = PairHarmonizer(settings.harmonization).run(raw1, raw2)
            decision = candidate.decision
            comparison1 = (
                candidate.t1.copy() if decision.status == "applied" else raw1.copy()
            )
            comparison2 = (
                candidate.t2.copy() if decision.status == "applied" else raw2.copy()
            )
            pif_mask = candidate.pif_mask.copy()
            transform_summary = dict(candidate.transform_summary)
        except Exception as error:
            comparison1, comparison2 = raw1.copy(), raw2.copy()
            pif_mask = np.zeros(raw1.shape[:2], dtype=np.uint8)
            decision = HarmonizationDecision(
                version=settings.harmonization.version,
                status="failed",
                reason_codes=[
                    "FAILED_HARMONIZATION_EXCEPTION",
                    "RAW_FALLBACK_USED",
                ],
                metrics=None,
                used_for_proposal=False,
            )
            transform_summary = {
                "error_type": type(error).__name__,
                "sharpness_adjustment_used": False,
            }
    else:
        comparison1, comparison2 = raw1.copy(), raw2.copy()
        pif_mask = np.zeros(raw1.shape[:2], dtype=np.uint8)
        decision = HarmonizationDecision(
            version=settings.harmonization.version,
            status="skipped",
            reason_codes=["SKIPPED_DISABLED", "RAW_FALLBACK_USED"],
            metrics=None,
            used_for_proposal=False,
        )

    prepared = ChangePreparedPair(
        raw_t1=raw1,
        raw_t2=raw2,
        comparison_t1=comparison1,
        comparison_t2=comparison2,
        pif_mask=pif_mask,
        validation=validated.report,
        decision=decision,
        transform_summary=transform_summary,
    )
    _publish_preparation_audit(prepared, output)
    return prepared


def preprocess_pair(
    sample: UnifiedSample,
    settings: AgentChangeSettings,
    artifact_dir: Path,
    *,
    data_root: Path,
) -> ChangePreprocessResult:
    """Run the legacy proposal path from one shared prepared pair."""

    np = _require_numpy()
    prepared = prepare_pair(sample, settings, artifact_dir, data_root=data_root)
    if prepared.raw_t1 is None or prepared.raw_t2 is None:
        return _preparation_result(prepared)

    proposals: list[ChangeProposal] = []
    score_map = np.zeros(prepared.raw_t1.shape[:2], dtype=np.float32)
    if settings.proposals.enabled:
        score_map, proposals = propose_changes(
            prepared.comparison_t1,
            prepared.comparison_t2,
            settings.proposals,
        )
    return publish_change_proposals(
        prepared,
        score_map=score_map,
        proposals=proposals,
        artifact_dir=artifact_dir,
        settings=settings,
    )


def publish_change_proposals(
    prepared: ChangePreparedPair,
    *,
    score_map: Any,
    proposals: list[ChangeProposal],
    artifact_dir: Path,
    settings: AgentChangeSettings,
    component_maps: dict[str, Any] | None = None,
) -> ChangePreprocessResult:
    """Publish legacy or V2 maps, overlays, crops, and serializable reports."""

    if prepared.raw_t1 is None or prepared.raw_t2 is None:
        raise ValueError("cannot publish proposals for an invalid prepared pair")
    np = _require_numpy()
    output = artifact_dir / "change_preprocess"
    output.mkdir(parents=True, exist_ok=True)
    files = _audit_files()
    is_v2 = component_maps is not None or any(
        proposal.source == "fused_change_v2" for proposal in proposals
    )
    map_key = "fused_change_map" if is_v2 else "difference_map"
    map_filename = "fused_change_map.png" if is_v2 else "difference_map.png"
    _write_image(output / map_filename, _map_artifact(score_map))
    overlay = render_overlay(prepared.comparison_t2, proposals)
    _write_image(output / "proposal_overlay.png", overlay)
    files[map_key] = f"change_preprocess/{map_filename}"
    files["proposal_overlay"] = "change_preprocess/proposal_overlay.png"

    for name, component in (component_maps or {}).items():
        if not name or any(
            not (character.isalnum() or character == "_") for character in name
        ):
            raise ValueError("component map names must be alphanumeric snake_case")
        filename = f"{name}.png"
        _write_image(output / filename, _map_artifact(component))
        files[name] = f"change_preprocess/{filename}"

    if prepared.decision.status == "applied" and settings.harmonization.save_artifacts:
        _write_image(output / "harmonized_t1.png", prepared.comparison_t1)
        _write_image(output / "harmonized_t2.png", prepared.comparison_t2)
        _write_image(output / "pif_mask.png", prepared.pif_mask)
        files.update(
            {
                "harmonized_t1": "change_preprocess/harmonized_t1.png",
                "harmonized_t2": "change_preprocess/harmonized_t2.png",
                "pif_mask": "change_preprocess/pif_mask.png",
            }
        )

    crops = output / "crops"
    updated: list[ChangeProposal] = []
    for proposal in proposals:
        x1, y1, x2, y2 = proposal.pixel_box
        evidence: list[str] = []
        for label, image in (("raw_t1", prepared.raw_t1), ("raw_t2", prepared.raw_t2)):
            filename = f"{proposal.proposal_id}_{label}.png"
            _write_image(crops / filename, image[y1:y2, x1:x2])
            evidence.append(f"change_preprocess/crops/{filename}")
        if prepared.decision.status == "applied":
            for label, image in (
                ("harmonized_t1", prepared.comparison_t1),
                ("harmonized_t2", prepared.comparison_t2),
            ):
                filename = f"{proposal.proposal_id}_{label}.png"
                _write_image(crops / filename, image[y1:y2, x1:x2])
                evidence.append(f"change_preprocess/crops/{filename}")
        updated.append(proposal.model_copy(update={"evidence_filenames": evidence}))

    _write_json(
        output / "proposals.json",
        [item.model_dump(mode="json") for item in updated],
    )
    files["proposals"] = "change_preprocess/proposals.json"
    files["harmonization_report"] = "change_preprocess/harmonization_report.json"
    result = ChangePreprocessResult(
        validation=prepared.validation,
        decision=prepared.decision,
        proposals=updated,
        artifact_files=files,
        transform_summary=prepared.transform_summary,
    )
    _write_json(output / "harmonization_report.json", result.model_dump(mode="json"))
    return result


def _audit_files() -> dict[str, str]:
    return {
        "validation_report": "change_preprocess/validation_report.json",
        "harmonization_report": "change_preprocess/harmonization_report.json",
    }


def _preparation_result(prepared: ChangePreparedPair) -> ChangePreprocessResult:
    return ChangePreprocessResult(
        validation=prepared.validation,
        decision=prepared.decision,
        proposals=[],
        artifact_files=_audit_files(),
        transform_summary=prepared.transform_summary,
    )


def _publish_preparation_audit(prepared: ChangePreparedPair, output: Path) -> None:
    _write_json(
        output / "harmonization_report.json",
        _preparation_result(prepared).model_dump(mode="json"),
    )


def _map_artifact(value: Any) -> Any:
    """Convert a scalar score/mask or RGB diagnostic into a uint8 image."""

    np = _require_numpy()
    array = np.asarray(value)
    if array.ndim not in {2, 3}:
        raise ValueError("change map artifact must be 2-D or 3-D")
    if array.dtype == np.uint8:
        return array.copy()
    if not np.isfinite(array).all():
        raise ValueError("change map artifact contains non-finite values")
    return np.round(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


def _write_image(path: Path, image: np.ndarray) -> None:
    """Atomically publish a uint8 image. / 原子发布 uint8 图像。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    Image.fromarray(image).save(temporary)
    temporary.replace(path)


def _write_json(path: Path, payload: object) -> None:
    """Atomically publish one UTF-8 JSON artifact. / 原子发布一份 UTF-8 JSON 产物。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
