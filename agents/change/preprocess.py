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
from agents.change.registration import RegisteredPair, RegistrationError, register_pair
from agents.change.schema import (
    ChangePreprocessResult,
    ChangeProposal,
    HarmonizationDecision,
    PairValidationReport,
    RegistrationReport,
    StructuralRescueCandidate,
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
    pif_valid: bool
    validation: PairValidationReport
    decision: HarmonizationDecision
    transform_summary: dict[str, object]
    registered_t1: Any = None
    registered_t2: Any = None
    registration_valid_mask: Any = None
    registration_report: RegistrationReport | None = None


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
    validated = PairValidator().validate(
        sample,
        data_root=data_root,
        registration_enabled=settings.registration.enabled,
    )
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
            pif_valid=False,
            validation=validated.report,
            decision=HarmonizationDecision(
                version=settings.harmonization.version,
                status="skipped",
                reason_codes=["SKIPPED_INVALID_PAIR", "RAW_FALLBACK_USED"],
                metrics=None,
                used_for_proposal=False,
            ),
            transform_summary={},
            registered_t1=None,
            registered_t2=None,
            registration_valid_mask=np.zeros((0, 0), dtype=bool),
            registration_report=None,
        )
        _publish_preparation_audit(prepared, output)
        return prepared

    raw1 = validated.t1.copy()
    raw2 = validated.t2.copy()
    try:
        registered: RegisteredPair = register_pair(
            raw1,
            raw2,
            metadata=sample.metadata,
            settings=settings.registration,
        )
    except RegistrationError:
        # ``quality_policy=fail`` is intentionally not converted into a
        # successful raw fallback. The agent boundary turns this stable code
        # into an AgentExecutionError before any VLM call.
        raise

    registration_report = registered.report
    if settings.registration.save_artifacts:
        _write_json(
            output / "registration_report.json",
            registration_report.model_dump(mode="json"),
        )
    if (
        registration_report.decision.used_for_comparison
        and registered.t1.shape == registered.t2.shape == raw1.shape
    ):
        geometry_t1 = registered.t1.copy()
        geometry_t2 = registered.t2.copy()
        registration_mask = registered.valid_overlap_mask.copy()
        geometry_comparable = True
    elif not settings.registration.enabled and raw1.shape == raw2.shape:
        # An explicitly disabled registration stage is the compatibility
        # request for the legacy same-canvas path.  Its canvas is therefore
        # fully valid even though no geometric quality estimate was made.
        geometry_t1 = raw1.copy()
        geometry_t2 = raw2.copy()
        registration_mask = np.ones(raw1.shape[:2], dtype=bool)
        geometry_comparable = True
    elif raw1.shape == raw2.shape:
        # A failed/rejected enabled registration must not be mistaken for a
        # trustworthy raw comparison merely because the arrays have matching
        # shapes.  Keep the raw fallback available for audit/VLM use, but
        # prevent dense proposal evidence from consuming it.
        geometry_t1 = raw1.copy()
        geometry_t2 = raw2.copy()
        registration_mask = np.zeros(raw1.shape[:2], dtype=bool)
        geometry_comparable = True
    else:
        geometry_t1 = None
        geometry_t2 = None
        registration_mask = np.zeros(raw1.shape[:2], dtype=bool)
        geometry_comparable = False

    transform_summary: dict[str, object] = {
        "registration_status": registration_report.decision.status,
        "registration_model": registration_report.decision.model,
        "registration_used_for_comparison": registration_report.decision.used_for_comparison,
        "registration_reason_codes": list(registration_report.decision.reason_codes),
        "registration_artifacts_saved": settings.registration.save_artifacts,
    }
    if geometry_comparable and settings.harmonization.enabled:
        if registration_report.decision.used_for_comparison:
            harmonization_mask = registration_mask
        else:
            # The legacy same-size path is comparable but deliberately has no
            # registration quality mask; use the complete raw canvas only
            # when registration is explicitly disabled.
            harmonization_mask = (
                np.ones(raw1.shape[:2], dtype=bool)
                if not settings.registration.enabled
                else np.zeros(raw1.shape[:2], dtype=bool)
            )
        if np.any(harmonization_mask):
            try:
                harmonizer = PairHarmonizer(settings.harmonization)
                if bool(np.all(harmonization_mask)):
                    candidate = harmonizer.run(geometry_t1, geometry_t2)
                else:
                    candidate = harmonizer.run(
                        geometry_t1,
                        geometry_t2,
                        valid_mask=harmonization_mask,
                    )
                decision = candidate.decision
                comparison1 = candidate.t1.copy()
                comparison2 = candidate.t2.copy()
                pif_mask = candidate.pif_mask.copy()
                pif_valid = candidate.pif_valid
                transform_summary.update(dict(candidate.transform_summary))
            except Exception as error:
                comparison1, comparison2 = geometry_t1.copy(), geometry_t2.copy()
                pif_mask = np.zeros(raw1.shape[:2], dtype=np.uint8)
                pif_valid = False
                decision = HarmonizationDecision(
                    version=settings.harmonization.version,
                    status="failed",
                    reason_codes=["FAILED_HARMONIZATION_EXCEPTION", "RAW_FALLBACK_USED"],
                    metrics=None,
                    used_for_proposal=False,
                )
                transform_summary.update(
                    {"error_type": type(error).__name__, "sharpness_adjustment_used": False}
                )
        else:
            comparison1, comparison2 = geometry_t1.copy(), geometry_t2.copy()
            pif_mask = np.zeros(raw1.shape[:2], dtype=np.uint8)
            pif_valid = False
            decision = HarmonizationDecision(
                version=settings.harmonization.version,
                status="skipped",
                reason_codes=["SKIPPED_REGISTRATION_UNUSABLE", "RAW_FALLBACK_USED"],
                metrics=None,
                used_for_proposal=False,
            )
    elif geometry_comparable:
        comparison1, comparison2 = geometry_t1.copy(), geometry_t2.copy()
        pif_mask = np.zeros(raw1.shape[:2], dtype=np.uint8)
        pif_valid = False
        decision = HarmonizationDecision(
            version=settings.harmonization.version,
            status="skipped",
            reason_codes=[
                "SKIPPED_DISABLED",
                *([] if settings.registration.enabled else ["REGISTRATION_DISABLED"]),
                "RAW_FALLBACK_USED",
            ],
            metrics=None,
            used_for_proposal=False,
        )
    else:
        comparison1, comparison2 = None, None
        pif_mask = np.zeros(raw1.shape[:2], dtype=np.uint8)
        pif_valid = False
        decision = HarmonizationDecision(
            version=settings.harmonization.version,
            status="skipped",
            reason_codes=[
                *(["SKIPPED_INVALID_PAIR"] if not validated.report.same_size else []),
                "SKIPPED_REGISTRATION_UNUSABLE",
                "RAW_FALLBACK_USED",
            ],
            metrics=None,
            used_for_proposal=False,
        )

    prepared = ChangePreparedPair(
        raw_t1=raw1,
        raw_t2=raw2,
        comparison_t1=comparison1,
        comparison_t2=comparison2,
        pif_mask=pif_mask,
        pif_valid=pif_valid,
        validation=validated.report,
        decision=decision,
        transform_summary=transform_summary,
        registered_t1=geometry_t1,
        registered_t2=geometry_t2,
        registration_valid_mask=registration_mask,
        registration_report=registration_report,
    )
    _publish_registration_artifacts(prepared, output, settings=settings)
    _publish_preparation_audit(prepared, output)
    return prepared


def _publish_registration_artifacts(
    prepared: ChangePreparedPair,
    output: Path,
    *,
    settings: AgentChangeSettings,
) -> None:
    """Publish registration-only artifacts without exposing raw replacements."""

    if prepared.registration_report is None or not settings.registration.save_artifacts:
        return
    np = _require_numpy()
    report = prepared.registration_report
    if (
        report.decision.used_for_comparison
        and report.decision.model != "identity"
        and prepared.registered_t2 is not None
    ):
        _write_image(output / "registered_t2.png", prepared.registered_t2)
        _write_image(
            output / "registration_overlap_mask.png",
            prepared.registration_valid_mask.astype(np.uint8) * 255,
        )


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
    if (
        prepared.raw_t1 is None
        or prepared.raw_t2 is None
        or prepared.comparison_t1 is None
        or prepared.comparison_t2 is None
    ):
        return _preparation_result(prepared)

    proposals: list[ChangeProposal] = []
    score_map = np.zeros(prepared.raw_t1.shape[:2], dtype=np.float32)
    if settings.proposals.enabled:
        score_map, proposals = propose_changes(
            prepared.comparison_t1,
            prepared.comparison_t2,
            settings.proposals,
            valid_mask=prepared.registration_valid_mask,
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
    component_masks: dict[str, Any] | None = None,
    diagnostics: dict[str, object] | None = None,
    rescue_candidates: list[StructuralRescueCandidate] | None = None,
) -> ChangePreprocessResult:
    """Publish legacy or V2 maps, overlays, crops, and serializable reports."""

    if prepared.raw_t1 is None or prepared.raw_t2 is None:
        raise ValueError("cannot publish proposals for an invalid prepared pair")
    np = _require_numpy()
    output = artifact_dir / "change_preprocess"
    output.mkdir(parents=True, exist_ok=True)
    files = _audit_files(prepared)
    is_v2 = component_maps is not None or component_masks is not None or any(
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

    diagnostics = diagnostics or {}
    pif_used = bool(
        diagnostics.get("pif_used_for_feature_alignment", False)
        or diagnostics.get("pif_used_for_threshold", False)
    )
    if pif_used and not prepared.pif_valid:
        raise ValueError("invalid PIF cannot be published as consumed V2 evidence")
    if prepared.decision.status == "applied" and settings.harmonization.save_artifacts:
        _write_image(output / "harmonized_t1.png", prepared.comparison_t1)
        _write_image(output / "harmonized_t2.png", prepared.comparison_t2)
        files.update(
            {
                "harmonized_t1": "change_preprocess/harmonized_t1.png",
                "harmonized_t2": "change_preprocess/harmonized_t2.png",
            }
        )
    publish_pif = pif_used or (
        settings.harmonization.save_artifacts
        and prepared.decision.status == "applied"
    )
    if publish_pif:
        _write_image(output / "pif_mask.png", prepared.pif_mask)
        files["pif_mask"] = "change_preprocess/pif_mask.png"

    crops = output / "crops"
    updated: list[ChangeProposal] = []
    for proposal in proposals:
        x1, y1, x2, y2 = proposal.pixel_box
        evidence: list[str] = []
        reference_crop = prepared.raw_t1[y1:y2, x1:x2]
        registered_t2 = _registered_t2_for_evidence(prepared)
        crop_images = [("raw_t1", prepared.raw_t1)]
        if registered_t2 is not None:
            crop_images.append(("registered_t2", registered_t2))
        else:
            raw_t2_evidence = _raw_t2_for_same_canvas_evidence(prepared)
            if raw_t2_evidence is not None:
                crop_images.append(("raw_t2", raw_t2_evidence))
        for label, image in crop_images:
            filename = f"{proposal.proposal_id}_{label}.png"
            _write_image(crops / filename, image[y1:y2, x1:x2])
            evidence.append(f"change_preprocess/crops/{filename}")
        published_mask_filename = proposal.mask_filename
        if component_masks is not None and proposal.mask_filename is not None:
            mask_basename = Path(proposal.mask_filename).name
            component_mask = component_masks.get(proposal.mask_filename)
            if component_mask is None:
                component_mask = component_masks.get(mask_basename)
            if component_mask is None:
                raise ValueError(
                    f"missing component mask for proposal {proposal.proposal_id}"
                )
            component_mask = _component_mask_artifact(
                component_mask,
                expected_shape=reference_crop.shape[:2],
            )
            _write_image(crops / mask_basename, component_mask)
            overlay_filename = f"{proposal.proposal_id}_mask_overlay.png"
            overlay_image = (
                registered_t2[y1:y2, x1:x2]
                if registered_t2 is not None
                else reference_crop
            )
            _write_image(
                crops / overlay_filename,
                _render_change_mask_overlay(
                    overlay_image,
                    component_mask,
                ),
            )
            published_mask_filename = f"change_preprocess/crops/{mask_basename}"
            overlay_relative = f"change_preprocess/crops/{overlay_filename}"
            evidence.append(overlay_relative)
            files[f"{proposal.proposal_id}_mask"] = published_mask_filename
            files[f"{proposal.proposal_id}_mask_overlay"] = overlay_relative
        if prepared.decision.status == "applied":
            for label, image in (
                ("harmonized_t1", prepared.comparison_t1),
                ("harmonized_t2", prepared.comparison_t2),
            ):
                filename = f"{proposal.proposal_id}_{label}.png"
                _write_image(crops / filename, image[y1:y2, x1:x2])
                evidence.append(f"change_preprocess/crops/{filename}")
        updated.append(
            proposal.model_copy(
                update={
                    "evidence_filenames": evidence,
                    "mask_filename": published_mask_filename,
                }
            )
        )

    _write_json(
        output / "proposals.json",
        [item.model_dump(mode="json") for item in updated],
    )
    files["proposals"] = "change_preprocess/proposals.json"
    candidate_payload = None if rescue_candidates is None else list(rescue_candidates)
    if candidate_payload is not None:
        _write_json(
            output / "building_rescue_candidates.json",
            [item.model_dump(mode="json") for item in candidate_payload],
        )
        files["building_rescue_candidates"] = (
            "change_preprocess/building_rescue_candidates.json"
        )
    files["harmonization_report"] = "change_preprocess/harmonization_report.json"
    result = ChangePreprocessResult(
        validation=prepared.validation,
        decision=prepared.decision,
        proposals=updated,
        artifact_files=files,
        transform_summary=prepared.transform_summary,
        diagnostics=diagnostics,
        registration=prepared.registration_report,
        rescue_candidates=candidate_payload or [],
    )
    _write_json(output / "harmonization_report.json", result.model_dump(mode="json"))
    return result


def _audit_files(prepared: ChangePreparedPair | None = None) -> dict[str, str]:
    files = {
        "validation_report": "change_preprocess/validation_report.json",
        "harmonization_report": "change_preprocess/harmonization_report.json",
    }
    report = prepared.registration_report if prepared is not None else None
    if (
        report is not None
        and report.decision.status != "skipped"
        and prepared is not None
        and prepared.transform_summary.get("registration_artifacts_saved", True)
    ):
        files["registration_report"] = "change_preprocess/registration_report.json"
        if report.decision.used_for_comparison and report.decision.model != "identity":
            files["registered_t2"] = "change_preprocess/registered_t2.png"
            files["registration_overlap_mask"] = (
                "change_preprocess/registration_overlap_mask.png"
            )
    return files


def _preparation_result(prepared: ChangePreparedPair) -> ChangePreprocessResult:
    return ChangePreprocessResult(
        validation=prepared.validation,
        decision=prepared.decision,
        proposals=[],
        artifact_files=_audit_files(prepared),
        transform_summary=prepared.transform_summary,
        diagnostics={"pif_valid": prepared.pif_valid},
        registration=prepared.registration_report,
    )


def _publish_preparation_audit(prepared: ChangePreparedPair, output: Path) -> None:
    _write_json(
        output / "harmonization_report.json",
        _preparation_result(prepared).model_dump(mode="json"),
    )


def _registered_t2_for_evidence(prepared: ChangePreparedPair) -> Any | None:
    report = prepared.registration_report
    if (
        report is None
        or not report.decision.used_for_comparison
        or report.decision.model == "identity"
        or prepared.registered_t2 is None
    ):
        return None
    return prepared.registered_t2


def _raw_t2_for_same_canvas_evidence(prepared: ChangePreparedPair) -> Any | None:
    """Return raw T2 only when its pixels share the T1 reference canvas.

    A raw T2 crop cannot inherit a T1 proposal box across different source
    sizes, nor after an enabled registration failure.  In those cases the
    full raw T2 remains available to the VLM, while local evidence uses only
    the reference T1 and any registered T2 artifact.
    """

    if prepared.raw_t1.shape[:2] != prepared.raw_t2.shape[:2]:
        return None
    report = prepared.registration_report
    if report is None:
        return prepared.raw_t2
    if "REGISTRATION_DISABLED" in report.decision.reason_codes:
        return prepared.raw_t2
    if report.decision.used_for_comparison and report.decision.model == "identity":
        return prepared.raw_t2
    return None


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


def _component_mask_artifact(value: Any, *, expected_shape: tuple[int, int]) -> Any:
    """Validate one crop-local component mask and normalize it to 0/255."""

    np = _require_numpy()
    mask = np.asarray(value)
    if mask.ndim != 2 or mask.shape != expected_shape:
        raise ValueError("change component mask shape does not match proposal crop")
    if not bool(np.all(np.isfinite(mask))):
        raise ValueError("change component mask contains non-finite values")
    return (mask != 0).astype(np.uint8) * 255


def _render_change_mask_overlay(image: Any, mask: Any) -> Any:
    """Overlay a crop-local component mask without altering raw evidence."""

    np = _require_numpy()
    overlay = np.asarray(image).copy()
    selected = np.asarray(mask) != 0
    if overlay.ndim != 3 or overlay.shape[:2] != selected.shape:
        raise ValueError("change mask overlay inputs are incompatible")
    color = np.asarray([255, 40, 40], dtype=np.float32)
    overlay[selected] = np.round(
        overlay[selected].astype(np.float32) * 0.55 + color * 0.45
    ).astype(np.uint8)
    return overlay


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
