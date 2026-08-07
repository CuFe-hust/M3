"""Change preprocessing orchestration and sample-scoped artifacts.

变化预处理编排与样本级产物。组合 pair 校验、一致化与差异提议；只在
artifact_dir 内写入派生产物（绝不修改源图片）；不调用视觉模型；写盘失败
显式向上暴露。numpy 与 cv2 是可选依赖（[change] extra），仅在编排执行时
惰性加载，模块导入本身不触发。
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from agents.change.difference_proposal import propose_changes, render_overlay
from agents.change.harmonizer import PairHarmonizer
from agents.change.pair_validator import PairValidator
from agents.change.schema import ChangePreprocessResult, HarmonizationDecision
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


def preprocess_pair(
    sample: UnifiedSample,
    settings: AgentChangeSettings,
    artifact_dir: Path,
    *,
    data_root: Path,
) -> ChangePreprocessResult:
    """Validate, harmonize, gate, propose, and publish auditable files.
    校验、一致化、门控、提议并发布可审计文件。"""
    np = _require_numpy()
    output = artifact_dir / "change_preprocess"
    output.mkdir(parents=True, exist_ok=True)
    validated = PairValidator().validate(sample, data_root=data_root)
    _write_json(output / "validation_report.json", validated.report.model_dump(mode="json"))
    files: dict[str, str] = {"validation_report": "change_preprocess/validation_report.json"}
    if not validated.report.valid or validated.t1 is None or validated.t2 is None:
        decision = HarmonizationDecision(
            version=settings.harmonization.version,
            status="skipped",
            reason_codes=["SKIPPED_INVALID_PAIR", "RAW_FALLBACK_USED"],
            metrics=None,
            used_for_proposal=False,
        )
        files["harmonization_report"] = "change_preprocess/harmonization_report.json"
        result = ChangePreprocessResult(
            validation=validated.report,
            decision=decision,
            proposals=[],
            artifact_files=files,
        )
        _write_json(output / "harmonization_report.json", result.model_dump(mode="json"))
        return result

    raw1, raw2 = validated.t1, validated.t2
    transform_summary: dict[str, object] = {}
    if settings.harmonization.enabled:
        try:
            candidate = PairHarmonizer(settings.harmonization).run(raw1, raw2)
            proposal1, proposal2 = candidate.t1, candidate.t2
            decision = candidate.decision
            transform_summary = candidate.transform_summary
            pif_mask = candidate.pif_mask
        except Exception as error:
            # Preserve the sample while exposing the preprocessing failure.
            # 保留样本并显式暴露预处理失败。
            proposal1, proposal2 = raw1, raw2
            pif_mask = np.zeros(raw1.shape[:2], dtype=np.uint8)
            decision = HarmonizationDecision(
                version=settings.harmonization.version,
                status="failed",
                reason_codes=["FAILED_HARMONIZATION_EXCEPTION", "RAW_FALLBACK_USED"],
                metrics=None,
                used_for_proposal=False,
            )
            transform_summary = {
                "error_type": type(error).__name__,
                "sharpness_adjustment_used": False,
            }
    else:
        proposal1, proposal2 = raw1, raw2
        pif_mask = np.zeros(raw1.shape[:2], dtype=np.uint8)
        decision = HarmonizationDecision(
            version=settings.harmonization.version,
            status="skipped",
            reason_codes=["SKIPPED_DISABLED", "RAW_FALLBACK_USED"],
            metrics=None,
            used_for_proposal=False,
        )

    proposals: list = []
    score_map = np.zeros(raw1.shape[:2], dtype=np.float32)
    if settings.proposals.enabled:
        score_map, proposals = propose_changes(proposal1, proposal2, settings.proposals)
    overlay = render_overlay(proposal2, proposals)
    _write_image(output / "difference_map.png", np.round(score_map * 255).astype(np.uint8))
    _write_image(output / "proposal_overlay.png", overlay)
    files.update(
        {
            "difference_map": "change_preprocess/difference_map.png",
            "proposal_overlay": "change_preprocess/proposal_overlay.png",
        }
    )
    if decision.status == "applied" and settings.harmonization.save_artifacts:
        _write_image(output / "harmonized_t1.png", proposal1)
        _write_image(output / "harmonized_t2.png", proposal2)
        _write_image(output / "pif_mask.png", pif_mask)
        files.update(
            {
                "harmonized_t1": "change_preprocess/harmonized_t1.png",
                "harmonized_t2": "change_preprocess/harmonized_t2.png",
                "pif_mask": "change_preprocess/pif_mask.png",
            }
        )

    crops = output / "crops"
    updated: list = []
    for proposal in proposals:
        x1, y1, x2, y2 = proposal.pixel_box
        evidence: list[str] = []
        for label, image in (("raw_t1", raw1), ("raw_t2", raw2)):
            filename = f"{proposal.proposal_id}_{label}.png"
            _write_image(crops / filename, image[y1:y2, x1:x2])
            evidence.append(f"change_preprocess/crops/{filename}")
        if decision.status == "applied":
            for label, image in (("harmonized_t1", proposal1), ("harmonized_t2", proposal2)):
                filename = f"{proposal.proposal_id}_{label}.png"
                _write_image(crops / filename, image[y1:y2, x1:x2])
                evidence.append(f"change_preprocess/crops/{filename}")
        updated.append(proposal.model_copy(update={"evidence_filenames": evidence}))

    _write_json(output / "proposals.json", [item.model_dump(mode="json") for item in updated])
    files["proposals"] = "change_preprocess/proposals.json"
    files["harmonization_report"] = "change_preprocess/harmonization_report.json"
    result = ChangePreprocessResult(
        validation=validated.report,
        decision=decision,
        proposals=updated,
        artifact_files=files,
        transform_summary=transform_summary,
    )
    _write_json(output / "harmonization_report.json", result.model_dump(mode="json"))
    return result


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
