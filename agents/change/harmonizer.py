"""PIF/LAB midpoint harmonization with conservative quality gating.

带保守质量门控的 PIF/LAB 中间域一致化。算法纯本地（不调用模型），输入图
绝不原地修改（始终从副本开始）；一致化失败时返回明确的 raw fallback
decision。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.change.schema import HarmonizationDecision, HarmonizationMetrics
from agents.change.settings import ChangeHarmonizationSettings
from agents.errors import OptionalDependencyMissingError


def _require_cv2():
    """Return the cv2 module or a stable optional-dependency error.
    返回 cv2 模块，缺失时抛出稳定的可选依赖错误。"""
    try:
        import cv2
    except ImportError as error:
        raise OptionalDependencyMissingError(
            "change", dependency="opencv-python-headless"
        ) from error
    return cv2


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
class HarmonizationCandidate:
    """In-memory candidate retained only when the gate accepts it.
    仅门控通过时采用的内存候选。"""

    t1: np.ndarray
    t2: np.ndarray
    pif_mask: np.ndarray
    pif_valid: bool
    decision: HarmonizationDecision
    transform_summary: dict[str, Any]


class PairHarmonizer:
    """Formalized deterministic version of the validated scratch algorithm.
    已验证临时算法的正式确定性版本。"""

    def __init__(self, settings: ChangeHarmonizationSettings) -> None:
        self.settings = settings
        calibration = settings.calibration_file
        if calibration and Path(calibration).is_file():
            try:
                payload = json.loads(Path(calibration).read_text(encoding="utf-8"))
                p05 = payload.get("pif_ratio", {}).get("p05")
                if isinstance(p05, (int, float)):
                    self.settings = settings.model_copy(
                        update={
                            "min_pif_ratio": max(settings.min_pif_ratio, float(p05))
                        }
                    )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                # Invalid calibration stays visible through CALIBRATION_NOT_AVAILABLE.
                # 无效校准通过 CALIBRATION_NOT_AVAILABLE 保持可见。
                self.settings = settings.model_copy(update={"calibration_file": None})

    def run(self, t1: np.ndarray, t2: np.ndarray) -> HarmonizationCandidate:
        cv2 = _require_cv2()
        np = _require_numpy()
        raw1, raw2 = t1.copy(), t2.copy()
        mask = estimate_pif_mask(raw1, raw2, self.settings)
        ratio = float(np.mean(mask > 0))
        calibration_available = bool(
            self.settings.calibration_file and Path(self.settings.calibration_file).is_file()
        )
        base_reasons = [] if calibration_available else ["CALIBRATION_NOT_AVAILABLE"]
        if int(np.count_nonzero(mask)) < self.settings.min_pif_pixels or ratio < self.settings.min_pif_ratio:
            decision = HarmonizationDecision(
                version=self.settings.version,
                status="skipped",
                reason_codes=[
                    *base_reasons,
                    "SKIPPED_INSUFFICIENT_PIF",
                    "RAW_FALLBACK_USED",
                ],
                metrics=None,
                used_for_proposal=False,
            )
            return HarmonizationCandidate(
                t1=raw1,
                t2=raw2,
                pif_mask=mask,
                pif_valid=False,
                decision=decision,
                transform_summary={"sharpness_adjustment_used": False},
            )

        lab1 = cv2.cvtColor(raw1, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab2 = cv2.cvtColor(raw2, cv2.COLOR_RGB2LAB).astype(np.float32)
        selected = mask > 0
        out1, out2 = lab1.copy(), lab2.copy()
        transforms: list[dict[str, float]] = []
        stable = True
        for channel in range(3):
            v1, v2 = lab1[..., channel][selected], lab2[..., channel][selected]
            q1 = np.percentile(v1, [5, 50, 95])
            q2 = np.percentile(v2, [5, 50, 95])
            target = (q1 + q2) * 0.5
            a1, b1 = _affine(v1, target)
            a2, b2 = _affine(v2, target)
            stable = (
                stable
                and all(np.isfinite([a1, b1, a2, b2]))
                and max(abs(a1), abs(a2)) <= self.settings.max_abs_gain
                and max(abs(b1), abs(b2)) <= self.settings.max_abs_offset
            )
            transforms.append(
                {
                    "channel": float(channel),
                    "t1_gain": a1,
                    "t1_offset": b1,
                    "t2_gain": a2,
                    "t2_offset": b2,
                }
            )
            out1[..., channel] = lab1[..., channel] * a1 + b1
            out2[..., channel] = lab2[..., channel] * a2 + b2

        clipped_ratio = float(
            np.mean((out1 <= 0) | (out1 >= 255)) + np.mean((out2 <= 0) | (out2 >= 255))
        ) * 0.5
        color1 = cv2.cvtColor(np.clip(out1, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
        color2 = cv2.cvtColor(np.clip(out2, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
        (
            h1,
            h2,
            sharpness_used,
            sharpness_safe,
            sharpness_ratio_before,
            sharpness_ratio_after,
        ) = _match_sharpness(color1, color2, self.settings)
        metrics = compute_metrics(raw1, raw2, h1, h2, mask)
        summary: dict[str, Any] = {
            "lab_transforms": transforms,
            "sharpness_adjustment_used": sharpness_used,
            "sharpness_adjustment_safe": sharpness_safe,
            "sharpness_adjustment_rolled_back": not sharpness_safe,
            "sharpness_ratio_before": sharpness_ratio_before,
            "sharpness_ratio_after": sharpness_ratio_after,
            "clipped_pixel_ratio": clipped_ratio,
        }

        reasons = list(base_reasons)
        if not sharpness_safe:
            # An unsafe optional blur is rolled back locally; it does not
            # invalidate safe color mapping. 不安全的可选模糊仅局部回退，
            # 不使安全的颜色映射整体失效。
            reasons.append("SHARPNESS_ADJUSTMENT_ROLLED_BACK")
        if not stable or clipped_ratio > self.settings.max_clipped_pixel_ratio:
            status, code = "rejected", "REJECTED_UNSTABLE_TRANSFORM"
        elif (
            self.settings.reject_when_pif_mad_worse
            and metrics.mad_pif_after > metrics.mad_pif_before * self.settings.max_pif_mad_degradation_ratio
        ):
            status, code = "rejected", "REJECTED_PIF_MAD_WORSE"
        else:
            status, code = "applied", "APPLIED_OK"
        reasons.append(code)
        used = status == "applied"
        if not used:
            reasons.append("RAW_FALLBACK_USED")
        decision = HarmonizationDecision(
            version=self.settings.version,
            status=status,
            reason_codes=reasons,
            metrics=metrics,
            used_for_proposal=used,
        )
        return HarmonizationCandidate(
            t1=h1 if used else raw1,
            t2=h2 if used else raw2,
            pif_mask=mask,
            pif_valid=True,
            decision=decision,
            transform_summary=summary,
        )


def estimate_pif_mask(
    t1: np.ndarray,
    t2: np.ndarray,
    settings: ChangeHarmonizationSettings,
) -> np.ndarray:
    """Estimate PIF once from raw inputs; callers reuse this exact mask.
    仅从原图估计一次 PIF，调用方复用同一掩膜。"""

    cv2 = _require_cv2()
    np = _require_numpy()
    g1 = cv2.cvtColor(t1, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(t2, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ksize = settings.pif_blur_ksize | 1
    low_diff = np.abs(
        cv2.GaussianBlur(g1, (ksize, ksize), 0) - cv2.GaussianBlur(g2, (ksize, ksize), 0)
    )
    grad1 = cv2.magnitude(
        cv2.Sobel(g1, cv2.CV_32F, 1, 0), cv2.Sobel(g1, cv2.CV_32F, 0, 1)
    )
    grad2 = cv2.magnitude(
        cv2.Sobel(g2, cv2.CV_32F, 1, 0), cv2.Sobel(g2, cv2.CV_32F, 0, 1)
    )
    grad_diff = np.abs(grad1 - grad2)
    low_thr = float(np.median(low_diff) + settings.pif_diff_k * _robust_mad(low_diff))
    grad_thr = float(np.median(grad_diff) + settings.pif_grad_k * _robust_mad(grad_diff))
    valid = (g1 > 10) & (g1 < 245) & (g2 > 10) & (g2 < 245)
    mask = ((low_diff <= low_thr) & (grad_diff <= grad_thr) & valid).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def compute_metrics(
    raw1: np.ndarray,
    raw2: np.ndarray,
    out1: np.ndarray,
    out2: np.ndarray,
    mask: np.ndarray,
) -> HarmonizationMetrics:
    """Compute before/after metrics with the same raw-derived PIF mask.
    使用同一个原图 PIF 掩膜计算前后指标。"""

    np = _require_numpy()
    a, b = _gray(raw1), _gray(raw2)
    x, y = _gray(out1), _gray(out2)
    selected = mask > 0
    return HarmonizationMetrics(
        pif_ratio=float(selected.mean()),
        mad_full_before=float(np.mean(np.abs(a - b))),
        mad_full_after=float(np.mean(np.abs(x - y))),
        mad_pif_before=float(np.mean(np.abs(a[selected] - b[selected]))),
        mad_pif_after=float(np.mean(np.abs(x[selected] - y[selected]))),
        corr_full_before=_corr(a, b),
        corr_full_after=_corr(x, y),
        corr_pif_before=_corr(a[selected], b[selected]),
        corr_pif_after=_corr(x[selected], y[selected]),
        pct_diff_gt20_before=float(np.mean(np.abs(a - b) > 20)),
        pct_diff_gt20_after=float(np.mean(np.abs(x - y) > 20)),
        lapvar_t1_before=_lapvar(raw1),
        lapvar_t2_before=_lapvar(raw2),
        lapvar_t1_after=_lapvar(out1),
        lapvar_t2_after=_lapvar(out2),
    )


def _match_sharpness(
    t1: np.ndarray,
    t2: np.ndarray,
    settings: ChangeHarmonizationSettings,
) -> tuple[np.ndarray, np.ndarray, bool, bool, float, float]:
    """Safely match sharpness by optional blur on the sharper image only.
    仅对更清晰图像执行可选模糊的安全锐度匹配。"""

    cv2 = _require_cv2()
    np = _require_numpy()
    before = [_lapvar(t1), _lapvar(t2)]
    ratio_before = max(before) / max(min(before), 1e-6)
    if not settings.match_sharpness:
        return t1, t2, False, True, ratio_before, ratio_before
    high, low = (0, 1) if before[0] > before[1] else (1, 0)
    ratio_before = max(before) / max(min(before), 1e-6)
    if ratio_before <= settings.sharpness_tolerance_ratio:
        return t1, t2, False, True, ratio_before, ratio_before
    images = [t1, t2]
    best, best_ratio = images[high], ratio_before
    for sigma in np.linspace(0.3, settings.max_blur_sigma, 6):
        candidate = cv2.GaussianBlur(images[high], (0, 0), float(sigma))
        ratio = max(_lapvar(candidate), before[low]) / max(
            min(_lapvar(candidate), before[low]), 1e-6
        )
        if ratio < best_ratio:
            best, best_ratio = candidate, ratio
    retained = _lapvar(best) / max(before[high], 1e-6)
    safe = best_ratio < ratio_before and retained >= settings.min_retained_lapvar_ratio
    if not safe:
        return t1, t2, False, False, ratio_before, ratio_before
    result = [t1.copy(), t2.copy()]
    result[high] = best
    return result[0], result[1], True, True, ratio_before, best_ratio


def _affine(values: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    np = _require_numpy()
    source = np.percentile(values, [5, 50, 95])
    gain = float((target[2] - target[0]) / max(source[2] - source[0], 1e-6))
    return gain, float(target[1] - gain * source[1])


def _gray(image: np.ndarray) -> np.ndarray:
    cv2 = _require_cv2()
    np = _require_numpy()
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)


def _lapvar(image: np.ndarray) -> float:
    cv2 = _require_cv2()
    return float(cv2.Laplacian(_gray(image), cv2.CV_32F).var())


def _robust_mad(values: np.ndarray) -> float:
    np = _require_numpy()
    median = np.median(values)
    return float(np.median(np.abs(values - median)) + 1e-6)


def _corr(first: np.ndarray, second: np.ndarray) -> float | None:
    np = _require_numpy()
    a, b = first.reshape(-1), second.reshape(-1)
    if a.size < 2 or float(np.std(a)) <= 1e-8 or float(np.std(b)) <= 1e-8:
        return None
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else None
