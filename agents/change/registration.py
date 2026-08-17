"""Conservative global geometric registration for Change V3.

The module owns only the geometric comparison frame. It never mutates decoded
source arrays, performs radiometric harmonization, or creates semantic claims.
OpenCV and NumPy are loaded lazily so core imports remain optional-dependency
safe.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from agents.change.schema import RegistrationDecision, RegistrationMetrics, RegistrationReport
from agents.change.settings import ChangeRegistrationSettings
from agents.errors import OptionalDependencyMissingError


RegistrationFailureCode = Literal[
    "REGISTRATION_INSUFFICIENT_MATCHES",
    "REGISTRATION_LOW_INLIER_RATIO",
    "REGISTRATION_HIGH_REPROJECTION_ERROR",
    "REGISTRATION_IMPLAUSIBLE_TRANSFORM",
    "REGISTRATION_LOW_COVERAGE",
    "REGISTRATION_FAILED_EXCEPTION",
]


class RegistrationError(RuntimeError):
    """Stable error raised when ``quality_policy='fail'`` is selected."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True)
class RegisteredPair:
    """Comparison-frame images and the audit record for one pair."""

    t1: Any
    t2: Any
    valid_overlap_mask: Any
    report: RegistrationReport


@dataclass(frozen=True)
class _Candidate:
    model: str
    matrix: Any
    inlier_mask: Any
    metrics: RegistrationMetrics
    reason_code: str | None


class GeometricRegistration:
    """Run conservative identity/similarity/affine/homography registration."""

    def __init__(self, settings: ChangeRegistrationSettings) -> None:
        self._settings = settings

    def run(
        self,
        t1: Any,
        t2: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> RegisteredPair:
        return register_pair(t1, t2, metadata=metadata, settings=self._settings)


def register_pair(
    t1: Any,
    t2: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
    settings: ChangeRegistrationSettings | None = None,
) -> RegisteredPair:
    """Register raw RGB arrays by warping T2 into the T1 canvas.

    Models are evaluated in conservative complexity order. A later model is
    considered only when the preceding model fails its quality gate; a small
    numerical improvement never justifies a more flexible transform.
    """

    np = _require_numpy()
    config = settings or ChangeRegistrationSettings()
    first = _validate_image(t1, name="t1", np=np)
    second = _validate_image(t2, name="t2", np=np)
    first_copy = first.copy()
    second_copy = second.copy()
    height_1, width_1 = first.shape[:2]
    height_2, width_2 = second.shape[:2]
    size_t1 = [width_1, height_1]
    size_t2 = [width_2, height_2]
    same_size = first.shape[:2] == second.shape[:2]

    if not config.enabled:
        return _identity_pair(
            first_copy,
            second_copy,
            size_t1=size_t1,
            size_t2=size_t2,
            reason_codes=["REGISTRATION_DISABLED"],
            status="skipped",
            used_for_comparison=False,
            version=config.version,
            np=np,
        )

    if config.prefer_metadata_alignment and same_size and _has_strong_alignment_metadata(metadata):
        return _identity_pair(
            first_copy,
            second_copy,
            size_t1=size_t1,
            size_t2=size_t2,
            reason_codes=["REGISTRATION_NOT_NEEDED", "METADATA_ALIGNMENT_USED"],
            status="skipped",
            used_for_comparison=True,
            version=config.version,
            np=np,
        )

    if same_size and bool(np.any(first)) and bool(np.array_equal(first, second)):
        return _identity_pair(
            first_copy,
            second_copy,
            size_t1=size_t1,
            size_t2=size_t2,
            reason_codes=["REGISTRATION_NOT_NEEDED", "IDENTICAL_INPUTS"],
            status="skipped",
            used_for_comparison=True,
            version=config.version,
            np=np,
        )

    try:
        cv2 = _require_cv2()
        gray_1 = _preprocess_gray(first_copy, cv2=cv2)
        gray_2 = _preprocess_gray(second_copy, cv2=cv2)
        keypoints_1, descriptors_1, detector_1 = _detect(gray_1, config, cv2=cv2)
        keypoints_2, descriptors_2, detector_2 = _detect(gray_2, config, cv2=cv2)
        matcher_name = detector_1
        diagnostics: dict[str, object] = {
            "detector": detector_1,
            "feature_count_t1": len(keypoints_1),
            "feature_count_t2": len(keypoints_2),
            "ratio_test": config.ratio_test,
        }
        if detector_1 != detector_2:
            keypoints_1, descriptors_1 = _detect_orb(gray_1, config, cv2=cv2)
            keypoints_2, descriptors_2 = _detect_orb(gray_2, config, cv2=cv2)
            matcher_name = "orb"
            diagnostics["detector"] = matcher_name
            diagnostics["fallback_reason"] = "SIFT_UNAVAILABLE_FOR_ONE_FRAME"
        elif detector_1 == "orb":
            diagnostics["fallback_reason"] = "SIFT_UNAVAILABLE_OR_FAILED"

        matches = _match_descriptors(
            descriptors_1,
            descriptors_2,
            detector=matcher_name,
            ratio_test=config.ratio_test,
            cv2=cv2,
        )
        source_points, destination_points = _matched_points(
            keypoints_1, keypoints_2, matches, np=np
        )
        if len(matches) < config.min_matches:
            return _failure_pair(
                first_copy,
                second_copy,
                size_t1=size_t1,
                size_t2=size_t2,
                reason_code="REGISTRATION_INSUFFICIENT_MATCHES",
                match_count=len(matches),
                diagnostics=diagnostics,
                settings=config,
                np=np,
            )

        candidates: list[_Candidate] = []
        identity = _evaluate_identity(
            source_points,
            destination_points,
            size_t1=size_t1,
            size_t2=size_t2,
            config=config,
            cv2=cv2,
            np=np,
        )
        candidates.append(identity)
        if identity.reason_code is None:
            return _accepted_pair(
                first_copy,
                second_copy,
                identity,
                size_t1=size_t1,
                size_t2=size_t2,
                diagnostics=diagnostics,
                version=config.version,
                cv2=cv2,
                np=np,
            )

        for model in _enabled_models(config):
            candidate = _fit_candidate(
                model,
                source_points,
                destination_points,
                size_t1=size_t1,
                size_t2=size_t2,
                config=config,
                cv2=cv2,
                np=np,
            )
            if candidate is None:
                continue
            candidates.append(candidate)
            if candidate.reason_code is None:
                return _accepted_pair(
                    first_copy,
                    second_copy,
                    candidate,
                    size_t1=size_t1,
                    size_t2=size_t2,
                    diagnostics=diagnostics,
                    version=config.version,
                    cv2=cv2,
                    np=np,
                )

        return _failure_pair(
            first_copy,
            second_copy,
            size_t1=size_t1,
            size_t2=size_t2,
            reason_code=_select_failure_reason(candidates),
            match_count=len(matches),
            metrics=candidates[-1].metrics if candidates else None,
            diagnostics=diagnostics,
            settings=config,
            np=np,
        )
    except RegistrationError:
        raise
    except Exception as error:
        return _failure_pair(
            first_copy,
            second_copy,
            size_t1=size_t1,
            size_t2=size_t2,
            reason_code="REGISTRATION_FAILED_EXCEPTION",
            match_count=0,
            diagnostics={"exception_type": type(error).__name__},
            settings=config,
            np=np,
        )


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise OptionalDependencyMissingError("change", dependency="numpy") from error
    return np


def _require_cv2() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise OptionalDependencyMissingError(
            "change", dependency="opencv-python-headless"
        ) from error
    return cv2


def _validate_image(value: Any, *, name: str, np: Any) -> Any:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"registration {name} must be an RGB HWC array")
    if any(int(dimension) <= 0 for dimension in image.shape[:2]):
        raise ValueError(f"registration {name} must be non-empty")
    if image.dtype != np.uint8:
        if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
            raise ValueError(f"registration {name} must be finite numeric RGB")
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _preprocess_gray(image: Any, *, cv2: Any) -> Any:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.GaussianBlur(gray, (3, 3), 0)


def _detect(gray: Any, settings: ChangeRegistrationSettings, *, cv2: Any) -> tuple[list[Any], Any, str]:
    if settings.feature_detector == "sift" and hasattr(cv2, "SIFT_create"):
        try:
            detector = cv2.SIFT_create(nfeatures=settings.max_features)
            keypoints, descriptors = detector.detectAndCompute(gray, None)
            if descriptors is not None and len(keypoints) >= 2:
                return keypoints, descriptors, "sift"
        except Exception:
            pass
    keypoints, descriptors = _detect_orb(gray, settings, cv2=cv2)
    return keypoints, descriptors, "orb"


def _detect_orb(gray: Any, settings: ChangeRegistrationSettings, *, cv2: Any) -> tuple[list[Any], Any]:
    detector = cv2.ORB_create(nfeatures=settings.max_features)
    return detector.detectAndCompute(gray, None)


def _match_descriptors(
    descriptors_1: Any,
    descriptors_2: Any,
    *,
    detector: str,
    ratio_test: float,
    cv2: Any,
) -> list[Any]:
    if descriptors_1 is None or descriptors_2 is None:
        return []
    norm = cv2.NORM_L2 if detector == "sift" else cv2.NORM_HAMMING
    matcher = cv2.BFMatcher(norm, crossCheck=False)
    raw_matches = matcher.knnMatch(descriptors_2, descriptors_1, k=2)
    retained: list[Any] = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < ratio_test * second.distance:
            retained.append(first)
    retained.sort(key=lambda item: (float(item.distance), int(item.queryIdx), int(item.trainIdx)))
    unique_queries: set[int] = set()
    unique_trains: set[int] = set()
    result: list[Any] = []
    for match in retained:
        if match.queryIdx in unique_queries or match.trainIdx in unique_trains:
            continue
        unique_queries.add(match.queryIdx)
        unique_trains.add(match.trainIdx)
        result.append(match)
    return result


def _matched_points(keypoints_1: list[Any], keypoints_2: list[Any], matches: list[Any], *, np: Any) -> tuple[Any, Any]:
    source = np.asarray([keypoints_2[item.queryIdx].pt for item in matches], dtype=np.float32)
    destination = np.asarray([keypoints_1[item.trainIdx].pt for item in matches], dtype=np.float32)
    return source, destination


def _enabled_models(settings: ChangeRegistrationSettings) -> tuple[str, ...]:
    models: list[str] = []
    if settings.allow_similarity:
        models.append("similarity")
    if settings.allow_affine:
        models.append("affine")
    if settings.allow_homography:
        models.append("homography")
    return tuple(models)


def _evaluate_identity(
    source: Any,
    destination: Any,
    *,
    size_t1: list[int],
    size_t2: list[int],
    config: ChangeRegistrationSettings,
    cv2: Any,
    np: Any,
) -> _Candidate:
    matrix = np.eye(3, dtype=np.float64)
    errors = np.linalg.norm(source - destination, axis=1)
    mask = errors <= config.max_median_reprojection_error
    metrics = _metrics(
        matrix,
        errors,
        mask,
        match_count=len(source),
        size_t1=size_t1,
        size_t2=size_t2,
        cv2=cv2,
        np=np,
    )
    return _Candidate("identity", matrix, mask, metrics, _quality_reason(metrics, config))


def _fit_candidate(
    model: str,
    source: Any,
    destination: Any,
    *,
    size_t1: list[int],
    size_t2: list[int],
    config: ChangeRegistrationSettings,
    cv2: Any,
    np: Any,
) -> _Candidate | None:
    threshold = max(1.0, config.max_median_reprojection_error)
    if model == "similarity":
        if len(source) < 2:
            return None
        matrix_2d, inliers = cv2.estimateAffinePartial2D(
            source, destination, method=cv2.RANSAC,
            ransacReprojThreshold=threshold, maxIters=4000,
            confidence=0.995, refineIters=10,
        )
    elif model == "affine":
        if len(source) < 3:
            return None
        matrix_2d, inliers = cv2.estimateAffine2D(
            source, destination, method=cv2.RANSAC,
            ransacReprojThreshold=threshold, maxIters=4000,
            confidence=0.995, refineIters=10,
        )
    elif model == "homography":
        if len(source) < 4:
            return None
        matrix_2d, inliers = cv2.findHomography(
            source, destination, method=cv2.RANSAC,
            ransacReprojThreshold=threshold, maxIters=4000,
            confidence=0.995,
        )
    else:
        raise ValueError(f"unsupported registration model: {model}")
    if matrix_2d is None or inliers is None:
        return None
    matrix = np.asarray(matrix_2d, dtype=np.float64)
    if matrix.shape == (2, 3):
        matrix = np.vstack([matrix, np.asarray([0.0, 0.0, 1.0])])
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return _Candidate(model, np.eye(3), np.zeros(len(source), dtype=bool), _empty_metrics(len(source)), "REGISTRATION_IMPLAUSIBLE_TRANSFORM")
    projected = _project(source, matrix, cv2=cv2, np=np)
    errors = np.linalg.norm(projected - destination, axis=1)
    inlier_mask = np.asarray(inliers).reshape(-1).astype(bool)
    metrics = _metrics(
        matrix, errors, inlier_mask, match_count=len(source),
        size_t1=size_t1, size_t2=size_t2, cv2=cv2, np=np,
    )
    reason = _quality_reason(metrics, config) or _plausibility_reason(metrics, config, size_t1=size_t1)
    return _Candidate(model, matrix, inlier_mask, metrics, reason)


def _project(points: Any, matrix: Any, *, cv2: Any, np: Any) -> Any:
    projected = cv2.perspectiveTransform(
        np.asarray(points, dtype=np.float32).reshape(-1, 1, 2),
        np.asarray(matrix, dtype=np.float64),
    )
    return projected.reshape(-1, 2).astype(np.float64)


def _metrics(
    matrix: Any,
    errors: Any,
    mask: Any,
    *,
    match_count: int,
    size_t1: list[int],
    size_t2: list[int],
    cv2: Any,
    np: Any,
) -> RegistrationMetrics:
    finite_errors = np.asarray(errors, dtype=np.float64)
    inlier_mask = np.asarray(mask, dtype=bool)
    inlier_errors = finite_errors[inlier_mask & np.isfinite(finite_errors)]
    if len(inlier_errors) == 0:
        inlier_errors = np.asarray([1e9], dtype=np.float64)
    inlier_count = int(np.count_nonzero(inlier_mask))
    scale_x, scale_y, rotation, tx, ty, perspective = _transform_facts(matrix, np=np)
    overlap = _overlap_ratio(matrix, size_t1=size_t1, size_t2=size_t2, cv2=cv2, np=np)
    return RegistrationMetrics(
        match_count=match_count,
        inlier_count=inlier_count,
        inlier_ratio=float(inlier_count / match_count) if match_count else 0.0,
        median_reprojection_error=float(np.median(inlier_errors)),
        p95_reprojection_error=float(np.percentile(inlier_errors, 95)),
        overlap_ratio=overlap,
        scale_x=scale_x,
        scale_y=scale_y,
        rotation_deg=rotation,
        translation_x=tx,
        translation_y=ty,
        perspective_magnitude=perspective,
    )


def _transform_facts(matrix: Any, *, np: Any) -> tuple[float, float, float, float, float, float]:
    value = np.asarray(matrix, dtype=np.float64)
    denominator = float(value[2, 2])
    if not math.isfinite(denominator) or abs(denominator) < 1e-12:
        return 0.0, 0.0, 0.0, 0.0, 0.0, float("inf")
    value = value / denominator
    linear = value[:2, :2]
    scale_x = float(np.linalg.norm(linear[:, 0]))
    scale_y = float(np.linalg.norm(linear[:, 1]))
    rotation = math.degrees(math.atan2(float(linear[1, 0]), float(linear[0, 0])))
    perspective = float(max(abs(value[2, 0]), abs(value[2, 1])))
    return scale_x, scale_y, rotation, float(value[0, 2]), float(value[1, 2]), perspective


def _overlap_ratio(matrix: Any, *, size_t1: list[int], size_t2: list[int], cv2: Any, np: Any) -> float:
    width, height = size_t1
    source_width, source_height = size_t2
    source_mask = np.ones((source_height, source_width), dtype=np.uint8)
    warped = cv2.warpPerspective(
        source_mask, np.asarray(matrix, dtype=np.float64), (width, height), flags=cv2.INTER_NEAREST
    )
    return float(np.count_nonzero(warped) / (width * height))


def _quality_reason(metrics: RegistrationMetrics, config: ChangeRegistrationSettings) -> str | None:
    if metrics.match_count < config.min_matches or metrics.inlier_count < config.min_inliers:
        return "REGISTRATION_INSUFFICIENT_MATCHES"
    if metrics.inlier_ratio < config.min_inlier_ratio:
        return "REGISTRATION_LOW_INLIER_RATIO"
    if metrics.median_reprojection_error > config.max_median_reprojection_error:
        return "REGISTRATION_HIGH_REPROJECTION_ERROR"
    if metrics.overlap_ratio < config.min_overlap_ratio:
        return "REGISTRATION_LOW_COVERAGE"
    return None


def _plausibility_reason(
    metrics: RegistrationMetrics,
    config: ChangeRegistrationSettings,
    *,
    size_t1: list[int],
) -> str | None:
    max_scale = config.max_scale_ratio
    if not (1.0 / max_scale <= metrics.scale_x <= max_scale and 1.0 / max_scale <= metrics.scale_y <= max_scale):
        return "REGISTRATION_IMPLAUSIBLE_TRANSFORM"
    if abs(metrics.rotation_deg) > config.max_rotation_deg:
        return "REGISTRATION_IMPLAUSIBLE_TRANSFORM"
    width, height = size_t1
    if abs(metrics.translation_x) / width > config.max_translation_ratio or abs(metrics.translation_y) / height > config.max_translation_ratio:
        return "REGISTRATION_IMPLAUSIBLE_TRANSFORM"
    if metrics.perspective_magnitude > config.max_perspective_magnitude:
        return "REGISTRATION_IMPLAUSIBLE_TRANSFORM"
    return None


def _select_failure_reason(candidates: list[_Candidate]) -> RegistrationFailureCode:
    reasons = [candidate.reason_code for candidate in candidates if candidate.reason_code]
    for code in (
        "REGISTRATION_IMPLAUSIBLE_TRANSFORM",
        "REGISTRATION_LOW_COVERAGE",
        "REGISTRATION_HIGH_REPROJECTION_ERROR",
        "REGISTRATION_LOW_INLIER_RATIO",
        "REGISTRATION_INSUFFICIENT_MATCHES",
    ):
        if code in reasons:
            return code  # type: ignore[return-value]
    return "REGISTRATION_FAILED_EXCEPTION"


def _empty_metrics(match_count: int) -> RegistrationMetrics:
    return RegistrationMetrics(match_count=match_count)


def _identity_pair(
    first: Any,
    second: Any,
    *,
    size_t1: list[int],
    size_t2: list[int],
    reason_codes: list[str],
    status: Literal["skipped", "applied"],
    used_for_comparison: bool,
    version: str,
    np: Any,
) -> RegisteredPair:
    same_size = first.shape[:2] == second.shape[:2]
    mask = np.ones(first.shape[:2], dtype=bool) if same_size else np.zeros(first.shape[:2], dtype=bool)
    report = RegistrationReport(
        decision=RegistrationDecision(
            version=version,
            status=status,
            model="identity",
            reason_codes=reason_codes,
            used_for_comparison=used_for_comparison,
        ),
        metrics=RegistrationMetrics(
            match_count=0,
            inlier_count=0,
            inlier_ratio=1.0 if same_size else 0.0,
            overlap_ratio=1.0 if same_size else 0.0,
        ),
        transform_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        source_size_t1=size_t1,
        source_size_t2=size_t2,
        output_size=size_t1,
        diagnostics={"matcher": "none"},
    )
    return RegisteredPair(first.copy(), second.copy(), mask, report)


def _accepted_pair(
    first: Any,
    second: Any,
    candidate: _Candidate,
    *,
    size_t1: list[int],
    size_t2: list[int],
    diagnostics: dict[str, object],
    version: str,
    cv2: Any,
    np: Any,
) -> RegisteredPair:
    width, height = size_t1
    warped = cv2.warpPerspective(
        second, np.asarray(candidate.matrix, dtype=np.float64), (width, height), flags=cv2.INTER_LINEAR
    )
    source_mask = np.ones(second.shape[:2], dtype=np.uint8)
    valid_mask = cv2.warpPerspective(
        source_mask, np.asarray(candidate.matrix, dtype=np.float64), (width, height), flags=cv2.INTER_NEAREST
    ).astype(bool)
    report = RegistrationReport(
        decision=RegistrationDecision(
            version=version,
            status="applied",
            model=candidate.model,  # type: ignore[arg-type]
            reason_codes=["REGISTRATION_APPLIED"],
            used_for_comparison=True,
        ),
        metrics=candidate.metrics,
        transform_matrix=np.asarray(candidate.matrix, dtype=np.float64).tolist(),
        source_size_t1=size_t1,
        source_size_t2=size_t2,
        output_size=size_t1,
        diagnostics=diagnostics,
    )
    return RegisteredPair(first.copy(), np.ascontiguousarray(warped), valid_mask, report)


def _failure_pair(
    first: Any,
    second: Any,
    *,
    size_t1: list[int],
    size_t2: list[int],
    reason_code: str,
    match_count: int,
    settings: ChangeRegistrationSettings,
    np: Any,
    metrics: RegistrationMetrics | None = None,
    diagnostics: dict[str, object] | None = None,
) -> RegisteredPair:
    if settings.quality_policy == "fail":
        raise RegistrationError(reason_code)
    report = RegistrationReport(
        decision=RegistrationDecision(
            version=settings.version,
            status="rejected" if reason_code != "REGISTRATION_FAILED_EXCEPTION" else "failed",
            model="none",
            reason_codes=[reason_code, "RAW_FALLBACK_USED"],
            used_for_comparison=False,
        ),
        metrics=metrics or RegistrationMetrics(match_count=match_count),
        transform_matrix=None,
        source_size_t1=size_t1,
        source_size_t2=size_t2,
        output_size=size_t1,
        diagnostics=diagnostics or {},
    )
    return RegisteredPair(
        first.copy(),
        second.copy(),
        np.zeros(first.shape[:2], dtype=bool),
        report,
    )


def _has_strong_alignment_metadata(metadata: Mapping[str, Any] | None) -> bool:
    if not metadata:
        return False
    return bool(
        metadata.get("geometry_aligned")
        or metadata.get("registration_id")
        or metadata.get("registration_status") == "aligned"
    )


__all__ = ["GeometricRegistration", "RegisteredPair", "RegistrationError", "register_pair"]
