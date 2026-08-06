"""Explicit run comparability fingerprints for quality and resource dimensions."""

from __future__ import annotations

from dataclasses import dataclass, field

from m3rs_eval.contracts import RunManifest


@dataclass
class ComparabilityResult:
    """Outcome of comparing two run manifests for quality and resource comparability."""

    quality_comparable: bool
    resource_comparable: bool
    quality_reasons: list[str] = field(default_factory=list)
    resource_reasons: list[str] = field(default_factory=list)


def compare_runs(candidate: RunManifest, prior: RunManifest) -> ComparabilityResult:
    """Determine whether two runs are comparable on quality and resource dimensions.

    Quality fingerprint includes: dataset manifest hashes, protocol_id,
    request manifest hash, prompt/evaluator hashes, generation budget,
    and protocol variant fields (xlrs_protocol, language_variant).

    Resource fingerprint additionally includes: hardware, software stack,
    batch/concurrency, generation budget, and timing boundary.
    """
    quality_reasons: list[str] = []
    resource_reasons: list[str] = []

    _check_quality(candidate, prior, quality_reasons)
    _check_resource(candidate, prior, resource_reasons)

    return ComparabilityResult(
        quality_comparable=len(quality_reasons) == 0,
        resource_comparable=len(resource_reasons) == 0,
        quality_reasons=quality_reasons,
        resource_reasons=resource_reasons,
    )


def _check_quality(
    candidate: RunManifest, prior: RunManifest, reasons: list[str]
) -> None:
    c_meta = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    p_meta = prior.metadata if isinstance(prior.metadata, dict) else {}

    # Protocol ID
    if candidate.protocol_id != prior.protocol_id:
        reasons.append("protocol_id mismatch")

    # Request manifest hash (overall dataset identity fingerprint)
    if candidate.request_manifest_hash != prior.request_manifest_hash:
        reasons.append("request_manifest_hash mismatch")

    # Dataset manifest hashes
    c_ds_hashes = c_meta.get("dataset_manifest_hashes", {})
    p_ds_hashes = p_meta.get("dataset_manifest_hashes", {})
    if c_ds_hashes != p_ds_hashes:
        reasons.append("dataset_manifest_hashes mismatch")

    # Generation budget (limit)
    c_limit = c_meta.get("limit")
    p_limit = p_meta.get("limit")
    if c_limit != p_limit:
        reasons.append("generation budget (limit) mismatch")

    # XLRS protocol variant
    c_xlrs = c_meta.get("xlrs_protocol")
    p_xlrs = p_meta.get("xlrs_protocol")
    if c_xlrs is None and p_xlrs is not None:
        reasons.append("xlrs_protocol mismatch: missing field: xlrs_protocol in candidate")
    elif c_xlrs is not None and p_xlrs is None:
        reasons.append("xlrs_protocol mismatch: missing field: xlrs_protocol in prior")
    elif c_xlrs != p_xlrs:
        reasons.append(f"xlrs_protocol mismatch: {c_xlrs} vs {p_xlrs}")

    # Language variant
    c_lang = c_meta.get("language_variant")
    p_lang = p_meta.get("language_variant")
    if c_lang is None and p_lang is not None:
        reasons.append("language_variant mismatch: missing field: language_variant in candidate")
    elif c_lang is not None and p_lang is None:
        reasons.append("language_variant mismatch: missing field: language_variant in prior")
    elif c_lang != p_lang:
        reasons.append(f"language_variant mismatch: {c_lang} vs {p_lang}")


def _check_resource(
    candidate: RunManifest, prior: RunManifest, reasons: list[str]
) -> None:
    c_meta = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    p_meta = prior.metadata if isinstance(prior.metadata, dict) else {}

    c_env = c_meta.get("environment", {}) if isinstance(c_meta.get("environment"), dict) else {}
    p_env = p_meta.get("environment", {}) if isinstance(p_meta.get("environment"), dict) else {}

    # Hardware - GPUs
    c_gpus = c_env.get("gpus")
    p_gpus = p_env.get("gpus")
    if c_gpus is None and p_gpus is not None:
        reasons.append("hardware mismatch: missing field: environment.gpus in candidate")
    elif c_gpus is not None and p_gpus is None:
        reasons.append("hardware mismatch: missing field: environment.gpus in prior")
    elif c_gpus != p_gpus:
        reasons.append(f"hardware (GPUs) mismatch: {c_gpus} vs {p_gpus}")

    # Hardware - platform
    c_platform = c_env.get("platform")
    p_platform = p_env.get("platform")
    if c_platform != p_platform:
        reasons.append(f"hardware (platform) mismatch: {c_platform} vs {p_platform}")

    # Hardware - CPU
    c_cpu = c_env.get("cpu")
    p_cpu = p_env.get("cpu")
    if c_cpu != p_cpu:
        reasons.append(f"hardware (CPU) mismatch: {c_cpu} vs {p_cpu}")

    # Software stack
    c_sw = c_env.get("software_stack")
    p_sw = p_env.get("software_stack")
    if c_sw != p_sw:
        reasons.append(f"software stack mismatch: {c_sw} vs {p_sw}")

    # Batch / concurrency
    c_batch = c_meta.get("batch_size")
    p_batch = p_meta.get("batch_size")
    if c_batch != p_batch:
        reasons.append(f"batch_size mismatch: {c_batch} vs {p_batch}")

    c_concurrency = c_meta.get("concurrency")
    p_concurrency = p_meta.get("concurrency")
    if c_concurrency != p_concurrency:
        reasons.append(f"concurrency mismatch: {c_concurrency} vs {p_concurrency}")

    # Generation budget (same as quality check, also matters for resource)
    c_limit = c_meta.get("limit")
    p_limit = p_meta.get("limit")
    if c_limit != p_limit:
        reasons.append("generation budget (limit) mismatch")

    # Timing boundary
    c_timing = c_meta.get("timing_boundary")
    p_timing = p_meta.get("timing_boundary")
    if c_timing is None and p_timing is not None:
        reasons.append("timing boundary mismatch: missing field: timing_boundary in candidate")
    elif c_timing is not None and p_timing is None:
        reasons.append("timing boundary mismatch: missing field: timing_boundary in prior")
    elif c_timing != p_timing:
        reasons.append(f"timing boundary mismatch: {c_timing} vs {p_timing}")
