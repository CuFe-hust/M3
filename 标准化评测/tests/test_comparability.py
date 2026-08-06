"""Tests for explicit run comparability fingerprints."""

import pytest

from m3rs_eval.comparability import ComparabilityResult, compare_runs
from m3rs_eval.contracts import RunManifest


def _make_manifest(
    run_id: str = "r1",
    *,
    hardware: str = "NVIDIA-A100",
    xlrs_protocol: str | None = None,
    protocol_id: str = "test-protocol",
    request_manifest_hash: str = "req001",
    dataset_hashes: dict | None = None,
    extra_meta: dict | None = None,
) -> RunManifest:
    """Convenience builder for test RunManifest objects."""
    metadata: dict = {
        "system_version": f"system-{run_id}",
        "limit": None,
        "doctor": {"passed": True},
        "environment": {
            "gpus": [hardware],
            "platform": "linux",
            "cpu": "Intel-Xeon",
        },
        "dataset_manifest_hashes": dataset_hashes or {"ds1": "hash123"},
    }
    if xlrs_protocol is not None:
        metadata["xlrs_protocol"] = xlrs_protocol
    if extra_meta:
        metadata.update(extra_meta)

    return RunManifest(
        run_id=run_id,
        status="complete",
        mode="full",
        protocol_id=protocol_id,
        created_at="2025-06-01T00:00:00+08:00",
        config_hash="abc123",
        request_manifest_hash=request_manifest_hash,
        eligible_for_history=True,
        protocol_hash="proto123",
        command_hash="cmd123",
        system_version_hash="sys123",
        metadata=metadata,
    )


class TestQualityComparability:
    """Step 3: quality comparability checks."""

    def test_quality_can_compare_when_hardware_differs(self):
        result = compare_runs(
            _make_manifest("r1", hardware="NVIDIA-A100"),
            _make_manifest("r2", hardware="NVIDIA-H100"),
        )
        assert result.quality_comparable
        assert not result.resource_comparable

    def test_lite_and_full_are_not_quality_comparable(self):
        result = compare_runs(
            _make_manifest("r1", xlrs_protocol="full"),
            _make_manifest("r2", xlrs_protocol="lite"),
        )
        assert not result.quality_comparable
        quality_text = " ".join(result.quality_reasons)
        assert "xlrs_protocol" in quality_text

    def test_same_manifest_is_fully_comparable(self):
        m = _make_manifest("r1")
        result = compare_runs(m, m)
        assert result.quality_comparable
        assert result.resource_comparable

    def test_different_protocol_id_not_quality_comparable(self):
        result = compare_runs(
            _make_manifest("r1", protocol_id="proto-v1"),
            _make_manifest("r2", protocol_id="proto-v2"),
        )
        assert not result.quality_comparable

    def test_missing_xlrs_protocol_yields_reason(self):
        """When one run has xlrs_protocol and the other does not, quality
        is incomparable and the reason mentions the missing field."""
        result = compare_runs(
            _make_manifest("r1", xlrs_protocol="full"),
            _make_manifest("r2"),  # no xlrs_protocol
        )
        assert not result.quality_comparable
        quality_text = " ".join(result.quality_reasons)
        assert "missing field" in quality_text
        assert "xlrs_protocol" in quality_text


class TestResourceComparability:
    """Step 4: resource comparability checks."""

    def test_different_hardware_not_resource_comparable(self):
        result = compare_runs(
            _make_manifest("r1", hardware="NVIDIA-A100"),
            _make_manifest("r2", hardware="NVIDIA-H100"),
        )
        assert not result.resource_comparable
        resource_text = " ".join(result.resource_reasons)
        assert "hardware" in resource_text

    def test_different_platform_not_resource_comparable(self):
        r1 = _make_manifest("r1", extra_meta={
            "environment": {
                "gpus": ["NVIDIA-A100"],
                "platform": "linux",
                "cpu": "Intel-Xeon",
            }
        })
        r2 = _make_manifest("r2", extra_meta={
            "environment": {
                "gpus": ["NVIDIA-A100"],
                "platform": "windows",
                "cpu": "Intel-Xeon",
            }
        })
        result = compare_runs(r1, r2)
        assert not result.resource_comparable

    def test_same_hardware_different_software_is_resource_incomparable(self):
        r1 = _make_manifest("r1", extra_meta={
            "environment": {
                "gpus": ["NVIDIA-A100"],
                "platform": "linux",
                "cpu": "Intel-Xeon",
                "software_stack": "CUDA 12.1",
            }
        })
        r2 = _make_manifest("r2", extra_meta={
            "environment": {
                "gpus": ["NVIDIA-A100"],
                "platform": "linux",
                "cpu": "Intel-Xeon",
                "software_stack": "CUDA 12.4",
            }
        })
        result = compare_runs(r1, r2)
        assert not result.resource_comparable
