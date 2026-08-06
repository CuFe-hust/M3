"""Factory for the four fixed benchmark dataset adapters."""

from __future__ import annotations

from collections.abc import Mapping

from m3rs_eval.config import EvaluationConfig
from m3rs_eval.registry import MetricRegistry

from .base import CheckResult, DatasetAdapter, DatasetError, DatasetMaterialization
from .levir_cc import LevirCCAdapter
from .mme_rs import MMERSAdapter
from .vrsbench import VRSBenchAdapter
from .xlrs_bench import XLRSBenchAdapter


def create_adapters(
    config: EvaluationConfig, protocol: Mapping[str, object], registry: MetricRegistry
) -> list[DatasetAdapter]:
    """Create one adapter for each protocol-locked benchmark without aliases."""
    return [
        LevirCCAdapter(config.datasets["levir_cc"], protocol, registry),
        VRSBenchAdapter(config.datasets["vrsbench"], protocol, registry),
        XLRSBenchAdapter(config.datasets["xlrs_bench"], protocol, registry),
        MMERSAdapter(config.datasets["mme_rs"], protocol, registry),
    ]


__all__ = [
    "CheckResult",
    "DatasetAdapter",
    "DatasetError",
    "DatasetMaterialization",
    "create_adapters",
]
