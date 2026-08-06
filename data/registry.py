"""Explicit dataset adapter registry; no module scanning, no entry points.

显式数据集适配器注册表；不扫描模块、不使用 entry point 自动发现。
支持规范名、别名、重复检测、列举与未知数据集错误；注册项为延迟 builder，
本阶段不得返回任何旧 Adapter 代理。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from data.adapters.base import DatasetAdapter, DatasetProbeError

AdapterBuilder = Callable[[], DatasetAdapter]


class DatasetRegistry:
    """Explicit registry mapping canonical names (and aliases) to builders.
    将规范名（及别名）显式映射到 builder 的注册表。"""

    def __init__(self) -> None:
        self._builders: dict[str, tuple[str, AdapterBuilder]] = {}
        self._aliases: dict[str, str] = {}

    @staticmethod
    def _normalize(name: str) -> str:
        """Canonical key: trimmed and case-folded. / 规范键：去空白并折叠大小写。"""
        return name.strip().casefold()

    def register(
        self,
        name: str,
        builder: AdapterBuilder,
        *,
        aliases: Iterable[str] = (),
    ) -> None:
        """Register a builder under a canonical name with optional aliases.
        Duplicate canonical names or aliases are rejected.
        以规范名注册 builder 并可带别名；重复的规范名或别名被拒绝。"""
        canonical = self._normalize(name)
        if canonical in self._builders or canonical in self._aliases:
            raise DatasetProbeError(f"duplicate adapter registration: {name!r}")
        alias_keys = [self._normalize(alias) for alias in aliases]
        for key in alias_keys:
            if key in self._builders or key in self._aliases:
                raise DatasetProbeError(f"duplicate adapter alias: {key!r}")
        self._builders[canonical] = (name, builder)
        for key in alias_keys:
            self._aliases[key] = canonical

    def get(self, name: str) -> DatasetAdapter:
        """Resolve a canonical name or alias and build the adapter instance.
        解析规范名或别名并构建适配器实例。"""
        key = self._normalize(name)
        entry = self._builders.get(key)
        if entry is None:
            mapped = self._aliases.get(key)
            entry = self._builders.get(mapped) if mapped is not None else None
        if entry is None:
            raise DatasetProbeError(
                f"Unsupported dataset {name!r}; supported={sorted(self.names())}"
            )
        return entry[1]()

    def names(self) -> tuple[str, ...]:
        """Sorted registered display names. / 排序后的注册显示名列表。"""
        return tuple(sorted(display for display, _ in self._builders.values()))


REGISTRY = DatasetRegistry()


def register_default_adapters(registry: DatasetRegistry = REGISTRY) -> None:
    """Explicitly register the audited built-in adapters (call once at bootstrap).
    显式注册经审计的内建适配器（在 bootstrap 阶段调用一次）。
    延迟 import 避免模块加载副作用；不扫描模块、不使用 entry point。"""
    from data.adapters.levir_cc import LEVIRCCAdapter
    from data.adapters.mme_realworld import MMERealWorldAdapter
    from data.adapters.vrsbench.adapter import VRSBenchAdapter
    from data.adapters.xlrs import XLRSAdapter

    registry.register("VRSBench", lambda: VRSBenchAdapter())
    registry.register("LEVIR-CC", lambda: LEVIRCCAdapter(), aliases=("LEVIR",))
    registry.register("MME-RealWorld", lambda: MMERealWorldAdapter(), aliases=("MME",))
    registry.register("XLRS-Bench", lambda: XLRSAdapter(), aliases=("XLRS",))
    registry.register(
        "XLRS-Bench-lite",
        lambda: XLRSAdapter(name="XLRS-Bench-lite"),
        aliases=("XLRS-lite",),
    )
