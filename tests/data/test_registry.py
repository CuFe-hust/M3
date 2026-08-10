"""Contract tests for the explicit dataset adapter registry.

显式适配器注册表测试：规范名、别名、重复检测、列举、未知数据集错误、
延迟 builder。注册表不扫描模块、不使用 entry point，不返回任何旧代理。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from data.adapters.base import DatasetProbeError
from data.registry import REGISTRY, DatasetRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]


class _Fake:
    """Minimal adapter stand-in for registration tests. / 注册测试的最小适配器替身。"""

    name = "fake"
    supported_tasks = {"general_vqa"}

    def probe(self, root: Path):
        return None

    def iter_samples(self, root: Path, split: str, task: str):
        return iter(())


def _fresh_registry() -> DatasetRegistry:
    return DatasetRegistry()


def test_register_and_get_by_canonical_name() -> None:
    registry = _fresh_registry()
    registry.register("VRSBench", lambda: _Fake())
    adapter = registry.get("VRSBench")
    assert adapter.name == "fake"
    assert registry.names() == ("VRSBench",)


def test_get_is_case_insensitive() -> None:
    registry = _fresh_registry()
    registry.register("LEVIR-CC", lambda: _Fake())
    assert registry.get("levir-cc").name == "fake"
    assert registry.get("  LEVIR-CC  ").name == "fake"


def test_aliases_resolve_to_canonical_name() -> None:
    registry = _fresh_registry()
    registry.register("VRSBench", lambda: _Fake(), aliases=("vrsbench", "VRSB"))
    assert registry.get("vrsbench").name == "fake"
    assert registry.get("VRSB").name == "fake"
    assert registry.names() == ("VRSBench",)


def test_duplicate_canonical_name_is_rejected() -> None:
    registry = _fresh_registry()
    registry.register("VRSBench", lambda: _Fake())
    with pytest.raises(DatasetProbeError, match="duplicate"):
        registry.register("VRSBench", lambda: _Fake())


def test_duplicate_alias_is_rejected() -> None:
    registry = _fresh_registry()
    registry.register("VRSBench", lambda: _Fake(), aliases=("vrs",))
    with pytest.raises(DatasetProbeError, match="duplicate"):
        registry.register("VRS", lambda: _Fake())


def test_unknown_dataset_raises_with_supported_list() -> None:
    registry = _fresh_registry()
    registry.register("VRSBench", lambda: _Fake())
    with pytest.raises(DatasetProbeError, match="Unsupported dataset"):
        registry.get("no-such-dataset")
    try:
        registry.get("no-such-dataset")
    except DatasetProbeError as error:
        assert "VRSBench" in str(error)


def test_lazy_builder_is_not_called_until_get() -> None:
    registry = _fresh_registry()
    calls: list[str] = []

    def build():
        calls.append("built")
        return _Fake()

    registry.register("lazy", build)
    assert calls == []
    registry.get("lazy")
    assert calls == ["built"]
    registry.get("lazy")
    assert calls == ["built", "built"]


def test_instance_can_be_registered_via_lambda() -> None:
    registry = _fresh_registry()
    instance = _Fake()
    registry.register("instance-dataset", lambda: instance)
    assert registry.get("instance-dataset") is instance


def test_registry_imports_no_legacy_packages() -> None:
    source = (REPO_ROOT / "data" / "registry.py").read_text(encoding="utf-8")
    assert "spacers_agent" not in source and "eval" not in source


def test_registry_does_not_scan_modules() -> None:
    source = (REPO_ROOT / "data" / "registry.py").read_text(encoding="utf-8")
    for marker in ("importlib", "pkgutil", "entry_points", "iter_modules", "walk_packages"):
        assert marker not in source, f"registry must not auto-discover via {marker}"


def test_registry_module_level_instance_is_empty_by_default() -> None:
    assert isinstance(REGISTRY, DatasetRegistry)
    assert REGISTRY.names() == ()


def test_registry_file_has_no_import_time_file_access() -> None:
    source = (REPO_ROOT / "data" / "registry.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        assert not isinstance(node, (ast.With, ast.Try)), (
            f"registry top-level {type(node).__name__} must not run at import"
        )
