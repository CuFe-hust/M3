from __future__ import annotations

import inspect

import pytest

from training.multimodal_sft.contracts import AdapterContractError, DataProfile, MultimodalModelAdapter
from training.multimodal_sft.data import JsonlDataProfile, profile_for
from training.multimodal_sft.profiles.change_agent import ChangeAgentDataProfile
from training.multimodal_sft.profiles.phase2 import Phase2DataProfile
from training.multimodal_sft.registry import AdapterRegistry, UnsupportedModelAdapter, default_registry


def test_builtin_adapters_satisfy_runtime_protocol() -> None:
    registry = default_registry()
    for name in ("qwen3_5", "qwen3_vl", "hf_generic_multimodal"):
        assert isinstance(registry.get(name), MultimodalModelAdapter)


def test_builtin_adapter_critical_signatures_are_explicit() -> None:
    adapter = default_registry().get("qwen3_5")

    def parameters(method: str) -> set[str]:
        return set(inspect.signature(getattr(adapter, method)).parameters)

    assert {"processor_dir", "local_files_only"} <= parameters("load_processor")
    assert {"processor", "processor_dir"} <= parameters("saved_processor_identity")
    assert {"processor", "episode", "max_seq_length", "return_tensors"} <= parameters("encode")
    assert {"output_dir", "local_files_only"} <= parameters("reload_exported")
    assert {"model", "processor", "change_fixture"} <= parameters("verify_export_forward")
    assert "change_fixture" in parameters("export_checkpoint")


def test_data_profiles_and_profile_for_satisfy_runtime_protocol() -> None:
    profiles = (
        JsonlDataProfile("fixture"),
        ChangeAgentDataProfile(),
        Phase2DataProfile(),
    )
    for profile in profiles:
        assert isinstance(profile, DataProfile)
    assert isinstance(profile_for("phase2"), DataProfile)
    assert isinstance(profile_for("change_agent"), DataProfile)


def test_generic_adapter_is_complete_but_fail_closed_for_unproven_capabilities() -> None:
    adapter = default_registry().get("hf_generic_multimodal")
    with pytest.raises(AdapterContractError, match="proven export reload contract"):
        adapter.reload_exported("unused")
    with pytest.raises(AdapterContractError, match="proven forward verification contract"):
        adapter.verify_export_forward(None, None)


def test_registry_rejects_incomplete_adapter_at_get_time() -> None:
    class IncompleteAdapter:
        name = "incomplete"

    registry = AdapterRegistry(include_builtins=False)
    registry.register("incomplete", IncompleteAdapter)
    with pytest.raises(UnsupportedModelAdapter, match="ADAPTER_CONTRACT_INCOMPLETE"):
        registry.get("incomplete")
