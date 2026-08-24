from __future__ import annotations

from training.multimodal_sft.registry import default_registry


def test_registry_resolves_auto_without_loading_weights(monkeypatch) -> None:
    registry = default_registry()
    assert set(registry.available()) == {"qwen3_vl", "qwen3_5", "hf_generic_multimodal"}
    assert registry.resolve
