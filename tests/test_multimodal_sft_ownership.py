from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from training.multimodal_sft.adapters import _hf
from training.multimodal_sft.parameter_plan import ParameterPlan


def _plan() -> ParameterPlan:
    return ParameterPlan(
        adapter_name="fixture",
        policy="lora_plus_projector",
        language_backbone="language",
        vision_backbone="vision",
        lora_module_paths=("language.layer",),
        full_train_module_paths=("connector",),
        full_train_parameter_names=("connector.weight",),
    )


def test_canonicalize_rejects_non_lora_adapter_tensor() -> None:
    with pytest.raises(ValueError, match="UNEXPECTED_NON_LORA_ADAPTER_STATE"):
        _hf.canonicalize_peft_lora_state_keys({"connector.weight": object()})


def test_lora_and_full_state_are_disjoint(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "adapter").mkdir()
    (tmp_path / "adapter" / "adapter_model.safetensors").write_bytes(b"fixture")
    (tmp_path / "model_trainable_state.safetensors").write_bytes(b"fixture")
    files = {
        "adapter_model.safetensors": {"language.layer.lora_A.weight": object()},
        "model_trainable_state.safetensors": {"connector.weight": object()},
    }
    monkeypatch.setattr(_hf, "_load_tensor_file", lambda path: files[path.name])
    result = _hf.validate_checkpoint_state(tmp_path, _plan())
    assert result["overlap_count"] == 0


def test_lora_config_does_not_assign_full_train_modules(monkeypatch) -> None:
    captured = {}

    class _Config:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class _Task:
        CAUSAL_LM = "causal"

    class _Peft:
        LoraConfig = _Config
        TaskType = _Task

        @staticmethod
        def get_peft_model(model, config):
            return model

    monkeypatch.setitem(__import__("sys").modules, "peft", _Peft)

    class _Parameter:
        requires_grad = True

    class _Model:
        def named_parameters(self):
            return iter((("language.layer.weight", _Parameter()), ("connector.weight", _Parameter())))

    _hf.apply_tuning_policy(_Model(), _plan(), SimpleNamespace(rank=1, alpha=2, dropout=0.0))
    assert captured["modules_to_save"] is None
