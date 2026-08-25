"""Contract tests for the model-family-agnostic multimodal SFT seam."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.multimodal_sft.adapters.qwen3_5 import Qwen35Adapter
from training.multimodal_sft.checkpoint import (
    build_training_manifest,
    read_manifest,
    validate_resume_compatibility,
    write_manifest,
)
from training.multimodal_sft.data import JsonlDataProfile
from training.multimodal_sft.parameter_plan import build_parameter_plan
from training.multimodal_sft.registry import default_registry


class _FakeParameter:
    requires_grad = True


class _FakeModule:
    def __init__(self) -> None:
        self._children: dict[str, _FakeModule] = {}
        self._parameters: dict[str, _FakeParameter] = {}

    def add_module(self, name: str, child: "_FakeModule") -> None:
        self._children[name] = child
        setattr(self, name, child)

    def add_parameter(self, name: str) -> None:
        parameter = _FakeParameter()
        self._parameters[name] = parameter
        setattr(self, name, parameter)

    def named_children(self):
        return self._children.items()

    def named_modules(self, prefix: str = ""):
        yield prefix, self
        for name, child in self._children.items():
            child_prefix = f"{prefix}.{name}" if prefix else name
            yield from child.named_modules(child_prefix)

    def named_parameters(self, prefix: str = ""):
        for name, parameter in self._parameters.items():
            yield (f"{prefix}.{name}" if prefix else name), parameter
        for name, child in self._children.items():
            child_prefix = f"{prefix}.{name}" if prefix else name
            yield from child.named_parameters(child_prefix)

    def parameters(self):
        return (value for _, value in self.named_parameters())

    def __iter__(self):
        return iter(self._children.values())


class _FakeLinear(_FakeModule):
    def __init__(self) -> None:
        super().__init__()
        self.add_parameter("weight")


class _FakeLayer(_FakeModule):
    def __init__(self) -> None:
        super().__init__()
        attention = _FakeModule()
        for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
            attention.add_module(name, _FakeLinear())
        self.add_module("linear_attn", attention)
        mlp = _FakeModule()
        for name in ("gate_proj", "up_proj", "down_proj"):
            mlp.add_module(name, _FakeLinear())
        self.add_module("mlp", mlp)


class _FakeLanguage(_FakeModule):
    def __init__(self) -> None:
        super().__init__()
        self.add_module("embed_tokens", _FakeLinear())
        self.add_module("layers", _FakeModule())
        self.layers.add_module("0", _FakeLayer())


class _FakeVisual(_FakeModule):
    def __init__(self) -> None:
        super().__init__()
        self.add_module("q_proj", _FakeLinear())  # same name trap
        self.add_module("merger", _FakeLinear())


class _FakeModel(_FakeModule):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=1, layer_types=["linear_attention"]))
        self.add_module("model", _FakeModule())
        self.model.add_module("language_model", _FakeLanguage())
        self.model.add_module("visual", _FakeVisual())


def test_qwen35_adapter_discovers_semantic_roles_without_vision_name_sniffing() -> None:
    plan = build_parameter_plan(_FakeModel(), Qwen35Adapter(), "lora_plus_projector")
    assert plan.lora_module_paths
    assert plan.full_train_module_paths == ("model.visual.merger",)
    assert all("visual" not in path for path in plan.lora_module_paths)
    assert all("q_proj" not in path for path in plan.lora_module_paths)


def test_registry_exposes_explicit_builtin_families() -> None:
    assert set(default_registry().available()) == {"qwen3_vl", "qwen3_5", "hf_generic_multimodal"}


def test_manifest_resume_identity_is_model_and_task_agnostic(tmp_path: Path) -> None:
    manifest = build_training_manifest(
        adapter_name="qwen3_5",
        model_identity={"model_type": "qwen3_5", "fingerprint": "identity"},
        task_profile="change_agent",
        data_contract={"target_schema": "ChangeInitialResult"},
        tuning_policy={"name": "lora_only"},
        parameter_plan={"language_lora_targets": ["model.language_model.layers.0.self_attn.in_proj_qkv"]},
    )
    write_manifest(tmp_path, manifest)
    loaded = read_manifest(tmp_path)
    validate_resume_compatibility(
        loaded,
        adapter_name="qwen3_5",
        model_identity={"model_type": "qwen3_5", "fingerprint": "identity"},
        task_profile="change_agent",
        tuning_policy={"name": "lora_only"},
    )
    assert json.loads((tmp_path / "multimodal_sft_training_manifest.json").read_text()) == loaded


def test_jsonl_profile_keeps_task_and_ordered_messages(tmp_path: Path) -> None:
    source = tmp_path / "episodes.jsonl"
    source.write_text(json.dumps({
        "task_profile": "change_agent",
        "target_schema": "ChangeInitialResult",
        "messages": [{"role": "user", "content": "T1/T2"}],
        "images": ["t1.png", "t2.png"],
    }) + "\n", encoding="utf-8")
    episodes = list(JsonlDataProfile("change_agent", required_target_schema="ChangeInitialResult").read(source))
    assert episodes[0].images == ("t1.png", "t2.png")


def test_generic_trainer_core_has_no_model_family_branch() -> None:
    source = Path("training/multimodal_sft/trainer_core.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "qwen" not in source.lower()
    assert not any(isinstance(node, ast.ImportFrom) and node.module and "transformers" in node.module for node in ast.walk(tree))
