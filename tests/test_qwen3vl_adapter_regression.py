from __future__ import annotations

import hashlib

from training.multimodal_sft.adapters.qwen3_vl import Qwen3VLAdapter
from training.multimodal_sft.parameter_plan import build_parameter_plan


class _Parameter:
    def __init__(self) -> None:
        self.requires_grad = True


class _Module:
    def __init__(self) -> None:
        self.children = {}
        self.parameters_ = {}

    def add(self, name, child) -> None:
        self.children[name] = child
        setattr(self, name, child)

    def named_modules(self, prefix=""):
        yield prefix, self
        for name, child in self.children.items():
            child_prefix = f"{prefix}.{name}" if prefix else name
            yield from child.named_modules(child_prefix)

    def named_parameters(self, prefix=""):
        for name, parameter in self.parameters_.items():
            yield (f"{prefix}.{name}" if prefix else name), parameter
        for name, child in self.children.items():
            child_prefix = f"{prefix}.{name}" if prefix else name
            yield from child.named_parameters(child_prefix)

    def parameters(self):
        return (parameter for _, parameter in self.named_parameters())


def _linear() -> _Module:
    module = _Module()
    module.parameters_["weight"] = _Parameter()
    return module


def _fixture() -> _Module:
    root = _Module()
    model = _Module()
    language = _Module()
    layers = _Module()
    layer = _Module()
    attention = _Module()
    mlp = _Module()
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        attention.add(name, _linear())
    for name in ("gate_proj", "up_proj", "down_proj"):
        mlp.add(name, _linear())
    layer.add("self_attn", attention)
    layer.add("mlp", mlp)
    layers.add("0", layer)
    language.add("layers", layers)
    language.add("embed_tokens", _linear())
    visual = _Module()
    visual.add("projector", _linear())
    model.add("language_model", language)
    model.add("visual", visual)
    root.add("model", model)
    return root


def test_qwen3vl_target_and_connector_regression_snapshot() -> None:
    plan = build_parameter_plan(_fixture(), Qwen3VLAdapter(), "lora_plus_projector")
    target_hash = hashlib.sha256("\n".join(plan.lora_module_paths).encode()).hexdigest()
    connector_hash = hashlib.sha256("\n".join(plan.full_train_module_paths).encode()).hexdigest()
    assert target_hash == "3a0a3147640bcc1454f1fb06c5e9f7b6a5c98f34c5a0f26d30124dd9d4477bc1"
    assert connector_hash == "62c2a6ee658c57fcc5098700c8b3922448480a9935204e20958d0d8f29aa0027"
