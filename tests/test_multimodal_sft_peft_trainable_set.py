from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
PretrainedConfig = pytest.importorskip("transformers").PretrainedConfig

from training.multimodal_sft.adapters import _hf
from training.multimodal_sft.contracts import CanonicalEpisode, ModelStructure
from training.multimodal_sft.data import JsonlDataProfile
from training.multimodal_sft.identity import processor_content_identity, processor_semantic_identity
from training.multimodal_sft.optimizer import OptimizerConfig, build_optimizer_groups
from training.multimodal_sft.parameter_plan import ParameterPlan
from training.multimodal_sft.trainer_core import GenericTrainerCore, TrainingConfig


class _TinyCausalLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = PretrainedConfig(model_type="tiny", is_encoder_decoder=False)
        self.language = torch.nn.Module()
        self.language.layer = torch.nn.Linear(2, 2, bias=True)
        self.connector = torch.nn.Linear(2, 2, bias=True)
        self.unrelated = torch.nn.Linear(2, 2, bias=True)

    def get_input_embeddings(self):
        return self.language.layer

    def get_output_embeddings(self):
        return self.language.layer

    def prepare_inputs_for_generation(self, input_ids=None, **kwargs):
        return {"input_ids": input_ids, **kwargs}

    def forward(self, x=None, input_ids=None, **kwargs):
        value = x if x is not None else input_ids
        value = self.language.layer(value) + self.connector(value) + self.unrelated(value)
        return SimpleNamespace(loss=(value - 1.0).pow(2).mean())


class _Processor:
    def save_pretrained(self, output_dir: str | Path) -> None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "processor_config.json").write_text("{}\n", encoding="utf-8")


def _plan() -> ParameterPlan:
    return ParameterPlan(
        adapter_name="tiny_peft",
        policy="lora_plus_projector",
        language_backbone="language",
        vision_backbone="vision",
        lora_module_paths=("language.layer",),
        full_train_module_paths=("connector",),
        full_train_parameter_names=("connector.weight", "connector.bias"),
    )


def _apply(model: torch.nn.Module):
    return _hf.apply_tuning_policy(model, _plan(), SimpleNamespace(rank=2, alpha=4, dropout=0.0))


def _actual(model: torch.nn.Module, canonical: str) -> torch.nn.Parameter:
    for name, parameter in model.named_parameters():
        if _hf.canonicalize_model_parameter_name(name) == canonical:
            return parameter
    raise AssertionError(f"missing parameter {canonical}")


class _TinyPeftAdapter:
    name = "tiny_peft"

    def discover_structure(self, model):
        return ModelStructure("language", "vision", {"language_lora_targets": ("language.layer",), "vision_connectors": ("connector",)})

    def apply_tuning_policy(self, model, parameter_plan, policy):
        self.last_model = _apply(model)
        return self.last_model

    def validate_trainable_parameters(self, model, parameter_plan):
        return _hf.validate_trainable_parameters(model, parameter_plan)

    def processor_identity(self, processor):
        return {**processor_semantic_identity(processor), "encoding_contract_version": "tiny_peft_v1"}

    def load_processor(self, processor_dir, *, local_files_only=True):
        return _Processor()

    def saved_processor_identity(self, processor, processor_dir):
        return {**processor_content_identity(processor_dir, processor), "encoding_contract_version": "tiny_peft_v1"}

    def validate_optimizer_parameters(self, model, parameter_plan, groups):
        return _hf.validate_optimizer_parameters(model, parameter_plan, groups)

    def encode(self, processor, episode, *, return_tensors="pt"):
        return {"x": torch.tensor([[float(episode.metadata["x"]), 1.0]])}

    def prepare_forward_inputs(self, batch):
        return batch

    def save_checkpoint(self, model, processor, output_dir):
        return _hf.save_checkpoint(model, processor, output_dir)

    def save_trainable_state(self, model, output_path, parameter_plan):
        return _hf.save_trainable_state(model, output_path, parameter_plan)

    def restore_trainable_state(self, *, model, checkpoint_dir, parameter_plan, manifest):
        return _hf.restore_trainable_state(model=model, checkpoint_dir=checkpoint_dir, parameter_plan=parameter_plan, manifest=manifest)

    def validate_checkpoint_state(self, checkpoint_dir, parameter_plan):
        return _hf.validate_checkpoint_state(checkpoint_dir, parameter_plan)

    def validate_checkpoint_ownership(self, model, checkpoint_dir, parameter_plan):
        return _hf.validate_checkpoint_ownership(model, checkpoint_dir, parameter_plan)


def _episodes(count: int):
    return [CanonicalEpisode("phase2", ({"role": "user", "content": "x"},), metadata={"x": index + 1}) for index in range(count)]


def _new_model() -> _TinyCausalLM:
    model = _TinyCausalLM()
    torch.manual_seed(123)
    for parameter in model.parameters():
        torch.nn.init.uniform_(parameter, -0.2, 0.2)
    return model


def _assert_nested_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
        return
    if isinstance(left, dict):
        assert isinstance(right, dict)
        assert list(left) == list(right)
        for key in left:
            _assert_nested_equal(left[key], right[key])
        return
    if isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
        return
    assert left == right


def test_real_peft_trainable_set_and_one_step_mutation() -> None:
    model = _apply(_new_model())
    plan = _plan()
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    lora = _hf.collect_lora_parameter_names(model)
    assert lora <= trainable
    assert any("lora_A" in name for name in lora)
    assert any("lora_B" in name for name in lora)
    assert _actual(model, "connector.weight").requires_grad
    assert _actual(model, "connector.bias").requires_grad
    assert not _actual(model, "language.layer.base_layer.weight").requires_grad
    assert not _actual(model, "language.layer.base_layer.bias").requires_grad
    assert not _actual(model, "unrelated.weight").requires_grad
    assert not _actual(model, "unrelated.bias").requires_grad
    _hf.validate_trainable_parameters(model, plan)
    groups, _stats = build_optimizer_groups(model, plan, OptimizerConfig(lora_lr=1e-2, connector_lr=1e-2))
    _hf.validate_optimizer_parameters(model, plan, groups)
    optimizer = torch.optim.AdamW(groups)
    base_before = _actual(model, "language.layer.base_layer.weight").detach().clone()
    lora_before = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if name in lora}
    connector_before = _actual(model, "connector.weight").detach().clone()
    optimizer.zero_grad(set_to_none=True)
    model(x=torch.ones((1, 2)))
    loss = model(x=torch.ones((1, 2))).loss
    loss.backward()
    optimizer.step()
    assert torch.equal(_actual(model, "language.layer.base_layer.weight"), base_before)
    assert any(not torch.equal(model.state_dict()[name], value) for name, value in lora_before.items())
    assert not torch.equal(_actual(model, "connector.weight"), connector_before)


def test_real_peft_continuous_resume_and_checkpoint_ownership(tmp_path: Path) -> None:
    def run(output_dir: Path, *, resume_from: Path | None = None, stop_after: int | None = None):
        adapter = _TinyPeftAdapter()
        trainer = GenericTrainerCore(adapter=adapter, data_profile=JsonlDataProfile("phase2"))
        model = _new_model()
        try:
            result = trainer.fit(
                model=model,
                processor=_Processor(),
                episodes=_episodes(6),
                config=TrainingConfig(output_dir=output_dir, epochs=1, max_steps=3, save_steps=2 if stop_after else 0, resume_from=resume_from, lora_lr=1e-2, connector_lr=1e-2, max_grad_norm=100.0, _test_stop_after_checkpoint_step=stop_after),
                policy="lora_plus_projector",
                model_identity={"model_type": "tiny_peft", "base_config_sha256": "cfg"},
            )
        except RuntimeError:
            raise
        return adapter, result

    continuous_adapter, continuous_result = run(tmp_path / "continuous")
    interrupted_dir = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="TEST_STOP_AFTER_CHECKPOINT"):
        run(interrupted_dir, stop_after=2)
    checkpoint = interrupted_dir / "checkpoint-2"
    resumed_adapter, resumed_result = run(interrupted_dir, resume_from=checkpoint)
    assert continuous_result.steps == resumed_result.steps == 3
    _hf.validate_trainable_parameters(continuous_adapter.last_model, _plan()) if hasattr(continuous_adapter, "last_model") else None
    for adapter in (continuous_adapter, resumed_adapter):
        assert not _actual(adapter.last_model, "language.layer.base_layer.weight").requires_grad
        assert not _actual(adapter.last_model, "language.layer.base_layer.bias").requires_grad
    for canonical in ("language.layer.base_layer.weight", "language.layer.base_layer.bias", "unrelated.weight", "unrelated.bias"):
        assert torch.equal(_actual(continuous_adapter.last_model, canonical), _actual(resumed_adapter.last_model, canonical))
    continuous_adapter_state = _hf._load_tensor_file(tmp_path / "continuous" / "adapter" / "adapter_model.safetensors")
    resumed_adapter_state = _hf._load_tensor_file(interrupted_dir / "adapter" / "adapter_model.safetensors")
    assert set(continuous_adapter_state) == set(resumed_adapter_state)
    for key in continuous_adapter_state:
        assert torch.equal(continuous_adapter_state[key], resumed_adapter_state[key])
    continuous_state = _hf._load_tensor_file(tmp_path / "continuous" / "model_trainable_state.safetensors")
    resumed_state = _hf._load_tensor_file(interrupted_dir / "model_trainable_state.safetensors")
    assert set(continuous_state) == set(resumed_state) == {"connector.weight", "connector.bias"}
    for key in continuous_state:
        assert torch.equal(continuous_state[key], resumed_state[key])
    _assert_nested_equal(
        torch.load(tmp_path / "continuous" / "optimizer.pt", weights_only=False),
        torch.load(interrupted_dir / "optimizer.pt", weights_only=False),
    )
    _assert_nested_equal(
        torch.load(tmp_path / "continuous" / "scheduler.pt", weights_only=False),
        torch.load(interrupted_dir / "scheduler.pt", weights_only=False),
    )
    assert json.loads((interrupted_dir / "trainer_state.json").read_text(encoding="utf-8"))["global_step"] == 3
    assert (tmp_path / "continuous" / "rng_state.pt").is_file()
