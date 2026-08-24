from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
torch = pytest.importorskip("torch")

from training.multimodal_sft.checkpoint import checkpoint_complete
from training.multimodal_sft.contracts import CanonicalEpisode, ModelStructure
from training.multimodal_sft.data import JsonlDataProfile
from training.multimodal_sft.parameter_plan import ParameterPlan
from training.multimodal_sft.trainer_core import GenericTrainerCore, TrainingConfig


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language = torch.nn.Linear(1, 1, bias=False)
        self.connector = torch.nn.Linear(1, 1, bias=False)

    def forward(self, x):
        value = self.language(x) + self.connector(x)
        return SimpleNamespace(loss=(value + torch.rand_like(value) * 0.01 - 1.0).pow(2).mean())


class _TinyAdapter:
    name = "tiny"

    def discover_structure(self, model):
        return ModelStructure("language", "vision", {"language_lora_targets": ("language",), "vision_connectors": ("connector",)})

    def apply_tuning_policy(self, model, parameter_plan, policy):
        for parameter in model.parameters():
            parameter.requires_grad = False
        for name, parameter in model.named_parameters():
            if name.startswith("language.") or name.startswith("connector."):
                parameter.requires_grad = True
        return model

    def validate_trainable_parameters(self, model, parameter_plan):
        assert any(parameter.requires_grad for parameter in model.parameters())

    def encode(self, processor, episode, *, return_tensors="pt"):
        return {"x": torch.tensor([[float(episode.metadata["x"])]])}

    def prepare_forward_inputs(self, batch):
        return batch

    def save_checkpoint(self, model, processor, output_dir):
        adapter_dir = Path(output_dir) / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        torch.save({name: value.detach().clone() for name, value in model.named_parameters() if name.startswith("language.")}, adapter_dir / "adapter_model.safetensors")
        (adapter_dir / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        (Path(output_dir) / "processor").mkdir(exist_ok=True)

    def save_trainable_state(self, model, output_path, parameter_plan):
        state = {name: value.detach().clone() for name, value in model.named_parameters() if name in parameter_plan.full_train_parameter_names}
        torch.save(state, output_path)

    def restore_trainable_state(self, *, model, checkpoint_dir, parameter_plan, manifest):
        root = Path(checkpoint_dir)
        adapter_state = torch.load(root / "adapter" / "adapter_model.safetensors", weights_only=False)
        full_state = torch.load(root / "model_trainable_state.safetensors", weights_only=False)
        expected_adapter = {name for name, _ in model.named_parameters() if name.startswith("language.")}
        assert set(adapter_state) == expected_adapter
        assert set(full_state) == set(parameter_plan.full_train_parameter_names)
        with torch.no_grad():
            for name, value in {**adapter_state, **full_state}.items():
                dict(model.named_parameters())[name].copy_(value)
        return model

    def validate_checkpoint_state(self, checkpoint_dir, parameter_plan):
        root = Path(checkpoint_dir)
        adapter_state = torch.load(root / "adapter" / "adapter_model.safetensors", weights_only=False)
        full_state = torch.load(root / "model_trainable_state.safetensors", weights_only=False)
        assert all("language." in name for name in adapter_state)
        assert set(full_state) == set(parameter_plan.full_train_parameter_names)
        assert not set(adapter_state) & set(full_state)
        return {"overlap_count": 0}


def _episodes(count: int) -> list[CanonicalEpisode]:
    return [CanonicalEpisode("phase2", ({"role": "user", "content": "x"},), metadata={"x": index + 1}) for index in range(count)]


def _run(tmp_path: Path, *, count: int, max_steps: int | None = None, resume_from: Path | None = None, save_steps: int = 0, save_total_limit: int | None = None, stop_after_checkpoint: int | None = None):
    model = _TinyModel()
    torch.manual_seed(123)
    for parameter in model.parameters():
        torch.nn.init.uniform_(parameter, -0.2, 0.2)
    trainer = GenericTrainerCore(adapter=_TinyAdapter(), data_profile=JsonlDataProfile("phase2"))
    result = trainer.fit(
        model=model,
        processor=object(),
        episodes=_episodes(count),
        config=TrainingConfig(output_dir=tmp_path, epochs=1, max_steps=max_steps, resume_from=resume_from, save_steps=save_steps, save_total_limit=save_total_limit, lora_lr=1e-2, connector_lr=1e-2, max_grad_norm=100.0, _test_stop_after_checkpoint_step=stop_after_checkpoint),
        policy="lora_plus_projector",
        model_identity={"model_type": "tiny", "base_config_sha256": "cfg"},
    )
    return model, result


def test_exact_resume_restores_lora_connector_and_position(tmp_path: Path) -> None:
    continuous, continuous_result = _run(tmp_path / "continuous", count=6, max_steps=3)
    interrupted_dir = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="TEST_STOP_AFTER_CHECKPOINT"):
        _run(interrupted_dir, count=6, max_steps=3, save_steps=2, stop_after_checkpoint=2)
    checkpoint = interrupted_dir / "checkpoint-2"
    assert checkpoint_complete(checkpoint)
    resumed, resumed_result = _run(interrupted_dir, count=6, max_steps=3, resume_from=checkpoint)
    for left, right in zip(continuous.parameters(), resumed.parameters()):
        assert torch.equal(left, right)
    assert continuous_result.steps == resumed_result.steps == 3
    state = json.loads((interrupted_dir / "trainer_state.json").read_text(encoding="utf-8"))
    assert state["global_step"] == 3
    assert state["next_micro_batch_index"] == 3


def test_checkpoint_serialization_does_not_change_rng_trajectory(tmp_path: Path) -> None:
    without_checkpoints, _ = _run(tmp_path / "without", count=6, max_steps=3, save_steps=0)
    with_checkpoints, _ = _run(tmp_path / "with", count=6, max_steps=3, save_steps=1)
    for left, right in zip(without_checkpoints.parameters(), with_checkpoints.parameters()):
        assert torch.equal(left, right)


def test_periodic_checkpoint_is_atomic_complete_and_rotated(tmp_path: Path) -> None:
    _run(tmp_path, count=6, max_steps=3, save_steps=1, save_total_limit=2)
    checkpoints = sorted(path.name for path in tmp_path.iterdir() if path.is_dir() and path.name.startswith("checkpoint-"))
    assert checkpoints == ["checkpoint-2", "checkpoint-3"]
    assert not list(tmp_path.glob(".checkpoint-*.tmp-*"))
    assert checkpoint_complete(tmp_path / "checkpoint-3")


def test_batching_is_fail_closed(tmp_path: Path) -> None:
    trainer = GenericTrainerCore(adapter=_TinyAdapter(), data_profile=JsonlDataProfile("phase2"))
    with pytest.raises(ValueError, match="GENERIC_BATCHING_NOT_YET_AVAILABLE"):
        trainer.fit(model=_TinyModel(), processor=object(), episodes=_episodes(1), config=TrainingConfig(output_dir=tmp_path, batch_size=2), policy="lora_plus_projector")
