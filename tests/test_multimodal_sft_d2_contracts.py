from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from training.multimodal_sft.contracts import CanonicalEpisode
from training.multimodal_sft.data import JsonlDataProfile
from training.multimodal_sft.trainer_core import GenericTrainerCore, TrainingConfig


class _BatchModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language = torch.nn.Linear(1, 1, bias=False)
        self.connector = torch.nn.Linear(1, 1, bias=False)

    def forward(self, x):
        value = self.language(x) + self.connector(x)
        return SimpleNamespace(loss=(value - 1.0).pow(2).mean())


class _BatchAdapter:
    name = "batch_fixture"

    def discover_structure(self, model):
        from training.multimodal_sft.contracts import ModelStructure

        return ModelStructure("language", "vision", {"language_lora_targets": ("language",), "vision_connectors": ("connector",)})

    def apply_tuning_policy(self, model, parameter_plan, policy):
        for parameter in model.parameters():
            parameter.requires_grad = True
        return model

    def validate_trainable_parameters(self, model, parameter_plan):
        assert any(parameter.requires_grad for parameter in model.parameters())

    def encode(self, processor, episode, *, max_seq_length=4096, return_tensors="pt"):
        assert max_seq_length == processor["max_seq_length"]
        return {"x": torch.tensor([[float(episode.metadata["x"])]])}

    def collate(self, encoded_examples):
        return {"x": torch.cat([item["x"] for item in encoded_examples], dim=0)}, [{} for _ in encoded_examples]

    def processor_identity(self, processor):
        return {"class": "batch_fixture", "chat_template_sha256": "fixture", "encoding_contract_version": "fixture_v1"}

    def load_processor(self, processor_dir, *, local_files_only=True):
        return {"max_seq_length": 321}

    def saved_processor_identity(self, processor, processor_dir):
        return {**self.processor_identity(processor), "content_sha256": "batch-fixture-content", "files": []}

    def prepare_forward_inputs(self, batch):
        return batch

    def save_checkpoint(self, model, processor, output_dir):
        root = Path(output_dir)
        (root / "adapter").mkdir(parents=True, exist_ok=True)
        torch.save({}, root / "adapter" / "adapter_model.safetensors")
        (root / "adapter" / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        (root / "processor").mkdir(exist_ok=True)

    def save_trainable_state(self, model, output_path, parameter_plan):
        torch.save({}, output_path)

    def validate_checkpoint_state(self, checkpoint_dir, parameter_plan):
        return {}


def _episodes(count: int, *, weighted: bool = False):
    return [
        CanonicalEpisode(
            "phase2",
            ({"role": "user", "content": "x"},),
            metadata={"x": index + 1, **({"sample_weight": 2.0} if weighted and index == 0 else {})},
        )
        for index in range(count)
    ]


@pytest.mark.parametrize(("batch_size", "expected_steps"), ((1, 3), (2, 2), (4, 1)))
def test_generic_core_uses_adapter_collate_for_batch_sizes(tmp_path, batch_size, expected_steps):
    adapter = _BatchAdapter()
    trainer = GenericTrainerCore(adapter=adapter, data_profile=JsonlDataProfile("phase2"))
    result = trainer.fit(
        model=_BatchModel(),
        processor={"max_seq_length": 321},
        episodes=_episodes(5),
        config=TrainingConfig(output_dir=tmp_path, batch_size=batch_size, gradient_accumulation_steps=2, max_seq_length=321),
        policy="lora_plus_projector",
        model_identity={"model_type": "batch_fixture"},
    )
    assert result.steps == expected_steps


def test_sample_weight_batching_fails_closed(tmp_path):
    trainer = GenericTrainerCore(adapter=_BatchAdapter(), data_profile=JsonlDataProfile("phase2"))
    with pytest.raises(ValueError, match="SAMPLE_WEIGHT_BATCHING_UNSUPPORTED"):
        trainer.fit(
            model=_BatchModel(),
            processor={"max_seq_length": 321},
            episodes=_episodes(2, weighted=True),
            config=TrainingConfig(output_dir=tmp_path, batch_size=2, max_seq_length=321),
            policy="lora_plus_projector",
            model_identity={"model_type": "batch_fixture"},
        )
