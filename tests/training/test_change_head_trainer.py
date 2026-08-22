from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from scripts.train_change_head import _load_yaml  # noqa: E402
from training.change_head.losses import change_head_loss  # noqa: E402
from training.change_head.trainer import (  # noqa: E402
    ChangeHeadTrainer,
    TrainingConfig,
    weighted_sample_indices,
)


class TinyNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.1))
        self.calls = 0

    def forward(self, *, valid_mask: torch.Tensor, **kwargs: object) -> torch.Tensor:
        del kwargs
        self.calls += 1
        return self.scale.expand_as(valid_mask)


def _batch() -> dict[str, object]:
    valid = torch.ones(1, 1, 3, 3)
    return {
        "network_inputs": {
            "expert_features": {},
            "semantic_probabilities": {},
            "expert_presence": {},
            "valid_mask": valid,
            "pif_mask": None,
        },
        "target_mask": torch.zeros(1, 1, 3, 3),
        "loss_valid_mask": valid,
    }


def test_swap_weight_zero_skips_second_forward() -> None:
    network = TinyNetwork()
    trainer = ChangeHeadTrainer(
        network,
        config=TrainingConfig(
            epochs=1,
            amp=False,
            swap_consistency_weight=0.0,
        ),
    )
    trainer.train_epoch([_batch()])
    assert network.calls == 1


def test_swap_weight_nonzero_runs_swapped_forward() -> None:
    network = TinyNetwork()
    trainer = ChangeHeadTrainer(
        network,
        config=TrainingConfig(
            epochs=1,
            amp=False,
            swap_consistency_weight=0.5,
        ),
    )
    trainer.train_epoch([_batch()])
    assert network.calls == 2


def test_all_declared_loss_weights_affect_total_loss() -> None:
    logits = torch.tensor([[[[-1.0, 2.0], [0.5, -0.5]]]], requires_grad=True)
    target = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    valid = torch.ones_like(target)
    values = []
    for field in ("bce_weight", "dice_weight", "boundary_weight"):
        kwargs = {"bce_weight": 1.0, "dice_weight": 1.0, "boundary_weight": 0.25}
        kwargs[field] *= 2.0
        values.append(float(change_head_loss(logits, target, valid, **kwargs)))
    assert len(set(values)) == 3


def test_optional_expert_dropout_never_drops_required() -> None:
    network = TinyNetwork()
    network.manifest = SimpleNamespace(
        experts=(
            SimpleNamespace(expert_id="required", required=True),
            SimpleNamespace(expert_id="optional", required=False),
        )
    )
    trainer = ChangeHeadTrainer(
        network,
        config=TrainingConfig(amp=False, optional_expert_dropout=0.99),
    )
    inputs = {
        "expert_features": {"required": 1, "optional": 2},
        "semantic_probabilities": {"required": 1, "optional": 2},
        "expert_presence": {"required": 1, "optional": 2},
    }
    dropped = trainer._drop_optional_experts(inputs)
    assert "required" in dropped["expert_features"]


def test_nochange_samples_remain_in_sampler() -> None:
    samples = [
        SimpleNamespace(tags=("no_change",)),
        SimpleNamespace(tags=("hard_case",)),
        SimpleNamespace(tags=("normal_changed",)),
    ]
    order = weighted_sample_indices(
        samples,
        tag_multipliers={"hard_case": 5.0, "no_change": 0.1},
        seed=42,
    )
    assert 0 in order and 1 in order and 2 in order


def test_unknown_training_config_fails(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("optimization:\n  epochs: 1\n  typo: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown optimization"):
        _load_yaml(path)


def test_one_tiny_epoch_reduces_loss() -> None:
    network = TinyNetwork()
    trainer = ChangeHeadTrainer(
        network,
        config=TrainingConfig(
            epochs=1,
            amp=False,
            learning_rate=0.1,
            dice_weight=0.0,
            boundary_weight=0.0,
            swap_consistency_weight=0.0,
        ),
    )
    batch = _batch()
    before = float(change_head_loss(network(**batch["network_inputs"]), batch["target_mask"], batch["loss_valid_mask"], dice_weight=0.0, boundary_weight=0.0))
    for _ in range(4):
        trainer.train_epoch([batch])
    after = float(change_head_loss(network(**batch["network_inputs"]), batch["target_mask"], batch["loss_valid_mask"], dice_weight=0.0, boundary_weight=0.0))
    assert after < before
