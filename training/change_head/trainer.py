"""Configurable trainer that consumes only frozen ChangeHead feature caches."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from training.change_head.losses import change_head_loss, swap_consistency_loss


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 4
    learning_rate: float = 0.0003
    weight_decay: float = 0.0001
    grad_clip_norm: float = 1.0
    amp: bool = True
    seed: int = 42
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    boundary_weight: float = 0.25
    swap_consistency_weight: float = 0.10
    swap_consistency_every_n_steps: int = 1
    max_pos_weight: float = 8.0
    optional_expert_dropout: float = 0.0
    sampling_default_weight: float = 1.0
    tag_multipliers: Mapping[str, float] = field(default_factory=dict)
    early_stopping_patience: int = 6
    early_stopping_metric: str = "val_pixel_f1"
    early_stopping_mode: str = "max"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def weighted_sample_indices(
    samples: Sequence[Any],
    *,
    default_weight: float = 1.0,
    tag_multipliers: Mapping[str, float],
    seed: int,
) -> list[int]:
    """Build a deterministic weighted order while retaining every class."""

    if not samples:
        return []
    rng = random.Random(seed)
    if default_weight <= 0.0:
        raise ValueError("default_weight must be positive")
    weights: list[float] = []
    for sample in samples:
        tags = getattr(sample, "tags", ())
        weight = float(default_weight)
        for tag in tags:
            weight *= max(0.0, float(tag_multipliers.get(tag, 1.0)))
        weights.append(max(weight, 1e-6))
    count = len(samples)
    order = list(range(count))
    for index, weight in enumerate(weights):
        repeats = max(1, int(round(weight)))
        order.extend([index] * (repeats - 1))
    rng.shuffle(order)
    # Keep no-change and ordinary changed examples present even if a caller
    # configured an aggressive hard-case multiplier.
    for index in range(count):
        if index not in order:
            order.append(index)
    return order


def _swap_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    swapped = dict(inputs)
    swapped["expert_features"] = {
        expert_id: (pair[1], pair[0])
        for expert_id, pair in inputs.get("expert_features", {}).items()
    }
    swapped["semantic_probabilities"] = {
        expert_id: (pair[1], pair[0])
        for expert_id, pair in inputs.get("semantic_probabilities", {}).items()
    }
    if inputs.get("rgb_t1") is not None or inputs.get("rgb_t2") is not None:
        swapped["rgb_t1"], swapped["rgb_t2"] = inputs.get("rgb_t2"), inputs.get("rgb_t1")
    return swapped


class ChangeHeadTrainer:
    def __init__(self, network: Any, *, config: TrainingConfig = TrainingConfig()) -> None:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("torch dependency missing") from error
        if config.swap_consistency_every_n_steps < 1:
            raise ValueError("swap_consistency_every_n_steps must be positive")
        if not 0.0 <= config.optional_expert_dropout < 1.0:
            raise ValueError("optional_expert_dropout must be in [0,1)")
        seed_everything(config.seed)
        self.network = network
        self.config = config
        self.optimizer = torch.optim.AdamW(
            network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self._step = 0
        scaler_enabled = bool(config.amp and torch.cuda.is_available())
        try:
            self._scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        except (AttributeError, TypeError):  # pragma: no cover - older torch
            self._scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    def _drop_optional_experts(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        probability = self.config.optional_expert_dropout
        if probability <= 0.0:
            return dict(inputs)
        manifest = getattr(self.network, "manifest", None)
        if manifest is None:
            return dict(inputs)
        dropped = dict(inputs)
        for expert in manifest.experts:
            if expert.required or random.random() >= probability:
                continue
            for field_name in ("expert_features", "semantic_probabilities", "expert_presence"):
                mapping = dict(dropped.get(field_name, {}))
                mapping.pop(expert.expert_id, None)
                dropped[field_name] = mapping
        return dropped

    @staticmethod
    def _as_batched_mask(value: Any) -> Any:
        """Convert single-sample [1,H,W] masks to training [B,1,H,W]."""

        if value.ndim == 3:
            return value.unsqueeze(0)
        if value.ndim != 4:
            raise ValueError("training masks must have shape [1,H,W] or [B,1,H,W]")
        return value

    @staticmethod
    def _pos_weight(target: Any, valid: Any, maximum: float) -> float:
        valid_mask = valid.bool()
        while valid_mask.ndim < target.ndim:
            valid_mask = valid_mask.unsqueeze(1)
        valid_mask = valid_mask.expand_as(target)
        positive = float((target.bool() & valid_mask).sum().detach().cpu())
        negative = float(((~target.bool()) & valid_mask).sum().detach().cpu())
        if positive <= 0.0:
            return 1.0
        return max(1.0, min(float(maximum), negative / positive))

    def train_epoch(self, batches: Iterable[dict[str, Any]]) -> float:
        import torch

        total = 0.0
        count = 0
        self.network.train()
        for batch in batches:
            self.optimizer.zero_grad(set_to_none=True)
            inputs = self._drop_optional_experts(batch["network_inputs"])
            loss_valid_mask = batch.get("loss_valid_mask", batch.get("valid_mask"))
            if loss_valid_mask is None:
                raise ValueError("training batch requires loss_valid_mask")
            loss_valid_mask = self._as_batched_mask(loss_valid_mask)
            target = self._as_batched_mask(batch["target_mask"])
            do_swap = (
                self.config.swap_consistency_weight > 0.0
                and self._step % self.config.swap_consistency_every_n_steps == 0
            )
            autocast_enabled = bool(self.config.amp and torch.cuda.is_available())
            with torch.autocast(device_type="cuda", enabled=autocast_enabled):
                logits = self.network(**inputs)
                swap_loss = None
                if do_swap:
                    swapped_logits = self.network(**_swap_inputs(inputs))
                    swap_loss = swap_consistency_loss(logits, swapped_logits, loss_valid_mask)
                loss = change_head_loss(
                    logits,
                    target,
                    loss_valid_mask,
                    bce_weight=self.config.bce_weight,
                    dice_weight=self.config.dice_weight,
                    boundary_weight=self.config.boundary_weight,
                    pos_weight=self._pos_weight(
                        target, loss_valid_mask, self.config.max_pos_weight
                    ),
                    swap_loss=swap_loss,
                    swap_weight=self.config.swap_consistency_weight,
                )
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("non-finite ChangeHead training loss")
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.grad_clip_norm)
            self._scaler.step(self.optimizer)
            self._scaler.update()
            total += float(loss.detach().cpu())
            count += 1
            self._step += 1
        return total / max(1, count)
