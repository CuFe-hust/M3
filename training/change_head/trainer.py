"""Minimal deterministic trainer for cached ChangeHead features."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable

from training.change_head.losses import change_head_loss


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 30
    learning_rate: float = 0.0003
    weight_decay: float = 0.0001
    grad_clip_norm: float = 1.0
    seed: int = 42


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


class ChangeHeadTrainer:
    def __init__(self, network: Any, *, config: TrainingConfig = TrainingConfig()) -> None:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("torch dependency missing") from error
        seed_everything(config.seed)
        self.network = network
        self.config = config
        self.optimizer = torch.optim.AdamW(
            network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )

    def train_epoch(self, batches: Iterable[dict[str, Any]]) -> float:
        total = 0.0
        count = 0
        self.network.train()
        for batch in batches:
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.network(**batch["network_inputs"])
            loss_valid_mask = batch.get("loss_valid_mask", batch.get("valid_mask"))
            if loss_valid_mask is None:
                raise ValueError("training batch requires loss_valid_mask")
            loss = change_head_loss(
                logits,
                batch["target_mask"],
                loss_valid_mask,
                pos_weight=float(batch.get("pos_weight", 1.0)),
            )
            loss.backward()
            import torch
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.grad_clip_norm)
            self.optimizer.step()
            total += float(loss.detach().cpu())
            count += 1
        return total / max(1, count)
