"""Minimal ChangeHead network; all torch imports remain optional/lazy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from models.change_head.manifest import ChangeHeadManifest

try:  # pragma: no cover - exercised when the optional extra is installed
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - default lightweight environment
    torch = None
    Tensor = Any
    nn = None
    F = None


if nn is not None:

    class MultiExpertSiameseChangeHead(nn.Module):
        """Small stable ABI network, replaceable internally by Phase F."""

        def __init__(self, manifest: ChangeHeadManifest) -> None:
            super().__init__()
            self.manifest = manifest
            self.projections = nn.ModuleDict({
                f"{expert.expert_id}__{stage}": nn.Conv2d(
                    expert.feature_channels_by_stage[stage], 1, kernel_size=1
                )
                for expert in manifest.experts
                for stage in expert.feature_stages
            })
            self.logit_bias = nn.Parameter(torch.zeros(1))

        def forward(
            self,
            *,
            expert_features: Mapping[str, tuple[Tensor, Tensor]],
            semantic_probabilities: Mapping[str, tuple[Tensor, Tensor]],
            expert_presence: Mapping[str, Tensor],
            valid_mask: Tensor,
            pif_mask: Tensor | None,
            rgb_t1: Tensor | None = None,
            rgb_t2: Tensor | None = None,
        ) -> Tensor:
            target_size = tuple(valid_mask.shape[-2:])
            pieces: list[Tensor] = []
            for expert in self.manifest.experts:
                presence = expert_presence[expert.expert_id].reshape(-1, 1, 1, 1)
                features = expert_features.get(expert.expert_id)
                if features is not None:
                    first, second = features
                    for index, stage in enumerate(expert.feature_stages):
                        diff = (first[index] - second[index]).abs()
                        diff = F.interpolate(diff, size=target_size, mode="bilinear", align_corners=False)
                        pieces.append(self.projections[f"{expert.expert_id}__{stage}"](diff) * presence)
                semantics = semantic_probabilities.get(expert.expert_id)
                if semantics is not None and expert.use_semantic_probabilities:
                    first, second = semantics
                    diff = (first - second).abs().mean(dim=1, keepdim=True)
                    pieces.append(F.interpolate(diff, size=target_size, mode="bilinear", align_corners=False) * presence)
            if not pieces:
                logits = self.logit_bias.expand(valid_mask.shape[0], 1, *target_size)
            else:
                logits = torch.stack(pieces, dim=0).mean(dim=0) + self.logit_bias
            if pif_mask is not None:
                logits = logits * pif_mask.to(dtype=logits.dtype).reshape(logits.shape[0], 1, *target_size)
            return logits * valid_mask.to(dtype=logits.dtype).reshape(logits.shape[0], 1, *target_size)

else:

    class MultiExpertSiameseChangeHead:  # type: ignore[no-redef]
        def __init__(self, manifest: ChangeHeadManifest) -> None:
            del manifest
            raise RuntimeError("LEARNED_CHANGE_INFERENCE_FAILED: torch dependency missing")
