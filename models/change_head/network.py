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
        """Small ChangeHead network with explicit mask semantics.

        ``valid_mask`` only establishes the output canvas here.  It is not
        applied to logits: validity is a post-sigmoid runtime concern and
        training losses mask invalid pixels explicitly.  A PIF mask is an
        optional context signal, never a change-validity gate.
        """

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
            self.use_pif_mask = bool(manifest.architecture.use_pif_mask)
            self.pif_fusion = (
                nn.Conv2d(1, 1, kernel_size=1)
                if self.use_pif_mask
                else None
            )
            self.logit_bias = nn.Parameter(torch.zeros(1))

        def forward(
            self,
            *,
            expert_features: Mapping[str, tuple[Tensor, Tensor]],
            semantic_probabilities: Mapping[str, tuple[Tensor, Tensor]],
            expert_presence: Mapping[str, Tensor],
            valid_mask: Tensor | None,
            pif_mask: Tensor | None,
            rgb_t1: Tensor | None = None,
            rgb_t2: Tensor | None = None,
        ) -> Tensor:
            target_size: tuple[int, int] | None = None
            if valid_mask is not None:
                target_size = tuple(valid_mask.shape[-2:])
            if target_size is None:
                for features in expert_features.values():
                    if features[0]:
                        target_size = tuple(features[0][0].shape[-2:])
                        break
            if target_size is None:
                for semantics in semantic_probabilities.values():
                    target_size = tuple(semantics[0].shape[-2:])
                    break
            if target_size is None:
                raise ValueError("ChangeHead requires a target spatial size")
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
            if self.use_pif_mask and pif_mask is not None:
                if pif_mask.ndim == 2:
                    pif_mask = pif_mask.unsqueeze(0).unsqueeze(0)
                elif pif_mask.ndim == 3:
                    pif_mask = pif_mask.unsqueeze(1)
                if pif_mask.ndim != 4 or pif_mask.shape[1] != 1:
                    raise ValueError("pif_mask must be [B,H,W] or [B,1,H,W]")
                if tuple(pif_mask.shape[-2:]) != target_size:
                    raise ValueError("pif_mask spatial shape must match target")
                assert self.pif_fusion is not None
                # PIF contributes context; it is deliberately not multiplied
                # into logits or interpreted as a change-validity mask.
                pieces.append(self.pif_fusion(pif_mask.to(dtype=torch.float32)))
            if not pieces:
                batch_size = 1
                if valid_mask is not None:
                    batch_size = int(valid_mask.shape[0]) if valid_mask.ndim >= 3 else 1
                logits = self.logit_bias.expand(batch_size, 1, *target_size)
            else:
                logits = torch.stack(pieces, dim=0).mean(dim=0) + self.logit_bias
            return logits

else:

    class MultiExpertSiameseChangeHead:  # type: ignore[no-redef]
        def __init__(self, manifest: ChangeHeadManifest) -> None:
            del manifest
            raise RuntimeError("LEARNED_CHANGE_INFERENCE_FAILED: torch dependency missing")
