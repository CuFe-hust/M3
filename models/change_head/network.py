"""Production multi-expert Siamese ChangeHead network."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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

    class ChangeHeadNetworkError(ValueError):
        """Stable construction/forward contract error for the learned head."""

        def __init__(self, code: str, message: str = "") -> None:
            self.code = code
            super().__init__(f"{code}: {message}".strip())


    class MultiExpertSiameseChangeHead(nn.Module):
        """Compact multi-scale, multi-expert Siamese change decoder.

        Backbones remain frozen and outside this module.  Each T1/T2 stage
        shares one projection, then the temporal representation is decoded by
        expert and stage.  The forward method returns raw logits only.
        """

        def __init__(self, manifest: ChangeHeadManifest) -> None:
            super().__init__()
            self.manifest = manifest
            architecture = manifest.architecture
            if architecture.use_rgb_pair:
                raise ChangeHeadNetworkError(
                    "LEARNED_CHANGE_UNSUPPORTED_RGB_INPUT",
                    "RGB pair is not implemented by this V1 head",
                )
            self.hidden_dim = architecture.hidden_dim
            self.semantic_dim = architecture.semantic_dim
            self.decoder_dim = architecture.decoder_dim
            self.use_pif_mask = bool(architecture.use_pif_mask)

            self.projections = nn.ModuleDict()
            self.projection_norms = nn.ModuleDict()
            self.temporal_fusions = nn.ModuleDict()
            self.stage_decoders = nn.ModuleDict()
            self.semantic_projections = nn.ModuleDict()
            self.semantic_decoders = nn.ModuleDict()
            self.expert_gates = nn.ModuleDict()
            for expert in manifest.experts:
                expert_key = self._expert_key(expert.expert_id)
                self.expert_gates[expert_key] = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(self.decoder_dim, 1, kernel_size=1),
                )
                if expert.use_semantic_probabilities:
                    # Aggregate class probabilities into a fixed three-channel
                    # temporal input so the ABI need not guess class counts.
                    self.semantic_projections[expert_key] = nn.Conv2d(
                        3, self.semantic_dim, kernel_size=1
                    )
                    self.semantic_decoders[expert_key] = nn.Conv2d(
                        self.semantic_dim, self.decoder_dim, kernel_size=1
                    )
                for stage in expert.feature_stages:
                    key = self._stage_key(expert.expert_id, stage)
                    channels = expert.feature_channels_by_stage[stage]
                    self.projections[key] = nn.Conv2d(
                        channels, self.hidden_dim, kernel_size=1, bias=True
                    )
                    self.projection_norms[key] = nn.GroupNorm(1, self.hidden_dim)
                    self.temporal_fusions[key] = nn.Sequential(
                        nn.Conv2d(5 * self.hidden_dim, self.hidden_dim, 3, padding=1),
                        nn.GroupNorm(1, self.hidden_dim),
                        nn.GELU(),
                        nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1),
                        nn.GroupNorm(1, self.hidden_dim),
                        nn.GELU(),
                    )
                    self.stage_decoders[key] = nn.Conv2d(
                        self.hidden_dim, self.decoder_dim, kernel_size=1
                    )
            self.pif_fusion = (
                nn.Conv2d(1, self.decoder_dim, kernel_size=1)
                if self.use_pif_mask
                else None
            )
            self.decoder = nn.Sequential(
                nn.Conv2d(self.decoder_dim, self.decoder_dim, 3, padding=1),
                nn.GroupNorm(1, self.decoder_dim),
                nn.GELU(),
                nn.Conv2d(self.decoder_dim, 1, kernel_size=1),
            )
            self.logit_bias = nn.Parameter(torch.zeros(1))

        @staticmethod
        def _expert_key(expert_id: str) -> str:
            return expert_id.replace(".", "__dot__").replace("-", "__dash__")

        @classmethod
        def _stage_key(cls, expert_id: str, stage: int) -> str:
            return f"{cls._expert_key(expert_id)}__{stage}"

        @staticmethod
        def _as_bchw(value: Tensor, *, name: str) -> Tensor:
            if value.ndim == 3:
                value = value.unsqueeze(0)
            if value.ndim != 4:
                raise ChangeHeadNetworkError(
                    "LEARNED_CHANGE_FEATURE_STAGE_MISMATCH",
                    f"{name} must be BCHW",
                )
            return value

        @staticmethod
        def _target_size(valid_mask: Tensor | None, expert_features: Mapping[str, Any]) -> tuple[int, int]:
            if valid_mask is not None:
                return tuple(int(value) for value in valid_mask.shape[-2:])
            for first, _ in expert_features.values():
                if first:
                    return tuple(int(value) for value in first[0].shape[-2:])
            raise ChangeHeadNetworkError("LEARNED_CHANGE_OUTPUT_INVALID", "missing target size")

        def _project(self, key: str, value: Tensor) -> Tensor:
            return F.gelu(self.projection_norms[key](self.projections[key](value)))

        def _semantic_context(
            self,
            expert_id: str,
            first: Tensor,
            second: Tensor,
            target_size: tuple[int, int],
        ) -> Tensor:
            first = self._as_bchw(first, name=f"{expert_id}:semantic_t1")
            second = self._as_bchw(second, name=f"{expert_id}:semantic_t2")
            if first.shape != second.shape or first.shape[1] < 1:
                raise ChangeHeadNetworkError("LEARNED_CHANGE_FEATURE_STAGE_MISMATCH", "semantic pair")
            first_mean = first.mean(dim=1, keepdim=True)
            second_mean = second.mean(dim=1, keepdim=True)
            delta = (second - first).abs().mean(dim=1, keepdim=True)
            context = torch.cat((first_mean, second_mean, delta), dim=1)
            context = self.semantic_projections[self._expert_key(expert_id)](context)
            context = self.semantic_decoders[self._expert_key(expert_id)](context)
            return F.interpolate(context, size=target_size, mode="bilinear", align_corners=False)

        def forward(
            self,
            *,
            expert_features: Mapping[str, tuple[Sequence[Tensor], Sequence[Tensor]]],
            semantic_probabilities: Mapping[str, tuple[Tensor, Tensor]] | None = None,
            expert_presence: Mapping[str, Tensor] | None = None,
            valid_mask: Tensor | None = None,
            pif_mask: Tensor | None = None,
            rgb_t1: Tensor | None = None,
            rgb_t2: Tensor | None = None,
        ) -> Tensor:
            del rgb_t1, rgb_t2
            semantic_probabilities = semantic_probabilities or {}
            expert_presence = expert_presence or {}
            target_size = self._target_size(valid_mask, expert_features)
            available: list[tuple[str, Tensor]] = []
            batch_size: int | None = None
            for expert in self.manifest.experts:
                expert_id = expert.expert_id
                features = expert_features.get(expert_id)
                presence = expert_presence.get(expert_id)
                present = features is not None
                if presence is not None:
                    present = present and bool(torch.any(presence > 0.5).item())
                if not present:
                    if expert.required:
                        raise ChangeHeadNetworkError(
                            "LEARNED_CHANGE_REQUIRED_EXPERT_MISSING", expert_id
                        )
                    continue
                assert features is not None
                first_features, second_features = features
                if len(first_features) != len(expert.feature_stages) or len(second_features) != len(expert.feature_stages):
                    raise ChangeHeadNetworkError("LEARNED_CHANGE_FEATURE_STAGE_MISMATCH", expert_id)
                expert_context: Tensor | None = None
                for index, stage in enumerate(expert.feature_stages):
                    first = self._as_bchw(first_features[index], name=f"{expert_id}:{stage}:t1")
                    second = self._as_bchw(second_features[index], name=f"{expert_id}:{stage}:t2")
                    if first.shape != second.shape:
                        raise ChangeHeadNetworkError("LEARNED_CHANGE_FEATURE_STAGE_MISMATCH", expert_id)
                    expected_channels = expert.feature_channels_by_stage[stage]
                    if int(first.shape[1]) != expected_channels:
                        raise ChangeHeadNetworkError("LEARNED_CHANGE_FEATURE_STAGE_MISMATCH", expert_id)
                    if batch_size is None:
                        batch_size = int(first.shape[0])
                    if int(first.shape[0]) != batch_size:
                        raise ChangeHeadNetworkError("LEARNED_CHANGE_FEATURE_STAGE_MISMATCH", "batch")
                    z1 = self._project(self._stage_key(expert_id, stage), first)
                    z2 = self._project(self._stage_key(expert_id, stage), second)
                    temporal = torch.cat((z1, z2, (z2 - z1).abs(), z1 * z2, 0.5 * (z1 + z2)), dim=1)
                    stage_context = self.temporal_fusions[self._stage_key(expert_id, stage)](temporal)
                    stage_context = self.stage_decoders[self._stage_key(expert_id, stage)](stage_context)
                    stage_context = F.interpolate(stage_context, size=target_size, mode="bilinear", align_corners=False)
                    expert_context = stage_context if expert_context is None else expert_context + stage_context
                assert expert_context is not None
                expert_context = expert_context / float(len(expert.feature_stages))
                if expert.use_semantic_probabilities:
                    semantics = semantic_probabilities.get(expert_id)
                    if semantics is None:
                        raise ChangeHeadNetworkError("LEARNED_CHANGE_SEMANTIC_INPUT_MISSING", expert_id)
                    expert_context = expert_context + self._semantic_context(
                        expert_id, semantics[0], semantics[1], target_size
                    )
                available.append((expert_id, expert_context))
            if not available:
                raise ChangeHeadNetworkError("LEARNED_CHANGE_REQUIRED_EXPERT_MISSING")
            gate_logits = torch.cat(
                [self.expert_gates[self._expert_key(expert_id)](context) for expert_id, context in available],
                dim=1,
            )
            weights = torch.softmax(gate_logits, dim=1)
            fused = sum(
                context * weights[:, index:index + 1]
                for index, (_, context) in enumerate(available)
            )
            if self.use_pif_mask:
                if pif_mask is None:
                    raise ChangeHeadNetworkError("LEARNED_CHANGE_CONTRACT_MISMATCH", "missing pif_mask")
                pif = self._as_bchw(pif_mask, name="pif_mask")
                if pif.shape[1] != 1 or tuple(pif.shape[-2:]) != target_size:
                    raise ChangeHeadNetworkError("LEARNED_CHANGE_OUTPUT_INVALID", "pif_mask shape")
                fused = fused + self.pif_fusion(pif.to(dtype=fused.dtype))  # type: ignore[operator]
            return self.decoder(fused) + self.logit_bias

else:

    class ChangeHeadNetworkError(ValueError):  # type: ignore[no-redef]
        pass


    class MultiExpertSiameseChangeHead:  # type: ignore[no-redef]
        def __init__(self, manifest: ChangeHeadManifest) -> None:
            del manifest
            raise RuntimeError("LEARNED_CHANGE_INFERENCE_FAILED: torch dependency missing")


__all__ = ["ChangeHeadNetworkError", "MultiExpertSiameseChangeHead"]
