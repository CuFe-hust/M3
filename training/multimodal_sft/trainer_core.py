"""Generic optimizer/training loop with no model-family knowledge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .checkpoint import build_training_manifest, write_manifest
from .contracts import DataProfile, MultimodalModelAdapter
from .parameter_plan import ParameterPlan, TuningPolicy, build_parameter_plan


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: str | Path
    epochs: int = 1
    learning_rate: float = 1e-4
    max_steps: int | None = None
    gradient_accumulation_steps: int = 1


@dataclass(frozen=True)
class TrainingResult:
    steps: int
    loss: float | None
    manifest_path: Path | None
    parameter_plan: ParameterPlan


class GenericTrainerCore:
    """Small dependency-light loop; adapters own model input semantics."""

    def __init__(self, *, adapter: MultimodalModelAdapter, data_profile: DataProfile) -> None:
        self.adapter = adapter
        self.data_profile = data_profile

    def build_plan(self, model: Any, policy: TuningPolicy | str, *, probe: Any | None = None) -> ParameterPlan:
        return build_parameter_plan(model, self.adapter, policy, probe=probe)

    def fit(
        self,
        *,
        model: Any,
        processor: Any,
        episodes: Iterable[Any],
        config: TrainingConfig,
        policy: TuningPolicy | str,
        probe: Any | None = None,
        model_identity: Mapping[str, Any] | None = None,
    ) -> TrainingResult:
        """Run a minimal generic loop over already canonical episodes.

        LoRA injection and model-specific optimizer preparation are adapter
        responsibilities.  This core only consumes the resulting model,
        encoded batches and a semantic parameter plan.
        """

        import torch

        plan = self.build_plan(model, policy, probe=probe)
        apply_policy = getattr(self.adapter, "apply_tuning_policy", None)
        if callable(apply_policy):
            model = apply_policy(model, plan, policy if isinstance(policy, TuningPolicy) else TuningPolicy.from_name(policy))
        selected_paths = (*plan.lora_module_paths, *plan.full_train_module_paths)
        params = [
            param for name, param in model.named_parameters()
            if any(("." + path + ".") in ("." + name + ".") for path in selected_paths)
            and bool(getattr(param, "requires_grad", True))
        ]
        if not params:
            raise ValueError("parameter plan selected no concrete model parameters")
        optimizer = torch.optim.AdamW(params, lr=config.learning_rate)
        model.train()
        steps = 0
        last_loss: float | None = None
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(config.epochs):
            del epoch
            for episode in episodes:
                self.data_profile.validate(episode)
                batch = self.adapter.encode(processor, episode)
                prepared = self.adapter.prepare_forward_inputs(batch)
                output = model(**prepared)
                loss = output["loss"] if isinstance(output, Mapping) else getattr(output, "loss", None)
                if loss is None:
                    raise ValueError("model output does not expose loss")
                (loss / config.gradient_accumulation_steps).backward()
                if (steps + 1) % config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                last_loss = float(loss.detach().cpu().item())
                steps += 1
                if config.max_steps is not None and steps >= config.max_steps:
                    break
            if config.max_steps is not None and steps >= config.max_steps:
                break
        save_checkpoint = getattr(self.adapter, "save_checkpoint", None)
        if not callable(save_checkpoint):
            raise ValueError("selected adapter does not implement checkpoint saving")
        save_checkpoint(model, processor, config.output_dir)
        manifest = build_training_manifest(
            adapter_name=str(getattr(self.adapter, "name", type(self.adapter).__name__)),
            model_identity=dict(model_identity or {}),
            task_profile=self.data_profile.name,
            data_contract={"profile": self.data_profile.name},
            tuning_policy=(policy.as_dict() if hasattr(policy, "as_dict") else {"name": str(policy)}),
            parameter_plan=plan.as_dict(),
            training={"steps": steps, "last_loss": last_loss},
        )
        manifest_path = write_manifest(config.output_dir, manifest)
        return TrainingResult(steps, last_loss, manifest_path, plan)
