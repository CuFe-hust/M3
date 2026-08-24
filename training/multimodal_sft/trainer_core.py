"""Generic SFT training orchestration with adapter-owned model semantics."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .checkpoint import (
    OPTIMIZER_FILENAME,
    PARAMETER_PLAN_FILENAME,
    SCHEDULER_FILENAME,
    TRAINER_STATE_FILENAME,
    build_training_manifest,
    read_compatible_manifest,
    validate_resume_compatibility,
    write_manifest,
)
from .contracts import DataProfile, MultimodalModelAdapter
from .optimizer import (
    OptimizerConfig,
    autocast_context,
    build_cosine_scheduler,
    build_optimizer_groups,
    clip_gradients,
)
from .parameter_plan import ParameterPlan, TuningPolicy, build_parameter_plan


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: str | Path
    epochs: int = 1
    lora_lr: float = 1e-4
    connector_lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    max_steps: int | None = None
    gradient_accumulation_steps: int = 1
    seed: int = 1234
    mixed_precision: str = "off"
    preflight_only: bool = False
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    logging_steps: int = 10
    smoke_gradients: bool = False
    repeat_group_key: str | None = None
    repeat_weights: Mapping[str, int] = field(default_factory=dict)
    resume_from: str | Path | None = None
    data_contract: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingResult:
    steps: int
    loss: float | None
    eval_loss: float | None
    manifest_path: Path | None
    parameter_plan: ParameterPlan
    optimizer_stats: Mapping[str, Any] = field(default_factory=dict)


class GenericTrainerCore:
    """Task/profile-aware trainer that never inspects model-family paths."""

    def __init__(self, *, adapter: MultimodalModelAdapter, data_profile: DataProfile) -> None:
        self.adapter = adapter
        self.data_profile = data_profile

    def build_plan(self, model: Any, policy: TuningPolicy | str, *, probe: Any | None = None) -> ParameterPlan:
        return build_parameter_plan(model, self.adapter, policy, probe=probe)

    @staticmethod
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

    def preflight(self, episodes: Iterable[Any], *, limit: int | None = None) -> dict[str, Any]:
        checked = 0
        for episode in episodes:
            self.data_profile.validate(episode)
            checked += 1
            if limit is not None and checked >= limit:
                break
        return {"checked": checked, "profile": self.data_profile.name, "passed": True}

    @staticmethod
    def _materialize(episodes: Iterable[Any], limit: int | None) -> list[Any]:
        result: list[Any] = []
        for episode in episodes:
            result.append(episode)
            if limit is not None and len(result) >= limit:
                break
        return result

    @staticmethod
    def _loss(output: Any) -> Any:
        if isinstance(output, Mapping):
            return output.get("loss")
        return getattr(output, "loss", None)

    @staticmethod
    def _repeat_rows(rows: Sequence[Any], *, group_key: str | None, weights: Mapping[str, int]) -> list[Any]:
        if not group_key or not weights:
            return list(rows)
        expanded: list[Any] = []
        for episode in rows:
            metadata = getattr(episode, "metadata", {}) or {}
            group = str(metadata.get(group_key, ""))
            count = int(weights.get(group, 1))
            if count < 1:
                raise ValueError("repeat weights must be positive")
            expanded.extend([episode] * count)
        return expanded

    @staticmethod
    def _write_log(root: Path, event: Mapping[str, Any]) -> None:
        root.mkdir(parents=True, exist_ok=True)
        with (root / "training_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(event), ensure_ascii=False, default=str) + "\n")

    def smoke_gradients(self, *, model: Any, processor: Any, episode: Any) -> dict[str, Any]:
        """Run one adapter-mediated forward/backward pass without updating weights."""

        import torch

        self.data_profile.validate(episode)
        model.train()
        batch = self.adapter.encode(processor, episode)
        output = model(**self.adapter.prepare_forward_inputs(batch))
        loss = self._loss(output)
        if loss is None:
            raise ValueError("gradient smoke requires a model loss")
        loss.backward()
        trainable = [parameter for parameter in model.parameters() if bool(getattr(parameter, "requires_grad", False))]
        with_grad = sum(parameter.grad is not None for parameter in trainable)
        if trainable and with_grad != len(trainable):
            raise ValueError(f"gradient smoke found missing gradients: {with_grad}/{len(trainable)}")
        for parameter in trainable:
            parameter.grad = None
        return {"passed": True, "trainable_parameters": len(trainable), "parameters_with_grad": with_grad, "loss": float(loss.detach().cpu().item())}

    def evaluate(self, *, model: Any, processor: Any, episodes: Sequence[Any]) -> float | None:
        import torch

        losses: list[float] = []
        was_training = bool(getattr(model, "training", False))
        model.eval()
        with torch.no_grad():
            for episode in episodes:
                self.data_profile.validate(episode)
                batch = self.adapter.encode(processor, episode)
                output = model(**self.adapter.prepare_forward_inputs(batch))
                loss = self._loss(output)
                if loss is not None:
                    losses.append(float(loss.detach().cpu().item()))
        if was_training:
            model.train()
        return sum(losses) / len(losses) if losses else None

    @staticmethod
    def _save_runtime_state(root: Path, *, step: int, epoch: int, loss: float | None, eval_loss: float | None, optimizer: Any, scheduler: Any) -> None:
        import torch

        root.mkdir(parents=True, exist_ok=True)
        torch.save(optimizer.state_dict(), root / OPTIMIZER_FILENAME)
        torch.save(scheduler.state_dict(), root / SCHEDULER_FILENAME)
        (root / TRAINER_STATE_FILENAME).write_text(
            json.dumps({"step": step, "epoch": epoch, "loss": loss, "eval_loss": eval_loss}, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _load_runtime_state(root: Path, optimizer: Any, scheduler: Any) -> dict[str, Any]:
        import torch

        state_path = root / TRAINER_STATE_FILENAME
        if not state_path.is_file():
            raise ValueError("resume checkpoint is missing trainer_state.json")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        try:
            optimizer_state = torch.load(root / OPTIMIZER_FILENAME, map_location="cpu", weights_only=False)
            scheduler_state = torch.load(root / SCHEDULER_FILENAME, map_location="cpu", weights_only=False)
        except TypeError:  # torch versions before the weights_only keyword
            optimizer_state = torch.load(root / OPTIMIZER_FILENAME, map_location="cpu")
            scheduler_state = torch.load(root / SCHEDULER_FILENAME, map_location="cpu")
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)
        return state

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
        eval_episodes: Iterable[Any] | None = None,
    ) -> TrainingResult:
        import torch

        self.seed_everything(config.seed)
        train_rows = self._materialize(episodes, None)
        train_rows = self._repeat_rows(train_rows, group_key=config.repeat_group_key, weights=config.repeat_weights)
        if config.max_train_samples is not None:
            train_rows = train_rows[: config.max_train_samples]
        eval_rows = self._materialize(eval_episodes or (), config.max_eval_samples)
        preflight = self.preflight(train_rows)
        if config.preflight_only:
            plan = self.build_plan(model, policy, probe=probe)
            return TrainingResult(
                steps=0,
                loss=None,
                eval_loss=None,
                manifest_path=None,
                parameter_plan=plan,
                optimizer_stats={"preflight": preflight},
            )
        if not train_rows:
            raise ValueError("training profile produced no episodes")

        selected_policy = policy if isinstance(policy, TuningPolicy) else TuningPolicy.from_name(policy)
        plan = self.build_plan(model, selected_policy, probe=probe)
        apply_policy = getattr(self.adapter, "apply_tuning_policy", None)
        if not callable(apply_policy):
            raise ValueError("selected adapter does not implement parameter-plan application")
        model = apply_policy(model, plan, selected_policy)
        validate_trainable = getattr(self.adapter, "validate_trainable_parameters", None)
        if callable(validate_trainable):
            validate_trainable(model, plan)
        if config.smoke_gradients:
            self.smoke_gradients(model=model, processor=processor, episode=train_rows[0])
        optimizer_config = OptimizerConfig(
            lora_lr=config.lora_lr,
            connector_lr=config.connector_lr,
            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,
            max_grad_norm=config.max_grad_norm,
            mixed_precision=config.mixed_precision,
        )
        groups, optimizer_stats = build_optimizer_groups(model, plan, optimizer_config)
        optimizer = torch.optim.AdamW(groups)
        updates_per_epoch = max(1, (len(train_rows) + config.gradient_accumulation_steps - 1) // config.gradient_accumulation_steps)
        total_steps = config.max_steps or max(1, updates_per_epoch * config.epochs)
        scheduler = build_cosine_scheduler(optimizer, total_steps, config.warmup_ratio)
        start_step = 0
        start_epoch = 0
        if config.resume_from:
            resume_root = Path(config.resume_from)
            manifest = read_compatible_manifest(
                resume_root,
                legacy_manifest_names=("phase2_training_manifest.json",),
            )
            if "checkpoint_type" not in manifest:
                legacy_profile = str(manifest.get("training_profile", self.data_profile.name))
                if legacy_profile != self.data_profile.name:
                    raise ValueError("resume legacy checkpoint task profile mismatch")
            else:
                validate_resume_compatibility(
                    manifest,
                    adapter_name=str(getattr(self.adapter, "name", type(self.adapter).__name__)),
                    model_identity=dict(model_identity or {}),
                    task_profile=self.data_profile.name,
                    tuning_policy=selected_policy.as_dict(),
                )
            state = self._load_runtime_state(resume_root, optimizer, scheduler)
            start_step = int(state.get("step", 0))
            start_epoch = int(state.get("epoch", 0))

        model.train()
        optimizer.zero_grad(set_to_none=True)
        last_loss: float | None = None
        current_step = start_step
        epoch = start_epoch
        for epoch in range(start_epoch, config.epochs):
            for index, episode in enumerate(train_rows):
                self.data_profile.validate(episode)
                batch = self.adapter.encode(processor, episode)
                with autocast_context(str(getattr(model, "device", "cpu")), config.mixed_precision):
                    output = model(**self.adapter.prepare_forward_inputs(batch))
                    loss = self._loss(output)
                    if loss is None:
                        raise ValueError("adapter/model forward did not return loss; labels are required")
                    sample_weight = float((getattr(episode, "metadata", {}) or {}).get("sample_weight", 1.0))
                    if sample_weight < 0:
                        raise ValueError("sample_weight must be non-negative")
                    scaled_loss = (loss * sample_weight) / config.gradient_accumulation_steps
                scaled_loss.backward()
                if (index + 1) % config.gradient_accumulation_steps == 0:
                    clip_gradients(model, config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    current_step += 1
                last_loss = float(loss.detach().cpu().item())
                if config.logging_steps > 0 and current_step and current_step % config.logging_steps == 0:
                    self._write_log(Path(config.output_dir), {"event": "train", "step": current_step, "epoch": epoch, "loss": last_loss})
                if config.max_steps is not None and current_step >= config.max_steps:
                    break
            if config.max_steps is not None and current_step >= config.max_steps:
                break
        eval_loss = self.evaluate(model=model, processor=processor, episodes=eval_rows) if eval_rows else None
        output_dir = Path(config.output_dir)
        save_checkpoint = getattr(self.adapter, "save_checkpoint", None)
        if not callable(save_checkpoint):
            raise ValueError("selected adapter does not implement checkpoint saving")
        save_checkpoint(model, processor, output_dir)
        save_trainable_state = getattr(self.adapter, "save_trainable_state", None)
        if not callable(save_trainable_state):
            raise ValueError("selected adapter does not implement trainable-state saving")
        save_trainable_state(model, output_dir / "model_trainable_state.safetensors")
        (output_dir / PARAMETER_PLAN_FILENAME).write_text(json.dumps(plan.as_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self._save_runtime_state(output_dir, step=current_step, epoch=epoch if train_rows else 0, loss=last_loss, eval_loss=eval_loss, optimizer=optimizer, scheduler=scheduler)
        self._write_log(output_dir, {"event": "eval", "step": current_step, "epoch": epoch, "loss": last_loss, "eval_loss": eval_loss})
        manifest = build_training_manifest(
            adapter_name=str(getattr(self.adapter, "name", type(self.adapter).__name__)),
            model_identity=dict(model_identity or {}),
            task_profile=self.data_profile.name,
            data_contract={
                **(dict(config.data_contract) or {"profile": self.data_profile.name}),
                "repeat_group_key": config.repeat_group_key,
                "repeat_weights": dict(config.repeat_weights),
            },
            tuning_policy=selected_policy.as_dict(),
            parameter_plan=plan.as_dict(),
            training={"steps": current_step, "last_loss": last_loss, "eval_loss": eval_loss, "seed": config.seed, "preflight": preflight},
        )
        manifest_path = write_manifest(output_dir, manifest)
        return TrainingResult(current_step, last_loss, eval_loss, manifest_path, plan, optimizer_stats)
