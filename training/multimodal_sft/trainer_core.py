"""Generic SFT orchestration with adapter-owned model semantics."""

from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .checkpoint import (
    OPTIMIZER_FILENAME,
    PARAMETER_PLAN_FILENAME,
    RNG_STATE_FILENAME,
    SCHEDULER_FILENAME,
    TRAINER_STATE_FILENAME,
    build_training_manifest,
    checkpoint_complete,
    read_compatible_manifest,
    validate_resume_compatibility,
    write_completion_marker,
    write_manifest,
)
from .contracts import DataProfile, MultimodalModelAdapter
from .optimizer import OptimizerConfig, autocast_context, build_cosine_scheduler, build_optimizer_groups, clip_gradients
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
    batch_size: int = 1
    seed: int = 1234
    mixed_precision: str = "off"
    preflight_only: bool = False
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    logging_steps: int = 10
    smoke_gradients: bool = False
    repeat_group_key: str | None = None
    repeat_weights: Mapping[str, int] = field(default_factory=dict)
    save_steps: int = 0
    save_total_limit: int | None = None
    resume_from: str | Path | None = None
    data_contract: Mapping[str, Any] = field(default_factory=dict)
    _test_stop_after_checkpoint_step: int | None = None


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

    @staticmethod
    def _capture_rng_state() -> dict[str, Any]:
        import torch
        payload: dict[str, Any] = {"python": random.getstate(), "torch": torch.get_rng_state()}
        try:
            import numpy as np
            payload["numpy"] = np.random.get_state()
        except ImportError:
            pass
        if torch.cuda.is_available():
            payload["cuda"] = torch.cuda.get_rng_state_all()
        return payload

    @staticmethod
    def _write_rng_state(root: Path, payload: Mapping[str, Any]) -> None:
        import torch
        torch.save(payload, root / RNG_STATE_FILENAME)

    @staticmethod
    def _save_rng_state(root: Path) -> None:
        GenericTrainerCore._write_rng_state(root, GenericTrainerCore._capture_rng_state())

    @staticmethod
    def _restore_rng_payload(payload: Mapping[str, Any]) -> None:
        import torch
        random.setstate(payload["python"])
        if "numpy" in payload:
            try:
                import numpy as np
                np.random.set_state(payload["numpy"])
            except ImportError:
                pass
        torch.set_rng_state(payload["torch"])
        if torch.cuda.is_available() and "cuda" in payload:
            torch.cuda.set_rng_state_all(payload["cuda"])

    @staticmethod
    def _restore_rng_state(root: Path) -> None:
        import torch
        path = root / RNG_STATE_FILENAME
        if not path.is_file():
            raise ValueError("resume checkpoint is missing rng_state.pt")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        GenericTrainerCore._restore_rng_payload(payload)

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
    def _save_runtime_state(root: Path, *, global_step: int, epoch_index: int, next_micro_batch_index: int, optimizer_updates_in_epoch: int, loss: float | None, eval_loss: float | None, optimizer: Any, scheduler: Any) -> None:
        import torch
        torch.save(optimizer.state_dict(), root / OPTIMIZER_FILENAME)
        torch.save(scheduler.state_dict(), root / SCHEDULER_FILENAME)
        state = {
            "global_step": int(global_step),
            "epoch_index": int(epoch_index),
            "next_micro_batch_index": int(next_micro_batch_index),
            "optimizer_updates_in_epoch": int(optimizer_updates_in_epoch),
            "step": int(global_step),
            "epoch": int(epoch_index),
            "loss": loss,
            "eval_loss": eval_loss,
        }
        (root / TRAINER_STATE_FILENAME).write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _load_runtime_state(root: Path, optimizer: Any, scheduler: Any) -> dict[str, Any]:
        import torch
        state_path = root / TRAINER_STATE_FILENAME
        if not state_path.is_file():
            raise ValueError("resume checkpoint is missing trainer_state.json")
        trainer_state = json.loads(state_path.read_text(encoding="utf-8"))
        try:
            optimizer_state = torch.load(root / OPTIMIZER_FILENAME, map_location="cpu", weights_only=False)
            scheduler_state = torch.load(root / SCHEDULER_FILENAME, map_location="cpu", weights_only=False)
        except TypeError:
            optimizer_state = torch.load(root / OPTIMIZER_FILENAME, map_location="cpu")
            scheduler_state = torch.load(root / SCHEDULER_FILENAME, map_location="cpu")
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)
        for group in optimizer.param_groups:
            for parameter in group.get("params", ()):
                parameter_state = optimizer.state.get(parameter, {})
                for key, value in list(parameter_state.items()):
                    if hasattr(value, "to"):
                        parameter_state[key] = value.to(device=parameter.device)
        return trainer_state

    @staticmethod
    def _training_plan(config: TrainingConfig, planned_total_steps: int) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "micro_batch_size": int(config.batch_size),
            "gradient_accumulation_steps": int(config.gradient_accumulation_steps),
            "planned_total_optimizer_steps": int(planned_total_steps),
            "resolved_epochs": int(config.epochs),
            "resolved_max_steps": config.max_steps,
            "scheduler_type": "cosine",
            "warmup_steps": int(planned_total_steps * config.warmup_ratio),
            "warmup_ratio": float(config.warmup_ratio),
            "lora_lr": float(config.lora_lr),
            "connector_lr": float(config.connector_lr),
            "weight_decay": float(config.weight_decay),
            "max_grad_norm": float(config.max_grad_norm),
            "mixed_precision": config.mixed_precision,
            "seed": int(config.seed),
        }

    @staticmethod
    def _effective_data_contract(config: TrainingConfig) -> dict[str, Any]:
        return {**dict(config.data_contract), "batch_size": int(config.batch_size), "repeat_group_key": config.repeat_group_key, "repeat_weights": dict(config.repeat_weights)}

    def _manifest(self, *, model_identity: Mapping[str, Any], plan: ParameterPlan, policy: TuningPolicy, config: TrainingConfig, training_plan: Mapping[str, Any], global_step: int, epoch_index: int, next_micro_batch_index: int, last_loss: float | None, eval_loss: float | None, preflight: Mapping[str, Any]) -> dict[str, Any]:
        return build_training_manifest(
            adapter_name=str(getattr(self.adapter, "name", type(self.adapter).__name__)),
            model_identity=dict(model_identity), task_profile=self.data_profile.name,
            data_contract=self._effective_data_contract(config), tuning_policy=policy.as_dict(),
            parameter_plan=plan.as_dict(), training={
                "global_step": int(global_step), "epoch_index": int(epoch_index),
                "next_micro_batch_index": int(next_micro_batch_index), "last_loss": last_loss,
                "eval_loss": eval_loss, "seed": config.seed, "preflight": dict(preflight),
            },
            training_plan=training_plan,
        )

    def _serialize_checkpoint(self, *, target: Path, model: Any, processor: Any, plan: ParameterPlan, manifest: Mapping[str, Any], optimizer: Any, scheduler: Any, global_step: int, epoch_index: int, next_micro_batch_index: int, optimizer_updates_in_epoch: int, last_loss: float | None, eval_loss: float | None) -> None:
        save_checkpoint = getattr(self.adapter, "save_checkpoint", None)
        save_trainable_state = getattr(self.adapter, "save_trainable_state", None)
        validate_checkpoint_state = getattr(self.adapter, "validate_checkpoint_state", None)
        if not callable(save_checkpoint) or not callable(save_trainable_state) or not callable(validate_checkpoint_state):
            raise ValueError("selected adapter does not implement checkpoint state contracts")
        rng_payload = self._capture_rng_state()
        target.mkdir(parents=True, exist_ok=True)
        save_checkpoint(model, processor, target)
        save_trainable_state(model, target / "model_trainable_state.safetensors", plan)
        (target / PARAMETER_PLAN_FILENAME).write_text(json.dumps(plan.as_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self._save_runtime_state(target, global_step=global_step, epoch_index=epoch_index, next_micro_batch_index=next_micro_batch_index, optimizer_updates_in_epoch=optimizer_updates_in_epoch, loss=last_loss, eval_loss=eval_loss, optimizer=optimizer, scheduler=scheduler)
        self._write_rng_state(target, rng_payload)
        self._write_log(target, {"event": "checkpoint", "step": global_step, "epoch": epoch_index, "next_micro_batch_index": next_micro_batch_index})
        write_manifest(target, manifest)
        validate_checkpoint_state(target, plan)
        validate_checkpoint_ownership = getattr(self.adapter, "validate_checkpoint_ownership", None)
        if callable(validate_checkpoint_ownership):
            validate_checkpoint_ownership(model, target, plan)
        write_completion_marker(target, global_step=global_step)
        if not checkpoint_complete(target):
            raise ValueError(f"checkpoint completeness validation failed: {target}")
        self._restore_rng_payload(rng_payload)

    @staticmethod
    def _rotate_periodic_checkpoints(output_dir: Path, limit: int | None) -> None:
        if limit is None:
            return
        candidates = []
        for path in output_dir.iterdir() if output_dir.is_dir() else ():
            if path.is_dir() and path.name.startswith("checkpoint-") and path.name[len("checkpoint-"):].isdigit():
                candidates.append((int(path.name[len("checkpoint-"):]), path))
        keep = max(0, int(limit))
        for _step, path in sorted(candidates)[:-keep] if keep else sorted(candidates):
            shutil.rmtree(path)

    def _save_periodic(self, *, output_dir: Path, step: int, model: Any, processor: Any, plan: ParameterPlan, manifest: Mapping[str, Any], optimizer: Any, scheduler: Any, epoch_index: int, next_micro_batch_index: int, optimizer_updates_in_epoch: int, last_loss: float | None, eval_loss: float | None, save_total_limit: int | None) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / f"checkpoint-{step}"
        if target.exists():
            raise ValueError(f"periodic checkpoint already exists: {target}")
        temp = output_dir / f".checkpoint-{step}.tmp-{os.getpid()}"
        if temp.exists():
            shutil.rmtree(temp)
        self._serialize_checkpoint(target=temp, model=model, processor=processor, plan=plan, manifest=manifest, optimizer=optimizer, scheduler=scheduler, global_step=step, epoch_index=epoch_index, next_micro_batch_index=next_micro_batch_index, optimizer_updates_in_epoch=optimizer_updates_in_epoch, last_loss=last_loss, eval_loss=eval_loss)
        os.replace(temp, target)
        self._rotate_periodic_checkpoints(output_dir, save_total_limit)
        return target

    def fit(self, *, model: Any, processor: Any, episodes: Iterable[Any], config: TrainingConfig, policy: TuningPolicy | str, probe: Any | None = None, model_identity: Mapping[str, Any] | None = None, eval_episodes: Iterable[Any] | None = None) -> TrainingResult:
        import torch
        if config.batch_size != 1:
            raise ValueError("GENERIC_BATCHING_NOT_YET_AVAILABLE: micro-batch size must be 1 before Phase 1D")
        if config.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if config.save_steps < 0:
            raise ValueError("save_steps must be zero or positive")
        self.seed_everything(config.seed)
        train_rows = self._materialize(episodes, None)
        train_rows = self._repeat_rows(train_rows, group_key=config.repeat_group_key, weights=config.repeat_weights)
        if config.max_train_samples is not None:
            train_rows = train_rows[: config.max_train_samples]
        eval_rows = self._materialize(eval_episodes or (), config.max_eval_samples)
        preflight = self.preflight(train_rows)
        if config.preflight_only:
            plan = self.build_plan(model, policy, probe=probe)
            return TrainingResult(0, None, None, None, plan, {"preflight": preflight})
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

        updates_per_epoch = max(1, (len(train_rows) + config.gradient_accumulation_steps - 1) // config.gradient_accumulation_steps)
        planned_total_steps = config.max_steps or max(1, updates_per_epoch * config.epochs)
        training_plan = self._training_plan(config, planned_total_steps)

        resume_root: Path | None = None
        start_step = start_epoch = start_index = optimizer_updates_in_epoch = 0
        if config.resume_from:
            resume_root = Path(config.resume_from)
            if not checkpoint_complete(resume_root):
                raise ValueError("resume checkpoint is incomplete or missing completion marker")
            resume_manifest = read_compatible_manifest(resume_root, legacy_manifest_names=("phase2_training_manifest.json",))
            if "checkpoint_type" not in resume_manifest:
                raise ValueError("legacy checkpoint requires an adapter-specific compatibility restore")
            validate_resume_compatibility(resume_manifest, adapter_name=str(getattr(self.adapter, "name", type(self.adapter).__name__)), model_identity=dict(model_identity or {}), task_profile=self.data_profile.name, tuning_policy=selected_policy.as_dict(), parameter_plan=plan.as_dict(), data_contract=self._effective_data_contract(config), training_plan=training_plan)
            restore = getattr(self.adapter, "restore_trainable_state", None)
            if not callable(restore):
                raise ValueError("selected adapter does not implement trainable-state restore")
            model = restore(model=model, checkpoint_dir=resume_root, parameter_plan=plan, manifest=dict(resume_manifest))
            if callable(validate_trainable):
                validate_trainable(model, plan)

        groups, optimizer_stats = build_optimizer_groups(model, plan, OptimizerConfig(lora_lr=config.lora_lr, connector_lr=config.connector_lr, weight_decay=config.weight_decay, warmup_ratio=config.warmup_ratio, max_grad_norm=config.max_grad_norm, mixed_precision=config.mixed_precision))
        validate_optimizer = getattr(self.adapter, "validate_optimizer_parameters", None)
        if callable(validate_optimizer):
            validate_optimizer(model, plan, groups)
        optimizer = torch.optim.AdamW(groups)
        scheduler = build_cosine_scheduler(optimizer, planned_total_steps, config.warmup_ratio)
        if resume_root is not None:
            state = self._load_runtime_state(resume_root, optimizer, scheduler)
            start_step = int(state.get("global_step", state.get("step", 0)))
            start_epoch = int(state.get("epoch_index", state.get("epoch", 0)))
            start_index = int(state.get("next_micro_batch_index", 0))
            optimizer_updates_in_epoch = int(state.get("optimizer_updates_in_epoch", 0))
            if start_index >= len(train_rows):
                start_epoch += 1
                start_index = 0
            self._restore_rng_state(resume_root)
        if config.smoke_gradients:
            self.smoke_gradients(model=model, processor=processor, episode=train_rows[start_index if start_index < len(train_rows) else 0])

        model.train()
        optimizer.zero_grad(set_to_none=True)
        last_loss: float | None = None
        current_step = start_step
        next_epoch_index = start_epoch
        next_micro_batch_index = start_index
        stop = current_step >= (config.max_steps or 2**63 - 1)
        for epoch_index in range(start_epoch, config.epochs):
            begin = start_index if epoch_index == start_epoch else 0
            for index in range(begin, len(train_rows)):
                episode = train_rows[index]
                self.data_profile.validate(episode)
                batch = self.adapter.encode(processor, episode)
                window_start = index - (index % config.gradient_accumulation_steps)
                window_size = min(config.gradient_accumulation_steps, len(train_rows) - window_start)
                with autocast_context(str(getattr(model, "device", "cpu")), config.mixed_precision):
                    output = model(**self.adapter.prepare_forward_inputs(batch))
                    loss = self._loss(output)
                    if loss is None:
                        raise ValueError("adapter/model forward did not return loss; labels are required")
                    sample_weight = float((getattr(episode, "metadata", {}) or {}).get("sample_weight", 1.0))
                    if sample_weight < 0:
                        raise ValueError("sample_weight must be non-negative")
                    scaled_loss = (loss * sample_weight) / window_size
                scaled_loss.backward()
                last_loss = float(loss.detach().cpu().item())
                end_window = ((index + 1) % config.gradient_accumulation_steps == 0) or index == len(train_rows) - 1
                if not end_window:
                    continue
                clip_gradients(model, config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                current_step += 1
                optimizer_updates_in_epoch += 1
                if index + 1 < len(train_rows):
                    next_epoch_index, next_micro_batch_index = epoch_index, index + 1
                else:
                    next_epoch_index, next_micro_batch_index = epoch_index + 1, 0
                if config.logging_steps > 0 and current_step % config.logging_steps == 0:
                    self._write_log(Path(config.output_dir), {"event": "train", "step": current_step, "epoch": epoch_index, "loss": last_loss})
                if config.save_steps and current_step % config.save_steps == 0:
                    manifest = self._manifest(model_identity=dict(model_identity or {}), plan=plan, policy=selected_policy, config=config, training_plan=training_plan, global_step=current_step, epoch_index=next_epoch_index, next_micro_batch_index=next_micro_batch_index, last_loss=last_loss, eval_loss=None, preflight=preflight)
                    self._save_periodic(output_dir=Path(config.output_dir), step=current_step, model=model, processor=processor, plan=plan, manifest=manifest, optimizer=optimizer, scheduler=scheduler, epoch_index=next_epoch_index, next_micro_batch_index=next_micro_batch_index, optimizer_updates_in_epoch=optimizer_updates_in_epoch, last_loss=last_loss, eval_loss=None, save_total_limit=config.save_total_limit)
                    if config._test_stop_after_checkpoint_step == current_step:
                        raise RuntimeError("TEST_STOP_AFTER_CHECKPOINT")
                if config.max_steps is not None and current_step >= config.max_steps:
                    stop = True
                    break
            start_index = 0
            if stop:
                break

        eval_loss = self.evaluate(model=model, processor=processor, episodes=eval_rows) if eval_rows else None
        output_dir = Path(config.output_dir)
        final_manifest = self._manifest(model_identity=dict(model_identity or {}), plan=plan, policy=selected_policy, config=config, training_plan=training_plan, global_step=current_step, epoch_index=next_epoch_index, next_micro_batch_index=next_micro_batch_index, last_loss=last_loss, eval_loss=eval_loss, preflight=preflight)
        self._serialize_checkpoint(target=output_dir, model=model, processor=processor, plan=plan, manifest=final_manifest, optimizer=optimizer, scheduler=scheduler, global_step=current_step, epoch_index=next_epoch_index, next_micro_batch_index=next_micro_batch_index, optimizer_updates_in_epoch=optimizer_updates_in_epoch, last_loss=last_loss, eval_loss=eval_loss)
        self._write_log(output_dir, {"event": "eval", "step": current_step, "epoch": next_epoch_index, "loss": last_loss, "eval_loss": eval_loss})
        return TrainingResult(current_step, last_loss, eval_loss, output_dir / "training_manifest.json", plan, optimizer_stats)
