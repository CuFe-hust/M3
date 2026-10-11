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
from .identity import base_weight_identity, materialize_processor_identity
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
    max_seq_length: int = 4096
    gradient_checkpointing: bool = False
    seed: int = 1234
    mixed_precision: str = "off"
    preflight_only: bool = False
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    logging_steps: int = 10
    eval_steps: int = 100
    smoke_gradients: bool = False
    smoke_gradients_only: bool = False
    repeat_group_key: str | None = None
    repeat_weights: Mapping[str, int] = field(default_factory=dict)
    save_steps: int = 0
    save_total_limit: int | None = None
    resume_from: str | Path | None = None
    data_contract: Mapping[str, Any] = field(default_factory=dict)
    image_roots: Any = None
    base_model_id: str | Path | None = None
    _test_stop_after_checkpoint_step: int | None = None


@dataclass(frozen=True)
class TrainingResult:
    steps: int
    loss: float | None
    eval_loss: float | None
    manifest_path: Path | None
    parameter_plan: ParameterPlan
    optimizer_stats: Mapping[str, Any] = field(default_factory=dict)
    final_adapter_path: Path | None = None


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

    def preflight(self, episodes: Iterable[Any], *, image_roots: Any = None, split: str = "train", epoch: int = 0, seed: int | str = 0, limit: int | None = None) -> dict[str, Any]:
        counts = {"checked": 0, "schema_errors": 0, "image_errors": 0, "prompt_errors": 0, "other_errors": 0}
        for episode in episodes:
            if limit is not None and counts["checked"] >= limit:
                break
            counts["checked"] += 1
            try:
                self._prepare_episode(episode, image_roots=image_roots, split=split, epoch=epoch, seed=seed)
            except Exception as exc:  # noqa: BLE001 - preflight reports all source errors
                code = str(getattr(exc, "code", "")) or str(exc)
                if "IMAGE_" in code or "IMAGE" in code or "UNKNOWN_IMAGE_SOURCE" in code or "UNSAFE_IMAGE_PATH" in code:
                    counts["image_errors"] += 1
                elif "PROMPT" in code:
                    counts["prompt_errors"] += 1
                elif isinstance(exc, (ValueError, KeyError, TypeError)):
                    counts["schema_errors"] += 1
                else:
                    counts["other_errors"] += 1
        counts["profile"] = self.data_profile.name
        counts["passed"] = not any(counts[key] for key in ("schema_errors", "image_errors", "prompt_errors", "other_errors"))
        if not counts["passed"]:
            raise ValueError(f"DATA_PREFLIGHT_FAILED: {counts}")
        return counts

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

    def _forward_loss(self, model: Any, batch: Mapping[str, Any]) -> Any:
        model_inputs = self._forward_inputs(model, batch)
        prepare_loss_inputs = getattr(self.adapter, "prepare_loss_inputs", None)
        compute_loss = getattr(self.adapter, "compute_loss", None)
        if callable(prepare_loss_inputs) and callable(compute_loss):
            model_inputs, loss_context = prepare_loss_inputs(model_inputs)
            return compute_loss(model(**model_inputs), loss_context)
        return self._loss(model(**model_inputs))

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

    def _prepare_episode(self, episode: Any, *, image_roots: Any, split: str, epoch: int, seed: int | str) -> Any:
        prepare = getattr(self.data_profile, "prepare", None)
        if callable(prepare):
            return prepare(episode, image_roots=image_roots, split=split, epoch=epoch, seed=seed)
        self.data_profile.validate(episode)
        return episode

    @staticmethod
    def _model_input_device(model: Any) -> Any:
        getter = getattr(model, "get_input_embeddings", None)
        if callable(getter):
            embedding = getter()
            weight = getattr(embedding, "weight", None)
            if weight is not None and getattr(weight, "device", None) is not None:
                return weight.device
        return getattr(model, "device", "cpu")

    @classmethod
    def _move_input_value(cls, value: Any, device: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: cls._move_input_value(item, device) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(cls._move_input_value(item, device) for item in value)
        if isinstance(value, list):
            return [cls._move_input_value(item, device) for item in value]
        move = getattr(value, "to", None)
        return move(device=device) if callable(move) else value

    def _forward_inputs(self, model: Any, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        prepared = self.adapter.prepare_forward_inputs(dict(batch))
        return self._move_input_value(prepared, self._model_input_device(model))

    def smoke_gradients(self, *, model: Any, processor: Any, episode: Any, parameter_plan: Any = None, image_roots: Any = None, split: str = "train", epoch: int = 0, seed: int | str = 0, max_seq_length: int = 4096) -> dict[str, Any]:
        import torch

        prepared = self._prepare_episode(episode, image_roots=image_roots, split=split, epoch=epoch, seed=seed)
        model.train()
        for parameter in model.parameters():
            parameter.grad = None
        encoded = self._encode(processor, prepared, max_seq_length=max_seq_length)
        batch, _meta = self._collate([encoded])
        loss = self._forward_loss(model, batch)
        if loss is None:
            raise ValueError("gradient smoke requires a model loss")
        if not bool(torch.isfinite(loss).all().item()):
            raise ValueError("gradient smoke found non-finite loss")
        loss.backward()
        named_parameters = list(model.named_parameters())
        trainable = [(name, parameter) for name, parameter in named_parameters if bool(getattr(parameter, "requires_grad", False))]
        with_grad = [(name, parameter) for name, parameter in trainable if parameter.grad is not None]
        if trainable and len(with_grad) != len(trainable):
            raise ValueError(f"gradient smoke found missing gradients: {len(with_grad)}/{len(trainable)}")
        nonfinite = [name for name, parameter in with_grad if not bool(torch.isfinite(parameter.grad).all().item())]
        if nonfinite:
            raise ValueError(f"gradient smoke found non-finite gradients: {nonfinite[:10]}")
        frozen_with_grad = [
            name for name, parameter in named_parameters
            if not bool(getattr(parameter, "requires_grad", False)) and parameter.grad is not None
        ]
        if frozen_with_grad:
            raise ValueError(f"gradient smoke found frozen-parameter gradients: {frozen_with_grad[:10]}")
        squared_norm = sum(float(parameter.grad.detach().float().pow(2).sum().cpu().item()) for _, parameter in with_grad)
        max_abs = max((float(parameter.grad.detach().float().abs().max().cpu().item()) for _, parameter in with_grad), default=0.0)
        full_train_paths = tuple(getattr(parameter_plan, "full_train_module_paths", ()) or ())
        connector_with_grad = sum(
            any(("." + path + ".") in ("." + name + ".") for path in full_train_paths)
            for name, _ in with_grad
        )
        lora_with_grad = len(with_grad) - connector_with_grad
        for _, parameter in trainable:
            parameter.grad = None
        return {
            "passed": True,
            "trainable_parameter_tensors": len(trainable),
            "parameter_tensors_with_grad": len(with_grad),
            "lora_parameter_tensors_with_grad": lora_with_grad,
            "connector_parameter_tensors_with_grad": connector_with_grad,
            "frozen_parameter_tensors_with_grad": 0,
            "nonfinite_gradient_tensors": 0,
            "gradient_l2_norm": squared_norm ** 0.5,
            "gradient_max_abs": max_abs,
            "loss": float(loss.detach().cpu().item()),
        }

    def evaluate(self, *, model: Any, processor: Any, episodes: Sequence[Any], image_roots: Any = None, epoch: int = 0, seed: int | str = 0, max_seq_length: int = 4096, batch_size: int = 1) -> float | None:
        import torch
        losses: list[float] = []
        was_training = bool(getattr(model, "training", False))
        model.eval()
        with torch.no_grad():
            prepared_rows = [self._prepare_episode(episode, image_roots=image_roots, split="validation", epoch=epoch, seed=seed) for episode in episodes]
            for start in range(0, len(prepared_rows), max(1, int(batch_size))):
                encoded = [self._encode(processor, episode, max_seq_length=max_seq_length) for episode in prepared_rows[start:start + max(1, int(batch_size))]]
                batch, _meta = self._collate(encoded)
                loss = self._forward_loss(model, batch)
                if loss is not None:
                    losses.append(float(loss.detach().cpu().item()))
        if was_training:
            model.train()
        return sum(losses) / len(losses) if losses else None

    def _encode(self, processor: Any, episode: Any, *, max_seq_length: int) -> Mapping[str, Any]:
        encode = getattr(self.adapter, "encode", None)
        if not callable(encode):
            raise ValueError("selected adapter does not implement encode")
        try:
            return encode(processor, episode, max_seq_length=max_seq_length)
        except TypeError as exc:
            if "max_seq_length" not in str(exc):
                raise
            return encode(processor, episode)

    def _collate(self, encoded_examples: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]:
        collate = getattr(self.adapter, "collate", None)
        if callable(collate):
            return collate(encoded_examples)
        if len(encoded_examples) != 1:
            raise ValueError("ADAPTER_COLLATE_REQUIRED_FOR_BATCHING")
        return encoded_examples[0], ({},)

    @staticmethod
    def _save_runtime_state(root: Path, *, global_step: int, epoch_index: int, next_micro_batch_index: int, next_sample_index: int, next_batch_index: int, optimizer_updates_in_epoch: int, loss: float | None, eval_loss: float | None, optimizer: Any, scheduler: Any) -> None:
        import torch
        torch.save(optimizer.state_dict(), root / OPTIMIZER_FILENAME)
        torch.save(scheduler.state_dict(), root / SCHEDULER_FILENAME)
        state = {
            "global_step": int(global_step),
            "epoch_index": int(epoch_index),
            "next_micro_batch_index": int(next_micro_batch_index),
            "next_sample_index": int(next_sample_index),
            "next_batch_index": int(next_batch_index),
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
            "max_seq_length": int(config.max_seq_length),
            "gradient_accumulation_steps": int(config.gradient_accumulation_steps),
            "planned_total_optimizer_steps": int(planned_total_steps),
            "resolved_epochs": int(config.epochs),
            "resolved_max_steps": config.max_steps,
            "scheduler_type": "cosine",
            "warmup_steps": int(planned_total_steps * config.warmup_ratio),
            "warmup_ratio": float(config.warmup_ratio),
            "lora_lr": float(config.lora_lr),
            "eval_steps": int(config.eval_steps),
            "connector_lr": float(config.connector_lr),
            "weight_decay": float(config.weight_decay),
            "max_grad_norm": float(config.max_grad_norm),
            "mixed_precision": config.mixed_precision,
            "seed": int(config.seed),
        }

    def _effective_data_contract(self, config: TrainingConfig) -> dict[str, Any]:
        profile_identity = getattr(self.data_profile, "identity_contract", None)
        resolved = dict(profile_identity(config.image_roots) if callable(profile_identity) else {})
        roots_contract = config.image_roots.contract() if callable(getattr(config.image_roots, "contract", None)) else None
        if roots_contract is not None:
            resolved["image_root_contract"] = roots_contract
        return {**resolved, **dict(config.data_contract), "batch_size": int(config.batch_size), "max_seq_length": int(config.max_seq_length), "repeat_group_key": config.repeat_group_key, "repeat_weights": dict(config.repeat_weights)}

    def _runtime_processor_identity(self, processor: Any) -> dict[str, Any]:
        semantic_fn = getattr(self.adapter, "processor_identity", None)
        semantic = dict(semantic_fn(processor)) if callable(semantic_fn) else {}
        encoding_contract = semantic.get("encoding_contract_version")
        if not encoding_contract:
            raise ValueError("RESUME_PROCESSOR_IDENTITY_UNPROVEN")
        content = materialize_processor_identity(
            processor,
            encoding_contract_version=str(encoding_contract),
        )
        # Adapter semantics are authoritative for the project encoding
        # contract; saved bytes remain authoritative for content identity.
        content.update({key: value for key, value in semantic.items() if key not in {"content_sha256", "files"}})
        return content

    def _load_canonical_resume_processor(
        self,
        checkpoint_dir: Path,
        manifest: Mapping[str, Any],
        runtime_processor: Any,
    ) -> tuple[Any, dict[str, Any]]:
        processor_dir = checkpoint_dir / "processor"
        if not processor_dir.is_dir():
            raise ValueError("RESUME_PROCESSOR_IDENTITY_UNPROVEN")
        expected = dict(manifest.get("processor", {}))
        if not expected.get("content_sha256") or not expected.get("encoding_contract_version"):
            raise ValueError("RESUME_PROCESSOR_IDENTITY_UNPROVEN")
        try:
            canonical = self.adapter.load_processor(processor_dir, local_files_only=True)
            canonical_identity = dict(self.adapter.saved_processor_identity(canonical, processor_dir))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("RESUME_PROCESSOR_IDENTITY_UNPROVEN") from exc
        semantic_keys = ("class", "tokenizer_class", "chat_template_sha256", "special_tokens_sha256", "special_token_ids")
        if any(expected.get(key) != canonical_identity.get(key) for key in semantic_keys):
            raise ValueError("RESUME_CHECKPOINT_PROCESSOR_IDENTITY_MISMATCH")
        if expected.get("content_sha256") != canonical_identity.get("content_sha256"):
            raise ValueError("RESUME_CHECKPOINT_PROCESSOR_IDENTITY_MISMATCH")
        if expected.get("encoding_contract_version") != canonical_identity.get("encoding_contract_version"):
            raise ValueError("RESUME_CHECKPOINT_PROCESSOR_IDENTITY_MISMATCH")
        runtime_identity = self._runtime_processor_identity(runtime_processor)
        return canonical, runtime_identity

    def _manifest(self, *, model_identity: Mapping[str, Any], plan: ParameterPlan, policy: TuningPolicy, config: TrainingConfig, training_plan: Mapping[str, Any], global_step: int, epoch_index: int, next_micro_batch_index: int, last_loss: float | None, eval_loss: float | None, preflight: Mapping[str, Any], processor: Any | None = None, next_sample_index: int = 0, next_batch_index: int = 0) -> dict[str, Any]:
        processor_identity = {}
        identity_fn = getattr(self.adapter, "processor_identity", None)
        if processor is not None and callable(identity_fn):
            processor_identity = dict(identity_fn(processor))
        return build_training_manifest(
            adapter_name=str(getattr(self.adapter, "name", type(self.adapter).__name__)),
            model_identity=dict(model_identity), task_profile=self.data_profile.name,
            data_contract=self._effective_data_contract(config), tuning_policy=policy.as_dict(),
            parameter_plan=plan.as_dict(), training={
                "global_step": int(global_step), "epoch_index": int(epoch_index),
                "next_micro_batch_index": int(next_micro_batch_index),
                "next_sample_index": int(next_sample_index), "next_batch_index": int(next_batch_index), "last_loss": last_loss,
                "eval_loss": eval_loss, "seed": config.seed, "preflight": dict(preflight),
            },
            processor_identity=processor_identity,
            training_plan=training_plan,
            base_model_id=str(config.base_model_id) if config.base_model_id is not None else None,
            base_weight_identity=model_identity.get("base_weight_identity"),
        )

    def _serialize_checkpoint(self, *, target: Path, model: Any, processor: Any, plan: ParameterPlan, manifest: Mapping[str, Any], optimizer: Any, scheduler: Any, global_step: int, epoch_index: int, next_micro_batch_index: int, next_sample_index: int = 0, next_batch_index: int = 0, optimizer_updates_in_epoch: int = 0, last_loss: float | None, eval_loss: float | None) -> None:
        save_checkpoint = getattr(self.adapter, "save_checkpoint", None)
        save_trainable_state = getattr(self.adapter, "save_trainable_state", None)
        validate_checkpoint_state = getattr(self.adapter, "validate_checkpoint_state", None)
        if not callable(save_checkpoint) or not callable(save_trainable_state) or not callable(validate_checkpoint_state):
            raise ValueError("selected adapter does not implement checkpoint state contracts")
        rng_payload = self._capture_rng_state()
        target.mkdir(parents=True, exist_ok=True)
        save_checkpoint(model, processor, target)
        saved_processor_identity = dict(self.adapter.saved_processor_identity(processor, target / "processor"))
        manifest_to_write = dict(manifest)
        manifest_to_write["processor"] = saved_processor_identity
        save_trainable_state(model, target / "model_trainable_state.safetensors", plan)
        (target / PARAMETER_PLAN_FILENAME).write_text(json.dumps(plan.as_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self._save_runtime_state(target, global_step=global_step, epoch_index=epoch_index, next_micro_batch_index=next_micro_batch_index, next_sample_index=next_sample_index, next_batch_index=next_batch_index, optimizer_updates_in_epoch=optimizer_updates_in_epoch, loss=last_loss, eval_loss=eval_loss, optimizer=optimizer, scheduler=scheduler)
        self._write_rng_state(target, rng_payload)
        self._write_log(target, {"event": "checkpoint", "step": global_step, "epoch": epoch_index, "next_micro_batch_index": next_micro_batch_index})
        write_manifest(target, manifest_to_write)
        validate_checkpoint_state(target, plan)
        validate_checkpoint_ownership = getattr(self.adapter, "validate_checkpoint_ownership", None)
        if callable(validate_checkpoint_ownership):
            validate_checkpoint_ownership(model, target, plan)
        write_completion_marker(target, global_step=global_step)
        if not checkpoint_complete(target):
            raise ValueError(f"checkpoint completeness validation failed: {target}")
        self._restore_rng_payload(rng_payload)

    def _save_final_adapter(self, *, model: Any, processor: Any, output_dir: Path) -> Path | None:
        model_saver = getattr(model, "save_pretrained", None)
        processor_saver = getattr(processor, "save_pretrained", None)
        if not callable(model_saver) or not callable(processor_saver):
            return None
        if output_dir.exists():
            raise ValueError(f"final adapter output already exists: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=False)
        model_saver(output_dir, safe_serialization=True)
        processor_saver(output_dir)
        if not (output_dir / "adapter_config.json").is_file():
            raise ValueError("final adapter is not a PEFT adapter")
        return output_dir

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

    def _save_periodic(self, *, output_dir: Path, step: int, model: Any, processor: Any, plan: ParameterPlan, manifest: Mapping[str, Any], optimizer: Any, scheduler: Any, epoch_index: int, next_micro_batch_index: int, next_sample_index: int = 0, next_batch_index: int = 0, optimizer_updates_in_epoch: int = 0, last_loss: float | None, eval_loss: float | None, save_total_limit: int | None) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / f"checkpoint-{step}"
        if target.exists():
            raise ValueError(f"periodic checkpoint already exists: {target}")
        temp = output_dir / f".checkpoint-{step}.tmp-{os.getpid()}"
        if temp.exists():
            shutil.rmtree(temp)
        self._serialize_checkpoint(target=temp, model=model, processor=processor, plan=plan, manifest=manifest, optimizer=optimizer, scheduler=scheduler, global_step=step, epoch_index=epoch_index, next_micro_batch_index=next_micro_batch_index, next_sample_index=next_sample_index, next_batch_index=next_batch_index, optimizer_updates_in_epoch=optimizer_updates_in_epoch, last_loss=last_loss, eval_loss=eval_loss)
        os.replace(temp, target)
        self._rotate_periodic_checkpoints(output_dir, save_total_limit)
        return target

    def fit(self, *, model: Any, processor: Any, episodes: Iterable[Any], config: TrainingConfig, policy: TuningPolicy | str, probe: Any | None = None, model_identity: Mapping[str, Any] | None = None, eval_episodes: Iterable[Any] | None = None) -> TrainingResult:
        import torch
        if config.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if config.max_seq_length < 1:
            raise ValueError("max_seq_length must be positive")
        if config.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if config.save_steps < 0:
            raise ValueError("save_steps must be zero or positive")
        if config.eval_steps < 0:
            raise ValueError("eval_steps must be zero or positive")
        if config.smoke_gradients_only and not config.smoke_gradients:
            raise ValueError("smoke_gradients_only requires smoke_gradients")
        self.seed_everything(config.seed)
        effective_model_identity = dict(model_identity or {})
        if config.base_model_id is not None:
            effective_model_identity["base_weight_identity"] = base_weight_identity(config.base_model_id)
        train_rows = self._materialize(episodes, None)
        train_rows = self._repeat_rows(train_rows, group_key=config.repeat_group_key, weights=config.repeat_weights)
        if config.max_train_samples is not None:
            train_rows = train_rows[: config.max_train_samples]
        eval_rows = self._materialize(eval_episodes or (), config.max_eval_samples)
        preflight = self.preflight(train_rows, image_roots=config.image_roots, split="train", epoch=0, seed=config.seed)
        if eval_rows:
            preflight["validation"] = self.preflight(eval_rows, image_roots=config.image_roots, split="validation", epoch=0, seed=config.seed)
        if config.preflight_only:
            plan = self.build_plan(model, policy, probe=probe)
            return TrainingResult(0, None, None, None, plan, {"preflight": preflight})
        if not train_rows:
            raise ValueError("training profile produced no episodes")
        if config.batch_size > 1:
            weighted = [row for row in train_rows if float((getattr(row, "metadata", {}) or {}).get("sample_weight", 1.0)) != 1.0]
            if weighted:
                raise ValueError("SAMPLE_WEIGHT_BATCHING_UNSUPPORTED")
        selected_policy = policy if isinstance(policy, TuningPolicy) else TuningPolicy.from_name(policy)
        plan = self.build_plan(model, selected_policy, probe=probe)
        processor_identity = {}
        processor_identity_fn = getattr(self.adapter, "processor_identity", None)
        if callable(processor_identity_fn):
            processor_identity = dict(processor_identity_fn(processor))
        apply_policy = getattr(self.adapter, "apply_tuning_policy", None)
        if not callable(apply_policy):
            raise ValueError("selected adapter does not implement parameter-plan application")
        model = apply_policy(model, plan, selected_policy)
        validate_trainable = getattr(self.adapter, "validate_trainable_parameters", None)
        if callable(validate_trainable):
            validate_trainable(model, plan)

        if config.gradient_checkpointing:
            enable_input_require_grads = getattr(model, "enable_input_require_grads", None)
            if callable(enable_input_require_grads):
                enable_input_require_grads()
            gradient_checkpointing_enable = getattr(model, "gradient_checkpointing_enable", None)
            if not callable(gradient_checkpointing_enable):
                raise ValueError("gradient_checkpointing is not supported by the selected model")
            try:
                gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                gradient_checkpointing_enable()
            model_config = getattr(model, "config", None)
            if model_config is not None and hasattr(model_config, "use_cache"):
                model_config.use_cache = False

        micro_batches_per_epoch = max(1, (len(train_rows) + config.batch_size - 1) // config.batch_size)
        updates_per_epoch = max(1, (micro_batches_per_epoch + config.gradient_accumulation_steps - 1) // config.gradient_accumulation_steps)
        planned_total_steps = config.max_steps or max(1, updates_per_epoch * config.epochs)
        training_plan = self._training_plan(config, planned_total_steps)

        resume_root: Path | None = None
        start_step = start_epoch = start_sample_index = optimizer_updates_in_epoch = 0
        start_batch_index = 0
        if config.resume_from:
            resume_root = Path(config.resume_from)
            if not checkpoint_complete(resume_root):
                raise ValueError("resume checkpoint is incomplete or missing completion marker")
            resume_manifest = read_compatible_manifest(resume_root, legacy_manifest_names=("phase2_training_manifest.json",))
            if "checkpoint_type" not in resume_manifest:
                raise ValueError("legacy checkpoint requires an adapter-specific compatibility restore")
            processor, runtime_processor_identity = self._load_canonical_resume_processor(resume_root, resume_manifest, processor)
            validate_resume_compatibility(resume_manifest, adapter_name=str(getattr(self.adapter, "name", type(self.adapter).__name__)), model_identity=effective_model_identity, task_profile=self.data_profile.name, tuning_policy=selected_policy.as_dict(), parameter_plan=plan.as_dict(), data_contract=self._effective_data_contract(config), training_plan=training_plan, processor_identity=runtime_processor_identity)
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
            start_sample_index = int(state.get("next_sample_index", state.get("next_micro_batch_index", 0)))
            start_batch_index = int(state.get("next_batch_index", start_sample_index // config.batch_size))
            optimizer_updates_in_epoch = int(state.get("optimizer_updates_in_epoch", 0))
            if start_sample_index >= len(train_rows):
                start_epoch += 1
                start_sample_index = 0
                start_batch_index = 0
            self._restore_rng_state(resume_root)
        smoke_result = None
        if config.smoke_gradients:
            smoke_index = start_sample_index if start_sample_index < len(train_rows) else 0
            smoke_result = self.smoke_gradients(model=model, processor=processor, episode=train_rows[smoke_index], parameter_plan=plan, image_roots=config.image_roots, epoch=start_epoch, seed=config.seed, max_seq_length=config.max_seq_length)
        if config.smoke_gradients_only:
            return TrainingResult(0, None, None, None, plan, {**optimizer_stats, "gradient_smoke": smoke_result})

        model.train()
        optimizer.zero_grad(set_to_none=True)
        last_loss: float | None = None
        eval_loss: float | None = None
        current_step = start_step
        next_epoch_index = start_epoch
        next_sample_index = start_sample_index
        next_batch_index = start_batch_index
        next_micro_batch_index = start_sample_index
        stop = current_step >= (config.max_steps or 2**63 - 1)
        for epoch_index in range(start_epoch, config.epochs):
            begin = start_sample_index if epoch_index == start_epoch else 0
            for batch_start in range(begin, len(train_rows), config.batch_size):
                batch_rows = train_rows[batch_start:batch_start + config.batch_size]
                prepared_rows = []
                for episode in batch_rows:
                    self.data_profile.validate(episode)
                    prepared_rows.append(self._prepare_episode(episode, image_roots=config.image_roots, split="train", epoch=epoch_index, seed=config.seed))
                encoded = [self._encode(processor, prepared, max_seq_length=config.max_seq_length) for prepared in prepared_rows]
                batch, _meta = self._collate(encoded)
                batch_index = batch_start // config.batch_size
                window_start = batch_index - (batch_index % config.gradient_accumulation_steps)
                total_batches = (len(train_rows) + config.batch_size - 1) // config.batch_size
                window_size = min(config.gradient_accumulation_steps, total_batches - window_start)
                with autocast_context(str(getattr(model, "device", "cpu")), config.mixed_precision):
                    loss = self._forward_loss(model, batch)
                    if loss is None:
                        raise ValueError("adapter/model forward did not return loss; labels are required")
                    sample_weight = float((getattr(batch_rows[0], "metadata", {}) or {}).get("sample_weight", 1.0)) if len(batch_rows) == 1 else 1.0
                    if sample_weight < 0:
                        raise ValueError("sample_weight must be non-negative")
                    scaled_loss = (loss * sample_weight) / window_size
                scaled_loss.backward()
                last_loss = float(loss.detach().cpu().item())
                end_window = ((batch_index + 1) % config.gradient_accumulation_steps == 0) or batch_start + len(batch_rows) >= len(train_rows)
                if not end_window:
                    continue
                clip_gradients(model, config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                current_step += 1
                optimizer_updates_in_epoch += 1
                if batch_start + len(batch_rows) < len(train_rows):
                    next_epoch_index, next_sample_index = epoch_index, batch_start + len(batch_rows)
                    next_batch_index = next_sample_index // config.batch_size
                else:
                    next_epoch_index, next_sample_index, next_batch_index = epoch_index + 1, 0, 0
                next_micro_batch_index = next_sample_index
                if config.logging_steps > 0 and current_step % config.logging_steps == 0:
                    self._write_log(Path(config.output_dir), {"event": "train", "step": current_step, "epoch": epoch_index, "loss": last_loss, "next_sample_index": next_sample_index, "next_batch_index": next_batch_index})
                if eval_rows and config.eval_steps and current_step % config.eval_steps == 0:
                    eval_loss = self.evaluate(model=model, processor=processor, episodes=eval_rows, image_roots=config.image_roots, epoch=next_epoch_index, seed=config.seed, max_seq_length=config.max_seq_length, batch_size=config.batch_size)
                    self._write_log(Path(config.output_dir), {"event": "eval", "step": current_step, "epoch": next_epoch_index, "loss": last_loss, "eval_loss": eval_loss})
                if config.save_steps and current_step % config.save_steps == 0:
                    manifest = self._manifest(model_identity=effective_model_identity, plan=plan, policy=selected_policy, config=config, training_plan=training_plan, global_step=current_step, epoch_index=next_epoch_index, next_micro_batch_index=next_micro_batch_index, next_sample_index=next_sample_index, next_batch_index=next_batch_index, last_loss=last_loss, eval_loss=eval_loss, preflight=preflight, processor=processor)
                    self._save_periodic(output_dir=Path(config.output_dir), step=current_step, model=model, processor=processor, plan=plan, manifest=manifest, optimizer=optimizer, scheduler=scheduler, epoch_index=next_epoch_index, next_micro_batch_index=next_micro_batch_index, next_sample_index=next_sample_index, next_batch_index=next_batch_index, optimizer_updates_in_epoch=optimizer_updates_in_epoch, last_loss=last_loss, eval_loss=eval_loss, save_total_limit=config.save_total_limit)
                    if config._test_stop_after_checkpoint_step == current_step:
                        raise RuntimeError("TEST_STOP_AFTER_CHECKPOINT")
                if config.max_steps is not None and current_step >= config.max_steps:
                    stop = True
                    break
            start_sample_index = 0
            if stop:
                break

        eval_loss = self.evaluate(model=model, processor=processor, episodes=eval_rows, image_roots=config.image_roots, epoch=next_epoch_index, seed=config.seed, max_seq_length=config.max_seq_length, batch_size=config.batch_size) if eval_rows else eval_loss
        output_dir = Path(config.output_dir)
        final_manifest = self._manifest(model_identity=effective_model_identity, plan=plan, policy=selected_policy, config=config, training_plan=training_plan, global_step=current_step, epoch_index=next_epoch_index, next_micro_batch_index=next_micro_batch_index, next_sample_index=next_sample_index, next_batch_index=next_batch_index, last_loss=last_loss, eval_loss=eval_loss, preflight=preflight, processor=processor)
        final_adapter_path = self._save_final_adapter(model=model, processor=processor, output_dir=output_dir / "final_adapter")
        if final_adapter_path is not None:
            final_manifest["final_adapter"] = str(final_adapter_path.name)
        self._write_log(output_dir, {"event": "eval", "step": current_step, "epoch": next_epoch_index, "loss": last_loss, "eval_loss": eval_loss})
        self._serialize_checkpoint(target=output_dir, model=model, processor=processor, plan=plan, manifest=final_manifest, optimizer=optimizer, scheduler=scheduler, global_step=current_step, epoch_index=next_epoch_index, next_micro_batch_index=next_micro_batch_index, next_sample_index=next_sample_index, next_batch_index=next_batch_index, optimizer_updates_in_epoch=optimizer_updates_in_epoch, last_loss=last_loss, eval_loss=eval_loss)
        return TrainingResult(current_step, last_loss, eval_loss, output_dir / "training_manifest.json", plan, optimizer_stats, final_adapter_path)
