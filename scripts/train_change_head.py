"""Train ChangeHead from the versioned frozen feature cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.change_head.manifest import ChangeHeadManifest  # noqa: E402
from training.change_head.evaluator import evaluate_probability_maps  # noqa: E402
from training.change_head.feature_cache import (  # noqa: E402
    CachedChangeTrainingSample,
    FeatureCache,
)
from training.change_head.trainer import (  # noqa: E402
    ChangeHeadTrainer,
    TrainingConfig,
    weighted_sample_indices,
)


_TOP_LEVEL = {
    "architecture",
    "data",
    "experts",
    "optimization",
    "loss",
    "optional_expert_dropout",
    "selection",
    "sampling",
    "early_stopping",
}
_SECTION_KEYS = {
    "architecture": {"name", "hidden_dim", "semantic_dim", "decoder_dim", "use_pif_mask", "use_rgb_pair", "optional_expert_dropout_supported"},
    "data": {"mask_frame"},
    "experts": {"expert_id", "required", "feature_stages", "use_semantic_probabilities", "missing_policy"},
    "optimization": {"epochs", "batch_size", "learning_rate", "weight_decay", "grad_clip_norm", "amp", "seed", "num_workers"},
    "loss": {"bce_weight", "dice_weight", "boundary_weight", "swap_consistency_weight", "swap_consistency_every_n_steps", "max_pos_weight"},
    "optional_expert_dropout": {"enabled", "probability"},
    "selection": {"primary_metric", "early_stop_patience"},
    "sampling": {"default_weight", "tag_multipliers", "optional_expert_dropout"},
    "early_stopping": {"patience", "metric", "mode"},
}


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("training config must be a mapping")
    unknown = set(value) - _TOP_LEVEL
    if unknown:
        raise ValueError(f"unknown training config sections: {sorted(unknown)}")
    for section, payload in value.items():
        if section == "experts":
            if not isinstance(payload, list):
                raise ValueError("experts config must be a list")
            for item in payload:
                if not isinstance(item, dict):
                    raise ValueError("expert config must be a mapping")
                extra = set(item) - _SECTION_KEYS["experts"]
                if extra:
                    raise ValueError(f"unknown experts config keys: {sorted(extra)}")
        elif not isinstance(payload, dict):
            raise ValueError(f"{section} config must be a mapping")
        else:
            extra = set(payload) - _SECTION_KEYS[section]
            if extra:
                raise ValueError(f"unknown {section} config keys: {sorted(extra)}")
    return value


def _resolve_cache_root(path: Path) -> Path:
    if (path / "index.jsonl").is_file():
        return path
    candidates = sorted(item for item in path.iterdir() if (item / "index.jsonl").is_file())
    if len(candidates) != 1:
        raise ValueError(f"cache root must contain exactly one fingerprint index: {path}")
    return candidates[0]


def _load_cache(path: Path) -> list[CachedChangeTrainingSample]:
    root = _resolve_cache_root(path)
    cache = FeatureCache(root)
    samples: list[CachedChangeTrainingSample] = []
    for line in (root / "index.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") not in {"written", "cached"}:
            continue
        sample = cache.read_sample(str(row["cache_key"]))
        if sample is None:
            raise ValueError(f"invalid cache sample: {row.get('sample_id')}")
        samples.append(sample)
    if not samples:
        raise ValueError(f"cache contains no usable samples: {path}")
    return samples


def _tensor(value: Any, *, device: str, add_channel: bool = False) -> Any:
    import torch

    array = np.asarray(value)
    tensor = torch.from_numpy(array.copy()).to(device=device, dtype=torch.float32)
    if add_channel and tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return tensor


def sample_to_batch(sample: CachedChangeTrainingSample, *, device: str) -> dict[str, Any]:
    expert_features: dict[str, tuple[list[Any], list[Any]]] = {}
    semantic_probabilities: dict[str, tuple[Any, Any]] = {}
    expert_presence: dict[str, Any] = {}
    for expert_id, expert in sample.experts.items():
        expert_features[expert_id] = (
            [_tensor(expert.first_features[stage], device=device) for stage in expert.feature_stages],
            [_tensor(expert.second_features[stage], device=device) for stage in expert.feature_stages],
        )
        if expert.first_semantic_probabilities is not None:
            semantic_probabilities[expert_id] = (
                _tensor(expert.first_semantic_probabilities, device=device),
                _tensor(expert.second_semantic_probabilities, device=device),
            )
        expert_presence[expert_id] = _tensor(np.ones((1,), dtype=np.float32), device=device)
    network_inputs: dict[str, Any] = {
        "expert_features": expert_features,
        "semantic_probabilities": semantic_probabilities,
        "expert_presence": expert_presence,
        "valid_mask": _tensor(sample.loss_valid_mask, device=device, add_channel=True),
        "pif_mask": None if sample.pif_mask is None else _tensor(sample.pif_mask, device=device),
    }
    if sample.comparison_t1 is not None:
        network_inputs["rgb_t1"] = _tensor(sample.comparison_t1, device=device)
        network_inputs["rgb_t2"] = _tensor(sample.comparison_t2, device=device)
    return {
        "network_inputs": network_inputs,
        "target_mask": _tensor(sample.target_change_mask, device=device, add_channel=True),
        "loss_valid_mask": _tensor(sample.loss_valid_mask, device=device, add_channel=True),
        "sample_id": sample.sample_id,
    }


def _signature(sample: CachedChangeTrainingSample) -> tuple[Any, ...]:
    return (
        tuple(sorted(sample.experts)),
        tuple(
            (expert_id, tuple((stage, tuple(expert.first_features[stage].shape)) for stage in expert.feature_stages))
            for expert_id, expert in sorted(sample.experts.items())
        ),
        tuple(np.asarray(sample.target_change_mask).shape),
    )


def _batches(
    samples: list[CachedChangeTrainingSample],
    indices: Iterable[int],
    *,
    batch_size: int,
    device: str,
) -> list[dict[str, Any]]:
    # Keep shape-compatible cache samples together.  A batch is otherwise
    # reduced to singleton samples rather than silently resizing feature grids.
    grouped: dict[tuple[Any, ...], list[CachedChangeTrainingSample]] = {}
    for index in indices:
        sample = samples[index]
        grouped.setdefault(_signature(sample), []).append(sample)
    batches: list[dict[str, Any]] = []
    for group in grouped.values():
        for start in range(0, len(group), max(1, batch_size)):
            chunk = group[start : start + max(1, batch_size)]
            if len(chunk) == 1:
                batches.append(sample_to_batch(chunk[0], device=device))
                continue
            # The network accepts BCHW tensors.  Merge each nested mapping.
            converted = [sample_to_batch(sample, device=device) for sample in chunk]
            first = converted[0]
            inputs = dict(first["network_inputs"])
            inputs["valid_mask"] = _stack([item["network_inputs"]["valid_mask"] for item in converted])
            inputs["pif_mask"] = None if first["network_inputs"]["pif_mask"] is None else _stack([item["network_inputs"]["pif_mask"] for item in converted])
            inputs["expert_presence"] = {
                expert_id: _stack([item["network_inputs"]["expert_presence"][expert_id] for item in converted])
                for expert_id in first["network_inputs"]["expert_presence"]
            }
            inputs["expert_features"] = {
                expert_id: (
                    [_stack([item["network_inputs"]["expert_features"][expert_id][0][stage] for item in converted]) for stage in range(len(first["network_inputs"]["expert_features"][expert_id][0]))],
                    [_stack([item["network_inputs"]["expert_features"][expert_id][1][stage] for item in converted]) for stage in range(len(first["network_inputs"]["expert_features"][expert_id][1]))],
                )
                for expert_id in first["network_inputs"]["expert_features"]
            }
            inputs["semantic_probabilities"] = {
                expert_id: (
                    _stack([item["network_inputs"]["semantic_probabilities"][expert_id][0] for item in converted]),
                    _stack([item["network_inputs"]["semantic_probabilities"][expert_id][1] for item in converted]),
                )
                for expert_id in first["network_inputs"]["semantic_probabilities"]
            }
            batches.append({
                "network_inputs": inputs,
                "target_mask": _stack([item["target_mask"] for item in converted]),
                "loss_valid_mask": _stack([item["loss_valid_mask"] for item in converted]),
                "sample_id": [item["sample_id"] for item in converted],
            })
    return batches


def _stack(values: list[Any]) -> Any:
    import torch

    return torch.cat(values, dim=0)


def _validate_manifest_config(config: dict[str, Any], manifest: ChangeHeadManifest) -> None:
    architecture = config.get("architecture", {})
    for key in ("name", "hidden_dim", "semantic_dim", "decoder_dim", "use_pif_mask", "use_rgb_pair"):
        if key in architecture and architecture[key] != getattr(manifest.architecture, key):
            raise ValueError(f"training config architecture mismatch: {key}")
    configured_ids = {str(item["expert_id"]) for item in config.get("experts", [])}
    if configured_ids and configured_ids != {expert.expert_id for expert in manifest.experts}:
        raise ValueError("training config experts do not match head manifest")


def _config_to_training(config: dict[str, Any]) -> TrainingConfig:
    optimization = config.get("optimization", {})
    loss = config.get("loss", {})
    sampling = config.get("sampling", {})
    dropout = config.get("optional_expert_dropout", {})
    early = config.get("early_stopping", {})
    selection = config.get("selection", {})
    sampling_dropout = sampling.get("optional_expert_dropout", {})
    probability = float(
        sampling_dropout.get("probability", dropout.get("probability", 0.0))
        if isinstance(sampling_dropout, dict)
        else dropout.get("probability", 0.0)
    )
    return TrainingConfig(
        epochs=int(optimization.get("epochs", 30)),
        batch_size=int(optimization.get("batch_size", 4)),
        learning_rate=float(optimization.get("learning_rate", 0.0003)),
        weight_decay=float(optimization.get("weight_decay", 0.0001)),
        grad_clip_norm=float(optimization.get("grad_clip_norm", 1.0)),
        amp=bool(optimization.get("amp", True)),
        seed=int(optimization.get("seed", 42)),
        num_workers=int(optimization.get("num_workers", 0)),
        bce_weight=float(loss.get("bce_weight", 1.0)),
        dice_weight=float(loss.get("dice_weight", 1.0)),
        boundary_weight=float(loss.get("boundary_weight", 0.25)),
        swap_consistency_weight=float(loss.get("swap_consistency_weight", 0.10)),
        swap_consistency_every_n_steps=int(loss.get("swap_consistency_every_n_steps", 1)),
        max_pos_weight=float(loss.get("max_pos_weight", 8.0)),
        optional_expert_dropout=probability if bool(dropout.get("enabled", True)) else 0.0,
        tag_multipliers={str(k): float(v) for k, v in sampling.get("tag_multipliers", {}).items()},
        early_stopping_patience=int(early.get("patience", selection.get("early_stop_patience", 6))),
        early_stopping_metric=str(early.get("metric", selection.get("primary_metric", "val_pixel_f1"))),
        early_stopping_mode=str(early.get("mode", "max")),
    )


def _save_safetensors(network: Any, path: Path) -> str:
    import hashlib

    try:
        from safetensors.torch import save_file
    except ImportError as error:
        raise RuntimeError("safetensors dependency is required for production checkpoint output") from error
    import torch

    state = {key: value.detach().to("cpu").contiguous() for key, value in network.state_dict().items()}
    save_file(state, str(path))
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evaluate(network: Any, samples: list[CachedChangeTrainingSample], *, device: str) -> tuple[dict[str, float], list[np.ndarray], list[np.ndarray]]:
    import torch

    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    network.eval()
    with torch.no_grad():
        for sample in samples:
            batch = sample_to_batch(sample, device=device)
            logits = network(**batch["network_inputs"])
            probability = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)
            probabilities.append(probability)
            targets.append(np.asarray(sample.target_change_mask, dtype=np.float32))
            valid_masks.append(np.asarray(sample.loss_valid_mask, dtype=bool))
    return evaluate_probability_maps(probabilities, targets, valid_masks), probabilities, targets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    config = _load_yaml(args.config)
    manifest = ChangeHeadManifest.model_validate(json.loads(args.manifest.read_text(encoding="utf-8")))
    _validate_manifest_config(config, manifest)
    training_config = _config_to_training(config)
    train_samples = _load_cache(args.train_cache)
    val_samples = _load_cache(args.val_cache)
    try:
        import torch
        from models.change_head.network import MultiExpertSiameseChangeHead
    except ImportError as error:
        raise RuntimeError("torch dependency is required for ChangeHead training") from error
    network = MultiExpertSiameseChangeHead(manifest).to(args.device)
    trainer = ChangeHeadTrainer(network, config=training_config)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "best").mkdir(exist_ok=True)
    (output_dir / "last").mkdir(exist_ok=True)
    try:
        import yaml

        (output_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
    except ImportError:
        (output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    log_path = output_dir / "train_log.jsonl"
    best_metric = float("-inf") if training_config.early_stopping_mode == "max" else float("inf")
    best_epoch = 0
    no_improvement = 0
    log_rows: list[dict[str, Any]] = []
    for epoch in range(1, training_config.epochs + 1):
        order = weighted_sample_indices(
            train_samples,
            tag_multipliers=training_config.tag_multipliers,
            seed=training_config.seed + epoch,
        )
        train_loss = trainer.train_epoch(
            _batches(train_samples, order, batch_size=training_config.batch_size, device=args.device)
        )
        metrics, probabilities, targets = _evaluate(network, val_samples, device=args.device)
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{key}": value for key, value in metrics.items()}}
        log_rows.append(row)
        log_path.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in log_rows), encoding="utf-8"
        )
        metric = float(row.get(training_config.early_stopping_metric, metrics.get("pixel_f1", 0.0)))
        improved = metric > best_metric if training_config.early_stopping_mode == "max" else metric < best_metric
        _save_safetensors(network, output_dir / "last" / "model.safetensors")
        if improved:
            best_metric = metric
            best_epoch = epoch
            no_improvement = 0
            weights_sha256 = _save_safetensors(
                network, output_dir / "best" / "model.safetensors"
            )
            locked_manifest = manifest.model_copy(
                update={"model_weights_sha256": weights_sha256}
            )
            (output_dir / "best" / "manifest.json").write_text(
                json.dumps(locked_manifest.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8"
            )
            (output_dir / "best" / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        else:
            no_improvement += 1
        if no_improvement >= training_config.early_stopping_patience:
            break
    np.save(output_dir / "val_logits.npy", np.asarray(probabilities, dtype=object), allow_pickle=True)
    np.save(output_dir / "val_targets.npy", np.asarray(targets, dtype=object), allow_pickle=True)
    summary = {
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "epochs_completed": len(log_rows),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "manifest_contract_version": manifest.input_contract_version,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
