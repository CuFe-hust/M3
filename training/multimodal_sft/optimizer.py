"""Generic optimizer and scheduler construction from semantic parameters."""

from __future__ import annotations

import contextlib
import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .parameter_plan import ParameterPlan


@dataclass(frozen=True)
class OptimizerConfig:
    lora_lr: float = 1e-4
    connector_lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    mixed_precision: str = "off"


def _under(name: str, paths: Iterable[str]) -> bool:
    dotted = "." + name + "."
    return any(("." + path + ".") in dotted for path in paths)


def _no_decay(name: str) -> bool:
    leaf = name.rsplit(".", 1)[-1].lower()
    segments = {segment.lower() for segment in name.split(".")}
    return leaf == "bias" or bool(segments & {"norm", "norm1", "norm2", "layernorm", "rmsnorm"})


def parameter_name_hash(names: Iterable[str]) -> str:
    payload = "\n".join(sorted(str(name) for name in names)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_optimizer_groups(
    model: Any,
    plan: ParameterPlan,
    config: OptimizerConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build deterministic LoRA/connector decay groups without name guessing."""

    groups: dict[str, dict[str, Any]] = {}
    for name, parameter in model.named_parameters():
        if not bool(getattr(parameter, "requires_grad", False)):
            continue
        is_connector = _under(name, plan.full_train_module_paths)
        is_lora = _under(name, plan.lora_module_paths) or ("lora" in name.lower() and not is_connector)
        if not is_connector and not is_lora:
            continue
        family = "connector" if is_connector else "lora"
        decay = "no_decay" if _no_decay(name) else "decay"
        key = f"{family}_{decay}"
        group = groups.setdefault(key, {"params": [], "lr": config.connector_lr if family == "connector" else config.lora_lr, "weight_decay": 0.0 if decay == "no_decay" else config.weight_decay, "name": key})
        group["params"].append(parameter)
    ordered = [groups[key] for key in ("lora_decay", "lora_no_decay", "connector_decay", "connector_no_decay") if key in groups]
    stats = {
        "groups": [
            {"name": group["name"], "lr": group["lr"], "weight_decay": group["weight_decay"], "count": len(group["params"])}
            for group in ordered
        ],
        "lora_parameter_hash": parameter_name_hash(name for name, parameter in model.named_parameters() if bool(getattr(parameter, "requires_grad", False)) and (_under(name, plan.lora_module_paths) or "lora" in name.lower())),
        "connector_parameter_hash": parameter_name_hash(name for name, parameter in model.named_parameters() if bool(getattr(parameter, "requires_grad", False)) and _under(name, plan.full_train_module_paths)),
    }
    if not ordered:
        raise ValueError("parameter plan produced no optimizer groups")
    return ordered, stats


def build_cosine_scheduler(optimizer: Any, num_training_steps: int, warmup_ratio: float) -> Any:
    """Warmup + cosine schedule, independent of any model family."""

    warmup_steps = max(0, int(num_training_steps * warmup_ratio))

    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining = max(1, num_training_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / remaining))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    import torch

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def autocast_context(device: str, mixed_precision: str):
    if mixed_precision == "off":
        return contextlib.nullcontext()
    import torch

    dtype = torch.float16 if mixed_precision == "float16" else torch.bfloat16
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    return torch.autocast(device_type=device_type, dtype=dtype)


def clip_gradients(model: Any, max_grad_norm: float) -> float:
    import torch

    parameters = [parameter for parameter in model.parameters() if bool(getattr(parameter, "requires_grad", False)) and parameter.grad is not None]
    if not parameters:
        return 0.0
    return float(torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm).detach().cpu().item())
