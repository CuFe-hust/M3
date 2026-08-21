"""Masked, numerically stable ChangeHead losses."""

from __future__ import annotations

from typing import Any


def _torch():
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as error:
        raise RuntimeError("torch dependency missing") from error
    return torch, functional


def masked_bce_with_logits(
    logits: Any,
    target: Any,
    valid: Any,
    *,
    pos_weight: float = 1.0,
) -> Any:
    torch, functional = _torch()
    loss = functional.binary_cross_entropy_with_logits(
        logits,
        target.float(),
        pos_weight=torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device),
        reduction="none",
    )
    mask = valid.bool()
    return (loss * mask).sum() / mask.float().sum().clamp_min(1.0)


def soft_dice_loss(logits: Any, target: Any, valid: Any, *, epsilon: float = 1e-6) -> Any:
    torch, _ = _torch()
    probability = torch.sigmoid(logits)
    mask = valid.float()
    intersection = (probability * target.float() * mask).sum()
    denominator = (probability * mask).sum() + (target.float() * mask).sum()
    return 1.0 - (2.0 * intersection + epsilon) / (denominator + epsilon)


def boundary_loss(logits: Any, target: Any, valid: Any) -> Any:
    torch, functional = _torch()
    target_float = target.float()
    dilated = functional.max_pool2d(target_float, kernel_size=3, stride=1, padding=1)
    eroded = -functional.max_pool2d(-target_float, kernel_size=3, stride=1, padding=1)
    boundary = (dilated - eroded).clamp(0.0, 1.0)
    weighted_valid = valid.bool() & boundary.bool()
    if not bool(weighted_valid.any()):
        return logits.sum() * 0.0
    return masked_bce_with_logits(logits, target, weighted_valid)


def swap_consistency_loss(logits_ab: Any, logits_ba: Any, valid: Any) -> Any:
    torch, _ = _torch()
    mask = valid.bool()
    if not bool(mask.any()):
        return logits_ab.sum() * 0.0
    return ((torch.sigmoid(logits_ab) - torch.sigmoid(logits_ba)).square() * mask).sum() / mask.float().sum()


def change_head_loss(
    logits: Any,
    target: Any,
    valid: Any,
    *,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    boundary_weight: float = 0.25,
    pos_weight: float = 1.0,
    swap_loss: Any | None = None,
    swap_weight: float = 0.10,
) -> Any:
    total = (
        bce_weight * masked_bce_with_logits(logits, target, valid, pos_weight=pos_weight)
        + dice_weight * soft_dice_loss(logits, target, valid)
        + boundary_weight * boundary_loss(logits, target, valid)
    )
    if swap_loss is not None:
        total = total + swap_weight * swap_loss
    return total

