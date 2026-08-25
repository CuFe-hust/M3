"""Qwen-family processor IO kept behind the adapter boundary.

The Phase 2 data module remains a compatibility surface for existing callers;
this module is the adapter-owned entry point used by generic SFT.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..contracts import AdapterContractError, CanonicalEpisode


def _legacy() -> Any:
    from scripts import qwen3vl_phase2_data

    return qwen3vl_phase2_data


def _episode_messages(episode: Any) -> list[dict[str, Any]]:
    return [dict(message) for message in episode.messages]


def encode_multimodal_episode(
    processor: Any,
    episode: Any,
    *,
    max_seq_length: int,
    return_tensors: str = "pt",
) -> dict[str, Any]:
    """Encode one prepared episode with assistant-only labels."""

    if return_tensors != "pt":
        raise AdapterContractError("Qwen multimodal encoding currently requires return_tensors='pt'")
    legacy = _legacy()
    return legacy.encode_multimodal_episode(
        processor,
        list(episode.images),
        _episode_messages(episode),
        int(max_seq_length),
        str(getattr(episode, "episode_id", "episode")),
    )


def collate(encoded_examples: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Right-pad text/labels and concatenate visual tensors."""

    return _legacy().Phase2DataCollator()(list(encoded_examples))


def validate_image_placeholder_gate(messages: list[dict[str, Any]], image_count: int) -> None:
    return _legacy()._validate_image_placeholder_gate(messages, image_count)


def assistant_mask(processor: Any, messages: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    return _legacy()._assistant_mask(processor, messages)


def align_labels(text_ids: Sequence[int], mask: Sequence[int], input_ids: Any) -> Any:
    return _legacy()._align_labels(text_ids, mask, input_ids)


def truncate_feature(
    processor: Any,
    images: Sequence[Any],
    messages: list[dict[str, Any]],
    episode_id: str,
    max_seq_length: int,
    delta: int,
) -> dict[str, Any]:
    return _legacy()._truncate_feature(
        processor, images, messages, episode_id, max_seq_length, delta
    )


def processor_identity(processor: Any) -> dict[str, Any]:
    """Return stable processor/chat-template identity for manifests/resume."""

    template = getattr(processor, "chat_template", None)
    if template is None:
        tokenizer = getattr(processor, "tokenizer", None)
        template = getattr(tokenizer, "chat_template", None)
    if template is None:
        template = ""
    canonical = json.dumps(template, ensure_ascii=False, sort_keys=True, default=str)
    return {
        "class": f"{type(processor).__module__}.{type(processor).__name__}",
        "chat_template_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "encoding_contract_version": "qwen_multimodal_v1",
    }


def probe_processor(processor: Any) -> tuple[set[str], list[str], dict[str, Any]]:
    """Prove capabilities with a real synthetic two-image processor call."""

    capabilities: set[str] = set()
    missing: list[str] = []
    details = processor_identity(processor)
    if not callable(getattr(processor, "apply_chat_template", None)):
        missing.append("chat_template")
    if not callable(processor):
        missing.append("processor_call")
    if missing:
        return capabilities, missing, details
    try:
        from PIL import Image

        images = [Image.new("RGB", (8, 8), (255, 0, 0)), Image.new("RGB", (8, 8), (0, 255, 0))]
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "probe"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        ]
        feature = _legacy().encode_multimodal_episode(
            processor, images, messages, 512, "processor-probe"
        )
        labels = feature["labels"]
        if not bool((labels != _legacy().IGNORE_INDEX).any()):
            raise AdapterContractError("processor probe produced no assistant labels")
        capabilities.update({"ordered_multi_image", "assistant_mask", "forward_labels"})
        details.update({
            "probe_image_count": 2,
            "probe_supervised_tokens": int((labels != _legacy().IGNORE_INDEX).sum().item()),
            "probe_input_length": int(feature["input_ids"].shape[0]),
        })
    except Exception as exc:  # noqa: BLE001 - probe must fail closed
        missing.extend(["ordered_multi_image", "assistant_mask", "forward_labels"])
        details["probe_error"] = f"{type(exc).__name__}: {exc}"
    return capabilities, sorted(set(missing)), details


def synthetic_verification_episode() -> CanonicalEpisode:
    """Return the smallest ordered two-image episode for export verification."""

    from PIL import Image

    return CanonicalEpisode(
        task_profile="export_verification",
        messages=(
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": "Describe the ordered image pair."},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "two images"}]},
        ),
        images=(Image.new("RGB", (16, 16), (255, 0, 0)), Image.new("RGB", (16, 16), (0, 255, 0))),
        target_schema="export_verification",
    )


def _load_change_fixture(path: str | Path) -> CanonicalEpisode:
    """Load one already-rendered canonical episode and its local images."""

    root = Path(path)
    if root.is_dir():
        candidates = sorted(root.glob("*.jsonl")) or sorted(root.glob("*.json"))
        if not candidates:
            raise AdapterContractError("EXPORT_CHANGE_FIXTURE_NOT_FOUND")
        root = candidates[0]
    try:
        text = root.read_text(encoding="utf-8")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            lines = [line for line in text.splitlines() if line.strip()]
            payload = json.loads(lines[0])
    except (OSError, IndexError, json.JSONDecodeError) as exc:
        raise AdapterContractError("EXPORT_CHANGE_FIXTURE_INVALID") from exc
    messages = payload.get("messages")
    image_items = payload.get("images")
    if not isinstance(messages, list) or not isinstance(image_items, list):
        raise AdapterContractError("EXPORT_CHANGE_FIXTURE_NOT_RENDERED")
    from PIL import Image

    images = []
    for item in image_items:
        image_path = item.get("path") if isinstance(item, Mapping) else item
        candidate = Path(str(image_path))
        if not candidate.is_absolute():
            candidate = root.parent / candidate
        if not candidate.is_file():
            raise AdapterContractError(f"EXPORT_CHANGE_FIXTURE_IMAGE_MISSING: {candidate}")
        images.append(Image.open(candidate).convert("RGB"))
    return CanonicalEpisode(
        task_profile=str(payload.get("task_profile", "change_agent")),
        messages=tuple(dict(message) for message in messages),
        images=tuple(images),
        target_schema=str(payload.get("target_schema", "ChangeInitialResult")),
        metadata=dict(payload.get("metadata", {})),
    )


def _forward_one(adapter: Any, model: Any, processor: Any, episode: CanonicalEpisode) -> dict[str, Any]:
    import torch

    encoded = adapter.encode(processor, episode, max_seq_length=512, return_tensors="pt")
    batch, _metadata = adapter.collate([encoded])
    prepared = adapter.prepare_forward_inputs(batch)
    model.eval()
    with torch.no_grad():
        output = model(**dict(prepared))
    loss = getattr(output, "loss", None)
    if loss is not None:
        finite = bool(torch.isfinite(loss).all().item())
    else:
        logits = output.get("logits") if isinstance(output, Mapping) else getattr(output, "logits", None)
        finite = logits is not None and bool(torch.isfinite(logits).all().item())
    if not finite:
        raise AdapterContractError("EXPORT_FORWARD_NONFINITE")
    return {"status": "PASS", "loss_finite": bool(finite)}


def verify_export_forward(
    adapter: Any,
    model: Any,
    processor: Any,
    *,
    change_fixture: str | Path | None = None,
) -> dict[str, Any]:
    synthetic = _forward_one(adapter, model, processor, synthetic_verification_episode())
    result: dict[str, Any] = {"synthetic_two_image_forward": "PASS"}
    if change_fixture is not None:
        _forward_one(adapter, model, processor, _load_change_fixture(change_fixture))
        result["change_fixture_forward"] = "PASS"
    return result


def __getattr__(name: str) -> Any:
    if name == "Phase2DataCollator":
        value = _legacy().Phase2DataCollator
        globals()[name] = value
        return value
    raise AttributeError(name)
