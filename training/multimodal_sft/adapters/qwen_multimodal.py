"""Qwen-family processor IO kept behind the adapter boundary.

The Phase 2 data module remains a compatibility surface for existing callers;
this module is the adapter-owned entry point used by generic SFT.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from ..contracts import AdapterContractError


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
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "probe"}]},
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


def __getattr__(name: str) -> Any:
    if name == "Phase2DataCollator":
        value = _legacy().Phase2DataCollator
        globals()[name] = value
        return value
    raise AttributeError(name)
