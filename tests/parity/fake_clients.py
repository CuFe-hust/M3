"""Deterministic recording clients used by the migration-only legacy harness.
迁移期旧路径 Harness 使用的确定性记录客户端。
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

from spacers_agent.evaluation import VQAAnswerJudgeResult


class RecordingFakeQwen:
    """Return stable schema payloads keyed by sample, model, and request ID.
    按样本、响应模型和请求 ID 返回稳定的 Schema 数据。
    """

    def __init__(self, run_root: Path, *, scenario: str) -> None:
        self.run_root = run_root
        self.scenario = scenario
        self.calls: list[dict[str, Any]] = []

    async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
        model_name = response_model.__name__
        key = f"{request_meta.sample_id}|{model_name}|{request_meta.request_id}"
        normalized_messages = normalize_messages(messages)
        self.calls.append(
            {
                "match_key": key,
                "response_model": model_name,
                "request_id": request_meta.request_id,
                "request_hash": request_meta.request_hash,
                "prompt_version": request_meta.prompt_version,
                "sample_id": request_meta.sample_id,
                "image_sha256": request_meta.image_sha256,
                "messages_sha256": stable_hash(normalized_messages),
                "image_inputs": collect_image_inputs(normalized_messages),
                "artifact_dir": relative_artifact_dir(request_meta.artifact_dir, self.run_root),
            }
        )
        if request_meta.sample_id == "primary_qwen_failure" and model_name == "AgentResult":
            raise RuntimeError("deterministic primary Qwen failure")
        if request_meta.sample_id == "counting_one_failed_tile" and request_meta.tile_id == "r000_c000":
            raise RuntimeError("deterministic tile failure")
        payload = response_payload(
            sample_id=str(request_meta.sample_id or ""),
            model_name=model_name,
            request_id=str(request_meta.request_id),
            tile_id=request_meta.tile_id,
            messages=messages,
        )
        return response_model.model_validate(payload)


class RecordingFakeDeepSeek:
    """Record the text-only Judge payload and return a stable verdict.
    记录纯文本 Judge 载荷并返回稳定结论。
    """

    judge_prompt = "deterministic text-only VQA judge"

    def __init__(self, run_root: Path, *, scenario: str) -> None:
        self.run_root = run_root
        self.scenario = scenario
        self.calls: list[dict[str, Any]] = []

    async def judge_json(self, payload, *, response_model, request_meta):
        key = f"{request_meta.sample_id}|{response_model.__name__}|{request_meta.request_id}"
        self.calls.append(
            {
                "match_key": key,
                "response_model": response_model.__name__,
                "request_id": request_meta.request_id,
                "request_hash": request_meta.request_hash,
                "prompt_version": request_meta.prompt_version,
                "sample_id": request_meta.sample_id,
                "image_sha256": request_meta.image_sha256,
                "payload": payload,
                "payload_sha256": stable_hash(payload),
                "artifact_dir": relative_artifact_dir(request_meta.artifact_dir, self.run_root),
            }
        )
        if request_meta.sample_id == "judge_failure":
            raise RuntimeError("deterministic Judge failure")
        return response_model.model_validate(
            VQAAnswerJudgeResult(score=1, concise_rationale="Deterministic text-only match.").model_dump()
        )


def normalize_messages(messages: Any) -> Any:
    """Replace every embedded image with a digest and byte length.
    将每个内嵌图像替换为摘要和字节长度。
    """

    if isinstance(messages, list):
        return [normalize_messages(item) for item in messages]
    if isinstance(messages, dict):
        if messages.get("type") == "image_url":
            image = messages.get("image_url", {})
            url = image.get("url", "") if isinstance(image, dict) else ""
            if isinstance(url, str) and url.startswith("data:") and "," in url:
                encoded = url.split(",", 1)[1]
                data = base64.b64decode(encoded)
                return {
                    "type": "image_url",
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "byte_length": len(data),
                }
        return {str(key): normalize_messages(item) for key, item in messages.items()}
    return messages


def collect_image_inputs(messages: Any) -> list[dict[str, Any]]:
    """Collect only normalized image descriptors from recorded messages.
    仅从已规范化消息中收集图像描述符。
    """

    collected: list[dict[str, Any]] = []
    if isinstance(messages, list):
        for item in messages:
            collected.extend(collect_image_inputs(item))
    elif isinstance(messages, dict):
        if messages.get("type") == "image_url" and "sha256" in messages:
            collected.append(dict(messages))
        else:
            for item in messages.values():
                collected.extend(collect_image_inputs(item))
    return collected


def response_payload(
    *,
    sample_id: str,
    model_name: str,
    request_id: str,
    tile_id: str | None,
    messages: Any,
) -> dict[str, Any]:
    """Resolve a fake response without relying on call order.
    在不依赖调用顺序的前提下解析 Fake 响应。
    """

    if model_name == "CountTargetSpec":
        return {
            "canonical_label": "building",
            "aliases": ["buildings"],
            "required_attributes": ["visible instance"],
            "excluded_attributes": ["shadow"],
            "inclusion_rule": "Count each distinct visible building once.",
            "exclusion_rule": "Exclude shadows and duplicates.",
        }
    if model_name == "TileCountResponse":
        resolved_tile = str(tile_id or request_id.rsplit(":", 1)[-1])
        return {
            "target": "building",
            "tile_id": resolved_tile,
            "points": [
                {
                    "local_id": "p1",
                    "x": 500,
                    "y": 500,
                    "confidence": 0.9,
                    "radius": 10,
                    "touches_crop_border": False,
                    "short_evidence": "deterministic visible building",
                }
            ],
            "reported_count": 1,
            "needs_split": False,
            "uncertainty": [],
        }
    if model_name == "_CountProposalResult":
        return {
            "agent_name": "general_vqa_agent",
            "answer": "2",
            "boxes": [[100, 100, 220, 220]],
            "evidence": ["one proposal box intentionally omitted"],
            "status": "completed",
        }
    if model_name == "SpatialCandidateReviewResult":
        return {
            "boxes": [
                ["small-vehicle", 100, 100, 220, 220],
                ["large-vehicle", 500, 500, 650, 650],
            ],
            "complete": True,
        }
    if model_name != "AgentResult":
        raise KeyError(f"No deterministic response for {sample_id}|{model_name}|{request_id}")
    if request_id.endswith(":count-localizer"):
        return expert_payload(
            "counting_agent",
            "2",
            evidence_items=[
                visual_box("large-vehicle", [100, 100, 220, 220]),
                visual_box("large-vehicle", [500, 500, 620, 620]),
            ],
        )
    if request_id.endswith(":spatial-candidate-review"):
        return expert_payload(
            "spatial_agent",
            "small vehicle",
            evidence_items=[
                visual_box("small-vehicle", [100, 100, 220, 220]),
                visual_box("large-vehicle", [500, 500, 650, 650]),
            ],
            geometry={"review": "complete"},
        )
    if sample_id == "partial_expert":
        return expert_payload("general_vqa_agent", "partial answer", status="partial")
    if sample_id == "failed_expert":
        return expert_payload("general_vqa_agent", "failed answer", status="failed")
    if "spatial_agent" in request_id:
        if sample_id == "vrsbench_extreme_category":
            return expert_payload("spatial_agent", "partial", status="partial")
        return expert_payload(
            "spatial_agent",
            "top-left",
            evidence_items=[visual_box("position-target", [100, 100, 240, 240])],
            geometry={"source": "deterministic spatial evidence"},
        )
    if "grounding_agent" in request_id:
        return expert_payload(
            "grounding_agent",
            "located",
            evidence_items=[visual_box("target", [200, 200, 400, 400])],
        )
    if "change_agent" in request_id:
        return expert_payload("change_agent", "a building appeared", geometry={"temporal_order": ["t1", "t2"]})
    if "caption_agent" in request_id:
        return expert_payload("caption_agent", "a concise remote-sensing caption")
    return expert_payload("general_vqa_agent", "yes")


def expert_payload(
    expert: str,
    answer: str,
    *,
    evidence_items: list[dict[str, Any]] | None = None,
    geometry: dict[str, Any] | None = None,
    status: str = "completed",
) -> dict[str, Any]:
    """Build one deterministic AgentResult payload.
    构建一条确定性的 AgentResult 载荷。
    """

    items = evidence_items or []
    return {
        "agent_name": expert,
        "answer": answer,
        "boxes": [item["box"] for item in items if item.get("box") is not None],
        "evidence": ["deterministic evidence"] if items else [],
        "evidence_items": items,
        "geometry": geometry or {},
        "status": status,
    }


def visual_box(label: str, box: list[int]) -> dict[str, Any]:
    """Create one normalized box observation.
    创建一条规范化框观测。
    """

    return {"label": label, "box": box, "confidence": 0.9, "image_id": "legacy-image"}


def stable_hash(value: Any) -> str:
    """Hash JSON-compatible data with a stable encoding.
    使用稳定编码计算 JSON 兼容数据的哈希。
    """

    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def relative_artifact_dir(value: Path | None, run_root: Path) -> str | None:
    """Record an artifact directory relative to the run root.
    记录相对于运行根目录的产物目录。
    """

    if value is None:
        return None
    try:
        return value.resolve().relative_to(run_root.resolve()).as_posix()
    except ValueError:
        return value.as_posix()
