#!/usr/bin/env python3
"""Refine visual-planner SFT labels with a text-only DeepSeek teacher.
使用纯文本 DeepSeek teacher 复标 Visual Planner SFT 数据。

The source dataset is always read-only. Live calls require ``--use-api`` and an API key supplied
through an environment variable or a no-echo terminal prompt. Every request contains only the
raw question; images, old targets, answers, provenance, and paths are excluded.
源数据始终保持只读。只有显式 ``--use-api``
且通过环境变量或无回显终端提示提供 API key 时才联网。
每个请求只包含原始问题，不包含图像、旧 target、答案、provenance 或路径。
"""

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agents.evidence_catalog import CatalogCategoryError, EvidenceCatalog
from agents.general_vqa.evidence.rendering import preview_from_path
from agents.schema import COUNTING_TASKS, VisualTaskPlan
from application.bootstrap import assemble_runtime
from application.settings import load_settings
from data.schema import TaskName
from evaluation.judges.base import build_judge_request_hash
from evaluation.judges.deepseek import (
    DeepSeekJudgeClient,
    DeepSeekJudgeError,
    JudgeTransportError,
)
from models.base import ModelCacheIdentity, RequestMeta
from models.cache import JsonResponseCache
from models.settings import DeepSeekSettings

RULE_VERSION = "visual-planner-text-refinement-v6.1"
TEACHER_PROMPT_VERSION = "deepseek-visual-planner-text-v6.1"
TRAINING_FORMAT = "visual-planner-compiled-chat-v1"
_USER_APPROVED_TARGET_FIELDS = frozenset(
    {"task", "needs_visual_assistance", "object_categories", "count_target"}
)
_SCENE_EVIDENCE_PROFILE = (
    "developed-space",
    "building",
    "road",
    "tree",
    "agriculture-land",
    "bareland",
    "rangeland",
    "water",
)
_TEXT_CATEGORY_ALIASES: dict[str, tuple[str, ...]] = {
    "river": ("water",),
    "ocean": ("water",),
    "lake": ("water",),
    "pond": ("water",),
    "body of water": ("water",),
    "forest": ("tree",),
    "woodland": ("tree",),
    "farmland": ("agriculture-land",),
    "cropland": ("agriculture-land",),
    "farm": ("agriculture-land",),
    "highway": ("road",),
    "street": ("road",),
    "terminal": ("airport",),
    "runway": ("airport",),
    "taxiway": ("airport",),
    "apron": ("airport",),
    "boarding bridge": ("airport",),
    "baseball field": ("baseball-diamond",),
    "track and field": ("ground-track-field",),
    "truck": ("large-vehicle",),
    "town": ("developed-space", "building", "road"),
    "city": ("developed-space", "building", "road"),
    "residential area": ("developed-space", "building", "road"),
    "industrial area": ("developed-space", "building", "road"),
    "court": (
        "basketball-court",
        "tennis-court",
    ),
}
_PILOT_TASK_QUOTAS = {
    "counting": 80,
    "general_vqa": 160,
    "scene_classification": 30,
    "spatial_relation": 30,
}
_STABLE_MERGE_ERRORS = frozenset({"COUNT_TARGET_MISSING_AFTER_TASK_GUARD"})
_ANSWER_LEAKAGE_MARKERS = (
    "identify the object category",
    "identify the category",
    "what kind of object",
    "what type of object",
    "which category",
    "what category",
    "category does",
    "classify the object",
    "name the object",
    "recognize the object",
    "object belongs to",
)
_ANSWER_LEAKAGE_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\b(?:identify|determine|select|choose)\b.{0,60}\b(?:category|class|type|kind)\b",
        r"\b(?:what|which)\b.{0,30}\b(?:category|class|type|kind)\b",
        r"\b(?:name|recognize)\b.{0,30}\b(?:object|feature|target)\b",
        r"\bwhat is (?:the )?(?:object|feature|target)\b",
    )
)


class TextPlanProposal(BaseModel):
    """Only the four user-approved fields judged by DeepSeek.
    仅包含由 DeepSeek 判断的四个用户授权字段。"""

    model_config = ConfigDict(extra="forbid")

    task: TaskName
    needs_visual_assistance: bool
    object_categories: list[str] = Field(default_factory=list, max_length=16)
    count_target: str | None = None

    @model_validator(mode="after")
    def validate_linkage(self) -> "TextPlanProposal":
        if self.needs_visual_assistance != bool(self.object_categories):
            raise ValueError("assistance and categories must be linked")
        if self.task in COUNTING_TASKS:
            if not isinstance(self.count_target, str) or not self.count_target.strip():
                raise ValueError("counting tasks require count_target")
            self.count_target = self.count_target.strip()
        elif self.count_target is not None:
            raise ValueError("non-counting tasks require null count_target")
        return self


@dataclass(frozen=True)
class NormalizedProposal:
    task: TaskName
    needs_visual_assistance: bool
    object_categories: tuple[str, ...]
    count_target: str | None
    decision_code: str


@dataclass(frozen=True)
class Episode:
    relative_file: str
    line_number: int
    record: dict[str, Any]
    question: str
    image: str


@dataclass(frozen=True)
class AnnotationResult:
    proposal: TextPlanProposal | None
    request_hash: str
    cache_hit: bool
    error_code: str | None = None


@dataclass(frozen=True)
class RuntimeProtocol:
    protocol_id: str
    document: dict[str, Any]
    executable_by_task: dict[str, tuple[str, ...]]


class _NoopVisionLanguageClient:
    """Assembly-only client; dataset refinement never invokes Qwen.
    仅用于 runtime 组装；数据复标绝不调用 Qwen。"""

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        """Declared offline identity required by the composition seam; it is
        never hashed against a real request. 组合 seam 要求的声明式离线身份；
        绝不用于真实请求哈希。"""

        return ModelCacheIdentity(
            model="offline-assembly-noop",
            generation={},
            client_version="noop-v1",
        )

    async def complete_json(self, **_: Any) -> Any:
        raise AssertionError("dataset refinement must not call Qwen")


def _urllib_label_transport(
    *,
    model: str,
    messages: list[dict[str, str]],
    api_key: str,
    base_url: str,
    timeout_seconds: int,
) -> str:
    """Call DeepSeek in non-thinking JSON mode for this labeling job only.
    仅为本复标任务以 non-thinking JSON 模式调用 DeepSeek。"""

    import urllib.error
    import urllib.request

    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 2048,
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    # Python.org macOS builds may not inherit the system CA bundle used by curl.
    # Python.org 的 macOS 构建可能不会继承 curl 使用的系统 CA bundle。
    ssl_context = ssl.create_default_context()
    for ca_bundle in (
        Path("/etc/ssl/cert.pem"),
        Path("/etc/ssl/certs/ca-certificates.crt"),
    ):
        if ca_bundle.is_file():
            ssl_context.load_verify_locations(cafile=str(ca_bundle))
            break
    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout_seconds,
            context=ssl_context,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise JudgeTransportError(
            "label transport http error", status_code=error.code
        ) from error
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        raise JudgeTransportError("label transport connection error") from error
    except json.JSONDecodeError as error:
        raise JudgeTransportError("label transport invalid response body") from error
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise JudgeTransportError("label transport missing assistant content") from error
    if not isinstance(content, str) or not content.strip():
        raise JudgeTransportError("label transport empty content")
    return content


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _plain_relative_path(value: str, *, where: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{where}:UNSAFE_PATH")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ValueError(f"{where}:UNSAFE_PATH")
    if any(part in {"", ".", ".."} for part in posix.parts):
        raise ValueError(f"{where}:UNSAFE_PATH")
    return posix


def _contained(root: Path, relative: str, *, where: str) -> Path:
    rel = _plain_relative_path(relative, where=where)
    candidate = (root / Path(*rel.parts)).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"{where}:UNSAFE_PATH")
    return candidate


def question_from_record(record: Mapping[str, Any]) -> str:
    """Return the raw question after validating one-or-more ordered images.
    校验一张或多张有序图像后返回原始问题。"""

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError("INVALID_MESSAGES")
    system, user = messages
    if system.get("role") != "system" or set(system) != {"role", "content_ref"}:
        raise ValueError("INVALID_SYSTEM_REFERENCE")
    content = user.get("content")
    if user.get("role") != "user" or not isinstance(content, list) or len(content) < 2:
        raise ValueError("INVALID_USER_CONTENT")
    image_blocks, text_block = content[:-1], content[-1]
    for image_block in image_blocks:
        if set(image_block) != {"type", "image"} or image_block.get("type") != "image":
            raise ValueError("INVALID_IMAGE_BLOCK")
    if set(text_block) != {"type", "text"} or text_block.get("type") != "text":
        raise ValueError("INVALID_TEXT_BLOCK")
    question = text_block.get("text")
    if not isinstance(question, str):
        raise ValueError("INVALID_QUESTION")
    return question


def image_paths_from_record(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Return validated ordered image references from the user message.
    返回 user 消息中经过校验的有序图像引用。"""

    question_from_record(record)
    content = record["messages"][1]["content"]
    return tuple(block["image"] for block in content[:-1])


def compile_training_messages(
    record: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    image_override: str | Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Resolve content_ref into inference-identical input plus assistant label.
    将 content_ref 展开为与推理一致的输入，并追加 assistant 标签。"""

    question = question_from_record(record)
    if image_override is None:
        images = image_paths_from_record(record)
    elif isinstance(image_override, str):
        images = (image_override,)
    else:
        images = tuple(image_override)
    if not images or any(not isinstance(image, str) or not image for image in images):
        raise ValueError("INVALID_IMAGE_BLOCK")
    target_text = record.get("target_text")
    system_prompt = protocol.get("system_prompt")
    if not isinstance(system_prompt, str) or not system_prompt:
        raise ValueError("INVALID_PROTOCOL_PROMPT")
    if not isinstance(target_text, str) or not target_text:
        raise ValueError("INVALID_TARGET_TEXT")
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                *({"type": "image", "image": image} for image in images),
                {"type": "text", "text": question},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": target_text}]},
    ]


def _materialize_training_preview(
    source_root: Path,
    output_root: Path,
    image_relative: str,
    *,
    max_side: int,
) -> str:
    source = _contained(source_root, image_relative, where="training image")
    data_url, digest = preview_from_path(source, max_side=max_side)
    prefix, separator, encoded = data_url.partition(",")
    if not separator or prefix != "data:image/png;base64":
        raise ValueError("INVALID_PREVIEW_DATA_URL")
    try:
        content = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ValueError("INVALID_PREVIEW_DATA_URL") from exc
    if _sha256_bytes(content) != digest:
        raise ValueError("PREVIEW_DIGEST_MISMATCH")
    relative = f"training_images/sha256/{digest}.png"
    target = _contained(output_root, relative, where="training preview")
    if target.is_file():
        if _sha256_file(target) != digest:
            raise ValueError("EXISTING_PREVIEW_DIGEST_MISMATCH")
    else:
        _atomic_write_bytes(target, content)
    return relative


def audit_source(source_root: Path) -> tuple[list[Episode], dict[str, Any]]:
    source = source_root.resolve()
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes: list[Episode] = []
    episode_ids: set[str] = set()
    protocol_ids: Counter[str] = Counter()
    image_paths: set[str] = set()
    task_counts: Counter[str] = Counter()
    dataset_root = source / "datasets"
    expected_files = {
        f"datasets/{group}/{filename}"
        for group, entry in manifest["datasets"].items()
        for filename in entry["files"]
    }
    actual_paths = sorted(dataset_root.glob("*/*.jsonl"))
    actual_files = {path.relative_to(source).as_posix() for path in actual_paths}
    if actual_files != expected_files:
        raise ValueError("MANIFEST_DATASET_FILE_SET_MISMATCH")
    for path in actual_paths:
        relative = path.relative_to(source).as_posix()
        expected = manifest["datasets"][path.parent.name]["files"][path.name]
        if path.stat().st_size != expected["bytes"] or _sha256_file(path) != expected["sha256"]:
            raise ValueError(f"MANIFEST_FILE_MISMATCH:{relative}")
        file_examples = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                file_examples += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"INVALID_JSONL:{relative}:{line_number}") from exc
                episode_id = record.get("episode_id")
                if not isinstance(episode_id, str) or not episode_id or episode_id in episode_ids:
                    raise ValueError("INVALID_OR_DUPLICATE_EPISODE_ID")
                episode_ids.add(episode_id)
                if record.get("source_group") != path.parent.name or record.get("split") != path.stem:
                    raise ValueError(f"SOURCE_OR_SPLIT_MISMATCH:{episode_id}")
                question = question_from_record(record)
                images = image_paths_from_record(record)
                image = record.get("image")
                if image != images[0]:
                    raise ValueError(f"IMAGE_REFERENCE_MISMATCH:{episode_id}")
                for image_reference in images:
                    image_path = _contained(source, image_reference, where="image")
                    if not image_path.is_file():
                        raise ValueError(f"IMAGE_MISSING:{episode_id}")
                protocol_ref = record["messages"][0]["content_ref"]
                protocol_path = _contained(source, protocol_ref, where="protocol")
                if not protocol_path.is_file() or protocol_path.name != f"{record.get('protocol_id')}.json":
                    raise ValueError(f"PROTOCOL_REFERENCE_MISMATCH:{episode_id}")
                try:
                    target = VisualTaskPlan.model_validate(record.get("target"))
                except ValidationError as exc:
                    raise ValueError(f"TARGET_SCHEMA_INVALID:{episode_id}") from exc
                if record.get("target_text") != _compact_json(record.get("target")):
                    raise ValueError(f"TARGET_TEXT_MISMATCH:{episode_id}")
                episodes.append(Episode(relative, line_number, record, question, image))
                protocol_ids[record["protocol_id"]] += 1
                image_paths.update(images)
                task_counts[target.task] += 1
        if file_examples != expected["examples"]:
            raise ValueError(f"MANIFEST_EXAMPLE_COUNT_MISMATCH:{relative}")
    if sum(protocol_ids.values()) != len(episodes):
        raise ValueError("AUDIT_COUNT_MISMATCH")
    if len(episodes) != sum(
        int(entry["examples"]) for entry in manifest["datasets"].values()
    ):
        raise ValueError("MANIFEST_TOTAL_COUNT_MISMATCH")
    for relative in sorted(image_paths):
        path = _contained(source, relative, where="image")
        name_digest = path.stem.casefold()
        if len(name_digest) == 64 and _sha256_file(path) != name_digest:
            raise ValueError("IMAGE_DIGEST_MISMATCH")
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError) as exc:
            raise ValueError("IMAGE_DECODE_FAILED") from exc
    return episodes, {
        "episodes": len(episodes),
        "unique_images": len(image_paths),
        "protocols": dict(sorted(protocol_ids.items())),
        "tasks": dict(sorted(task_counts.items())),
        "manifest_sha256": _sha256_file(manifest_path),
    }


def _pilot_question_bucket(episode: Episode) -> str:
    task = episode.record["target"]["task"]
    question = " ".join(episode.question.casefold().split())
    if task == "counting":
        return (
            "known_count_target"
            if episode.record["target"]["needs_visual_assistance"]
            else "constrained_or_unknown_count_target"
        )
    if task != "general_vqa":
        return task
    if _answer_leakage(question):
        return "category_answer_leakage"
    if "bounding box" in question or "reference bounding" in question:
        return "box_attribute_or_state"
    if re.search(r"\b(relative to|in relation to|position of|where is|route from)\b", question):
        return "local_relation"
    if re.search(
        r"\b(economic|history|historical|suitable|benefit|industry|land use)\b",
        question,
    ):
        return "global_reasoning"
    return "other_general_vqa"


def select_stratified_pilot(episodes: Sequence[Episode]) -> list[Episode]:
    """Select the frozen 300-row task/source/template pilot deterministically.
    确定性选取冻结的 300 条 task/source/template pilot。"""

    selected: list[Episode] = []
    for task, quota in _PILOT_TASK_QUOTAS.items():
        buckets: dict[tuple[str, str, str], list[Episode]] = defaultdict(list)
        for episode in episodes:
            if episode.record["target"]["task"] != task:
                continue
            key = (
                episode.record["source_group"],
                episode.record["protocol_id"],
                _pilot_question_bucket(episode),
            )
            buckets[key].append(episode)
        positions = {key: 0 for key in buckets}
        task_selection: list[Episode] = []
        while len(task_selection) < quota:
            progressed = False
            for key in sorted(buckets):
                position = positions[key]
                if position >= len(buckets[key]):
                    continue
                task_selection.append(buckets[key][position])
                positions[key] = position + 1
                progressed = True
                if len(task_selection) == quota:
                    break
            if not progressed:
                raise ValueError(f"PILOT_QUOTA_UNAVAILABLE:{task}")
        selected.extend(task_selection)
    return selected


def build_runtime_protocol(project_root: Path, config_path: Path) -> RuntimeProtocol:
    """Assemble the current runtime without inference and freeze its planner.
    不执行推理地组装当前 runtime，并冻结其 planner。"""

    settings = load_settings(config_path, environ={})
    components = assemble_runtime(
        settings,
        project_root=project_root,
        prompts_root=project_root / "prompts",
        qwen_client=_NoopVisionLanguageClient(),
    )
    planner = components.visual_task_planner
    system_prompt = planner.system_prompt
    binding = json.loads(system_prompt.split("planner_binding=", 1)[1])
    response_schema = VisualTaskPlan.model_json_schema()
    core = {
        "protocol_version": "visual-task-plan-v5",
        "system_prompt_file": "prompts/visual_task_plan_v5.md",
        "system_prompt_sha256": _sha256_bytes(system_prompt.encode("utf-8")),
        "system_prompt": system_prompt,
        "planner_binding": binding,
        "response_model": "VisualTaskPlan",
        "response_schema_sha256": _sha256_bytes(
            _canonical_json(response_schema).encode("utf-8")
        ),
        "response_schema": response_schema,
    }
    protocol_id = "protocol-" + _sha256_bytes(_canonical_json(core).encode("utf-8"))[:16]
    document = {"protocol_id": protocol_id, **core}
    executable = {
        task: tuple(categories)
        for task, categories in binding["task_executable_categories"].items()
    }
    return RuntimeProtocol(protocol_id, document, executable)


def build_teacher_prompt(
    runtime_system_prompt: str,
    annotation_rubric: str,
) -> str:
    """Wrap the complete runtime rules with the four-field teacher adapter.
    用四字段 teacher 适配层包装完整运行时规则。"""

    if not runtime_system_prompt.strip() or "planner_binding=" not in runtime_system_prompt:
        raise ValueError("INCOMPLETE_RUNTIME_SYSTEM_PROMPT")
    if not annotation_rubric.strip():
        raise ValueError("EMPTY_ANNOTATION_RUBRIC")
    response_schema = _canonical_json(TextPlanProposal.model_json_schema())
    return (
        "--- BEGIN COMPLETE AUTHORITATIVE RUNTIME PROMPT ---\n"
        + runtime_system_prompt
        + "\n--- END COMPLETE AUTHORITATIVE RUNTIME PROMPT ---\n\n"
        + "--- BEGIN TEXT-ONLY FOUR-FIELD ADAPTER RUBRIC ---\n"
        + annotation_rubric.rstrip()
        + "\n--- END TEXT-ONLY FOUR-FIELD ADAPTER RUBRIC ---\n\n"
        + "--- BEGIN FINAL EXACT FOUR-FIELD RESPONSE SCHEMA ---\n"
        + response_schema
        + "\n--- END FINAL EXACT FOUR-FIELD RESPONSE SCHEMA ---\n"
    )


def bind_teacher_protocol(
    runtime_protocol: RuntimeProtocol,
    teacher_prompt: str,
) -> RuntimeProtocol:
    """Bind the exact teacher prompt and schema into the derived protocol.
    将精确 teacher prompt 与 schema 绑定到派生 protocol。"""

    core = {
        key: deepcopy(value)
        for key, value in runtime_protocol.document.items()
        if key != "protocol_id"
    }
    global_leaves = _global_executable_leaves(
        runtime_protocol.executable_by_task,
        tuple(runtime_protocol.document["planner_binding"]["canonical_leaf_categories"]),
    )
    annotation_policy = {
        "version": "task-independent-object-evidence-v1",
        "global_executable_categories": list(global_leaves),
        "task_gate": False,
        "evidence_is_auxiliary": True,
        "detector_miss_implies_absence": False,
        "max_categories": 8,
    }
    core["annotation_evidence_policy"] = annotation_policy
    core["system_prompt"] = (
        core["system_prompt"].rstrip()
        + "\nannotation_evidence_policy="
        + _canonical_json(annotation_policy)
    )
    core["system_prompt_sha256"] = _sha256_bytes(
        core["system_prompt"].encode("utf-8")
    )
    response_schema = TextPlanProposal.model_json_schema()
    core["refinement_teacher"] = {
        "prompt_version": TEACHER_PROMPT_VERSION,
        "prompt_sha256": _sha256_bytes(teacher_prompt.encode("utf-8")),
        "response_model": "TextPlanProposal",
        "response_schema_sha256": _sha256_bytes(
            _canonical_json(response_schema).encode("utf-8")
        ),
        "response_schema": response_schema,
    }
    protocol_id = "protocol-" + _sha256_bytes(
        _canonical_json(core).encode("utf-8")
    )[:16]
    return RuntimeProtocol(
        protocol_id=protocol_id,
        document={"protocol_id": protocol_id, **core},
        executable_by_task=runtime_protocol.executable_by_task,
    )


def _global_executable_leaves(
    executable_by_task: Mapping[str, Sequence[str]],
    catalog_order: Sequence[str],
) -> tuple[str, ...]:
    """Return the callable submodel-leaf union in stable catalog order.
    按稳定 catalog 顺序返回可调用子模型叶子类别并集。"""

    available = {
        category
        for categories in executable_by_task.values()
        for category in categories
    }
    return tuple(category for category in catalog_order if category in available)


def _singular_token(token: str) -> str:
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("es"):
        stem = token[:-2]
        if stem.endswith(("s", "x", "z", "ch", "sh")):
            return stem
    if len(token) > 2 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _semantic_tokens(value: str) -> tuple[str, ...]:
    return tuple(
        _singular_token(token)
        for token in re.findall(r"[a-z0-9]+", value.casefold().replace("_", "-"))
    )


def _question_evidence_categories(
    question: str,
    *,
    task: TaskName,
    catalog: EvidenceCatalog,
    global_executable: Sequence[str],
) -> tuple[str, ...]:
    """Extract callable auxiliary categories without claiming image presence.
    提取可调用辅助类别，但不声称图中一定存在对应目标。"""

    executable = set(global_executable)
    requested: set[str] = set()

    question_tokens = _semantic_tokens(question)
    candidates: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for leaf in catalog.leaf_categories:
        candidates.append((_semantic_tokens(leaf), (leaf,)))
    for parent in catalog.parent_categories:
        candidates.append((_semantic_tokens(parent), catalog.expand_target(parent)))
    for alias, target in catalog.aliases.items():
        candidates.append((_semantic_tokens(alias), catalog.expand_target(target)))
    for alias, targets in _TEXT_CATEGORY_ALIASES.items():
        candidates.append((_semantic_tokens(alias), targets))

    occupied: set[int] = set()
    for phrase, targets in sorted(candidates, key=lambda item: len(item[0]), reverse=True):
        if not phrase or len(phrase) > len(question_tokens):
            continue
        for start in range(len(question_tokens) - len(phrase) + 1):
            span = set(range(start, start + len(phrase)))
            if span & occupied or question_tokens[start : start + len(phrase)] != phrase:
                continue
            requested.update(target for target in targets if target in executable)
            occupied.update(span)
            break
    explicit = [
        leaf for leaf in catalog.leaf_categories if leaf in requested and leaf in executable
    ][:8]
    selected = set(explicit)
    if task == "scene_classification":
        for leaf in _SCENE_EVIDENCE_PROFILE:
            if leaf in executable and len(selected) < 8:
                selected.add(leaf)
    return tuple(leaf for leaf in catalog.leaf_categories if leaf in selected)


def _answer_leakage(question: str) -> bool:
    normalized = " ".join(question.casefold().split())
    return any(marker in normalized for marker in _ANSWER_LEAKAGE_MARKERS) or any(
        pattern.search(normalized) for pattern in _ANSWER_LEAKAGE_PATTERNS
    )


def _deterministic_text_task(question: str) -> TaskName | None:
    """Apply only high-precision text rules that must not depend on evidence support.
    仅应用不得受 evidence 支持度影响的高精度文本规则。"""

    normalized = " ".join(question.casefold().split())
    if re.search(r"\b(how many|number of|amount of|cardinality of)\b", normalized):
        return "counting"
    if re.search(
        r"\b(relative to|in relation to|parallel to|shortest (?:driving |walking )?(?:route|path) from)\b",
        normalized,
    ):
        return "spatial_relation"
    # Treat open-ended descriptions of a named local region as region captions.
    # 将对明确局部区域的开放式描述归为区域 caption。
    if re.search(r"^(?:how would you describe|describe)\b", normalized):
        return "caption"
    if re.search(r"\b(urban|rural)\b", normalized) and re.search(
        r"\b(area|setting|scene|surroundings?|environment)\b",
        normalized,
    ):
        return "scene_classification"
    localized_type = re.search(
        r"\btype of (?:facility|terrain|infrastructure|scene)\b", normalized
    )
    localized_relation = re.search(
        r"\b(?:immediately|directly|to the (?:right|left)|above|below)\b",
        normalized,
    )
    if localized_type and localized_relation:
        return "general_vqa"
    if re.search(
        r"\bmain type of (?:facility|terrain|infrastructure|scene)\b", normalized
    ):
        return "scene_classification"
    if re.search(
        r"\btype of (?:facility|terrain|infrastructure|scene)\b.{0,32}"
        r"\b(?:depicted|shown|visible) in (?:the )?(?:image|scene)\b",
        normalized,
    ):
        return "scene_classification"
    return None


def normalize_proposal(
    proposal: TextPlanProposal,
    *,
    question: str,
    catalog: EvidenceCatalog,
    executable_by_task: Mapping[str, Sequence[str]],
) -> NormalizedProposal:
    task = _deterministic_text_task(question) or proposal.task
    if task == "grounding" and not re.search(
        r"\b(bounding box|bbox|coordinates?|pixel|point|roi|localization geometry)\b",
        question.casefold(),
    ):
        task = "general_vqa"
    count_target = proposal.count_target if task in COUNTING_TASKS else None
    if task in COUNTING_TASKS and not count_target:
        raise ValueError("COUNT_TARGET_MISSING_AFTER_TASK_GUARD")
    if _answer_leakage(question) and task != "scene_classification":
        return NormalizedProposal(
            task=task,
            needs_visual_assistance=False,
            object_categories=(),
            count_target=count_target,
            decision_code="category_is_requested_answer",
        )
    global_executable = _global_executable_leaves(
        executable_by_task,
        catalog.leaf_categories,
    )
    executable = set(global_executable)
    requested = set(
        _question_evidence_categories(
            question,
            task=task,
            catalog=catalog,
            global_executable=global_executable,
        )
    )
    if task in COUNTING_TASKS:
        assert count_target is not None
        canonical_target = catalog.canonicalize_alias(count_target)
        expected = catalog.expand_target(canonical_target)
        if expected:
            count_target = canonical_target
            requested.update(leaf for leaf in expected if leaf in executable)
        requested.update(
            _question_evidence_categories(
                count_target,
                task=task,
                catalog=catalog,
                global_executable=global_executable,
            )
        )

    for raw in proposal.object_categories:
        try:
            leaves = catalog.expand_target(raw)
        except CatalogCategoryError:
            continue
        requested.update(leaf for leaf in leaves if leaf in executable)

    ordered = [leaf for leaf in catalog.leaf_categories if leaf in requested][:8]
    if not ordered:
        return NormalizedProposal(
            task,
            False,
            (),
            count_target,
            "no_callable_category",
        )
    return NormalizedProposal(
        task=task,
        object_categories=tuple(ordered),
        needs_visual_assistance=True,
        count_target=count_target,
        decision_code="assistance_enabled",
    )


def merge_target(
    old_target: Mapping[str, Any],
    proposal: TextPlanProposal,
    *,
    question: str,
    catalog: EvidenceCatalog,
    executable_by_task: Mapping[str, Sequence[str]],
) -> VisualTaskPlan:
    normalized = normalize_proposal(
        proposal,
        question=question,
        catalog=catalog,
        executable_by_task=executable_by_task,
    )
    updated = deepcopy(dict(old_target))
    updated.update(
        {
            "task": normalized.task,
            "needs_visual_assistance": normalized.needs_visual_assistance,
            "object_categories": list(normalized.object_categories),
            "count_target": normalized.count_target,
        }
    )
    plan = VisualTaskPlan.model_validate(updated)
    if plan.needs_visual_assistance:
        executable = set(
            _global_executable_leaves(executable_by_task, catalog.leaf_categories)
        )
        if any(not catalog.is_leaf(category) for category in plan.object_categories):
            raise ValueError("NON_LEAF_CATEGORY")
        if any(category not in executable for category in plan.object_categories):
            raise ValueError("CAPABILITY_UNAVAILABLE")
    return plan


def _annotate_one(
    question: str,
    *,
    client: DeepSeekJudgeClient,
    model: str,
    prompt: str,
    cache: JsonResponseCache,
) -> AnnotationResult:
    payload = {"question": question}
    question_id = _sha256_bytes(question.encode("utf-8"))
    request_hash = build_judge_request_hash(
        model=model,
        prompt_text=prompt,
        prompt_version=TEACHER_PROMPT_VERSION,
        sample_id=question_id,
        payload=payload,
        response_schema=TextPlanProposal.model_json_schema(),
    )
    if not question:
        return AnnotationResult(
            None,
            request_hash,
            False,
            "TEXT_TEACHER_EMPTY_QUESTION_UNSUPPORTED",
        )
    try:
        cache_hit = cache.load(request_hash) is not None
        proposal = client.judge_json(
            payload,
            response_model=TextPlanProposal,
            request_meta=RequestMeta(
                request_id=f"question-{question_id[:16]}:deepseek-text-plan",
                request_hash=request_hash,
                prompt_version=TEACHER_PROMPT_VERSION,
                sample_id=f"question-{question_id[:16]}",
            ),
            system_prompt=prompt,
            repair_with_original_payload=True,
        )
    except DeepSeekJudgeError as exc:
        return AnnotationResult(
            None,
            request_hash,
            locals().get("cache_hit", False),
            str(exc),
        )
    except Exception as exc:
        return AnnotationResult(
            None,
            request_hash,
            locals().get("cache_hit", False),
            type(exc).__name__,
        )
    return AnnotationResult(proposal, request_hash, cache_hit)


def annotate_questions(
    questions: Sequence[str],
    *,
    client: DeepSeekJudgeClient,
    model: str,
    prompt: str,
    cache: JsonResponseCache,
    concurrency: int,
) -> dict[str, AnnotationResult]:
    unique = list(dict.fromkeys(questions))
    results: dict[str, AnnotationResult] = {}
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="deepseek-label") as pool:
        futures = {
            pool.submit(
                _annotate_one,
                question,
                client=client,
                model=model,
                prompt=prompt,
                cache=cache,
            ): question
            for question in unique
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results


def duplicate_region_conflicts(episodes: Sequence[Episode]) -> set[str]:
    groups: dict[tuple[str, str], list[Episode]] = defaultdict(list)
    for episode in episodes:
        groups[(episode.image, episode.question)].append(episode)
    conflicts: set[str] = set()
    for group in groups.values():
        regions = {
            _canonical_json(episode.record["target"].get("region_request"))
            for episode in group
        }
        if len(regions) > 1:
            conflicts.update(episode.record["episode_id"] for episode in group)
    return conflicts


def _git_identity(project_root: Path) -> dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        return {"head": None, "dirty": None}
    return {"head": head, "dirty": dirty}


def build_run_identity(
    *,
    project_root: Path,
    config_path: Path,
    source_audit: Mapping[str, Any],
    protocol: RuntimeProtocol,
    prompt: str,
    settings: DeepSeekSettings,
    limit: int | None,
    concurrency: int,
    selection_mode: str = "full",
) -> dict[str, Any]:
    files = {
        relative: _sha256_file(project_root / relative)
        for relative in (
            "agents/schema.py",
            "agents/evidence_catalog.json",
            "agents/counting/expert_catalog.json",
            "prompts/visual_task_plan_v5.md",
            "prompts/visual_task_plan_text_teacher_v6.md",
        )
    }
    files["runtime_config"] = _sha256_file(config_path)
    identity = {
        "rule_version": RULE_VERSION,
        "teacher_prompt_version": TEACHER_PROMPT_VERSION,
        "teacher_prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
        "teacher_model": settings.model,
        "teacher_base_url": settings.base_url,
        "teacher_timeout_seconds": settings.timeout_seconds,
        "teacher_max_retries": settings.max_retries,
        "teacher_thinking": "disabled",
        "concurrency": concurrency,
        "response_schema_sha256": _sha256_bytes(
            _canonical_json(TextPlanProposal.model_json_schema()).encode("utf-8")
        ),
        "source_manifest_sha256": source_audit["manifest_sha256"],
        "target_protocol_id": protocol.protocol_id,
        "limit": limit,
        "selection_mode": selection_mode,
        "git": _git_identity(project_root),
        "input_files": files,
    }
    identity["refinement_run_id"] = "refine-" + _sha256_bytes(
        _canonical_json(identity).encode("utf-8")
    )[:16]
    return identity


def _prepare_output(output_root: Path, identity: Mapping[str, Any], *, resume: bool) -> None:
    identity_path = output_root / "audit" / "refinement_run.json"
    if output_root.exists() and not resume:
        raise ValueError("OUTPUT_EXISTS_USE_RESUME")
    if identity_path.is_file():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing != identity:
            raise ValueError("RESUME_IDENTITY_MISMATCH")
    elif output_root.exists() and any(output_root.iterdir()):
        raise ValueError("RESUME_IDENTITY_MISSING")
    _atomic_write_json(identity_path, identity)


def _link_images(source_root: Path, output_root: Path, *, copy_images: bool) -> str:
    policy = "copy" if copy_images else "hardlink-with-copy-fallback"
    for source in sorted((source_root / "images").rglob("*")):
        if source.is_symlink():
            raise ValueError("SOURCE_IMAGE_SYMLINK_FORBIDDEN")
        if not source.is_file():
            continue
        relative = source.relative_to(source_root)
        target = output_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file():
            if _sha256_file(target) == _sha256_file(source):
                continue
            raise ValueError("EXISTING_IMAGE_DIGEST_MISMATCH")
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            if copy_images:
                shutil.copy2(source, temporary)
            else:
                try:
                    os.link(source, temporary)
                except OSError:
                    shutil.copy2(source, temporary)
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()
    return policy


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    content = "".join(_canonical_json(row) + "\n" for row in rows)
    _atomic_write_text(path, content)
    encoded = content.encode("utf-8")
    return {"examples": len(rows), "bytes": len(encoded), "sha256": _sha256_bytes(encoded)}


def compile_dataset(
    *,
    source_root: Path,
    output_root: Path,
    episodes: Sequence[Episode],
    annotations: Mapping[str, AnnotationResult],
    protocol: RuntimeProtocol,
    catalog: EvidenceCatalog,
    identity: Mapping[str, Any],
    copy_images: bool,
) -> dict[str, Any]:
    conflicts = duplicate_region_conflicts(episodes)
    accepted_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    training_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    decisions: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    task_counts: Counter[str] = Counter()
    assistance_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    preview_paths: dict[str, str] = {}
    preview_max_side = protocol.document["planner_binding"]["preview_max_side"]
    if not isinstance(preview_max_side, int) or preview_max_side <= 0:
        raise ValueError("INVALID_PREVIEW_MAX_SIDE")
    for episode in episodes:
        episode_id = episode.record["episode_id"]
        annotation = annotations[episode.question]
        error_code = annotation.error_code
        if episode_id in conflicts:
            error_code = "DUPLICATE_REGION_CONFLICT"
        if annotation.proposal is None or error_code is not None:
            quarantine.append(
                {
                    "episode_id": episode_id,
                    "source_file": episode.relative_file,
                    "line_number": episode.line_number,
                    "error_code": error_code or "ANNOTATION_MISSING",
                    "request_hash": annotation.request_hash,
                }
            )
            continue
        try:
            normalized = normalize_proposal(
                annotation.proposal,
                question=episode.question,
                catalog=catalog,
                executable_by_task=protocol.executable_by_task,
            )
            target = merge_target(
                episode.record["target"],
                annotation.proposal,
                question=episode.question,
                catalog=catalog,
                executable_by_task=protocol.executable_by_task,
            )
        except (ValidationError, ValueError, CatalogCategoryError) as exc:
            stable_error = str(exc)
            quarantine.append(
                {
                    "episode_id": episode_id,
                    "source_file": episode.relative_file,
                    "line_number": episode.line_number,
                    "error_code": (
                        stable_error
                        if stable_error in _STABLE_MERGE_ERRORS
                        else type(exc).__name__
                    ),
                    "request_hash": annotation.request_hash,
                }
            )
            continue
        old_target = episode.record["target"]
        final_target = target.model_dump(mode="json")
        if set(final_target) != set(old_target):
            raise ValueError("TARGET_FIELD_SET_CHANGED")
        changed_target_fields = sorted(
            key
            for key in final_target
            if final_target[key] != old_target[key]
        )
        if not set(changed_target_fields).issubset(_USER_APPROVED_TARGET_FIELDS):
            raise ValueError("UNAPPROVED_TARGET_FIELD_CHANGE")
        record = deepcopy(episode.record)
        record["protocol_id"] = protocol.protocol_id
        record["protocol_version"] = "visual-task-plan-v5"
        record["messages"][0]["content_ref"] = f"protocols/{protocol.protocol_id}.json"
        record["target"] = final_target
        record["target_text"] = _canonical_json(record["target"])
        accepted_by_file[episode.relative_file].append(record)
        training_image = preview_paths.get(record["image"])
        if training_image is None:
            training_image = _materialize_training_preview(
                source_root,
                output_root,
                record["image"],
                max_side=preview_max_side,
            )
            preview_paths[record["image"]] = training_image
        training_record = {
            "schema_version": 1,
            "format": TRAINING_FORMAT,
            "episode_id": episode_id,
            "image": training_image,
            "messages": compile_training_messages(
                record,
                protocol.document,
                image_override=training_image,
            ),
            "source_group": record["source_group"],
            "split": record["split"],
        }
        training_by_file[episode.relative_file].append(training_record)
        task_counts[target.task] += 1
        assistance_counts[str(target.needs_visual_assistance).lower()] += 1
        category_counts.update(target.object_categories)
        decisions.append(
            {
                "episode_id": episode_id,
                "request_hash": annotation.request_hash,
                "cache_hit": annotation.cache_hit,
                "old_task": old_target["task"],
                "new_task": target.task,
                "old_assistance": old_target["needs_visual_assistance"],
                "new_assistance": target.needs_visual_assistance,
                "old_count_target": old_target["count_target"],
                "new_count_target": target.count_target,
                "new_categories": target.object_categories,
                "changed_target_fields": changed_target_fields,
                "review_required": bool(
                    target.needs_visual_assistance
                    or "task" in changed_target_fields
                    or "count_target" in changed_target_fields
                ),
                "decision_code": normalized.decision_code,
            }
        )
    protocol_path = output_root / "protocols" / f"{protocol.protocol_id}.json"
    _atomic_write_json(protocol_path, protocol.document)
    dataset_stats: dict[str, Any] = {}
    training_stats: dict[str, Any] = {}
    for relative, rows in sorted(accepted_by_file.items()):
        dataset_rel = PurePosixPath(relative)
        stats = _write_jsonl(output_root / Path(*dataset_rel.parts), rows)
        dataset_stats[relative] = stats
        training_rel = PurePosixPath("training") / dataset_rel.relative_to("datasets")
        training_stats[training_rel.as_posix()] = _write_jsonl(
            output_root / Path(*training_rel.parts), training_by_file[relative]
        )
    _write_jsonl(output_root / "audit" / "label_decisions.jsonl", decisions)
    _write_jsonl(output_root / "audit" / "quarantine.jsonl", quarantine)
    link_policy = _link_images(source_root, output_root, copy_images=copy_images)
    source_manifest = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    manifest = deepcopy(source_manifest)
    manifest["description"] = "Text-only DeepSeek-refined visual-planner episodes"
    manifest["protocol_policy"] = (
        "one content-addressed runtime-v5-plus-annotation-policy protocol"
    )
    manifest["target_policy"] = (
        "text-only teacher proposal plus task-independent callable-submodel gates"
    )
    manifest["protocols"] = {
        protocol.protocol_id: {
            "path": f"protocols/{protocol.protocol_id}.json",
            "system_prompt_sha256": protocol.document["system_prompt_sha256"],
            "response_schema_sha256": protocol.document["response_schema_sha256"],
            "protocol_version": "visual-task-plan-v5",
            "planner_binding": protocol.document["planner_binding"],
            **(
                {"refinement_teacher": protocol.document["refinement_teacher"]}
                if "refinement_teacher" in protocol.document
                else {}
            ),
        }
    }
    for group, entry in manifest["datasets"].items():
        files: dict[str, Any] = {}
        splits: dict[str, int] = {}
        for relative, stats in dataset_stats.items():
            rel = PurePosixPath(relative)
            if rel.parent.name == group:
                files[rel.name] = stats
                splits[rel.stem] = stats["examples"]
        entry["files"] = files
        entry["splits"] = splits
        entry["examples"] = sum(splits.values())
        entry["embedded_image_blocks"] = entry["examples"]
        entry["protocol_ids"] = [protocol.protocol_id] if files else []
    manifest["images"]["embedded_block_count"] = len(decisions)
    manifest["image_alias_policy"] = link_policy
    manifest["refinement"] = {
        "refinement_run_id": identity["refinement_run_id"],
        "source_manifest_sha256": identity["source_manifest_sha256"],
        "accepted": len(decisions),
        "quarantine": len(quarantine),
        "source_total": len(episodes),
        "teacher_model": identity["teacher_model"],
        "teacher_prompt_version": TEACHER_PROMPT_VERSION,
        "rule_version": RULE_VERSION,
        "training_format": TRAINING_FORMAT,
        "training_files": training_stats,
    }
    _atomic_write_json(output_root / "manifest.json", manifest)
    distribution = {
        "accepted": len(decisions),
        "quarantine": len(quarantine),
        "tasks": dict(sorted(task_counts.items())),
        "assistance": dict(sorted(assistance_counts.items())),
        "categories": dict(sorted(category_counts.items())),
    }
    _atomic_write_json(output_root / "audit" / "distribution.json", distribution)
    _atomic_write_json(
        output_root / "audit" / "training_contract.json",
        {
            "format": TRAINING_FORMAT,
            "system_prompt": (
                "current VisualTaskPlanner.system_prompt plus frozen "
                "task-independent annotation evidence policy"
            ),
            "user_content_order": ["image", "raw_question"],
            "image_preprocessing": "exact runtime preview_from_path PNG bytes",
            "preview_max_side": preview_max_side,
            "preview_images": len(preview_paths),
            "preview_bytes_verified": True,
            "assistant_supervision": "canonical compact VisualTaskPlan JSON",
            "semantic_messages_verified": True,
            "processor_tokenization_verified": False,
            "remaining_requirement": "training must use the same Qwen processor/chat template and assistant-only loss mask as deployment",
        },
    )
    _atomic_write_text(
        output_root / "README.md",
        "# Refined visual-planner training data\n\n"
        f"- Source episodes: {len(episodes)}\n"
        f"- Accepted: {len(decisions)}\n"
        f"- Quarantine: {len(quarantine)}\n"
        f"- Protocol: `{protocol.protocol_id}`\n"
        f"- Teacher: `{identity['teacher_model']}` (text-only; raw question only)\n"
        f"- Deterministic rule: `{RULE_VERSION}`\n"
        "- Evidence policy: task-independent global callable-submodel leaves; auxiliary only.\n"
        f"- Runtime catalog: `{protocol.document['planner_binding']['catalog_version']}`\n"
        "- `datasets/` keeps compact content references; `training/` contains resolved system/user/assistant chat records.\n"
        "- `training_images/` contains the exact deterministic PNG previews used by the inference planner.\n"
        "- API keys, images, old targets, answers, and provenance were never sent to the teacher.\n"
        "- Quarantined rows are excluded from training and remain traceable in `audit/quarantine.jsonl`.\n",
    )
    return distribution


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/local.yaml"))
    parser.add_argument("--use-api", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument(
        "--prompt-api-key",
        action="store_true",
        help="read the API key from a no-echo terminal prompt instead of the environment",
    )
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--timeout-seconds", type=int)
    parser.add_argument("--max-retries", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.concurrency <= 0 or args.concurrency > 256:
        raise SystemExit("--concurrency must be within 1..256")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.pilot and args.limit is not None:
        raise SystemExit("--pilot and --limit are mutually exclusive")
    project_root = Path(__file__).resolve().parents[1]
    source_root = args.source.resolve()
    episodes, source_audit = audit_source(source_root)
    if args.pilot:
        episodes = select_stratified_pilot(episodes)
    elif args.limit is not None:
        episodes = episodes[: args.limit]
    runtime_protocol = build_runtime_protocol(project_root, args.config.resolve())
    catalog = EvidenceCatalog.from_file(project_root / "agents" / "evidence_catalog.json")
    annotation_rubric = (
        project_root / "prompts" / "visual_task_plan_text_teacher_v6.md"
    ).read_text(encoding="utf-8")
    prompt = build_teacher_prompt(
        runtime_protocol.document["system_prompt"],
        annotation_rubric,
    )
    protocol = bind_teacher_protocol(runtime_protocol, prompt)
    base_settings = load_settings(args.config.resolve(), environ={}).models.deepseek
    settings = base_settings.model_copy(
        update={
            **({"model": args.model} if args.model else {}),
            **({"base_url": args.base_url} if args.base_url else {}),
            **({"timeout_seconds": args.timeout_seconds} if args.timeout_seconds else {}),
            **({"max_retries": args.max_retries} if args.max_retries is not None else {}),
            "api_key_env": args.api_key_env,
        }
    )
    summary = {
        "source_audit": source_audit,
        "selected_episodes": len(episodes),
        "unique_questions": len({episode.question for episode in episodes}),
        "target_protocol_id": protocol.protocol_id,
        "catalog_version": protocol.document["planner_binding"]["catalog_version"],
        "general_vqa_executable": len(protocol.executable_by_task["general_vqa"]),
    }
    if not args.use_api:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.output is None:
        raise SystemExit("--output is required with --use-api")
    output_root = args.output.resolve()
    if output_root == source_root or output_root.is_relative_to(source_root):
        raise SystemExit("output work directory must be outside the source directory")
    api_key = (
        getpass.getpass("DeepSeek API key: ")
        if args.prompt_api_key
        else os.environ.get(args.api_key_env, "")
    )
    if not api_key:
        source = "terminal prompt" if args.prompt_api_key else args.api_key_env
        raise SystemExit(f"missing API key from: {source}")
    identity = build_run_identity(
        project_root=project_root,
        config_path=args.config.resolve(),
        source_audit=source_audit,
        protocol=protocol,
        prompt=prompt,
        settings=settings,
        limit=args.limit,
        concurrency=args.concurrency,
        selection_mode="stratified-pilot-v1" if args.pilot else "full",
    )
    _prepare_output(output_root, identity, resume=args.resume)
    _atomic_write_json(
        output_root / "audit" / "input_audit.json",
        {**source_audit, "selected_episodes": len(episodes)},
    )
    cache = JsonResponseCache(output_root / "audit" / "api_cache")
    client = DeepSeekJudgeClient(
        settings,
        api_key=api_key,
        judge_prompt=prompt,
        repair_prompt=(
            prompt
            + "\n\nYour previous response failed structural validation. Return only a "
            "corrected JSON object matching the four-field adapter contract."
        ),
        cache=cache,
        transport=_urllib_label_transport,
    )
    annotations = annotate_questions(
        [episode.question for episode in episodes],
        client=client,
        model=settings.model,
        prompt=prompt,
        cache=cache,
        concurrency=args.concurrency,
    )
    distribution = compile_dataset(
        source_root=source_root,
        output_root=output_root,
        episodes=episodes,
        annotations=annotations,
        protocol=protocol,
        catalog=catalog,
        identity=identity,
        copy_images=args.copy_images,
    )
    print(json.dumps({**summary, **distribution}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
