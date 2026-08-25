"""Formal offline multi-source ChangeAgent SFT corpus builder.

正式离线多源 ChangeAgent SFT corpus 构建器。Pair, split, exclusions and
source selection remain source-spec owned; only target serialization is v2.
pair、split、排除项和 source 选择保持由 source spec 权威控制。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml

from agents.change.schema import CANONICAL_NO_CHANGE
from training.multimodal_sft.change_target_contract import (
    CHANGE_SFT_EPISODE_SCHEMA_VERSION,
    CHANGE_TARGET_CONTRACT_NAME,
    CHANGE_TARGET_CONTRACT_VERSION,
    canonical_change_initial_result,
    change_target_contract_descriptor,
    change_target_contract_identity,
)


SOURCE_SPEC_SCHEMA_VERSION = 1
ALLOWED_TASKS = {"change_caption", "change_qa"}
FORMAL_TRAIN_ORDERING_POLICY = "sha256_episode_id_v1"
FORMAL_TRAIN_ORDERING_SEED = 1234
VALIDATION_ORDERING_POLICY = "builder_source_order_v1"


class ChangeCorpusBuildError(ValueError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}" if detail else code)


def formal_train_order_key(
    episode: dict[str, Any],
    *,
    seed: int = FORMAL_TRAIN_ORDERING_SEED,
) -> tuple[str, str]:
    """Return a cross-process deterministic pseudo-random train ordering key."""

    episode_id = str(episode.get("episode_id") or "")
    if not episode_id:
        raise ChangeCorpusBuildError("EPISODE_ID_REQUIRED_FOR_ORDERING")
    digest = hashlib.sha256(f"{int(seed)}\0{episode_id}".encode("utf-8")).hexdigest()
    return digest, episode_id


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def builder_git_identity() -> dict[str, Any]:
    """Bind a formal corpus to the committed builder implementation."""

    repository = Path(__file__).resolve().parents[2]

    def git(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(repository), *args],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ChangeCorpusBuildError("BUILDER_GIT_IDENTITY_UNAVAILABLE") from exc
        return result.stdout.strip()

    commit = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    return {
        "commit": commit,
        "tree": tree,
        "working_tree_clean": not bool(git("status", "--porcelain", "--untracked-files=no")),
    }


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ChangeCorpusBuildError("SOURCE_READ_ERROR", str(path)) from exc
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError:
        parsed = []
        for line_no, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ChangeCorpusBuildError("SOURCE_JSON_INVALID", f"{path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise ChangeCorpusBuildError("SOURCE_ROW_INVALID", f"{path}:{line_no}")
            parsed.append(row)
    if isinstance(parsed, dict):
        for key in ("images", "records", "data", "samples", "annotations", "items"):
            value = parsed.get(key)
            if isinstance(value, list) and (key != "images" or all(isinstance(item, dict) for item in value)):
                parsed = value
                break
        else:
            parsed = [parsed]
    if not isinstance(parsed, list) or not all(isinstance(row, dict) for row in parsed):
        raise ChangeCorpusBuildError("SOURCE_SCHEMA_INVALID", str(path))
    return parsed


def load_source_spec(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise ChangeCorpusBuildError("SOURCE_SPEC_MISSING", str(source))
    try:
        raw = source.read_text(encoding="utf-8")
        spec = json.loads(raw) if source.suffix.lower() == ".json" else yaml.safe_load(raw)
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ChangeCorpusBuildError("SOURCE_SPEC_INVALID", str(source)) from exc
    if not isinstance(spec, dict) or spec.get("schema_version") != SOURCE_SPEC_SCHEMA_VERSION:
        raise ChangeCorpusBuildError("SOURCE_SPEC_VERSION")
    canonical = spec.get("canonical_dataset")
    if not isinstance(canonical, dict) or canonical.get("type") != "levir":
        raise ChangeCorpusBuildError("CANONICAL_DATASET_INVALID")
    for key in ("captions", "image_root"):
        value = canonical.get(key)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ChangeCorpusBuildError("CANONICAL_PATH_NOT_ABSOLUTE", key)
    exclusions = spec.get("exclusions")
    if not isinstance(exclusions, dict) or not isinstance(exclusions.get("file"), str):
        raise ChangeCorpusBuildError("EXCLUSION_FILE_REQUIRED")
    split_policy = spec.get("split_policy")
    if not isinstance(split_policy, dict) or split_policy.get("authority") != "levir_official":
        raise ChangeCorpusBuildError("OFFICIAL_SPLIT_AUTHORITY_REQUIRED")
    if split_policy.get("include_test", False):
        raise ChangeCorpusBuildError("TEST_SPLIT_MUST_BE_EXCLUDED")
    sources = spec.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ChangeCorpusBuildError("SOURCES_REQUIRED")
    for item in sources:
        if not isinstance(item, dict) or not item.get("id") or not item.get("kind"):
            raise ChangeCorpusBuildError("SOURCE_ENTRY_INVALID")
        if item.get("kind") == "multi_turn" and item.get("enabled", False):
            raise ChangeCorpusBuildError("MULTITURN_V1_DISABLED")
        if item.get("enabled", True) and item.get("task") not in ALLOWED_TASKS:
            raise ChangeCorpusBuildError("EXPLICIT_TASK_REQUIRED", str(item.get("id")))
        if item.get("enabled", True):
            item_path = item.get("path")
            if not isinstance(item_path, str) or not Path(item_path).is_absolute():
                raise ChangeCorpusBuildError("SOURCE_PATH_NOT_ABSOLUTE", str(item.get("id")))
    return spec


def _split(value: Any) -> str:
    normalized = normalize_text(value).lower()
    if normalized in {"val", "valid", "validation", "dev"}:
        return "validation"
    return "test" if normalized == "test" else "train"


def _safe_rel(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        return None
    value = value.replace("\\", "/")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return None
    return path.as_posix()


def _field(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def _pair_refs(row: dict[str, Any], split: str, image_root: Path) -> tuple[str | None, str | None]:
    t1 = _field(row, "t1", "image_t1", "before", "image_a", "A")
    t2 = _field(row, "t2", "image_t2", "after", "image_b", "B")
    images = row.get("images") or row.get("image") or row.get("image_paths")
    if isinstance(images, list) and len(images) >= 2:
        t1 = t1 or (images[0].get("path") if isinstance(images[0], dict) else images[0])
        t2 = t2 or (images[1].get("path") if isinstance(images[1], dict) else images[1])
    filename = _field(row, "filename", "file_name", "image")
    if t1 is None and isinstance(filename, str):
        raw_split = normalize_text(_field(row, "split", "set", "subset", "filepath")).lower()
        folders = list(dict.fromkeys((["val"] if raw_split in {"validation", "valid", "dev"} else []) + [raw_split, split]))
        candidates = [(f"A/{filename}", f"B/{filename}")]
        for folder in folders:
            if folder:
                candidates.extend(((f"{folder}/A/{filename}", f"{folder}/B/{filename}"), (f"images/{folder}/A/{filename}", f"images/{folder}/B/{filename}")))
        for candidate_t1, candidate_t2 in candidates:
            if (image_root / candidate_t1).is_file() and (image_root / candidate_t2).is_file():
                t1, t2 = candidate_t1, candidate_t2
                break
        else:
            t1, t2 = candidates[0]
    return _safe_rel(t1), _safe_rel(t2)


def _stable_id(row: dict[str, Any], filename: str) -> str:
    return normalize_text(_field(row, "source_record_id", "id", "image_id", "imgid", "filename", "file_name") or filename)


def build_pair_registry(spec: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    canonical = spec["canonical_dataset"]
    root = Path(canonical["image_root"])
    registry: list[dict[str, Any]] = []
    aliases: dict[str, set[str]] = {}
    seen: set[str] = set()
    for row in _read_json_or_jsonl(Path(canonical["captions"])):
        split = _split(_field(row, "split", "set", "subset", "filepath"))
        filename = normalize_text(_field(row, "filename", "file_name", "image", "name") or "")
        source_id = _stable_id(row, filename)
        if not filename and not source_id:
            raise ChangeCorpusBuildError("PAIR_ID_MISSING")
        canonical_id = f"levir:{split}:{filename or source_id}"
        if canonical_id in seen:
            raise ChangeCorpusBuildError("PAIR_ID_DUPLICATE", canonical_id)
        t1, t2 = _pair_refs(row, split, root)
        if not t1 or not t2:
            raise ChangeCorpusBuildError("PAIR_PATH_MISSING", canonical_id)
        pair = {"canonical_pair_id": canonical_id, "source_split": split, "training_eligible": split != "test", "source_record_id": source_id, "filename": filename, "t1_path": t1, "t2_path": t2}
        registry.append(pair)
        seen.add(canonical_id)
        for alias in {canonical_id, source_id, filename, Path(filename).stem if filename else "", Path(t1).name}:
            if alias:
                aliases.setdefault(normalize_text(alias), set()).add(canonical_id)
    return registry, aliases


def load_exclusions(spec: dict[str, Any], aliases: dict[str, set[str]]) -> dict[str, Any]:
    path = Path(spec["exclusions"]["file"])
    if not path.is_file():
        raise ChangeCorpusBuildError("EXCLUSION_FILE_MISSING", str(path))
    lines = path.read_text(encoding="utf-8").splitlines()
    ids = [line.strip() for line in lines if line.strip()]
    if len(ids) != len(set(ids)):
        raise ChangeCorpusBuildError("EXCLUSION_DUPLICATE")
    mapped: dict[str, str] = {}
    unmatched: list[str] = []
    ambiguous: list[str] = []
    for item in ids:
        hits = aliases.get(normalize_text(item), set())
        if len(hits) == 1:
            mapped[item] = next(iter(hits))
        elif hits:
            ambiguous.append(item)
        else:
            unmatched.append(item)
    if unmatched:
        raise ChangeCorpusBuildError("EXCLUSION_UNMATCHED", ",".join(unmatched[:5]))
    if ambiguous:
        raise ChangeCorpusBuildError("EXCLUSION_AMBIGUOUS", ",".join(ambiguous[:5]))
    return {"path": str(path), "sha256": sha256_file(path), "line_count": len(lines), "unique_count": len(ids), "mapped_count": len(mapped), "unmatched": unmatched, "ambiguous": ambiguous, "mapped_ids": mapped}


def _answer(value: Any) -> str:
    if isinstance(value, dict):
        value = _field(value, "raw", "caption", "text", "answer", "value")
    return normalize_text(value)


def _conversation(row: dict[str, Any]) -> tuple[str, str, int]:
    turns = _field(row, "turns", "conversations", "messages")
    if not isinstance(turns, list):
        return normalize_text(_field(row, "question", "instruction", "query")), _answer(_field(row, "answer", "response", "output")), 0
    if len(turns) > 2:
        return (normalize_text(_field(turns[0], "value", "content", "text")) if turns else ""), "", len(turns)
    question = normalize_text(_field(row, "question", "instruction", "query"))
    answer = _answer(_field(row, "answer", "response", "output"))
    if len(turns) == 2:
        question = question or normalize_text(_field(turns[0], "value", "content", "text"))
        answer = answer or _answer(_field(turns[1], "value", "content", "text"))
    return question, answer, len(turns)


def _resolve_pair(row: dict[str, Any], aliases: dict[str, set[str]]) -> str:
    values = [normalize_text(row[key]) for key in ("canonical_pair_id", "pair_id", "parent_id", "image_id", "filename", "file_name", "id") if row.get(key) is not None]
    images = row.get("images") or row.get("image") or row.get("image_paths")
    image_values: list[str] = []
    if isinstance(images, list):
        for image in images:
            value = image.get("path") if isinstance(image, dict) else image
            if value:
                image_values.extend((normalize_text(value), Path(str(value)).name, Path(str(value)).stem))
    def resolve(candidates: list[str]) -> set[str]:
        hits: set[str] = set()
        for value in candidates:
            hits.update(aliases.get(value, set()))
            hits.update(aliases.get(Path(value).name, set()))
            hits.update(aliases.get(Path(value).stem, set()))
        return hits
    hits = resolve(image_values) if image_values else resolve(values)
    if len(hits) != 1:
        raise ChangeCorpusBuildError("PAIR_UNMATCHED" if not hits else "PAIR_AMBIGUOUS")
    return next(iter(hits))


def _target(answer: str) -> dict[str, Any]:
    if answer.upper() in {"NO_CHANGE", "NO CHANGE"}:
        answer = CANONICAL_NO_CHANGE
    return canonical_change_initial_result({"agent_name": "change_agent", "answer": answer, "boxes": [], "evidence_items": [], "geometry": {}, "status": "completed"})


def _episode(source: dict[str, Any], row: dict[str, Any], row_index: str, pair: dict[str, Any], task: str, question: str, answer: str, source_sha: str) -> dict[str, Any]:
    source_id = str(source["id"])
    pair_id = pair["canonical_pair_id"]
    return {
        "schema_version": CHANGE_SFT_EPISODE_SCHEMA_VERSION,
        "episode_id": f"{source_id}/{pair_id}/{row_index}",
        "parent_sample_id": pair_id,
        "dataset": "LEVIR-CC" if source["kind"] == "levir_caption" else "ChangeChat",
        "split": pair["source_split"], "task": task, "input_contract": "semantic_pair_v1", "question": question,
        "images": [{"image_source": "levir", "path": pair["t1_path"], "role": "raw_full_t1"}, {"image_source": "levir", "path": pair["t2_path"], "role": "raw_full_t2"}],
        "request_payload": {"decision_stage": "initial", "question": question, "task": task, "temporal_roles": ["t1", "t2"], "image_manifest": [{"index": "0", "role": "raw_full_t1"}, {"index": "1", "role": "raw_full_t2"}]},
        "target": {"response_schema": CHANGE_TARGET_CONTRACT_NAME, "contract_version": CHANGE_TARGET_CONTRACT_VERSION, "result": _target(answer)},
        "augmentation_policy": {"temporal_geometry": "locked_identity", "photometric": "disabled"},
        "provenance": {"source_id": source_id, "source_file_sha256": source_sha, "source_row_index": row_index, "original_source_record_id": _stable_id(row, str(row.get("filename") or row.get("id") or row_index)), "answer_origin": "human", "original_image_refs": row.get("images") or [row.get("t1"), row.get("t2")], "canonical_pair_id": pair_id},
    }


def _supervision_key(episode: dict[str, Any]) -> tuple[str, str, str, str]:
    return (episode["parent_sample_id"], episode["task"], normalize_text(episode.get("question")), normalize_text(episode["target"]["result"]["answer"]).lower())


def inspect_source(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    rows = _read_json_or_jsonl(source)
    questions: list[str] = []
    answers: list[str] = []
    image_types: Counter[str] = Counter()
    lengths: Counter[int] = Counter()
    geometry = 0
    for row in rows:
        question, answer, length = _conversation(row)
        if question and len(questions) < 10:
            questions.append(question)
        if answer and len(answers) < 10:
            answers.append(answer)
        lengths[length] += 1
        image_types[type(row.get("images") or row.get("image") or row.get("image_paths") or []).__name__] += 1
        geometry += int(any(key in row for key in ("geometry", "boxes", "mask", "coordinates", "bbox")))
    return {"path": str(source), "sha256": sha256_file(source), "row_count": len(rows), "conversation_length": {str(key): value for key, value in lengths.items()}, "sample_questions": questions, "sample_answers": answers, "image_ref_types": dict(image_types), "structured_geometry_rows": geometry}


def _expand_source_rows(kind: str, rows: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    expanded: list[tuple[str, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        if kind != "levir_caption":
            expanded.append((str(index), row))
            continue
        captions = row.get("captions") or row.get("sentences") or row.get("answers")
        captions = [captions] if isinstance(captions, str) else captions
        if not isinstance(captions, list):
            captions = [_field(row, "caption", "answer")]
        captions = [item for item in captions if item not in (None, "")]
        if not captions:
            expanded.append((str(index), row))
        for caption_index, caption in enumerate(captions):
            clone = dict(row)
            clone["answer"] = _answer(caption)
            expanded.append((f"{index}:{caption_index}", clone))
    return expanded


def _write_jsonl(root: Path, name: str, rows: Iterable[dict[str, Any]]) -> str:
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n" for row in rows)
    (root / name).write_text(payload, encoding="utf-8", newline="\n")
    return sha256_bytes(payload.encode("utf-8"))


def _build_into(spec_path: Path, output: Path, prompt_ref: str) -> dict[str, Any]:
    spec = load_source_spec(spec_path)
    registry, aliases = build_pair_registry(spec)
    exclusions = load_exclusions(spec, aliases)
    excluded_pairs = set(exclusions["mapped_ids"].values())
    prompt_path = Path(prompt_ref)
    if not prompt_path.is_file():
        prompt_path = Path(__file__).resolve().parents[2] / "prompts" / f"{prompt_ref}.md"
    if not prompt_path.is_file():
        raise ChangeCorpusBuildError("PROMPT_MISSING", prompt_ref)
    prompt_text = prompt_path.read_text(encoding="utf-8")
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    row_map: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    seen: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    registry_by_id = {item["canonical_pair_id"]: item for item in registry}
    for source in spec["sources"]:
        item_summary: dict[str, Any] = {"source_id": source["id"], "kind": source["kind"], "task": source.get("task"), "enabled": source.get("enabled", True), "rows": 0, "accepted": 0, "rejected": 0, "test_excluded": 0, "duplicates_removed": 0, "excluded_source_intentionally": False}
        if not source.get("enabled", True):
            item_summary.update(excluded_source_intentionally=True, reason=source.get("reason", "disabled_by_source_spec"))
            summary.append(item_summary)
            continue
        source_path = Path(source["path"])
        source_sha = sha256_file(source_path)
        rows = _read_json_or_jsonl(source_path)
        item_summary["rows"] = len(rows)
        for index, row in _expand_source_rows(source["kind"], rows):
            try:
                lookup = row if source["kind"] != "levir_caption" else {"filename": row.get("filename") or row.get("file_name"), "id": row.get("id"), "image_id": row.get("image_id")}
                pair_id = _resolve_pair(lookup, aliases)
                pair = registry_by_id[pair_id]
                if pair_id in excluded_pairs:
                    raise ChangeCorpusBuildError("EXCLUDED_CANONICAL_PAIR")
                if pair["source_split"] == "test":
                    item_summary["test_excluded"] += 1
                    rejected.append({"source_id": source["id"], "source_row_index": index, "canonical_pair_id": pair_id, "reason": "TEST_SPLIT_EXCLUDED"})
                    row_map.append({"source_id": source["id"], "source_row_index": index, "canonical_pair_id": pair_id, "split": "test", "status": "excluded"})
                    continue
                question, answer, turns = _conversation(row)
                if turns > 2:
                    raise ChangeCorpusBuildError("CONTEXT_DEPENDENT_MULTITURN")
                if not answer:
                    raise ChangeCorpusBuildError("MISSING_ANSWER")
                task = str(source["task"])
                explicit = row.get("task") or row.get("task_type")
                if explicit and explicit != task:
                    raise ChangeCorpusBuildError("SOURCE_TASK_SCHEMA_MISMATCH")
                if task == "change_qa" and not question:
                    raise ChangeCorpusBuildError("MISSING_QUESTION")
                if source["kind"] == "change_localization" and any(key in row for key in ("geometry", "boxes", "mask", "coordinates", "bbox")):
                    raise ChangeCorpusBuildError("STRUCTURED_LOCALIZATION_OUT_OF_V1")
                episode = _episode(source, row, index, pair, task, question, answer, source_sha)
                key = _supervision_key(episode)
                if key in seen:
                    seen[key]["provenance"].setdefault("duplicate_sources", []).append({"source_id": source["id"], "source_row_index": index})
                    item_summary["duplicates_removed"] += 1
                    continue
                seen[key] = episode
                (validation if pair["source_split"] == "validation" else train).append(episode)
                item_summary["accepted"] += 1
                row_map.append({"source_id": source["id"], "source_row_index": index, "canonical_pair_id": pair_id, "split": pair["source_split"], "status": "accepted", "task": task})
            except ChangeCorpusBuildError as exc:
                item_summary["rejected"] += 1
                rejected.append({"source_id": source["id"], "source_row_index": index, "reason": exc.code})
                row_map.append({"source_id": source["id"], "source_row_index": index, "status": "rejected", "reason": exc.code})
        summary.append(item_summary)
    train.sort(
        key=lambda episode: formal_train_order_key(
            episode,
            seed=FORMAL_TRAIN_ORDERING_SEED,
        )
    )
    train_pairs = {row["parent_sample_id"] for row in train}
    validation_pairs = {row["parent_sample_id"] for row in validation}
    if train_pairs & validation_pairs:
        raise ChangeCorpusBuildError("GLOBAL_SPLIT_LEAKAGE")
    output.mkdir(parents=True, exist_ok=True)
    train_sha = _write_jsonl(output, "train.jsonl", train)
    validation_sha = _write_jsonl(output, "validation.jsonl", validation)
    rejected_sha = _write_jsonl(output, "rejected.jsonl", rejected)
    registry_sha = _write_jsonl(output, "pair_registry.jsonl", registry)
    row_map_sha = _write_jsonl(output, "changechat_row_map.jsonl", row_map)
    source_summary_payload = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    (output / "source_summary.json").write_text(source_summary_payload, encoding="utf-8", newline="\n")
    descriptor_payload = json.dumps(change_target_contract_descriptor(), ensure_ascii=False, indent=2) + "\n"
    (output / "target_contract.json").write_text(descriptor_payload, encoding="utf-8", newline="\n")
    source_shas = {str(item["id"]): sha256_file(Path(item["path"])) for item in spec["sources"] if item.get("enabled", True) and item.get("path")}
    manifest = {
        "schema_version": CHANGE_SFT_EPISODE_SCHEMA_VERSION,
        "tool": "scripts/build_change_qwen_sft_corpus.py",
        "builder_git": builder_git_identity(),
        "source_spec_sha256": sha256_file(spec_path),
        "canonical_dataset": {"captions_sha256": sha256_file(Path(spec["canonical_dataset"]["captions"])), "image_root": spec["canonical_dataset"]["image_root"]},
        "change_prompt": {"ref": prompt_ref, "sha256": sha256_bytes(prompt_text.encode("utf-8"))},
        "target_contract": change_target_contract_identity(),
        "ordering": {
            "train": {
                "policy": FORMAL_TRAIN_ORDERING_POLICY,
                "seed": FORMAL_TRAIN_ORDERING_SEED,
                "key": "episode_id",
            },
            "validation": {"policy": VALIDATION_ORDERING_POLICY},
        },
        "exclusions": {key: value for key, value in exclusions.items() if key != "mapped_ids"},
        "global_split_policy": spec["split_policy"], "source_file_sha256": source_shas, "pair_registry_sha256": registry_sha,
        "outputs": {"train.jsonl_sha256": train_sha, "validation.jsonl_sha256": validation_sha, "rejected.jsonl_sha256": rejected_sha, "pair_registry.jsonl_sha256": registry_sha, "changechat_row_map.jsonl_sha256": row_map_sha, "source_summary.json_sha256": sha256_bytes(source_summary_payload.encode("utf-8")), "target_contract.json_sha256": sha256_bytes(descriptor_payload.encode("utf-8"))},
        "counts": {"pair_registry": len(registry), "unique_train_pairs": len(train_pairs), "unique_validation_pairs": len(validation_pairs), "train_episodes": len(train), "validation_episodes": len(validation), "duplicates_removed": sum(item["duplicates_removed"] for item in summary), "rejected": len(rejected), "by_source": {item["source_id"]: item for item in summary}, "by_task": dict(Counter(row["task"] for row in train + validation)), "by_split": {"train": len(train), "validation": len(validation), "test": 0}},
        "multi_turn_policy": {"enabled": False, "reason": "context_dependent_multiturn_v1", "source_excluded_intentionally": any(item["excluded_source_intentionally"] for item in summary)},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def build_corpus(source_spec: str | Path, output_dir: str | Path, prompt_ref: str) -> dict[str, Any]:
    """Build into a new directory and publish atomically. / 在新目录构建并原子发布。"""

    destination = Path(output_dir)
    if destination.exists():
        raise ChangeCorpusBuildError("OUTPUT_EXISTS", str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        manifest = _build_into(Path(source_spec), temporary, prompt_ref)
        os.replace(temporary, destination)
        return manifest
    except Exception:
        # Only remove the task-owned temporary directory. / 只清理本任务创建的临时目录。
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
