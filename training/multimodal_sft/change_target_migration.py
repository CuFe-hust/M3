"""Read-only audit and safe-reference migration for Change SFT targets.

Change SFT 目标的只读审计、安全参考迁移与精确 corpus 对比。
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from training.multimodal_sft.change_target_contract import (
    CHANGE_SFT_EPISODE_SCHEMA_VERSION,
    CHANGE_TARGET_CONTRACT_NAME,
    CHANGE_TARGET_CONTRACT_VERSION,
    canonical_change_initial_result,
    change_target_contract_descriptor,
    change_target_contract_identity,
)


class ChangeTargetMigrationError(ValueError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}" if detail else code)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        raise ChangeTargetMigrationError("CORPUS_FILE_MISSING", str(path))
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ChangeTargetMigrationError("CORPUS_JSON_INVALID", f"{path.name}:{line_no}") from exc
            if not isinstance(row, dict):
                raise ChangeTargetMigrationError("CORPUS_ROW_INVALID", f"{path.name}:{line_no}")
            yield line_no, row


def _semantic_content(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return any(_semantic_content(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_semantic_content(item) for item in value)
    return True


def recursive_diff(left: Any, right: Any, path: str = "") -> list[str]:
    """Return deterministic JSON pointer-like changed paths. / 返回确定性差异路径。"""

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}/{key}"
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(recursive_diff(left[key], right[key], child))
        return paths
    if isinstance(left, list) and isinstance(right, list):
        paths = []
        for index in range(max(len(left), len(right))):
            child = f"{path}/{index}"
            if index >= len(left) or index >= len(right):
                paths.append(child)
            else:
                paths.extend(recursive_diff(left[index], right[index], child))
        return paths
    return [] if left == right else [path or "/"]


def _allowed_target_diff(path: str) -> bool:
    return path == "/target/result/evidence" or bool(
        re.fullmatch(r"/target/result/evidence_items/\d+/confidence", path)
    )


def audit_target_contract(
    *, train: str | Path, validation: str | Path, manifest: str | Path
) -> dict[str, Any]:
    """Audit v1/v2 rows without modifying inputs. / 只读审计 v1/v2 行。"""

    files = {"train": Path(train), "validation": Path(validation)}
    manifest_path = Path(manifest)
    if not manifest_path.is_file():
        raise ChangeTargetMigrationError("MANIFEST_MISSING", str(manifest_path))
    counters: Counter[str] = Counter()
    schemas: Counter[str] = Counter()
    diff_paths: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for split, path in files.items():
        for line_no, row in _jsonl(path):
            counters["total_rows"] += 1
            counters[f"{split}_rows"] += 1
            schemas[str(row.get("schema_version"))] += 1
            target = row.get("target")
            raw = target.get("result") if isinstance(target, dict) else None
            if not isinstance(raw, dict):
                counters["schema_type_errors"] += 1
                if len(errors) < 20:
                    errors.append({"split": split, "line": line_no, "episode_id": row.get("episode_id"), "code": "INVALID_TARGET_RESULT"})
                continue
            if "evidence" in raw:
                counters["result_evidence_rows"] += 1
                if _semantic_content(raw["evidence"]):
                    counters["evidence_nonempty_rows"] += 1
                else:
                    counters["evidence_empty_rows"] += 1
            items = raw.get("evidence_items")
            if isinstance(items, list):
                counters["evidence_items_total"] += len(items)
                for item in items:
                    if isinstance(item, dict) and "confidence" in item:
                        counters["evidence_item_confidence_fields"] += 1
            try:
                canonical = canonical_change_initial_result(raw)
            except Exception as exc:  # noqa: BLE001 - report stable type only
                counters["schema_type_errors"] += 1
                if len(errors) < 20:
                    errors.append({"split": split, "line": line_no, "episode_id": row.get("episode_id"), "code": "INVALID_TARGET_SCHEMA", "error_type": type(exc).__name__})
                continue
            paths = recursive_diff(raw, canonical, "/target/result")
            if paths:
                counters["canonicalization_changed_rows"] += 1
            other = [path for path in paths if not _allowed_target_diff(path)]
            counters["other_noncanonical_diff_count"] += len(other)
            diff_paths.update(paths)
            if paths and len(examples) < 20:
                examples.append({"split": split, "line": line_no, "episode_id": row.get("episode_id"), "diff_paths": paths})
    for key in (
        "total_rows", "train_rows", "validation_rows", "result_evidence_rows",
        "evidence_empty_rows", "evidence_nonempty_rows", "evidence_items_total",
        "evidence_item_confidence_fields", "canonicalization_changed_rows",
        "other_noncanonical_diff_count", "schema_type_errors",
    ):
        counters.setdefault(key, 0)
    migration_allowed = (
        counters["evidence_nonempty_rows"] == 0
        and counters["other_noncanonical_diff_count"] == 0
        and counters["schema_type_errors"] == 0
    )
    return {
        "schema_version": 1,
        "status": "PASS" if migration_allowed else "TARGET_CONTRACT_MIGRATION_REVIEW_REQUIRED",
        "migration_allowed": migration_allowed,
        "input": {
            "train": str(files["train"]), "train_sha256": _sha256_file(files["train"]),
            "validation": str(files["validation"]), "validation_sha256": _sha256_file(files["validation"]),
            "manifest": str(manifest_path), "manifest_sha256": _sha256_file(manifest_path),
        },
        "counts": dict(counters),
        "episode_schema_versions": dict(sorted(schemas.items())),
        "diff_paths": dict(sorted(diff_paths.items())),
        "examples": examples,
        "errors": errors,
        "current_target_contract": change_target_contract_identity(),
    }


def write_json_atomic(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    os.replace(temporary, target)


def canonical_v2_row(row: Mapping[str, Any]) -> dict[str, Any]:
    migrated = copy.deepcopy(dict(row))
    target = migrated.get("target")
    if not isinstance(target, dict) or not isinstance(target.get("result"), dict):
        raise ChangeTargetMigrationError("INVALID_TARGET_RESULT", str(row.get("episode_id") or ""))
    migrated["schema_version"] = CHANGE_SFT_EPISODE_SCHEMA_VERSION
    target["response_schema"] = CHANGE_TARGET_CONTRACT_NAME
    target["contract_version"] = CHANGE_TARGET_CONTRACT_VERSION
    target["result"] = canonical_change_initial_result(target["result"])
    return migrated


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            payload = json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
            handle.write(payload)
            digest.update(payload.encode("utf-8"))
            count += 1
    return count, digest.hexdigest()


def migrate_reference_corpus(
    *, old_dir: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Create a safe canonical reference mirror, never an authoritative rebuild.

    创建安全的规范参考镜像；它不能替代从原始 source 的权威重建。
    """

    old = Path(old_dir)
    destination = Path(output_dir)
    if destination.exists():
        raise ChangeTargetMigrationError("OUTPUT_EXISTS", str(destination))
    audit = audit_target_contract(train=old / "train.jsonl", validation=old / "validation.jsonl", manifest=old / "manifest.json")
    if not audit["migration_allowed"]:
        raise ChangeTargetMigrationError("TARGET_CONTRACT_MIGRATION_REVIEW_REQUIRED")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        outputs: dict[str, Any] = {}
        for split in ("train", "validation"):
            count, digest = _write_rows(temporary / f"{split}.jsonl", (canonical_v2_row(row) for _, row in _jsonl(old / f"{split}.jsonl")))
            outputs[f"{split}_rows"] = count
            outputs[f"{split}.jsonl_sha256"] = digest
        for name in ("rejected.jsonl", "pair_registry.jsonl", "changechat_row_map.jsonl", "source_summary.json"):
            source = old / name
            if source.is_file():
                shutil.copyfile(source, temporary / name)
                outputs[f"{name}_sha256"] = _sha256_file(source)
        descriptor_payload = json.dumps(change_target_contract_descriptor(), ensure_ascii=False, indent=2) + "\n"
        (temporary / "target_contract.json").write_text(descriptor_payload, encoding="utf-8", newline="\n")
        outputs["target_contract.json_sha256"] = hashlib.sha256(descriptor_payload.encode("utf-8")).hexdigest()
        manifest = json.loads((old / "manifest.json").read_text(encoding="utf-8"))
        manifest["schema_version"] = CHANGE_SFT_EPISODE_SCHEMA_VERSION
        manifest["target_contract"] = change_target_contract_identity()
        manifest.setdefault("migration_reference", {})
        manifest["migration_reference"] = {"authoritative_training_corpus": False, "source_manifest_sha256": _sha256_file(old / "manifest.json")}
        manifest_outputs = manifest.setdefault("outputs", {})
        for key in ("train.jsonl_sha256", "validation.jsonl_sha256", "target_contract.json_sha256"):
            manifest_outputs[key] = outputs[key]
        (temporary / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        report = {"status": "PASS", "authoritative_training_corpus": False, "audit": audit, "outputs": outputs, "target_contract": change_target_contract_identity()}
        write_json_atomic(temporary / "migration_report.json", report)
        os.replace(temporary, destination)
        return report
    except Exception:
        # Only remove the task-owned temporary directory. / 只清理本任务创建的临时目录。
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _rows_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for _, row in _jsonl(path):
        episode_id = str(row.get("episode_id") or "")
        if not episode_id or episode_id in rows:
            raise ChangeTargetMigrationError("DUPLICATE_OR_MISSING_EPISODE_ID", episode_id)
        rows[episode_id] = row
    return rows


def compare_corpora(*, old_dir: str | Path, new_dir: str | Path) -> dict[str, Any]:
    """Prove every new row equals the allowlisted v2 transform. / 证明每行仅含白名单变化。"""

    old_root, new_root = Path(old_dir), Path(new_dir)
    split_reports: dict[str, Any] = {}
    unexpected_paths: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    total_expected = 0
    total_unexpected = 0
    for split in ("train", "validation"):
        old_rows = _rows_by_id(old_root / f"{split}.jsonl")
        new_rows = _rows_by_id(new_root / f"{split}.jsonl")
        missing = sorted(set(old_rows) - set(new_rows))
        added = sorted(set(new_rows) - set(old_rows))
        unexpected = 0
        for episode_id in sorted(set(old_rows) & set(new_rows)):
            expected = canonical_v2_row(old_rows[episode_id])
            paths = recursive_diff(expected, new_rows[episode_id])
            if paths:
                unexpected += 1
                unexpected_paths.update(paths)
                if len(examples) < 20:
                    examples.append({"split": split, "episode_id": episode_id, "unexpected_paths": paths})
            else:
                total_expected += 1
        total_unexpected += unexpected + len(missing) + len(added)
        split_reports[split] = {"old_rows": len(old_rows), "new_rows": len(new_rows), "missing_episode_ids": len(missing), "added_episode_ids": len(added), "unexpected_diff_rows": unexpected, "old_unique_pairs": len({row.get("parent_sample_id") for row in old_rows.values()}), "new_unique_pairs": len({row.get("parent_sample_id") for row in new_rows.values()}), "by_task_old": dict(Counter(str(row.get("task")) for row in old_rows.values())), "by_task_new": dict(Counter(str(row.get("task")) for row in new_rows.values()))}
    supporting: dict[str, Any] = {}
    for name in ("rejected.jsonl", "pair_registry.jsonl", "changechat_row_map.jsonl", "source_summary.json"):
        old_path, new_path = old_root / name, new_root / name
        old_exists, new_exists = old_path.is_file(), new_path.is_file()
        if old_exists and new_exists:
            old_sha, new_sha = _sha256_file(old_path), _sha256_file(new_path)
            supporting[name] = {"old_present": True, "new_present": True, "old_sha256": old_sha, "new_sha256": new_sha, "identical": old_sha == new_sha}
            total_unexpected += int(old_sha != new_sha)
        else:
            supporting[name] = {
                "old_present": old_exists,
                "new_present": new_exists,
                "old_sha256": _sha256_file(old_path) if old_exists else None,
                "new_sha256": _sha256_file(new_path) if new_exists else None,
                "identical": old_exists == new_exists,
            }
            total_unexpected += int(old_exists != new_exists)
    return {"status": "PASS" if total_unexpected == 0 else "CORPUS_MIGRATION_UNEXPECTED_DIFF", "expected_transform_rows": total_expected, "unexpected_diff_count": total_unexpected, "splits": split_reports, "supporting_artifacts": supporting, "unexpected_paths": dict(sorted(unexpected_paths.items())), "examples": examples, "target_contract": change_target_contract_identity()}
