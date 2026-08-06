"""Shared dataset materialization primitives for the locked benchmark scopes."""

from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

from m3rs_eval.config import DatasetConfig
from m3rs_eval.contracts import RequestRecord, write_jsonl
from m3rs_eval.registry import MetricRegistry


class DatasetError(ValueError):
    """Raised when a configured dataset cannot satisfy its locked protocol."""


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class DatasetMaterialization:
    dataset: str
    requests_path: Path
    references_path: Path
    expected_samples: int
    task_counts: dict[str, int]
    manifest_hash: str
    coverage: dict[str, Any]


@dataclass(frozen=True)
class MaterializedExample:
    request: dict[str, Any]
    reference: dict[str, Any]
    annotation_path: Path
    source_label: str


class DatasetAdapter(Protocol):
    dataset: str

    def preflight(self) -> list[CheckResult]: ...

    def materialize(
        self, mode: str, limit: int | None, destination: Path
    ) -> DatasetMaterialization: ...

    def evaluate(
        self,
        materialization: DatasetMaterialization,
        predictions_path: Path,
        registry: MetricRegistry,
        *,
        context: Any,
        log_dir: Path,
    ) -> Any: ...


class BaseDatasetAdapter:
    """Keeps dataset parsing separate from contract-safe JSONL emission."""

    dataset: str
    expected_files: tuple[str, ...]
    supported_profiles = frozenset({"fixture", "official"})

    def __init__(
        self,
        config: DatasetConfig,
        protocol: Mapping[str, Any],
        registry: MetricRegistry,
    ) -> None:
        self.config = config
        self.protocol = protocol
        self.registry = registry

    def preflight(self) -> list[CheckResult]:
        try:
            self._load_examples()
        except DatasetError as error:
            return [CheckResult(name=self.dataset, passed=False, detail=str(error))]
        return [CheckResult(name=self.dataset, passed=True, detail="dataset layout and scope verified")]

    def materialize(
        self, mode: str, limit: int | None, destination: Path
    ) -> DatasetMaterialization:
        if mode not in {"smoke", "full"}:
            raise DatasetError(f"{self.dataset}: mode must be 'smoke' or 'full'")
        if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0):
            raise DatasetError(f"{self.dataset}: limit must be a positive integer or None")
        if mode == "full" and limit is not None:
            raise DatasetError(f"{self.dataset}: full mode does not permit a limit")

        examples = self._load_examples()
        self._validate_unique_sample_ids(examples)
        selected = self._select_examples(examples, mode, limit)
        if not selected:
            raise DatasetError(f"{self.dataset}: locked scope contains no materializable samples")

        requests = [self._request_record(example.request) for example in selected]
        request_ids = [request.sample_id for request in requests]
        reference_ids = [str(example.reference["sample_id"]) for example in selected]
        if request_ids != reference_ids:
            raise DatasetError(f"{self.dataset}: requests and references lost sample alignment")

        destination = Path(destination)
        requests_path = destination / f"{self.dataset}_requests.jsonl"
        references_path = destination / f"{self.dataset}_references.jsonl"
        write_jsonl(requests_path, requests)
        _write_raw_jsonl(references_path, [example.reference for example in selected])

        task_counts = _task_counts(request.task for request in requests)
        coverage = self._coverage(selected, examples, task_counts)
        manifest_hash = _manifest_hash(
            self._asset_root(),
            {example.annotation_path for example in selected},
            requests,
        )
        return DatasetMaterialization(
            dataset=self.dataset,
            requests_path=requests_path,
            references_path=references_path,
            expected_samples=len(selected),
            task_counts=task_counts,
            manifest_hash=manifest_hash,
            coverage=coverage,
        )

    def evaluate(
        self,
        materialization: DatasetMaterialization,
        predictions_path: Path,
        registry: MetricRegistry,
        *,
        context: Any,
        log_dir: Path,
    ) -> Any:
        """Delegate to the canonical tolerant Task 5 evaluator."""
        from m3rs_eval.evaluation import MetricContext, evaluate_materialization

        if not isinstance(context, MetricContext):
            raise DatasetError(f"{self.dataset}: context must be a MetricContext")
        if registry is not self.registry:
            raise DatasetError(f"{self.dataset}: evaluate registry must match adapter registry")
        return evaluate_materialization(
            self,
            materialization,
            Path(predictions_path),
            registry,
            context=context,
            log_dir=Path(log_dir),
        )

    def _load_examples(self) -> list[MaterializedExample]:
        self._require_profile_layout()
        examples = self._parse_examples()
        if not examples:
            raise DatasetError(f"{self.dataset}: no records were parsed from profile '{self.config.profile}'")
        self._validate_image_files(examples)
        return examples

    def _asset_root(self) -> Path:
        root = (self.config.asset_root or self.config.root).resolve()
        if not root.is_dir():
            raise DatasetError(f"{self.dataset}: configured asset root '{root}' is not a directory")
        return root

    def _validate_image_files(self, examples: Iterable[MaterializedExample]) -> None:
        asset_root = self._asset_root()
        for example in examples:
            for image in example.request["images"]:
                image_path = Path(image).resolve()
                try:
                    image_path.relative_to(asset_root)
                except ValueError as error:
                    raise DatasetError(
                        f"{self.dataset}: image '{image}' at {example.source_label} escapes asset root "
                        f"'{asset_root}'"
                    ) from error
                try:
                    mode = image_path.stat().st_mode
                except OSError as error:
                    raise DatasetError(
                        f"{self.dataset}: image '{image}' at {example.source_label} is missing"
                    ) from error
                if not stat.S_ISREG(mode):
                    raise DatasetError(
                        f"{self.dataset}: image '{image}' at {example.source_label} is not a regular file"
                    )

    def _validate_unique_sample_ids(self, examples: Iterable[MaterializedExample]) -> None:
        seen: dict[str, str] = {}
        for example in examples:
            sample_id = str(example.request["sample_id"])
            previous = seen.get(sample_id)
            if previous is not None:
                raise DatasetError(
                    f"{self.dataset}: duplicate generated sample_id '{sample_id}' at "
                    f"{example.source_label}; first seen at {previous}"
                )
            seen[sample_id] = example.source_label

    def _select_examples(
        self, examples: list[MaterializedExample], mode: str, limit: int | None
    ) -> list[MaterializedExample]:
        if mode != "smoke" or limit is None:
            return examples
        groups: dict[str, list[MaterializedExample]] = {}
        for example in examples:
            groups.setdefault(self._scope_label(example), []).append(example)
        return [example for scope in sorted(groups) for example in groups[scope][:limit]]

    def _scope_dimensions(self, example: MaterializedExample) -> dict[str, str]:
        return {"task": str(example.request["task"])}

    def _scope_label(self, example: MaterializedExample) -> str:
        return "|".join(f"{key}={value}" for key, value in self._scope_dimensions(example).items())

    def _require_profile_layout(self) -> None:
        if self.config.profile not in self.supported_profiles:
            supported = ", ".join(sorted(self.supported_profiles))
            raise DatasetError(
                f"{self.dataset}: unsupported adapter profile '{self.config.profile}' at root "
                f"'{self.config.root}'; supported profiles: {supported}"
            )
        missing = [name for name in self.expected_files if not (self.config.root / name).is_file()]
        if missing:
            expected = ", ".join(self.expected_files)
            raise DatasetError(
                f"{self.dataset}: profile '{self.config.profile}' expects file(s) {expected} "
                f"under configured root '{self.config.root}'; missing: {', '.join(missing)}"
            )

    def _parse_examples(self) -> list[MaterializedExample]:
        raise NotImplementedError

    def _coverage(
        self,
        selected: list[MaterializedExample],
        total: list[MaterializedExample],
        task_counts: dict[str, int],
    ) -> dict[str, Any]:
        selected_counts = _scope_counts(selected, self._scope_label)
        total_counts = _scope_counts(total, self._scope_label)
        return {
            "profile": self.config.profile,
            "tasks": task_counts,
            "requested_samples": len(selected),
            "total_samples": len(total),
            "scopes": {
                scope: {"selected": selected_counts.get(scope, 0), "total": total_counts[scope]}
                for scope in sorted(total_counts)
            },
        }

    def _request_record(self, raw: Mapping[str, Any]) -> RequestRecord:
        payload = dict(raw)
        payload["request_hash"] = _request_hash(payload, self._asset_root())
        return RequestRecord.from_dict(payload)

    def _annotation_json(self, name: str) -> tuple[Path, Any]:
        path = self.config.root / name
        try:
            return path, json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise DatasetError(f"{self.dataset}: could not parse annotation file '{path}': {error}") from error

    def _annotation_jsonl(self, name: str) -> tuple[Path, list[dict[str, Any]]]:
        path = self.config.root / name
        rows: list[dict[str, Any]] = []
        try:
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    raise DatasetError(f"{self.dataset}: '{path}' line {number} is blank")
                parsed = json.loads(line)
                if not isinstance(parsed, dict):
                    raise DatasetError(f"{self.dataset}: '{path}' line {number} is not an object")
                rows.append(parsed)
        except (OSError, json.JSONDecodeError) as error:
            raise DatasetError(f"{self.dataset}: could not parse annotation file '{path}': {error}") from error
        return path, rows


def _request_hash(payload: Mapping[str, Any], asset_root: Path) -> str:
    encoded = json.dumps(
        _portable_request_payload(payload, asset_root),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_raw_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _task_counts(tasks: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task] = counts.get(task, 0) + 1
    return dict(sorted(counts.items()))


def _scope_counts(
    examples: Iterable[MaterializedExample], scope_label: Any
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in examples:
        scope = scope_label(example)
        counts[scope] = counts.get(scope, 0) + 1
    return counts


def _manifest_hash(asset_root: Path, annotation_paths: Iterable[Path], requests: Iterable[RequestRecord]) -> str:
    annotations: list[dict[str, Any]] = []
    for path in annotation_paths:
        try:
            relative = path.resolve().relative_to(asset_root).as_posix()
            payload = path.read_bytes()
        except (OSError, ValueError) as error:
            raise DatasetError(f"could not hash annotation file '{path}': {error}") from error
        annotations.append(
            {
                "annotation_path": relative,
                "annotation_sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
            }
        )
    manifest = {
        "annotations": sorted(annotations, key=lambda entry: entry["annotation_path"]),
        "requests": [_portable_request_payload(request.to_dict(), asset_root) for request in requests],
    }
    encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _portable_request_payload(payload: Mapping[str, Any], asset_root: Path) -> dict[str, Any]:
    portable = dict(payload)
    images = payload.get("images")
    if not isinstance(images, (list, tuple)):
        raise DatasetError("request images must be a list before manifest hashing")
    portable["images"] = [_portable_image_path(image, asset_root) for image in images]
    return portable


def _portable_image_path(image: object, asset_root: Path) -> str:
    image_path = Path(str(image)).resolve()
    try:
        relative = image_path.relative_to(asset_root)
    except ValueError as error:
        raise DatasetError(
            f"image '{image}' escapes deterministic asset root '{asset_root}'"
        ) from error
    try:
        mode = image_path.stat().st_mode
    except OSError as error:
        raise DatasetError(f"image '{image}' is missing while hashing manifest") from error
    if not stat.S_ISREG(mode):
        raise DatasetError(f"image '{image}' is not a regular file while hashing manifest")
    return relative.as_posix()
