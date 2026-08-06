"""Create explicitly non-formal fake-system input without altering request evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from m3rs_eval.contracts import ContractError, RequestRecord, read_jsonl


class FixtureInputError(ValueError):
    """Raised when isolated fixture command input cannot be built safely."""


@dataclass(frozen=True)
class FixtureCommandInput:
    path: Path
    ephemeral: bool = True
    eligible_for_history: bool = False
    formal: bool = False


def prepare_fixture_command_input(
    requests_path: Path,
    references_path: Path,
    destination: Path,
    *,
    profile: str,
    formal_execution: bool,
) -> FixtureCommandInput:
    """Join fixture requests/references into an ephemeral fake-system-only JSONL file."""
    if profile != "fixture" or formal_execution:
        raise FixtureInputError("fixture command input is forbidden outside non-formal fixture execution")
    try:
        requests = read_jsonl(Path(requests_path), RequestRecord, unique_key="sample_id")
    except ContractError as error:
        raise FixtureInputError(f"invalid persisted requests: {error}") from error
    references = _read_references(Path(references_path))
    request_ids = [request.sample_id for request in requests]
    if set(request_ids) != set(references) or len(request_ids) != len(references):
        raise FixtureInputError("fixture requests and references must have one-to-one sample IDs")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for request in requests:
            payload = request.to_dict()
            payload["fixture_prediction"] = _prediction_hint(references[request.sample_id])
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    return FixtureCommandInput(path=target)


def _read_references(path: Path) -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise FixtureInputError(f"could not read fixture references: {path}") from error
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise FixtureInputError(f"fixture references line {line_number} is invalid JSON") from error
        sample_id = row.get("sample_id") if isinstance(row, dict) else None
        if not isinstance(sample_id, str) or not sample_id or sample_id in references:
            raise FixtureInputError(f"fixture references line {line_number} has an invalid or duplicate sample_id")
        references[sample_id] = row
    return references


def _prediction_hint(reference: dict[str, Any]) -> str | dict[str, Any]:
    answer = reference.get("answer")
    if isinstance(answer, str) and answer:
        return answer
    boxes = reference.get("boxes")
    if isinstance(boxes, list) and boxes:
        return {"boxes": boxes}
    raise FixtureInputError("fixture reference requires a nonempty answer or boxes")
