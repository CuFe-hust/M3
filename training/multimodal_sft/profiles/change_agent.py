"""Real ChangeAgent preparation profile for the generic trainer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from agents.change.prompt_contract import INITIAL_RESPONSE_SUFFIX, evidence_label
from training.multimodal_sft.change_target_contract import (
    CHANGE_SFT_EPISODE_SCHEMA_VERSION,
    CHANGE_TARGET_CONTRACT_NAME,
    CHANGE_TARGET_CONTRACT_VERSION,
    canonical_change_initial_result,
    change_target_contract_identity,
)
from training.multimodal_sft.contracts import ImageRef, PreparedMultimodalEpisode
from training.multimodal_sft.image_roots import ImageRootRegistry


class ChangeAgentDataError(ValueError):
    def __init__(self, code: str, episode_id: str = "") -> None:
        self.code = str(code)
        self.episode_id = str(episode_id)
        super().__init__(f"{self.code}: {self.episode_id}" if episode_id else self.code)


class ChangeAgentDataProfile:
    name = "change_agent"

    def __init__(self, *, data_manifest: str | Path | None = None, prompt_ref: str | None = None, prompt_file: str | Path | None = None) -> None:
        self.data_manifest = Path(data_manifest) if data_manifest else None
        self.prompt_ref = str(prompt_ref) if prompt_ref else None
        self.prompt_file = Path(prompt_file) if prompt_file else None
        self._manifest: dict[str, Any] | None = None
        self._prompt_text_cache: str | None = None
        self._last_files: dict[str, Path] = {}
        if self.data_manifest is not None:
            self._manifest = self._read_manifest(self.data_manifest)

    @staticmethod
    def _read_manifest(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise ChangeAgentDataError("DATA_MANIFEST_MISSING")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ChangeAgentDataError("DATA_MANIFEST_INVALID") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("change_prompt"), dict):
            raise ChangeAgentDataError("DATA_MANIFEST_INVALID")
        contract = payload.get("target_contract")
        if not isinstance(contract, dict):
            raise ChangeAgentDataError("TARGET_CONTRACT_IDENTITY_MISSING")
        expected = change_target_contract_identity()
        if any(contract.get(key) != expected[key] for key in expected):
            raise ChangeAgentDataError("TARGET_CONTRACT_IDENTITY_MISMATCH")
        return payload

    @staticmethod
    def _sha256_bytes(value: bytes) -> str:
        return hashlib.sha256(value).hexdigest()

    @classmethod
    def _sha256_file(cls, path: Path) -> str:
        return cls._sha256_bytes(path.read_bytes())

    def _verify_file_identity(self, path: Path, split: str) -> None:
        if self._manifest is None:
            raise ChangeAgentDataError("DATA_MANIFEST_REQUIRED")
        key = "train.jsonl_sha256" if split == "train" else "validation.jsonl_sha256"
        expected = self._manifest.get("outputs", {}).get(key)
        actual = self._sha256_file(path)
        if expected != actual:
            raise ChangeAgentDataError("DATA_MANIFEST_SHA_MISMATCH")
        self._last_files[split] = path.resolve()

    def _prompt_text(self) -> str:
        if self._prompt_text_cache is not None:
            return self._prompt_text_cache
        if self._manifest is None:
            raise ChangeAgentDataError("DATA_MANIFEST_REQUIRED")
        prompt_meta = self._manifest["change_prompt"]
        manifest_ref = str(prompt_meta.get("ref") or "")
        ref = self.prompt_ref or manifest_ref
        candidate: Path | None = self.prompt_file
        if candidate is None and ref:
            maybe = Path(ref)
            if maybe.is_file():
                candidate = maybe
            else:
                repo_prompt = Path(__file__).resolve().parents[3] / "prompts" / f"{ref}.md"
                if repo_prompt.is_file():
                    candidate = repo_prompt
        if candidate is None or not candidate.is_file():
            raise ChangeAgentDataError("CHANGE_PROMPT_SHA_MISMATCH")
        text = candidate.read_text(encoding="utf-8")
        if str(prompt_meta.get("sha256")) != self._sha256_bytes(text.encode("utf-8")):
            raise ChangeAgentDataError("CHANGE_PROMPT_SHA_MISMATCH")
        if self.prompt_ref and manifest_ref and self.prompt_ref != manifest_ref:
            raise ChangeAgentDataError("CHANGE_PROMPT_SHA_MISMATCH")
        self._prompt_text_cache = text
        return text

    @staticmethod
    def _read_rows(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            raise ChangeAgentDataError("DATA_FILE_MISSING")
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ChangeAgentDataError("DATA_JSON_INVALID") from exc
            if not isinstance(row, dict):
                raise ChangeAgentDataError("DATA_ROW_INVALID")
            rows.append(row)
        return rows

    def read(self, path: str | Path) -> Iterable[dict[str, Any]]:
        source = Path(path)
        split = "validation" if source.name.startswith(("validation", "val")) else "train"
        self._verify_file_identity(source, split)
        return iter(self._read_rows(source))

    def validate(self, episode: Mapping[str, Any]) -> None:
        episode_id = str(episode.get("episode_id") or "")
        if episode.get("schema_version") != CHANGE_SFT_EPISODE_SCHEMA_VERSION:
            raise ChangeAgentDataError("SCHEMA_VERSION", episode_id)
        if episode.get("task") not in {"change_caption", "change_qa"}:
            raise ChangeAgentDataError("UNKNOWN_TASK", episode_id)
        if episode.get("input_contract") not in {"semantic_pair_v1", "runtime_initial_v1"}:
            raise ChangeAgentDataError("INVALID_INPUT_CONTRACT", episode_id)
        images = episode.get("images")
        if not isinstance(images, list) or len(images) < 2:
            raise ChangeAgentDataError("MISSING_T2", episode_id)
        roles = tuple(item.get("role") for item in images[:2] if isinstance(item, dict))
        if roles != ("raw_full_t1", "raw_full_t2"):
            raise ChangeAgentDataError("INVALID_ROLE_ORDER", episode_id)
        if not isinstance(episode.get("request_payload"), dict):
            raise ChangeAgentDataError("REQUEST_PAYLOAD_INVALID", episode_id)
        self._canonical_target_result(episode)

    def _canonical_target_result(self, episode: Mapping[str, Any]) -> dict[str, Any]:
        """Require an already-canonical v2 target. / 要求已规范化的 v2 目标。"""

        episode_id = str(episode.get("episode_id") or "")
        target = episode.get("target")
        if not isinstance(target, dict) or target.get("response_schema") != CHANGE_TARGET_CONTRACT_NAME:
            raise ChangeAgentDataError("INVALID_TARGET_SCHEMA", episode_id)
        if target.get("contract_version") != CHANGE_TARGET_CONTRACT_VERSION:
            raise ChangeAgentDataError("TARGET_CONTRACT_VERSION_MISMATCH", episode_id)
        raw = target.get("result")
        if not isinstance(raw, Mapping):
            raise ChangeAgentDataError("INVALID_TARGET_SCHEMA", episode_id)
        try:
            canonical = canonical_change_initial_result(raw)
        except Exception as exc:  # noqa: BLE001 - stable profile boundary
            raise ChangeAgentDataError("INVALID_TARGET_SCHEMA", episode_id) from exc
        if raw != canonical:
            raise ChangeAgentDataError("NONCANONICAL_TARGET_RESULT", episode_id)
        return canonical

    def render_messages(self, episode: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
        """Render the exact training conversation without loading image bytes."""

        self.validate(episode)
        canonical_result = self._canonical_target_result(episode)
        prompt_text = self._prompt_text()
        user: list[dict[str, Any]] = [{"type": "text", "text": "Decision stage: initial. Compare the next two authoritative raw images first."}]
        for image in episode["images"]:
            user.append({"type": "text", "text": evidence_label(str(image["role"]))})
            user.append({"type": "image"})
        user.append({"type": "text", "text": json.dumps(episode["request_payload"], ensure_ascii=False, separators=(",", ":"))})
        return (
            {"role": "system", "content": prompt_text + "\n\n" + INITIAL_RESPONSE_SUFFIX},
            {"role": "user", "content": user},
            {"role": "assistant", "content": [{"type": "text", "text": json.dumps(canonical_result, ensure_ascii=False, separators=(",", ":"))}]},
        )

    def prepare(self, episode: Mapping[str, Any], *, image_roots: Any, split: str, epoch: int, seed: int | str) -> PreparedMultimodalEpisode:
        self.validate(episode)
        canonical_result = self._canonical_target_result(episode)
        if str(episode.get("split")) != split:
            raise ChangeAgentDataError("SPLIT_MISMATCH", str(episode.get("episode_id") or ""))
        registry = image_roots if isinstance(image_roots, ImageRootRegistry) else ImageRootRegistry(dict(image_roots or {}))
        resolved_images = []
        refs: list[ImageRef] = []
        for image in episode["images"]:
            source = str(image["image_source"])
            relative = str(image["path"])
            resolved_images.append(registry.load_rgb(source, relative))
            refs.append(ImageRef(source, str(registry.resolve(source, relative)), str(image["role"])))
        messages = self.render_messages(episode)
        return PreparedMultimodalEpisode(
            episode_id=str(episode["episode_id"]), task_profile=self.name, messages=messages,
            images=tuple(resolved_images), image_roles=tuple(ref.role for ref in refs),
            target_schema="ChangeInitialResult",
            metadata={
                "image_refs": tuple(refs),
                "request_payload": episode["request_payload"],
                "target": {
                    "response_schema": CHANGE_TARGET_CONTRACT_NAME,
                    "contract_version": CHANGE_TARGET_CONTRACT_VERSION,
                    "result": canonical_result,
                },
                "prompt_ref": self._manifest["change_prompt"]["ref"] if self._manifest else None,
                "epoch": int(epoch),
                "seed": str(seed),
            },
        )

    def identity_contract(self, image_roots: Any) -> dict[str, Any]:
        registry = image_roots if isinstance(image_roots, ImageRootRegistry) else ImageRootRegistry(dict(image_roots or {}))
        if self._manifest is None:
            return {
                "data_profile": self.name,
                "target_contract": change_target_contract_identity(),
                "image_root_contract": registry.contract(),
            }
        train = self._last_files.get("train")
        validation = self._last_files.get("validation")
        self._prompt_text()
        return {
            "data_profile": self.name,
            "target_contract": change_target_contract_identity(),
            "data_manifest_sha256": self._sha256_file(self.data_manifest) if self.data_manifest else None,
            "train_file_sha256": self._sha256_file(train) if train else None,
            "validation_file_sha256": self._sha256_file(validation) if validation else None,
            "prompt_ref": self._manifest["change_prompt"]["ref"],
            "prompt_text_sha256": self._manifest["change_prompt"]["sha256"],
            "image_sources": sorted(registry.roots),
            "image_root_contract": registry.contract(),
        }
