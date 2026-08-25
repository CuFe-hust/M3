"""Canonical ChangeAgent SFT target serialization identity.

ChangeAgent SFT 目标的规范序列化身份。Runtime validators may accept legacy
input fields, while formal SFT rows must equal the current public serialization.
运行时可兼容读取旧字段，但正式 SFT 行必须等于当前公共序列化结果。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from agents.change.schema import ChangeInitialResult
from agents.schema import VisualEvidence


CHANGE_SFT_EPISODE_SCHEMA_VERSION = 2
CHANGE_TARGET_CONTRACT_NAME = "ChangeInitialResult"
CHANGE_TARGET_CONTRACT_VERSION = "change_initial_result_v2_no_legacy_evidence"


def canonical_change_initial_result(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the public JSON serialization after legacy-input normalization.

    对兼容输入完成规范化后，返回当前公共 JSON 序列化。
    """

    return ChangeInitialResult.model_validate(dict(value)).model_dump(
        mode="json",
        by_alias=True,
        exclude_none=False,
    )


def change_target_contract_descriptor() -> dict[str, Any]:
    """Return the deterministic formal-output descriptor. / 返回确定性正式输出描述。"""

    return {
        "name": CHANGE_TARGET_CONTRACT_NAME,
        "version": CHANGE_TARGET_CONTRACT_VERSION,
        "episode_schema_version": CHANGE_SFT_EPISODE_SCHEMA_VERSION,
        "serialization_json_schema": ChangeInitialResult.model_json_schema(mode="serialization"),
        "result_fields": list(ChangeInitialResult.model_fields),
        "visual_evidence_fields": list(VisualEvidence.model_fields),
        "legacy_input_only_fields": {
            "result": ["evidence"],
            "visual_evidence": ["confidence"],
        },
    }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def change_target_contract_identity() -> dict[str, Any]:
    """Return the compact identity embedded in corpus/checkpoint manifests.

    返回写入 corpus/checkpoint manifest 的紧凑身份。
    """

    descriptor = change_target_contract_descriptor()
    return {
        "name": CHANGE_TARGET_CONTRACT_NAME,
        "version": CHANGE_TARGET_CONTRACT_VERSION,
        "sha256": hashlib.sha256(_canonical_json(descriptor)).hexdigest(),
        "episode_schema_version": CHANGE_SFT_EPISODE_SCHEMA_VERSION,
        "result_fields": descriptor["result_fields"],
        "visual_evidence_fields": descriptor["visual_evidence_fields"],
    }
