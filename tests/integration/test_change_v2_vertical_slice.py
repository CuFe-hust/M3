from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from agents.base import AgentContext
from agents.change.agent import ChangeAgent
from agents.change.settings import (
    AgentChangeSettings,
    ChangeProposalSettings,
    ChangeSemanticSettings,
)
from agents.errors import AgentExecutionError
from agents.visual_base import PromptBinding
from data.schema import GroundTruth, ImageRef, UnifiedSample
from models.base import DenseSemanticOutput, ModelCacheIdentity


_PROMPT = PromptBinding(
    text="Raw T1/T2 evidence is authoritative; auxiliary maps are hints only.",
    version="v2",
)


class _Budget:
    def __init__(self) -> None:
        self.qwen_calls = 0

    def reserve_qwen(self) -> None:
        self.qwen_calls += 1

    def reserve_deepseek(self) -> None:
        pass


class _QwenClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return ModelCacheIdentity(
            model="qwen-logical-test",
            generation={"temperature": 0.0},
            client_version="test-v1",
        )

    async def complete_json(self, *, messages, response_model, request_meta, **kwargs):
        self.calls.append(
            {"messages": messages, "request_hash": request_meta.request_hash}
        )
        return response_model.model_validate(
            {
                "agent_name": "change_agent",
                "answer": "A localized change requires raw-image confirmation.",
                "status": "completed",
            }
        )


class _DenseClient:
    def __init__(
        self,
        *,
        mismatch: bool = False,
        model: str = "segformer-logical-test",
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.outputs = _dense_outputs(mismatch=mismatch)
        self.model = model

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return ModelCacheIdentity(
            model=self.model,
            generation={"backend": "fake"},
            client_version="test-v1",
        )

    def infer(self, image: Any, **kwargs: Any) -> DenseSemanticOutput:
        self.calls.append({"image": image, **kwargs})
        return self.outputs[len(self.calls) - 1]


def _dense_outputs(*, mismatch: bool) -> tuple[DenseSemanticOutput, DenseSemanticOutput]:
    rng = np.random.default_rng(79)
    first_features = rng.normal(size=(8, 16, 16)).astype(np.float32)
    second_features = first_features.copy()
    second_features[:, 4:8, 4:8] *= np.float32(-1.0)
    first_probabilities = np.empty((2, 16, 16), dtype=np.float32)
    first_probabilities[0] = 0.95
    first_probabilities[1] = 0.05
    second_probabilities = first_probabilities.copy()
    second_probabilities[0, 4:8, 4:8] = 0.05
    second_probabilities[1, 4:8, 4:8] = 0.95

    def output(probabilities: np.ndarray, features: np.ndarray, size: tuple[int, int]):
        return DenseSemanticOutput(
            probabilities=probabilities,
            features=features,
            semantic_stride=(4.0, 4.0),
            feature_stride=(4.0, 4.0),
            original_size=size,
            class_names=("stable", "candidate_change"),
            diagnostics={},
        )

    return (
        output(first_probabilities, first_features, (64, 64)),
        output(
            second_probabilities,
            second_features,
            (63, 64) if mismatch else (64, 64),
        ),
    )


def _sample(root: Path, *, invalid: bool = False) -> UnifiedSample:
    root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), (80, 90, 100)).save(root / "t1.png")
    Image.new("RGB", (64, 64), (80, 90, 100)).save(root / "t2.png")
    images = [
        ImageRef(image_id="t1", path="t1.png", role="t1"),
        ImageRef(image_id="t2", path="t2.png", role="t2"),
    ]
    if invalid:
        Image.new("RGB", (64, 64), (80, 90, 100)).save(root / "context.png")
        images.append(ImageRef(image_id="c", path="context.png", role="context"))
    return UnifiedSample(
        sample_id="change-v2",
        dataset="offline",
        split="test",
        task="change_caption",
        images=images,
        question="Describe the change.",
        ground_truth=GroundTruth(answers=["reference must not enter payload"]),
    )


def _settings(*, enabled: bool, policy: str = "fallback_legacy") -> AgentChangeSettings:
    return AgentChangeSettings(
        semantic=ChangeSemanticSettings(
            enabled=enabled,
            local_match_radius=0,
            min_pif_feature_cells=16,
            failure_policy=policy,
        ),
        proposals=ChangeProposalSettings(
            min_component_area_ratio=0.001,
            max_component_area_ratio=0.50,
            mask_close_kernel=1,
        ),
    )


def _run(
    root: Path,
    *,
    settings: AgentChangeSettings,
    dense_client: _DenseClient | None,
    invalid: bool = False,
) -> tuple[Any, _QwenClient, _Budget]:
    qwen = _QwenClient()
    budget = _Budget()
    execution = asyncio.run(
        ChangeAgent(
            qwen,
            semantic_client=dense_client,
            prompt=_PROMPT,
            settings=settings,
        ).run(
            _sample(root, invalid=invalid),
            AgentContext(
                artifact_dir=root / "artifacts",
                qwen_client=None,
                call_budget=budget,
                data_root=root,
            ),
        )
    )
    return execution, qwen, budget


def _payload(qwen: _QwenClient) -> dict[str, Any]:
    user_content = qwen.calls[-1]["messages"][1]["content"]
    return json.loads(user_content[-1]["text"])


def test_disabled_path_calls_no_dense_model_and_preserves_v1_source(tmp_path: Path) -> None:
    dense = _DenseClient()

    execution, qwen, budget = _run(
        tmp_path,
        settings=_settings(enabled=False),
        dense_client=dense,
    )

    assert dense.calls == []
    assert len(qwen.calls) == 1
    assert budget.qwen_calls == 1
    assert execution.trace["semantic_status"] == "disabled"
    assert execution.trace["proposal_source"] == "difference_map_v1"
    artifact_root = tmp_path / "artifacts" / "change_preprocess"
    assert (artifact_root / "difference_map.png").is_file()
    assert (artifact_root / "proposal_overlay.png").is_file()
    assert (artifact_root / "proposals.json").is_file()
    for v2_filename in (
        "low_level_difference_map.png",
        "feature_residual_map.png",
        "semantic_difference_map.png",
        "fused_change_map.png",
        "binary_change_mask.png",
    ):
        assert not (artifact_root / v2_filename).exists()


def test_change_v1_import_path_does_not_load_semantic_runtime() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import agents.change.agent; import application.bootstrap; "
                "print(sorted({'torch', 'transformers', "
                "'models.segformer_transformers'} & set(sys.modules)))"
            ),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "[]"


def test_enabled_vertical_slice_calls_two_dense_frames_and_one_qwen(tmp_path: Path) -> None:
    dense = _DenseClient()

    execution, qwen, budget = _run(
        tmp_path,
        settings=_settings(enabled=True),
        dense_client=dense,
    )

    assert len(dense.calls) == 2
    assert len(qwen.calls) == 1
    assert budget.qwen_calls == 1
    assert execution.trace["semantic_status"] == "success"
    assert execution.trace["perception_mode"] == "fused_v2"
    assert execution.trace["semantic_enabled"] is True
    assert execution.trace["semantic_reason_codes"] == []
    assert execution.trace["semantic_model"] == "segformer-logical-test"
    assert execution.trace["semantic_client_version"] == "test-v1"
    assert execution.trace["semantic_model_revision"] is None
    assert execution.trace["proposal_source"] == "fused_change_v2"
    assert execution.trace["segformer_model"] == "segformer-logical-test"
    assert execution.trace["feature_residual_version"] == "pif_robust_local_cosine_v1"
    assert execution.trace["semantic_difference_version"] == "confidence_weighted_js_v1"
    assert execution.trace["fusion_version"] == "weighted_pif_robust_fusion_v1"
    assert execution.trace["fusion_effective_weights"]
    assert execution.trace["threshold_mode"] in {"pif_robust", "no_change_floor"}
    assert isinstance(execution.trace["threshold_value"], float)
    assert execution.trace["pif_feature_cells"] > 0
    assert all(
        not Path(relative).is_absolute()
        for relative in execution.trace["perception_artifacts"].values()
    )
    assert str(tmp_path) not in json.dumps(execution.trace)

    payload = _payload(qwen)
    assert payload["proposals"]
    assert payload["proposals"][0]["source"] == "fused_change_v2"
    assert set(payload["proposals"][0]["component_scores"]) == {
        "low_level",
        "feature",
        "semantic",
        "fused",
    }
    assert payload["perception"]["semantic_model"] == "segformer-logical-test"
    assert str(tmp_path) not in json.dumps(payload)

    roles = [item["role"] for item in payload["image_manifest"]]
    assert roles[:5] == [
        "raw_full_t1",
        "raw_full_t2",
        "harmonized_t1",
        "harmonized_t2",
        "proposal_overlay",
    ]
    crop_roles = roles[5:]
    assert "change_000:change_000_raw_t1" in crop_roles
    assert "change_000:change_000_raw_t2" in crop_roles
    assert "change_000:change_000_mask_overlay" in crop_roles
    assert crop_roles.index("change_000:change_000_raw_t1") < crop_roles.index(
        "change_000:change_000_raw_t2"
    ) < crop_roles.index("change_000:change_000_mask_overlay")
    assert (
        sum(
            role.endswith("_raw_t1") or role.endswith("_raw_t2")
            for role in crop_roles
        )
        >= 2
    )

    proposal_box = payload["proposals"][0]["box"]
    expected_change_box = [250, 250, 500, 500]
    assert _box_intersection_area(proposal_box, expected_change_box) > 0

    proposal_file = tmp_path / "artifacts" / "change_preprocess" / "proposals.json"
    proposals = json.loads(proposal_file.read_text(encoding="utf-8"))
    assert all(not Path(path).is_absolute() for path in proposals[0]["evidence_filenames"])
    assert not Path(proposals[0]["mask_filename"]).is_absolute()
    assert all(isinstance(value, float) for value in proposals[0]["component_scores"].values())
    mask_path = tmp_path / "artifacts" / proposals[0]["mask_filename"]
    assert mask_path.is_file()
    expected_v2_artifacts = {
        "low_level_difference_map.png",
        "feature_residual_map.png",
        "semantic_difference_map.png",
        "fused_change_map.png",
        "binary_change_mask.png",
        "proposal_overlay.png",
        "proposals.json",
        "crops/change_000_mask.png",
        "crops/change_000_mask_overlay.png",
    }
    artifact_root = tmp_path / "artifacts" / "change_preprocess"
    assert all((artifact_root / relative).is_file() for relative in expected_v2_artifacts)


def test_invalid_pair_calls_neither_dense_nor_qwen(tmp_path: Path) -> None:
    dense = _DenseClient()
    qwen = _QwenClient()
    budget = _Budget()

    with pytest.raises(AgentExecutionError, match="INVALID_CHANGE_PAIR"):
        asyncio.run(
            ChangeAgent(
                qwen,
                semantic_client=dense,
                prompt=_PROMPT,
                settings=_settings(enabled=True),
            ).run(
                _sample(tmp_path, invalid=True),
                AgentContext(
                    artifact_dir=tmp_path / "artifacts",
                    qwen_client=None,
                    call_budget=budget,
                    data_root=tmp_path,
                ),
            )
        )

    assert dense.calls == []
    assert qwen.calls == []
    assert budget.qwen_calls == 0


def test_missing_client_fallback_calls_qwen_once(tmp_path: Path) -> None:
    execution, qwen, budget = _run(
        tmp_path,
        settings=_settings(enabled=True),
        dense_client=None,
    )

    assert len(qwen.calls) == 1
    assert budget.qwen_calls == 1
    assert execution.trace["semantic_status"] == "fallback"
    assert execution.trace["semantic_reason_code"] == "SEGFORMER_CLIENT_MISSING"
    assert execution.trace["semantic_reason_codes"] == ["SEGFORMER_CLIENT_MISSING"]
    assert execution.trace["perception_mode"] == "fallback_legacy"
    assert execution.trace["proposal_source"] == "difference_map_v1"
    artifact_root = tmp_path / "artifacts" / "change_preprocess"
    for filename in (
        "low_level_difference_map.png",
        "feature_residual_map.png",
        "semantic_difference_map.png",
        "fused_change_map.png",
        "binary_change_mask.png",
    ):
        assert not (artifact_root / filename).exists()


def test_missing_client_fail_policy_calls_no_qwen(tmp_path: Path) -> None:
    qwen = _QwenClient()
    budget = _Budget()

    with pytest.raises(AgentExecutionError, match="SEGFORMER_CLIENT_MISSING"):
        asyncio.run(
            ChangeAgent(
                qwen,
                semantic_client=None,
                prompt=_PROMPT,
                settings=_settings(enabled=True, policy="fail"),
            ).run(
                _sample(tmp_path),
                AgentContext(
                    artifact_dir=tmp_path / "artifacts",
                    qwen_client=None,
                    call_budget=budget,
                    data_root=tmp_path,
                ),
            )
        )

    assert qwen.calls == []
    assert budget.qwen_calls == 0


def test_pair_grid_mismatch_falls_back_deterministically(tmp_path: Path) -> None:
    dense = _DenseClient(mismatch=True)

    execution, qwen, budget = _run(
        tmp_path,
        settings=_settings(enabled=True),
        dense_client=dense,
    )

    assert len(dense.calls) == 2
    assert len(qwen.calls) == 1
    assert budget.qwen_calls == 1
    assert execution.trace["semantic_status"] == "fallback"
    assert execution.trace["semantic_reason_code"] == "SEGFORMER_PAIR_GRID_MISMATCH"
    assert execution.trace["proposal_source"] == "difference_map_v1"


def test_segformer_logical_identity_participates_in_qwen_request_hash(
    tmp_path: Path,
) -> None:
    first_execution, first_qwen, _ = _run(
        tmp_path / "first",
        settings=_settings(enabled=True),
        dense_client=_DenseClient(model="segformer-logical-a"),
    )
    second_execution, second_qwen, _ = _run(
        tmp_path / "second",
        settings=_settings(enabled=True),
        dense_client=_DenseClient(model="segformer-logical-b"),
    )

    assert first_execution.trace["segformer_model"] == "segformer-logical-a"
    assert second_execution.trace["segformer_model"] == "segformer-logical-b"
    assert first_qwen.calls[0]["request_hash"] != second_qwen.calls[0]["request_hash"]


def test_repeated_fake_run_has_stable_trace_and_artifact_contract(tmp_path: Path) -> None:
    first, _, _ = _run(
        tmp_path / "first",
        settings=_settings(enabled=True),
        dense_client=_DenseClient(),
    )
    second, _, _ = _run(
        tmp_path / "second",
        settings=_settings(enabled=True),
        dense_client=_DenseClient(),
    )

    assert first.trace == second.trace


def _box_intersection_area(first: list[int], second: list[int]) -> int:
    return max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0,
        min(first[3], second[3]) - max(first[1], second[1]),
    )
