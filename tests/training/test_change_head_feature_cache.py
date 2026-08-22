from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agents.change.perception import SemanticExpertBinding
from models.base import (
    LEARNED_CHANGE_INPUT_CONTRACT_VERSION,
    LEARNED_CHANGE_OUTPUT_CONTRACT_VERSION,
    ModelCacheIdentity,
    DenseSemanticPyramidOutput,
)
from models.change_head.manifest import ChangeHeadManifest
from scripts import cache_change_head_features as cache_script
from training.change_head.feature_cache import (
    CachedChangeTrainingSample,
    CachedExpertFeaturePair,
    FeatureCache,
    build_feature_cache_key,
)


def _manifest(*, optional: bool = False) -> ChangeHeadManifest:
    expert = {
        "expert_id": "expert_1",
        "logical_model_id": "SegFormer-MiT-B2:iSAID:local",
        "weights_sha256": "a" * 64,
        "class_names_sha256": "b" * 64,
        "feature_stages": (1, 2),
        "feature_channels_by_stage": {1: 2, 2: 3},
        "required": True,
        "use_semantic_probabilities": True,
        "missing_policy": "error",
    }
    experts = (expert,)
    if optional:
        experts = (
            expert,
            {
                **expert,
                "expert_id": "expert_2",
                "required": False,
                "missing_policy": "zero_with_presence_mask",
            },
        )
    return ChangeHeadManifest.model_validate(
        {
            "input_contract_version": LEARNED_CHANGE_INPUT_CONTRACT_VERSION,
            "output_contract_version": LEARNED_CHANGE_OUTPUT_CONTRACT_VERSION,
            "architecture": {
                "name": "multi_expert_siamese_change_head_v1",
                "hidden_dim": 16,
                "semantic_dim": 8,
                "decoder_dim": 8,
                "optional_expert_dropout_supported": True,
                "use_pif_mask": True,
                "use_rgb_pair": False,
            },
            "experts": experts,
            "pipeline_fingerprint": "c" * 64,
            "model_weights_sha256": "d" * 64,
            "created_from_git_commit": "test",
            "training_manifest_sha256": "e" * 64,
        }
    )


def _sample() -> CachedChangeTrainingSample:
    return CachedChangeTrainingSample(
        sample_id="s1",
        image_size=(4, 3),
        experts={
            "expert_1": CachedExpertFeaturePair(
                expert_id="expert_1",
                logical_model_id="SegFormer-MiT-B2:iSAID:local",
                weights_sha256="a" * 64,
                class_map_sha256="b" * 64,
                feature_stages=(1, 2),
                first_features={1: np.ones((2, 3, 4), dtype=np.float32), 2: np.ones((3, 2, 2), dtype=np.float32)},
                second_features={1: np.zeros((2, 3, 4), dtype=np.float32), 2: np.zeros((3, 2, 2), dtype=np.float32)},
                first_semantic_probabilities=np.ones((2, 3, 4), dtype=np.float32),
                second_semantic_probabilities=np.zeros((2, 3, 4), dtype=np.float32),
            )
        },
        target_change_mask=np.array([[0, 1], [0, 0]], dtype=np.float32),
        loss_valid_mask=np.array([[True, False], [True, True]]),
        pif_mask=np.ones((2, 2), dtype=np.uint8),
        comparison_t1=None,
        comparison_t2=None,
        dataset_name="toy",
        split="train",
        tags=("hard_case",),
        input_pipeline_fingerprint="c" * 64,
    )


def test_cache_key_changes_when_backbone_sha_changes(tmp_path: Path) -> None:
    (tmp_path / "t1.png").write_bytes(b"t1")
    (tmp_path / "t2.png").write_bytes(b"t2")
    kwargs = {
        "sample_id": "s1",
        "t1_path": tmp_path / "t1.png",
        "t2_path": tmp_path / "t2.png",
        "pipeline_fingerprint": "c" * 64,
        "experts": [{"expert_id": "e", "weights_sha256": "a" * 64, "feature_stages": [1]}],
    }
    first = build_feature_cache_key(**kwargs)
    kwargs["experts"] = [{"expert_id": "e", "weights_sha256": "b" * 64, "feature_stages": [1]}]
    assert first != build_feature_cache_key(**kwargs)


def test_cache_key_changes_when_feature_stages_or_fingerprint_change(tmp_path: Path) -> None:
    (tmp_path / "t1.png").write_bytes(b"t1")
    (tmp_path / "t2.png").write_bytes(b"t2")
    base = dict(
        sample_id="s1",
        t1_path=tmp_path / "t1.png",
        t2_path=tmp_path / "t2.png",
        pipeline_fingerprint="c" * 64,
        experts=[{"expert_id": "e", "weights_sha256": "a" * 64, "feature_stages": [1]}],
    )
    first = build_feature_cache_key(**base)
    changed_stages = {**base, "experts": [{**base["experts"][0], "feature_stages": [1, 2]}]}
    changed_fingerprint = {**base, "pipeline_fingerprint": "d" * 64}
    assert first != build_feature_cache_key(**changed_stages)
    assert first != build_feature_cache_key(**changed_fingerprint)


def test_cached_feature_shapes_match_declared_stages(tmp_path: Path) -> None:
    cache = FeatureCache(tmp_path)
    cache.write_sample("a" * 64, _sample())
    loaded = cache.read_sample("a" * 64)
    assert loaded is not None
    assert loaded.experts["expert_1"].feature_stages == (1, 2)
    assert loaded.experts["expert_1"].first_features[1].shape == (2, 3, 4)
    assert loaded.experts["expert_1"].second_features[2].shape == (3, 2, 2)


def _binding() -> SemanticExpertBinding:
    return SemanticExpertBinding(
        expert_id="expert_1",
        logical_model_id="SegFormer-MiT-B2:iSAID:local",
        priority=1,
        role="generic",
        neutral_labels=frozenset(),
        transient_labels=frozenset(),
        persistent_labels=frozenset(),
        client=SimpleNamespace(
            cache_identity=ModelCacheIdentity(
                model="SegFormer-MiT-B2:iSAID:local",
                generation={"weights_sha256": "a" * 64},
                client_version="test",
            )
        ),
        class_names=("background",),
        class_names_sha256="b" * 64,
        weights_sha256="a" * 64,
    )


def _example() -> SimpleNamespace:
    first = DenseSemanticPyramidOutput(
        probabilities=np.ones((1, 3, 3), dtype=np.float32),
        features_by_stage={1: np.ones((2, 3, 3), dtype=np.float32), 2: np.ones((3, 2, 2), dtype=np.float32)},
        semantic_stride=(1.0, 1.0),
        feature_strides_by_stage={1: (1.0, 1.0), 2: (1.5, 1.5)},
        original_size=(3, 3),
        class_names=("background",),
        diagnostics={},
        weights_sha256="a" * 64,
    )
    second = DenseSemanticPyramidOutput(
        probabilities=np.zeros((1, 3, 3), dtype=np.float32),
        features_by_stage={1: np.zeros((2, 3, 3), dtype=np.float32), 2: np.zeros((3, 2, 2), dtype=np.float32)},
        semantic_stride=(1.0, 1.0),
        feature_strides_by_stage={1: (1.0, 1.0), 2: (1.5, 1.5)},
        original_size=(3, 3),
        class_names=("background",),
        diagnostics={},
        weights_sha256="a" * 64,
    )
    run = SimpleNamespace(first_output=first, second_output=second, weights_sha256="a" * 64)
    prepared = SimpleNamespace(
        raw_t1=np.zeros((3, 3, 3), dtype=np.uint8),
        pif_mask=np.zeros((3, 3), dtype=np.uint8),
        comparison_t1=np.zeros((3, 3, 3), dtype=np.uint8),
        comparison_t2=np.ones((3, 3, 3), dtype=np.uint8),
    )
    record = SimpleNamespace(
        sample_id="s1", source_dataset="toy", split="train", tags=("no_change",)
    )
    return SimpleNamespace(
        prepared=prepared,
        record=record,
        target_mask=np.zeros((3, 3), dtype=np.float32),
        loss_valid_mask=np.ones((3, 3), dtype=bool),
        run=run,
    )


def test_same_expert_runs_once_and_pif_is_not_loss_validity(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_infer(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return _example().run

    monkeypatch.setattr(cache_script, "infer_semantic_expert_pair", fake_infer)
    example = _example()
    sample, missing = cache_script._build_cached_sample(
        example,
        manifest=_manifest(),
        bindings_by_id={"expert_1": _binding()},
        semantic_settings=SimpleNamespace(),
        pipeline_fingerprint="c" * 64,
    )
    assert calls == 1
    assert not missing
    assert np.array_equal(sample.loss_valid_mask, np.ones((3, 3), dtype=bool))
    assert np.array_equal(sample.pif_mask, np.zeros((3, 3), dtype=np.uint8))


def test_required_expert_missing_fails_and_optional_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(cache_script.RequiredExpertFailure):
        cache_script._build_cached_sample(
            _example(),
            manifest=_manifest(),
            bindings_by_id={},
            semantic_settings=SimpleNamespace(),
            pipeline_fingerprint="c" * 64,
        )
    optional_manifest = _manifest(optional=True)
    monkeypatch.setattr(cache_script, "infer_semantic_expert_pair", lambda *args, **kwargs: _example().run)
    sample, missing = cache_script._build_cached_sample(
        _example(),
        manifest=optional_manifest,
        bindings_by_id={"expert_1": _binding()},
        semantic_settings=SimpleNamespace(),
        pipeline_fingerprint="c" * 64,
    )
    assert set(sample.experts) == {"expert_1"}
    assert missing == ["expert_2"]
