from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from models.base import (  # noqa: E402
    LEARNED_CHANGE_INPUT_CONTRACT_VERSION,
    LEARNED_CHANGE_OUTPUT_CONTRACT_VERSION,
)
from models.change_head.manifest import ChangeHeadManifest  # noqa: E402
from scripts import train_change_head as train_script  # noqa: E402
from training.change_head.feature_cache import (  # noqa: E402
    CachedChangeTrainingSample,
    CachedExpertFeaturePair,
)
from training.change_head.losses import change_head_loss  # noqa: E402


def _manifest() -> ChangeHeadManifest:
    return ChangeHeadManifest.model_validate(
        {
            "input_contract_version": LEARNED_CHANGE_INPUT_CONTRACT_VERSION,
            "output_contract_version": LEARNED_CHANGE_OUTPUT_CONTRACT_VERSION,
            "architecture": {
                "name": "multi_expert_siamese_change_head_v1",
                "hidden_dim": 4,
                "semantic_dim": 4,
                "decoder_dim": 4,
                "optional_expert_dropout_supported": True,
                "use_pif_mask": True,
                "use_rgb_pair": False,
            },
            "experts": [{
                "expert_id": "expert_1",
                "logical_model_id": "SegFormer-MiT-B2:iSAID:local",
                "weights_sha256": "a" * 64,
                "class_names_sha256": "b" * 64,
                "feature_stages": [1],
                "feature_channels_by_stage": {"1": 2},
                "required": True,
                "use_semantic_probabilities": False,
                "missing_policy": "error",
            }],
            "pipeline_fingerprint": "c" * 64,
            "model_weights_sha256": "d" * 64,
            "created_from_git_commit": "test",
            "training_manifest_sha256": "e" * 64,
        }
    )


def _sample(sample_id: str = "sample-1") -> CachedChangeTrainingSample:
    features = np.ones((2, 4, 4), dtype=np.float32)
    expert = CachedExpertFeaturePair(
        expert_id="expert_1",
        logical_model_id="SegFormer-MiT-B2:iSAID:local",
        weights_sha256="a" * 64,
        class_map_sha256="b" * 64,
        feature_stages=(1,),
        first_features={1: features},
        second_features={1: features * 2},
        first_semantic_probabilities=None,
        second_semantic_probabilities=None,
    )
    return CachedChangeTrainingSample(
        sample_id=sample_id,
        image_size=(4, 4),
        experts={"expert_1": expert},
        target_change_mask=np.zeros((4, 4), dtype=np.float32),
        loss_valid_mask=np.ones((4, 4), dtype=bool),
        pif_mask=np.ones((4, 4), dtype=np.float32),
        comparison_t1=None,
        comparison_t2=None,
        dataset_name="synthetic",
        split="val",
        tags=("hard_case",),
        input_pipeline_fingerprint="f" * 64,
    )


def test_cached_pif_mask_batches_to_b1hw() -> None:
    sample = _sample()
    single = train_script.sample_to_batch(sample, device="cpu")
    assert tuple(single["network_inputs"]["pif_mask"].shape) == (1, 4, 4)
    batch = train_script._batches(
        [sample, _sample("sample-2")], [0, 1], batch_size=2, device="cpu"
    )[0]
    assert tuple(batch["network_inputs"]["pif_mask"].shape) == (2, 1, 4, 4)
    assert tuple(batch["target_mask"].shape) == (2, 1, 4, 4)
    assert tuple(batch["loss_valid_mask"].shape) == (2, 1, 4, 4)


def test_cache_sample_with_pif_reaches_network() -> None:
    from models.change_head.network import MultiExpertSiameseChangeHead

    sample = _sample()
    batch = train_script._batches(
        [sample, _sample("sample-2")], [0, 1], batch_size=2, device="cpu"
    )[0]
    network = MultiExpertSiameseChangeHead(_manifest())
    logits = network(**batch["network_inputs"])
    loss = change_head_loss(logits, batch["target_mask"], batch["loss_valid_mask"])
    loss.backward()
    assert tuple(logits.shape) == (2, 1, 4, 4)


def test_best_validation_artifacts_keep_raw_logits_and_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation = train_script.EvaluationArtifacts(
        metrics={"pixel_f1": 0.9, "pixel_iou": 0.82},
        logits=[np.array([[-2.0, 0.0], [2.0, 1.0]], dtype=np.float32)],
        probabilities=[np.array([[0.1, 0.5], [0.9, 0.7]], dtype=np.float32)],
        targets=[np.array([[0, 0], [1, 1]], dtype=np.float32)],
        valid_masks=[np.array([[1, 0], [1, 1]], dtype=bool)],
        tags=[["hard_case", "building_edge"]],
        sample_ids=["val-001"],
    )
    monkeypatch.setattr(
        train_script,
        "_save_safetensors",
        lambda network, path: (path.write_bytes(b"weights") or "d" * 64),
    )
    directory = tmp_path / "best"
    train_script._save_checkpoint_bundle(
        directory=directory,
        network=object(),
        manifest=_manifest(),
        row={"epoch": 2, "val_pixel_f1": 0.9},
        evaluation=evaluation,
        epoch=2,
    )
    assert np.array_equal(
        np.load(directory / "val_logits.npy")[0], evaluation.logits[0]
    )
    assert np.array_equal(
        np.load(directory / "val_valid_masks.npy")[0], evaluation.valid_masks[0]
    )
    assert json.loads((directory / "val_tags.json").read_text()) == evaluation.tags
    assert json.loads((directory / "val_sample_ids.json").read_text()) == evaluation.sample_ids
    metrics = json.loads((directory / "metrics.json").read_text())
    assert metrics["best_epoch"] == 2
    assert metrics["pixel_f1"] == 0.9


def test_unknown_selection_metric_fails_fast() -> None:
    with pytest.raises(train_script.TrainingConfigError, match="Unknown selection metric"):
        train_script._config_to_training(
            {"selection": {"primary_metric": "agent_proxy_score"}}
        )
