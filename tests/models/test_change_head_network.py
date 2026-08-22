from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from models.base import (  # noqa: E402
    LEARNED_CHANGE_INPUT_CONTRACT_VERSION,
    LEARNED_CHANGE_OUTPUT_CONTRACT_VERSION,
)
from models.change_head.manifest import ChangeHeadManifest  # noqa: E402
from models.change_head.network import MultiExpertSiameseChangeHead  # noqa: E402
from training.change_head.losses import masked_bce_with_logits  # noqa: E402


def _manifest(*, use_pif_mask: bool) -> ChangeHeadManifest:
    value = "a" * 64
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
                "use_pif_mask": use_pif_mask,
                "use_rgb_pair": False,
            },
            "experts": ({
                "expert_id": "expert_1",
                "logical_model_id": "SegFormer-MiT-B2:iSAID:local",
                "weights_sha256": value,
                "class_names_sha256": "b" * 64,
                "feature_stages": (1,),
                "feature_channels_by_stage": {1: 2},
                "required": True,
                "use_semantic_probabilities": False,
                "missing_policy": "error",
            },),
            "pipeline_fingerprint": "c" * 64,
            "model_weights_sha256": "d" * 64,
            "created_from_git_commit": "test",
            "training_manifest_sha256": "e" * 64,
        }
    )


def _inputs() -> dict[str, object]:
    first = torch.zeros(1, 2, 3, 3)
    second = torch.zeros(1, 2, 3, 3)
    second[:, 0, 1, 1] = 2.0
    return {
        "expert_features": {"expert_1": ([first], [second])},
        "semantic_probabilities": {},
        "expert_presence": {"expert_1": torch.ones(1)},
        "valid_mask": torch.ones(1, 3, 3),
    }


def _fixed_network(*, use_pif_mask: bool) -> MultiExpertSiameseChangeHead:
    network = MultiExpertSiameseChangeHead(_manifest(use_pif_mask=use_pif_mask))
    with torch.no_grad():
        network.projections["expert_1__1"].weight.fill_(1.0)
        network.projections["expert_1__1"].bias.zero_()
        network.logit_bias.fill_(0.25)
        if network.pif_fusion is not None:
            network.pif_fusion.weight.fill_(0.5)
            network.pif_fusion.bias.zero_()
    return network


def test_pif_zero_does_not_force_half_probability() -> None:
    network = _fixed_network(use_pif_mask=True)
    inputs = _inputs()
    inputs["pif_mask"] = torch.zeros(1, 3, 3)
    logits = network(**inputs)
    probability = torch.sigmoid(logits)
    assert not torch.allclose(probability, torch.full_like(probability, 0.5))
    inputs["expert_features"]["expert_1"][1][0][:, 0, 0, 0] = 3.0  # type: ignore[index]
    changed = torch.sigmoid(network(**inputs))
    assert not torch.allclose(probability, changed)


def test_pif_is_not_change_validity_gate_when_disabled() -> None:
    network = _fixed_network(use_pif_mask=False)
    inputs = _inputs()
    without_pif = network(**inputs)
    with_zero_pif = network(**inputs, pif_mask=torch.zeros(1, 3, 3))
    with_one_pif = network(**inputs, pif_mask=torch.ones(1, 3, 3))
    assert torch.equal(without_pif, with_zero_pif)
    assert torch.equal(without_pif, with_one_pif)


def test_invalid_pixels_do_not_contribute_to_loss() -> None:
    logits = torch.zeros(1, 1, 2, 2)
    valid = torch.tensor([[True, False], [False, False]])
    target_a = torch.zeros(1, 1, 2, 2)
    target_b = target_a.clone()
    target_b[:, :, 0, 1:] = 1.0
    target_b[:, :, 1, :] = 1.0
    loss_a = masked_bce_with_logits(logits, target_a, valid)
    loss_b = masked_bce_with_logits(logits, target_b, valid)
    assert torch.equal(loss_a, loss_b)
