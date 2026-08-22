from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from models.base import (  # noqa: E402
    DenseSemanticPyramidOutput,
    LEARNED_CHANGE_INPUT_CONTRACT_VERSION,
    LEARNED_CHANGE_OUTPUT_CONTRACT_VERSION,
    LearnedChangeExpertPair,
    LearnedChangeRequest,
)
from models.change_head.calibration import ChangeHeadCalibration  # noqa: E402
from models.change_head.checkpoint import LoadedChangeHeadCheckpoint  # noqa: E402
from models.change_head.manifest import ChangeHeadManifest  # noqa: E402
from models.change_head.network import MultiExpertSiameseChangeHead  # noqa: E402
from models.change_head.runtime import TorchLearnedChangeClient  # noqa: E402


def _manifest() -> ChangeHeadManifest:
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
                "use_pif_mask": True,
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


def _output(features: np.ndarray) -> DenseSemanticPyramidOutput:
    return DenseSemanticPyramidOutput(
        probabilities=np.zeros((1, 3, 3), dtype=np.float32),
        features_by_stage={1: features},
        semantic_stride=(1.0, 1.0),
        feature_strides_by_stage={1: (1.0, 1.0)},
        original_size=(3, 3),
        class_names=("background",),
        diagnostics={},
        weights_sha256="a" * 64,
    )


def _client() -> TorchLearnedChangeClient:
    manifest = _manifest()
    network = MultiExpertSiameseChangeHead(manifest)
    with torch.no_grad():
        network.projections["expert_1__1"].weight.fill_(1.0)
        network.projections["expert_1__1"].bias.zero_()
        network.logit_bias.fill_(0.25)
        assert network.pif_fusion is not None
        network.pif_fusion.weight.fill_(0.5)
        network.pif_fusion.bias.zero_()
    checkpoint = LoadedChangeHeadCheckpoint(
        root=Path("."),
        manifest=manifest,
        calibration=ChangeHeadCalibration(
            temperature=1.0,
            rescue_probability_threshold=0.8,
            rescue_min_component_area_ratio=0.01,
            validation_reliability=0.9,
        ),
        state_dict=network.state_dict(),
    )
    return TorchLearnedChangeClient(checkpoint, device="cpu")


def _request(*, valid_mask: np.ndarray) -> LearnedChangeRequest:
    first = np.zeros((2, 3, 3), dtype=np.float32)
    second = np.zeros((2, 3, 3), dtype=np.float32)
    second[0, 1, 1] = 2.0
    pair = LearnedChangeExpertPair(
        expert_id="expert_1",
        logical_model_id="SegFormer-MiT-B2:iSAID:local",
        weights_sha256="a" * 64,
        class_names_sha256="b" * 64,
        first=_output(first),
        second=_output(second),
    )
    return LearnedChangeRequest(
        image_size=(3, 3),
        experts={"expert_1": pair},
        valid_mask=valid_mask,
        pif_mask=np.ones((3, 3), dtype=np.float32),
        pif_valid=True,
    )


def test_invalid_pixels_are_zero_after_runtime() -> None:
    valid = np.ones((3, 3), dtype=np.float32)
    valid[0, 0] = 0.0
    output = _client().infer(_request(valid_mask=valid))
    assert output.probability_map[0, 0] == 0.0
    assert output.probability_map[1, 1] > 0.5
    assert output.uncertainty_map is not None
    assert output.uncertainty_map[0, 0] == 0.0

