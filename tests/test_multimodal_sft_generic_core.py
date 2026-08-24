from __future__ import annotations

from pathlib import Path

from training.multimodal_sft.checkpoint import checkpoint_complete, read_compatible_manifest
from training.multimodal_sft.optimizer import OptimizerConfig, build_optimizer_groups
from training.multimodal_sft.parameter_plan import ParameterPlan


class _Parameter:
    def __init__(self, *, requires_grad: bool = True) -> None:
        self.requires_grad = requires_grad


class _Model:
    def __init__(self) -> None:
        self._items = [
            ("language.layer.lora_A", _Parameter()),
            ("language.layer.lora_B", _Parameter()),
            ("connector.weight", _Parameter()),
            ("frozen.weight", _Parameter(requires_grad=False)),
        ]

    def named_parameters(self):
        return iter(self._items)


def test_generic_files_have_no_model_family_literals() -> None:
    forbidden = ("qwen3", "qwen", "q_proj", "k_proj", "v_proj", "in_proj", "deepstack", "merger", "image_grid_thw", "mm_token_type_ids")
    for name in ("trainer_core.py", "optimizer.py", "checkpoint.py"):
        source = (Path("training/multimodal_sft") / name).read_text(encoding="utf-8").lower()
        assert not any(token.lower() in source for token in forbidden), name


def test_optimizer_groups_are_deterministic_and_use_connector_lr() -> None:
    model = _Model()
    plan = ParameterPlan(
        adapter_name="fixture",
        policy="lora_plus_projector",
        language_backbone="language",
        vision_backbone="vision",
        lora_module_paths=("language.layer",),
        full_train_module_paths=("connector",),
    )
    groups, stats = build_optimizer_groups(model, plan, OptimizerConfig(lora_lr=2e-4, connector_lr=3e-5))
    assert [group["name"] for group in groups] == ["lora_decay", "connector_decay"]
    assert [group["lr"] for group in groups] == [2e-4, 3e-5]
    assert stats["lora_parameter_hash"]
    assert stats["connector_parameter_hash"]


def test_legacy_manifest_is_read_only_through_explicit_compatibility_name(tmp_path) -> None:
    (tmp_path / "legacy_manifest.json").write_text('{"schema_version": 1, "training_profile": "phase2"}\n', encoding="utf-8")
    manifest = read_compatible_manifest(tmp_path, legacy_manifest_names=("legacy_manifest.json",))
    assert manifest["training_profile"] == "phase2"
    assert not checkpoint_complete(tmp_path)
