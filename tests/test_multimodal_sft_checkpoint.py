from __future__ import annotations

from training.multimodal_sft.checkpoint import build_training_manifest, identity_fingerprint, validate_resume_compatibility


def _manifest():
    plan = {"adapter_name": "tiny", "parameter_names": ["language.weight"], "full_train_parameter_names": ["connector.weight"]}
    return build_training_manifest(adapter_name="tiny", model_identity={"model_type": "tiny", "base_config_sha256": "cfg"}, task_profile="phase2", data_contract={"schema": "v1"}, tuning_policy={"name": "lora_plus_projector"}, parameter_plan=plan)


def test_resume_identity_and_parameter_plan_gate() -> None:
    manifest = _manifest()
    plan = manifest["parameter_plan"]
    validate_resume_compatibility(manifest, adapter_name="tiny", model_identity={"model_type": "tiny", "base_config_sha256": "cfg"}, task_profile="phase2", tuning_policy={"name": "lora_plus_projector"}, parameter_plan=plan, data_contract={"schema": "v1"})
    altered = dict(plan)
    altered["parameter_names"] = ["different.weight"]
    try:
        validate_resume_compatibility(manifest, adapter_name="tiny", model_identity={"model_type": "tiny", "base_config_sha256": "cfg"}, task_profile="phase2", tuning_policy={"name": "lora_plus_projector"}, parameter_plan=altered)
    except ValueError as exc:
        assert "parameter plan" in str(exc)
    else:
        raise AssertionError("altered parameter plan was accepted")
