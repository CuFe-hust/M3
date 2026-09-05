from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts import finetune_multimodal_sft as cli
from training.multimodal_sft.contracts import AdapterProbe, ModelIdentity
from training.multimodal_sft.trainer_core import GenericTrainerCore


def test_cli_propagates_resume_and_smoke_only(monkeypatch, tmp_path: Path, capsys) -> None:
    captured = {}
    probe = AdapterProbe("fixture", ModelIdentity("fixture"), frozenset())

    class _Adapter:
        name = "fixture"

        def load(self, *args, **kwargs):
            return object(), object(), probe

    class _Registry:
        def resolve(self, *args, **kwargs):
            return _Adapter(), probe

    profile = SimpleNamespace(name="change_agent", read=lambda path: iter([{"episode_id": "x"}]))
    monkeypatch.setattr(cli, "profile_for", lambda *args, **kwargs: profile)
    monkeypatch.setattr(cli, "default_registry", lambda: _Registry())
    monkeypatch.setattr(cli.ImageRootRegistry, "from_specs", lambda specs: SimpleNamespace(roots={}))

    def _fit(self, **kwargs):
        captured["config"] = kwargs["config"]
        plan = SimpleNamespace(as_dict=lambda: {
            "adapter_name": "fixture",
            "policy": "lora_plus_projector",
            "lora_module_paths": ["language.one", "language.two"],
            "full_train_module_paths": ["vision.connector"],
            "structure": {"details": {"actual_target_count": 2}},
        })
        return SimpleNamespace(steps=0, manifest_path=None, final_adapter_path=None, parameter_plan=plan, optimizer_stats={"gradient_smoke": {"passed": True}})

    monkeypatch.setattr(GenericTrainerCore, "fit", _fit)
    resume = tmp_path / "checkpoint-10"
    exit_code = cli.main([
        "--model-id", "fixture",
        "--data-profile", "change_agent",
        "--train-file", str(tmp_path / "train.jsonl"),
        "--data-manifest", str(tmp_path / "manifest.json"),
        "--resume-from", str(resume),
        "--smoke-gradients-only",
    ])

    assert exit_code == 0
    assert captured["config"].resume_from == str(resume)
    assert captured["config"].smoke_gradients is True
    assert captured["config"].smoke_gradients_only is True
    output = capsys.readouterr().out
    assert '"gradient_smoke"' in output
    assert '"lora_module_count": 2' in output
    assert '"full_train_module_paths"' in output


def test_grounding_cli_applies_training_defaults(monkeypatch, tmp_path: Path) -> None:
    captured = {}
    probe = AdapterProbe("fixture", ModelIdentity("fixture"), frozenset())

    class _Adapter:
        name = "fixture"

        def load(self, *args, **kwargs):
            return object(), object(), probe

    class _Registry:
        def resolve(self, *args, **kwargs):
            return _Adapter(), probe

    profile = SimpleNamespace(name="grounding", read=lambda path: iter([{"episode_id": "x"}]))
    monkeypatch.setattr(cli, "profile_for", lambda *args, **kwargs: profile)
    monkeypatch.setattr(cli, "default_registry", lambda: _Registry())
    monkeypatch.setattr(cli.ImageRootRegistry, "from_specs", lambda specs: SimpleNamespace(roots={}))

    def _fit(self, **kwargs):
        captured["config"] = kwargs["config"]
        plan = SimpleNamespace(as_dict=lambda: {
            "adapter_name": "fixture",
            "policy": "lora_only",
            "lora_module_paths": ["language.one"],
            "full_train_module_paths": [],
            "structure": {"details": {}},
        })
        return SimpleNamespace(
            steps=0,
            manifest_path=None,
            final_adapter_path=None,
            parameter_plan=plan,
            optimizer_stats={},
        )

    monkeypatch.setattr(GenericTrainerCore, "fit", _fit)
    exit_code = cli.main([
        "--model-id", "fixture",
        "--data-profile", "grounding",
        "--train-file", str(tmp_path / "train.jsonl"),
    ])

    assert exit_code == 0
    config = captured["config"]
    assert config.epochs == 2
    assert config.gradient_accumulation_steps == 16
    assert config.eval_steps == 34
    assert config.save_steps == 34
    assert config.connector_lr == 0.0
