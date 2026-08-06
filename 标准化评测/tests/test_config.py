from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from m3rs_eval.config import (
    ConfigError,
    DatasetConfig,
    load_config,
    redact_config,
    serialize_resolved_config,
)


def test_load_config_rejects_empty_dataset_path(tmp_path):
    path = tmp_path / "server.yaml"
    path.write_text(
        "protocol_path: protocol.yaml\ndatasets:\n  levir_cc:\n    root: ''\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="datasets.levir_cc.root"):
        load_config(path)


def test_redact_config_omits_nested_secret_names_and_values():
    raw = {"system": {"environment": {"API_TOKEN": "abc", "CUDA_VISIBLE_DEVICES": "0"}}}
    assert redact_config(raw)["system"]["environment"] == {
        "CUDA_VISIBLE_DEVICES": "0",
    }


def test_fixture_config_resolves_the_existing_metric_registry():
    config = load_config(Path(__file__).parents[1] / "configs" / "fixture.yaml")

    assert config.metric_registry_path.is_file()
    assert config.metric_registry_path.name == "metrics.yaml"
    assert config.datasets["levir_cc"].profile == "fixture"
    assert config.datasets["levir_cc"].asset_root == config.project_root / "test_fixtures"
    assert config.datasets["levir_cc"].official_scorer_command is None
    for dataset in ("levir_cc", "vrsbench", "xlrs_bench"):
        assert config.datasets[dataset].official_scorer_output.is_file()
    assert config.datasets["mme_rs"].official_scorer_output is None


def test_load_config_accepts_explicit_sensitive_argv_positions(tmp_path, fixture_config):
    project_root = Path(__file__).parents[1]
    source = yaml.safe_load((project_root / "configs" / "fixture.yaml").read_text(encoding="utf-8"))
    source["project_root"] = str(fixture_config.project_root)
    source["protocol_path"] = str(fixture_config.protocol_path)
    source["metric_registry_path"] = str(fixture_config.metric_registry_path)
    source["output_root"] = str(tmp_path / "runs")
    source["model_weights"] = str(fixture_config.model_weights)
    source["system"]["working_directory"] = str(fixture_config.system.working_directory)
    source["system"]["sensitive_argument_positions"] = [1]
    source["datasets"]["levir_cc"]["official_scorer_command"] = [
        "python",
        "--opaque",
        "scorer-secret",
    ]
    source["datasets"]["levir_cc"].pop("official_scorer_output")
    source["datasets"]["levir_cc"]["scorer_sensitive_argument_positions"] = [2]
    path = tmp_path / "fixture.yaml"
    path.write_text(yaml.safe_dump(source), encoding="utf-8")

    config = load_config(path)

    assert config.system.sensitive_argument_positions == (1,)
    assert config.datasets["levir_cc"].scorer_sensitive_argument_positions == (2,)


def test_serialize_resolved_config_redacts_command_and_environment_secrets(fixture_config):
    config = replace(
        fixture_config,
        system=replace(
            fixture_config.system,
            command=(
                "python",
                "--token=inline-secret",
                "--password",
                "next-secret",
                "--ordinary",
                "visible",
                "--opaque",
                "explicit-secret",
            ),
            sensitive_argument_positions=(7,),
            environment={"DB_CREDENTIAL": "database-secret", "BATCH": "2"},
        ),
    )

    serialized = serialize_resolved_config(config)

    command = serialized["system"]["command"]
    assert command == [
        "python",
        "<redacted-argument>",
        "<redacted-argument>",
        "<redacted-argument>",
        "--ordinary",
        "visible",
        "--opaque",
        "<redacted-argument>",
    ]
    assert serialized["system"]["environment"] == {"BATCH": "2"}
    assert "inline-secret" not in repr(serialized)
    assert "next-secret" not in repr(serialized)
    assert "explicit-secret" not in repr(serialized)
    assert "DB_CREDENTIAL" not in repr(serialized)


def test_serialize_resolved_config_redacts_every_official_scorer_command(fixture_config):
    datasets = dict(fixture_config.datasets)
    datasets["levir_cc"] = DatasetConfig(
        root=datasets["levir_cc"].root,
        asset_root=datasets["levir_cc"].asset_root,
        profile="official",
        official_scorer_command=(
            "python",
            "score.py",
            "--token=official-secret",
            "--credential",
            "next-official-secret",
            "--format",
            "json",
            "--opaque",
            "explicit-official-secret",
        ),
        scorer_sensitive_argument_positions=(8,),
    )

    serialized = serialize_resolved_config(replace(fixture_config, datasets=datasets))

    command = serialized["datasets"]["levir_cc"]["official_scorer_command"]
    assert command == [
        "python",
        "score.py",
        "<redacted-argument>",
        "<redacted-argument>",
        "<redacted-argument>",
        "--format",
        "json",
        "--opaque",
        "<redacted-argument>",
    ]
    serialized_json = yaml.safe_dump(serialized)
    assert "official-secret" not in serialized_json
    assert "next-official-secret" not in serialized_json
    assert "explicit-official-secret" not in serialized_json


def test_load_config_rejects_protocol_metric_outside_registry(tmp_path):
    project_root = Path(__file__).parents[2]
    registry_path = project_root / "指标字典" / "outputs" / "019fc7cf-aa6b-7143-b248-647f0db1037d" / "metrics.yaml"
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(
        "protocol_id: test\n"
        f"metric_namespace: '{registry_path.as_posix()}'\n"
        "datasets:\n"
        "  levir_cc:\n"
        "    required_metric_ids: [not.a.registered.metric]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "project_root: .\n"
        "protocol_path: protocol.yaml\n"
        f"metric_registry_path: '{registry_path.as_posix()}'\n"
        "output_root: runs\n"
        "system_version: test\n"
        "model_name: test\n"
        "model_weights: model.bin\n"
        "training_data_version: test\n"
        "operator: test\n"
        "system:\n"
        "  command: [python]\n"
        "  working_directory: .\n"
        "  timeout_seconds: 1\n"
        "  environment: {}\n"
        "datasets:\n"
        "  levir_cc: {root: .}\n"
        "  vrsbench: {root: .}\n"
        "  xlrs_bench: {root: .}\n"
        "  mme_rs: {root: .}\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="not.a.registered.metric"):
        load_config(config_path)


def test_load_config_rejects_nonexact_official_scorer_placeholder(tmp_path, fixture_config):
    source = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "fixture.yaml").read_text(encoding="utf-8")
    )
    source["project_root"] = str(fixture_config.project_root)
    source["protocol_path"] = str(fixture_config.protocol_path)
    source["metric_registry_path"] = str(fixture_config.metric_registry_path)
    source["output_root"] = str(tmp_path / "runs")
    source["model_weights"] = str(fixture_config.model_weights)
    source["system"]["working_directory"] = str(fixture_config.system.working_directory)
    source["datasets"]["levir_cc"]["official_scorer_command"] = [
        "python",
        "score.py",
        "--input={predictions_jsonl}",
    ]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(source), encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown or nonexact"):
        load_config(path)


def test_fixture_scorer_output_and_timeout_are_typed(tmp_path, fixture_config):
    source = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "fixture.yaml").read_text(encoding="utf-8")
    )
    source["project_root"] = str(fixture_config.project_root)
    source["protocol_path"] = str(fixture_config.protocol_path)
    source["metric_registry_path"] = str(fixture_config.metric_registry_path)
    source["output_root"] = str(tmp_path / "runs")
    source["model_weights"] = str(fixture_config.model_weights)
    source["system"]["working_directory"] = str(fixture_config.system.working_directory)
    scorer_output = tmp_path / "scores.json"
    scorer_output.write_text('{"scorer_version":"v1","metrics":[]}', encoding="utf-8")
    source["datasets"]["levir_cc"]["official_scorer_output"] = str(scorer_output)
    source["datasets"]["levir_cc"]["official_scorer_timeout_seconds"] = 17
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(source), encoding="utf-8")

    config = load_config(path)

    assert config.datasets["levir_cc"].official_scorer_output == scorer_output
    assert config.datasets["levir_cc"].official_scorer_timeout_seconds == 17


def _write_config_source(tmp_path, fixture_config, source):
    source["project_root"] = str(fixture_config.project_root)
    source["protocol_path"] = str(fixture_config.protocol_path)
    source["metric_registry_path"] = str(fixture_config.metric_registry_path)
    source["output_root"] = str(tmp_path / "runs")
    source["model_weights"] = str(fixture_config.model_weights)
    source["system"]["working_directory"] = str(fixture_config.system.working_directory)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    return path


def test_official_profile_forbids_fixture_output_ingestion(tmp_path, fixture_config):
    source = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "fixture.yaml").read_text(encoding="utf-8")
    )
    source["datasets"]["levir_cc"]["profile"] = "official"

    with pytest.raises(ConfigError, match="official_scorer_output.*fixture-only"):
        load_config(_write_config_source(tmp_path, fixture_config, source))


def test_official_caption_profile_requires_command_version_and_commit_pins(
    tmp_path, fixture_config
):
    source = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "fixture.yaml").read_text(encoding="utf-8")
    )
    dataset = source["datasets"]["levir_cc"]
    dataset["profile"] = "official"
    dataset.pop("official_scorer_output")
    dataset.pop("official_scorer_expected_version", None)
    dataset["official_scorer_command"] = ["python", "score.py", "{output_json}"]

    with pytest.raises(ConfigError, match="expected_version"):
        load_config(_write_config_source(tmp_path, fixture_config, source))

    dataset["official_scorer_expected_version"] = "levir-official-v1"
    with pytest.raises(ConfigError, match="expected_commit"):
        load_config(_write_config_source(tmp_path, fixture_config, source))

    dataset["official_scorer_expected_commit"] = "0123456789abcdef"
    loaded = load_config(_write_config_source(tmp_path, fixture_config, source))
    assert loaded.datasets["levir_cc"].official_scorer_expected_version == "levir-official-v1"
    assert loaded.datasets["levir_cc"].official_scorer_expected_commit == "0123456789abcdef"


def test_fixture_output_requires_a_pinned_expected_version(tmp_path, fixture_config):
    source = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "fixture.yaml").read_text(encoding="utf-8")
    )
    source["datasets"]["vrsbench"].pop("official_scorer_expected_version", None)

    with pytest.raises(ConfigError, match="expected_version"):
        load_config(_write_config_source(tmp_path, fixture_config, source))
