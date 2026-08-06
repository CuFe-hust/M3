from __future__ import annotations

import shutil
from dataclasses import replace

import yaml

from m3rs_eval.preflight import run_doctor


def _isolated_config(fixture_config, tmp_path):
    return replace(fixture_config, output_root=tmp_path / "runs")


def test_doctor_rejects_training_split_in_formal_protocol(fixture_config, tmp_path):
    config = _isolated_config(fixture_config, tmp_path)
    protocol = yaml.safe_load(config.protocol_path.read_text(encoding="utf-8"))
    protocol["datasets"]["levir_cc"]["formal_split"] = "train"
    protocol_path = tmp_path / "leaky_protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol), encoding="utf-8")

    report = run_doctor(replace(config, protocol_path=protocol_path))

    assert not report.passed
    assert any(check.code == "DATASET_SPLIT_LEAKAGE" for check in report.checks)


def test_doctor_reports_git_unavailable_as_warning(fixture_config, tmp_path):
    report = run_doctor(_isolated_config(fixture_config, tmp_path))

    assert report.passed
    assert report.exit_code == 0
    assert any(
        check.code == "GIT_UNAVAILABLE" and check.level == "warning"
        for check in report.checks
    )


def test_doctor_rejects_missing_system_executable(fixture_config, tmp_path):
    config = _isolated_config(fixture_config, tmp_path)
    system = replace(
        config.system,
        command=("m3rs-command-that-does-not-exist", "{input_jsonl}", "{output_jsonl}"),
    )

    report = run_doctor(replace(config, system=system))

    assert not report.passed
    assert any(check.code == "COMMAND_NOT_EXECUTABLE" for check in report.checks)


def test_doctor_rejects_output_root_that_is_a_file(fixture_config, tmp_path):
    output_root = tmp_path / "runs"
    output_root.write_text("not a directory", encoding="utf-8")

    report = run_doctor(replace(fixture_config, output_root=output_root))

    assert not report.passed
    assert any(check.code == "OUTPUT_NOT_WRITABLE" for check in report.checks)


def test_doctor_rejects_disk_below_threshold(fixture_config, tmp_path, monkeypatch):
    config = _isolated_config(fixture_config, tmp_path)
    monkeypatch.setattr(
        "m3rs_eval.preflight.shutil.disk_usage",
        lambda _path: shutil._ntuple_diskusage(total=100, used=99, free=1),
    )

    report = run_doctor(config)

    assert not report.passed
    assert any(check.code == "DISK_SPACE_LOW" for check in report.checks)


def test_doctor_result_is_json_serializable(fixture_config, tmp_path):
    report = run_doctor(_isolated_config(fixture_config, tmp_path))

    payload = report.to_dict()

    assert payload["passed"] is True
    assert payload["exit_code"] == 0
    assert payload["checks"]


def test_doctor_rejects_missing_official_scorer_executable(fixture_config, tmp_path):
    config = _isolated_config(fixture_config, tmp_path)
    levir = replace(
        config.datasets["levir_cc"],
        profile="official",
        official_scorer_output=None,
        official_scorer_command=(
            "m3rs-scorer-that-does-not-exist",
            "{references_jsonl}",
            "{predictions_jsonl}",
            "{output_json}",
        ),
        official_scorer_working_directory=tmp_path,
        official_scorer_expected_commit="fixture-commit",
    )

    report = run_doctor(
        replace(config, datasets={**config.datasets, "levir_cc": levir})
    )

    assert not report.passed
    assert any(check.code == "OFFICIAL_SCORER_COMMAND_INVALID" for check in report.checks)
