import json
import subprocess
import sys
from pathlib import Path

import pytest

from spacers_agent.events import EventWriter
from spacers_agent.run_store import RunStore
from spacers_agent.settings import AppSettings, CountingSettings, load_settings


def test_settings_apply_non_secret_environment_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("runs:\n  root: local-runs\n", encoding="utf-8")

    settings = load_settings(
        config_path,
        environ={"QWEN_MODEL": "offline-qwen", "DATASET_ROOT": "fixtures/dataset"},
    )

    assert settings.models.qwen.model == "offline-qwen"
    assert settings.paths.dataset_root == Path("fixtures/dataset")
    assert settings.runs.root == Path("local-runs")


def test_settings_reads_local_dotenv_without_exposing_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    (tmp_path / ".env").write_text("QWEN_MODEL=dotenv-qwen\nQWEN_API_KEY=secret\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    settings = load_settings(config_path)

    assert settings.models.qwen.model == "dotenv-qwen"
    assert "secret" not in settings.model_dump_json()


def test_sequential_counting_rejects_parallel_concurrency() -> None:
    with pytest.raises(ValueError, match="concurrency=1"):
        CountingSettings(sequential=True, concurrency=2)


def test_run_store_writes_manifest_snapshots_and_safe_event(tmp_path: Path) -> None:
    prompt = tmp_path / "count_v1.md"
    prompt.write_text("<!-- version: v1 -->\nPrompt", encoding="utf-8")
    settings = AppSettings.model_validate({"runs": {"root": str(tmp_path / "runs")}})
    store = RunStore(settings.runs.root, Path.cwd())

    manifest = store.create_run(settings, prompt_paths=[prompt], run_id="phase1-test", dataset="fixture", split="dev")

    run_dir = tmp_path / "runs" / manifest.run_id
    saved_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest["run_id"] == "phase1-test"
    assert (run_dir / "config.snapshot.yaml").is_file()
    assert (run_dir / "prompts.snapshot" / "count_v1.md").read_text(encoding="utf-8").endswith("Prompt")
    assert "RUN_CREATED" in (run_dir / "events.jsonl").read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="secret fields"):
        EventWriter(run_dir / "events.jsonl").write("INVALID", details={"api_key": "secret"})


def test_cli_help_offline_health_and_run_initialization(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"runs:\n  root: {str(tmp_path / 'runs')}\n", encoding="utf-8")
    help_result = subprocess.run(
        [sys.executable, "-m", "spacers_agent.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    health_result = subprocess.run(
        [sys.executable, "-m", "spacers_agent.cli", "health", "qwen"],
        check=False,
        capture_output=True,
        text=True,
    )
    run_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "spacers_agent.cli",
            "--config",
            str(config_path),
            "run-init",
            "--run-id",
            "cli-test",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert help_result.returncode == 0
    assert "run-init" in help_result.stdout
    assert health_result.returncode == 0
    assert "deferred_until_phase_2_and_explicit_authorization" in health_result.stdout
    assert run_result.returncode == 0
    assert (tmp_path / "runs" / "cli-test" / "manifest.json").is_file()
