from __future__ import annotations

import json

import pytest

from m3rs_eval.contracts import RunManifest
from m3rs_eval.state import InvalidTransition, RunStateStore


def manifest(status: str = "created") -> RunManifest:
    return RunManifest(
        run_id="20260805T120000+0800__fixture-v1__12345678",
        status=status,
        mode="smoke",
        protocol_id="official_full_v1",
        created_at="2026-08-05T12:00:00+08:00",
        config_hash="config-hash",
        request_manifest_hash="pending",
        eligible_for_history=False,
        protocol_hash="protocol-hash",
        command_hash="command-hash",
        system_version_hash="system-version-hash",
        metadata={},
    )


def test_complete_cannot_transition_back_to_running(tmp_path):
    store = RunStateStore.create(tmp_path, manifest())
    store.transition("preflight_passed")
    store.transition("inference_running")
    store.transition("evaluating")
    store.transition("complete")

    with pytest.raises(InvalidTransition):
        store.transition("inference_running")


def test_manifest_updates_are_atomic(tmp_path):
    store = RunStateStore.create(tmp_path, manifest())

    store.transition("preflight_passed")

    assert not list(tmp_path.glob("*.tmp"))
    payload = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert payload["status"] == "preflight_passed"


def test_state_machine_rejects_skipped_transition(tmp_path):
    store = RunStateStore.create(tmp_path, manifest())

    with pytest.raises(InvalidTransition):
        store.transition("evaluating")


@pytest.mark.parametrize("terminal", ["complete", "incomplete", "failed"])
def test_terminal_runs_are_immutable(tmp_path, terminal):
    store = RunStateStore.create(tmp_path, manifest())
    store.transition("preflight_passed")
    store.transition("inference_running")
    store.transition("evaluating")
    store.transition(terminal)

    with pytest.raises(InvalidTransition):
        store.update(metadata={"changed": True})


def test_load_validates_and_preserves_manifest(tmp_path):
    created = RunStateStore.create(tmp_path, manifest())
    created.transition("preflight_passed")

    loaded = RunStateStore.load(tmp_path)

    assert loaded.manifest == created.manifest
    assert loaded.run_dir == tmp_path


def test_create_refuses_to_overwrite_existing_manifest(tmp_path):
    RunStateStore.create(tmp_path, manifest())

    with pytest.raises(FileExistsError):
        RunStateStore.create(tmp_path, manifest())
