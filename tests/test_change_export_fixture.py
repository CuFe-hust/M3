from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import export_multimodal_sft_checkpoint as export_cli
from scripts.prepare_change_qwen_sft import _answer_result
from training.multimodal_sft.change_export_fixture import (
    CHANGE_EXPORT_FIXTURE_SELECTION_POLICY,
    ChangeExportFixtureError,
    build_change_export_fixture,
)
from training.multimodal_sft.change_target_contract import (
    CHANGE_TARGET_CONTRACT_VERSION,
    change_target_contract_identity,
)
from training.multimodal_sft.image_roots import ImageRootRegistry
from training.multimodal_sft.profiles.change_agent import ChangeAgentDataProfile


def _image(path: Path, color: tuple[int, int, int]) -> None:
    Image = pytest.importorskip("PIL.Image")
    Image.new("RGB", (8, 6), color).save(path)


def _row(episode_id: str, pair: str, t1: str, t2: str) -> dict:
    images = [
        {"image_source": "levir", "path": t1, "role": "raw_full_t1"},
        {"image_source": "levir", "path": t2, "role": "raw_full_t2"},
    ]
    return {
        "schema_version": 2,
        "episode_id": episode_id,
        "parent_sample_id": pair,
        "dataset": "LEVIR-CC",
        "split": "validation",
        "task": "change_caption",
        "input_contract": "semantic_pair_v1",
        "question": "",
        "images": images,
        "request_payload": {
            "decision_stage": "initial", "task": "change_caption",
            "image_manifest": [{"index": "0", "role": "raw_full_t1"}, {"index": "1", "role": "raw_full_t2"}],
        },
        "target": {
            "response_schema": "ChangeInitialResult",
            "contract_version": CHANGE_TARGET_CONTRACT_VERSION,
            "result": _answer_result("A building appeared."),
        },
    }


def _fixture_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "images"
    root.mkdir()
    for name, color in (("a1.png", (1, 2, 3)), ("a2.png", (4, 5, 6)), ("z1.png", (7, 8, 9)), ("z2.png", (10, 11, 12))):
        _image(root / name, color)
    rows = [
        _row("z-episode", "pair-z", "z1.png", "z2.png"),
        _row("a-episode", "pair-a", "a1.png", "a2.png"),
    ]
    validation = tmp_path / "validation.jsonl"
    validation.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    prompt = tmp_path / "prompt.md"
    prompt.write_text("stable prompt\n", encoding="utf-8")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "change_prompt": {"ref": "change_dual_path_v9", "sha256": hashlib.sha256(prompt.read_text(encoding="utf-8").encode()).hexdigest()},
        "target_contract": change_target_contract_identity(),
        "outputs": {"train.jsonl_sha256": digest(validation), "validation.jsonl_sha256": digest(validation)},
    }), encoding="utf-8")
    return root, validation, manifest, prompt


def test_export_fixture_is_rendered_deterministically_with_t1_t2(tmp_path: Path) -> None:
    root, validation, manifest, prompt = _fixture_inputs(tmp_path)
    output = tmp_path / "fixture.json"
    fixture = build_change_export_fixture(
        validation=validation,
        manifest=manifest,
        image_roots=ImageRootRegistry({"levir": root}),
        prompt_ref="change_dual_path_v9",
        prompt_file=prompt,
        output=output,
    )
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == fixture
    assert fixture["selection_policy"] == CHANGE_EXPORT_FIXTURE_SELECTION_POLICY
    assert fixture["metadata"]["episode_id"] == "a-episode"
    assert fixture["target_schema"] == "ChangeInitialResult"
    assert len(fixture["messages"]) == 3
    assert [item["role"] for item in fixture["images"]] == ["raw_full_t1", "raw_full_t2"]
    assert all(Path(item["path"]).is_absolute() and Path(item["path"]).is_file() for item in fixture["images"])
    profile = ChangeAgentDataProfile(data_manifest=manifest, prompt_ref="change_dual_path_v9", prompt_file=prompt)
    selected = min(list(profile.read(validation)), key=lambda row: row["episode_id"])
    assert fixture["messages"] == list(profile.render_messages(selected))
    with pytest.raises(ChangeExportFixtureError, match="EXPORT_FIXTURE_OUTPUT_EXISTS"):
        build_change_export_fixture(
            validation=validation, manifest=manifest, image_roots={"levir": root},
            prompt_ref="change_dual_path_v9", prompt_file=prompt, output=output,
        )


def test_generic_export_cli_passes_change_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured = {}
    adapter = SimpleNamespace(name="qwen3_5")

    class _Registry:
        def resolve(self, *args, **kwargs):
            return adapter, SimpleNamespace()

    class _Exporter:
        def __init__(self, selected):
            assert selected is adapter

        def export(self, **kwargs):
            captured.update(kwargs)
            return {"status": "PASS"}

    monkeypatch.setattr(export_cli, "default_registry", lambda: _Registry())
    monkeypatch.setattr(export_cli, "GenericExporter", _Exporter)
    fixture = tmp_path / "fixture.json"
    fixture.write_text("{}", encoding="utf-8")
    assert export_cli.main([
        "--model-id", "model", "--checkpoint-dir", "checkpoint", "--output-dir", "output",
        "--model-adapter", "qwen3_5", "--verify-forward", "--change-fixture", str(fixture),
    ]) == 0
    assert captured["change_fixture"] == str(fixture)
    assert captured["verify_forward"] is True
