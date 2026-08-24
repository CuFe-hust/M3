from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.multimodal_sft.contracts import PreparedMultimodalEpisode
from training.multimodal_sft.data import JsonlDataProfile
from training.multimodal_sft.image_roots import ImageRootError, ImageRootRegistry
from training.multimodal_sft.profiles.change_agent import ChangeAgentDataError, ChangeAgentDataProfile
from training.multimodal_sft.profiles.phase2 import Phase2DataProfile
from training.multimodal_sft.trainer_core import GenericTrainerCore


def _image(path: Path, color: tuple[int, int, int]) -> None:
    Image = pytest.importorskip("PIL.Image")
    image = Image.new("RGB", (8, 6), color)
    image.save(path)


def _change_row() -> dict:
    from scripts.prepare_change_qwen_sft import _answer_result

    images = [
        {"image_source": "changechat", "path": "t1.png", "role": "raw_full_t1"},
        {"image_source": "changechat", "path": "t2.png", "role": "raw_full_t2"},
    ]
    return {
        "schema_version": 1, "episode_id": "change/train/sample/change_caption/0", "parent_sample_id": "sample",
        "dataset": "ChangeChat", "split": "train", "task": "change_caption", "input_contract": "semantic_pair_v1",
        "question": "", "images": images,
        "request_payload": {"decision_stage": "initial", "task": "change_caption", "image_manifest": [{"index": "0", "role": "raw_full_t1"}, {"index": "1", "role": "raw_full_t2"}]},
        "target": {"response_schema": "ChangeInitialResult", "result": _answer_result("no change")},
    }


def _write_change_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    _image(tmp_path / "t1.png", (10, 20, 30))
    _image(tmp_path / "t2.png", (30, 20, 10))
    prompt = tmp_path / "prompt.md"
    prompt.write_text("stable prompt\n", encoding="utf-8")
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps(_change_row(), ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    prompt_digest = hashlib.sha256(prompt.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    manifest.write_text(json.dumps({"change_prompt": {"ref": "prompt.md", "sha256": prompt_digest}, "outputs": {"train.jsonl_sha256": digest(train), "validation.jsonl_sha256": digest(train)}}, indent=2) + "\n", encoding="utf-8")
    return train, prompt, manifest


def test_change_profile_prepares_real_source_schema_and_t1_t2(tmp_path: Path) -> None:
    train, prompt, manifest = _write_change_fixture(tmp_path)
    profile = ChangeAgentDataProfile(data_manifest=manifest, prompt_file=prompt)
    rows = list(profile.read(train))
    assert "messages" not in rows[0]
    prepared = profile.prepare(rows[0], image_roots=ImageRootRegistry({"changechat": tmp_path}), split="train", epoch=0, seed=42)
    assert isinstance(prepared, PreparedMultimodalEpisode)
    assert prepared.image_roles == ("raw_full_t1", "raw_full_t2")
    assert [image.mode for image in prepared.images] == ["RGB", "RGB"]
    assert prepared.messages[0]["content"].endswith("Set agent_name to change_agent and status to completed.")
    assert prepared.messages[1]["content"][1]["text"] == "AUTHORITATIVE RAW T1 - earlier full scene"
    assert prepared.metadata["request_payload"]["decision_stage"] == "initial"


def test_change_prompt_and_data_manifest_sha_are_fail_closed(tmp_path: Path) -> None:
    train, prompt, manifest = _write_change_fixture(tmp_path)
    profile = ChangeAgentDataProfile(data_manifest=manifest, prompt_file=prompt)
    list(profile.read(train))
    prompt.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ChangeAgentDataError, match="CHANGE_PROMPT_SHA_MISMATCH"):
        ChangeAgentDataProfile(data_manifest=manifest, prompt_file=prompt).prepare(_change_row(), image_roots=ImageRootRegistry({"changechat": tmp_path}), split="train", epoch=0, seed=0)
    train.write_text(train.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ChangeAgentDataError, match="DATA_MANIFEST_SHA_MISMATCH"):
        list(profile.read(train))


def test_image_root_registry_rejects_unknown_escape_missing_and_decode(tmp_path: Path) -> None:
    _image(tmp_path / "ok.png", (1, 2, 3))
    registry = ImageRootRegistry({"known": tmp_path})
    assert registry.resolve("known", "ok.png").name == "ok.png"
    for source, relative, code in (("unknown", "ok.png", "UNKNOWN_IMAGE_SOURCE"), ("known", "../ok.png", "UNSAFE_IMAGE_PATH"), ("known", "missing.png", "IMAGE_MISSING")):
        with pytest.raises(ImageRootError, match=code):
            registry.resolve(source, relative)
    (tmp_path / "bad.png").write_bytes(b"not an image")
    with pytest.raises(ImageRootError, match="IMAGE_DECODE_ERROR"):
        registry.load_rgb("known", "bad.png")
    outside = tmp_path.parent / "outside.png"
    _image(outside, (4, 5, 6))
    try:
        (tmp_path / "escape.png").symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ImageRootError, match="UNSAFE_IMAGE_PATH"):
        registry.resolve("known", "escape.png")


def test_phase2_profile_preserves_old_render_and_seed_contract(tmp_path: Path) -> None:
    _image(tmp_path / "image.png", (20, 30, 40))
    row = {
        "episode_id": "phase2/sample/0", "parent_episode_id": "sample", "image_source": "phase2", "image": "image.png",
        "task_kind": "vqa_self_attention", "augmentation_policy": {"geometry": "orientation_locked"},
        "turns": [{"user_text": "What changed?", "assistant_text": "Nothing", "input_boxes": [], "target_boxes": []}],
    }
    profile = Phase2DataProfile(augmentation_enabled=True)
    prepared_a = profile.prepare(row, image_roots=ImageRootRegistry({"phase2": tmp_path}), split="train", epoch=2, seed="seed")
    prepared_b = profile.prepare(row, image_roots=ImageRootRegistry({"phase2": tmp_path}), split="train", epoch=2, seed="seed")
    assert prepared_a.messages == prepared_b.messages
    assert prepared_a.metadata["augmentation"]["group_seed"] == prepared_b.metadata["augmentation"]["group_seed"]
    assert prepared_a.image_roles == ("image",)


def test_generic_trainer_prepare_boundary_is_profile_owned() -> None:
    class _Profile(JsonlDataProfile):
        def __init__(self):
            super().__init__("phase2")
            self.called = False

        def prepare(self, episode, **kwargs):
            self.called = True
            return "prepared"

    profile = _Profile()
    core = GenericTrainerCore(adapter=SimpleNamespace(), data_profile=profile)
    assert core._prepare_episode("source", image_roots=None, split="train", epoch=0, seed=1) == "prepared"
    assert profile.called
