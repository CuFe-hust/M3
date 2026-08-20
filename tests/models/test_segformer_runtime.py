"""Offline SegFormer runtime and migrated metadata regression tests.

SegFormer 离线运行时与迁移元数据回归测试。真实 GB 级依赖和权重不会加载。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from models.base import (
    ModelAssetHashMismatchError,
    ModelAssetMissingError,
    ModelAssetPointerError,
)
from models.segformer_transformers import SegFormerRuntime
from models.settings import ModelSettings, SegFormerSettings


_ISAID_LABELS = (
    "background",
    "storage_tank",
    "Large_Vehicle",
    "Small_Vehicle",
    "plane",
    "ship",
    "Swimming_pool",
    "Harbor",
    "tennis_court",
    "Ground_Track_Field",
    "Soccer_ball_field",
    "baseball_diamond",
    "Bridge",
    "basketball_court",
    "Roundabout",
    "Helicopter",
)


class _FakeModel:
    def __init__(self, label_count: int) -> None:
        self.config = SimpleNamespace(num_labels=label_count)


def _checkpoint(
    root: Path,
    *,
    labels: tuple[str, ...] = ("background", "vehicle"),
    weight_bytes: bytes = b"fake-segformer-weights",
    with_classes: bool = True,
) -> tuple[Path, str]:
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["SegformerForSemanticSegmentation"],
                "model_type": "segformer",
                "id2label": {
                    str(index): f"LABEL_{index}" for index in range(len(labels))
                },
            }
        ),
        encoding="utf-8",
    )
    if with_classes:
        (root / "classes.json").write_text(
            json.dumps(
                {
                    "num_classes": len(labels),
                    "id2name": {
                        str(index): name for index, name in enumerate(labels)
                    },
                    "name2id": {
                        name: index for index, name in enumerate(labels)
                    },
                }
            ),
            encoding="utf-8",
        )
    (root / "model.safetensors").write_bytes(weight_bytes)
    return root, hashlib.sha256(weight_bytes).hexdigest()


def _settings(path: Path, digest: str, **overrides: Any) -> SegFormerSettings:
    values: dict[str, Any] = {
        "model_path": path,
        "logical_model_id": "SegFormer:test:local",
        "weights_sha256": digest,
        "device": "cpu",
    }
    values.update(overrides)
    return SegFormerSettings(**values)


def _runtime(
    path: Path,
    digest: str,
    *,
    loads: list[str] | None = None,
) -> SegFormerRuntime:
    def loader(settings: SegFormerSettings) -> tuple[Any, Any, str]:
        if loads is not None:
            loads.append(settings.logical_model_id)
        return _FakeModel(2), object(), "cpu"

    return SegFormerRuntime(
        _settings(path, digest),
        loader=loader,
        inference_runner=lambda model, processor, image, device, count: [
            [0, 1],
            [1, 0],
        ],
    )


def test_import_is_lazy_and_does_not_load_heavy_runtimes() -> None:
    script = (
        "import sys; import models.segformer_transformers; "
        "print(sorted({'torch','transformers','onnxruntime'} & set(sys.modules)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "[]"


def test_core_models_import_does_not_load_torch_or_transformers() -> None:
    script = (
        "import sys; import models; "
        "print(sorted({'torch','transformers'} & set(sys.modules)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "[]"


def test_model_settings_declare_authoritative_classes_without_filesystem_access(
    tmp_path: Path,
) -> None:
    settings = ModelSettings()
    assert settings.segformer_isaid.classes_filename == "classes.json"
    assert settings.segformer_oem.classes_filename == "classes.json"
    assert settings.segformer_isaid.allow_download is False
    assert settings.segformer_oem.allow_download is False
    assert str(settings.segformer_isaid.model_path) not in (
        settings.segformer_isaid.logical_model_id
    )

    missing = SegFormerSettings(
        model_path=tmp_path / "missing-checkpoint",
        logical_model_id="segformer-missing-test",
        classes_filename="missing-classes.json",
    )
    assert missing.classes_filename == "missing-classes.json"


def test_classes_filename_does_not_change_logical_model_identity() -> None:
    with_classes = SegFormerSettings(classes_filename="classes.json")
    without_classes = SegFormerSettings(classes_filename=None)
    assert with_classes.logical_model_id == without_classes.logical_model_id


def test_predict_wraps_preprocess_inference_and_class_mapping(tmp_path: Path) -> None:
    path, digest = _checkpoint(tmp_path / "checkpoint")
    result = _runtime(path, digest).predict(Image.new("RGB", (2, 2)))
    assert (result.width, result.height) == (2, 2)
    assert result.class_ids == (0, 1)
    assert result.class_name(0) == "background"
    assert result.class_name(1) == "vehicle"
    assert result.logical_model_id == "SegFormer:test:local"
    assert result.device == "cpu"


def test_relative_checkpoint_path_and_config_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, digest = _checkpoint(tmp_path / "relative-checkpoint")
    monkeypatch.chdir(tmp_path)
    result = _runtime(Path(path.name), digest).predict(Image.new("RGB", (2, 2)))
    assert result.class_ids == (0, 1)


def test_missing_weight_has_stable_model_asset_error(tmp_path: Path) -> None:
    path, digest = _checkpoint(tmp_path / "checkpoint")
    (path / "model.safetensors").unlink()
    with pytest.raises(ModelAssetMissingError, match="model.safetensors"):
        _runtime(path, digest).load()


def test_git_lfs_pointer_is_rejected_before_loader(tmp_path: Path) -> None:
    pointer = (
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:" + b"a" * 64 + b"\nsize 123\n"
    )
    path, digest = _checkpoint(tmp_path / "checkpoint", weight_bytes=pointer)
    with pytest.raises(
        ModelAssetPointerError,
        match="actual binary has not been downloaded",
    ):
        _runtime(path, digest).load()


def test_wrong_sha256_is_rejected_before_loader(tmp_path: Path) -> None:
    path, _ = _checkpoint(tmp_path / "checkpoint")
    with pytest.raises(ModelAssetHashMismatchError, match="digest mismatch"):
        _runtime(path, "0" * 64).load()


def test_concurrent_calls_load_one_checkpoint_once(tmp_path: Path) -> None:
    path, digest = _checkpoint(tmp_path / "checkpoint")
    loads: list[str] = []
    runtime = _runtime(path, digest, loads=loads)
    barrier = threading.Barrier(12)
    results: list[tuple[str, ...]] = []

    def worker() -> None:
        barrier.wait()
        results.append(runtime.labels)

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert loads == ["SegFormer:test:local"]
    assert results == [("background", "vehicle")] * 12


def test_failed_load_does_not_poison_runtime(tmp_path: Path) -> None:
    path, digest = _checkpoint(tmp_path / "checkpoint")
    attempts = 0

    def loader(settings: SegFormerSettings) -> tuple[Any, Any, str]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient")
        return _FakeModel(2), object(), "cpu"

    runtime = SegFormerRuntime(_settings(path, digest), loader=loader)
    with pytest.raises(RuntimeError, match="transient"):
        runtime.load()
    assert runtime.loaded is False
    runtime.load()
    assert runtime.loaded is True
    assert attempts == 2


def test_oem_falls_back_to_checkpoint_config_labels(tmp_path: Path) -> None:
    path, digest = _checkpoint(tmp_path / "checkpoint", with_classes=False)
    runtime = SegFormerRuntime(
        _settings(path, digest, classes_filename=None),
        loader=lambda settings: (_FakeModel(2), object(), "cpu"),
    )
    assert runtime.labels == ("LABEL_0", "LABEL_1")


def test_migrated_isaid_class_order_is_frozen() -> None:
    payload = json.loads(
        Path("models/segformer_mitb2_isaid/classes.json").read_text(encoding="utf-8")
    )
    labels = tuple(payload["id2name"][str(index)] for index in range(16))
    assert labels == _ISAID_LABELS
    assert payload["name2id"] == {
        name: index for index, name in enumerate(_ISAID_LABELS)
    }
