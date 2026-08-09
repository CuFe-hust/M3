"""C2 contract tests for the generic local SegFormer client."""

from __future__ import annotations

import builtins
import hashlib
import socket
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from models.base import ModelAssetHashMismatchError, ModelAssetMissingError
from models.segformer_transformers import (
    SegFormerDependencyError,
    SegFormerDeviceError,
    SegFormerInferenceError,
    SegFormerLoadError,
    SegFormerMetadataError,
    SegFormerTransformersClient,
    _load_transformers_runtime,
    _run_transformers_inference,
)
from models.settings import SegFormerSettings

_CLASS_MAP = {0: "background", 1: "vehicle"}


class _FakeModel:
    def __init__(self, channels: int = 2) -> None:
        self.config = SimpleNamespace(num_labels=channels)
        self.to_calls: list[str] = []
        self.eval_calls = 0

    def to(self, device: str) -> "_FakeModel":
        self.to_calls.append(device)
        return self

    def eval(self) -> None:
        self.eval_calls += 1


def _settings(root: Path, *, weight_bytes: bytes = b"segformer", **overrides: Any):
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.safetensors").write_bytes(weight_bytes)
    values: dict[str, Any] = {
        "model_path": root,
        "logical_model_id": "segformer-test-local",
        "weights_sha256": hashlib.sha256(weight_bytes).hexdigest(),
        "device": "cpu",
        "revision": "revision-1",
    }
    values.update(overrides)
    return SegFormerSettings(**values)


def _prediction(width: int = 3, height: int = 2):
    class_ids = np.zeros((height, width), dtype=np.int64)
    class_ids[:, -1] = 1
    confidence = np.full((height, width), 0.75, dtype=np.float32)
    return class_ids, confidence


def test_module_import_does_not_import_heavy_dependencies() -> None:
    script = (
        "import sys; import models.segformer_transformers; "
        "print(sorted({'torch','transformers'} & set(sys.modules)))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "[]"


def test_constructor_does_not_load_and_first_predict_loads_once(tmp_path: Path) -> None:
    calls: list[str] = []

    def loader(settings: SegFormerSettings):
        calls.append(settings.logical_model_id)
        return _FakeModel(), object(), "cpu"

    client = SegFormerTransformersClient(
        _settings(tmp_path / "checkpoint"),
        _CLASS_MAP,
        loader=loader,
        inference_runner=lambda *args: _prediction(),
    )

    assert client.loaded is False
    assert calls == []
    first = client.predict(Image.new("RGB", (3, 2)))
    second = client.predict(Image.new("RGB", (3, 2)))

    assert calls == ["segformer-test-local"]
    assert client.loaded is True
    assert first.width == second.width == 3


def test_constructor_does_not_touch_missing_checkpoint(tmp_path: Path) -> None:
    settings = SegFormerSettings(
        model_path=tmp_path / "not-present",
        logical_model_id="segformer-construction-only",
        weights_sha256="0" * 64,
        device="cpu",
    )

    client = SegFormerTransformersClient(settings, _CLASS_MAP)

    assert client.loaded is False


def test_same_logical_identity_reuses_loaded_runtime_across_clients(tmp_path: Path) -> None:
    calls: list[str] = []

    def loader(settings: SegFormerSettings):
        calls.append(settings.logical_model_id)
        return _FakeModel(), object(), "cpu"

    settings = _settings(tmp_path / "checkpoint")
    clients = [
        SegFormerTransformersClient(
            settings,
            _CLASS_MAP,
            loader=loader,
            inference_runner=lambda *args: _prediction(),
        )
        for _ in range(2)
    ]

    for client in clients:
        client.predict(Image.new("RGB", (3, 2)))

    assert calls == ["segformer-test-local"]


def test_auto_loader_is_local_only_and_uses_auto_classes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, dict[str, Any]] = {}
    model = _FakeModel()

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, source: str, **kwargs: Any):
            calls["model"] = {"source": source, **kwargs}
            return model

    class FakeAutoProcessor:
        @classmethod
        def from_pretrained(cls, source: str, **kwargs: Any):
            calls["processor"] = {"source": source, **kwargs}
            return object()

    torch_module = ModuleType("torch")
    torch_module.cuda = SimpleNamespace(is_available=lambda: False)
    torch_module.float16 = "float16"
    torch_module.bfloat16 = "bfloat16"
    torch_module.float32 = "float32"
    transformers_module = ModuleType("transformers")
    transformers_module.AutoImageProcessor = FakeAutoProcessor
    transformers_module.AutoModelForSemanticSegmentation = FakeAutoModel
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network used")),
    )
    settings = _settings(tmp_path / "checkpoint")

    loaded_model, _, device = _load_transformers_runtime(settings)

    assert loaded_model is model
    assert device == "cpu"
    assert calls["model"]["local_files_only"] is True
    assert calls["processor"]["local_files_only"] is True
    assert calls["model"]["revision"] == "revision-1"
    assert model.to_calls == ["cpu"]
    assert model.eval_calls == 1


def test_download_cannot_be_enabled() -> None:
    with pytest.raises(ValueError):
        SegFormerSettings(allow_download=True)


class _FakeTensor:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    @property
    def shape(self):
        return self.value.shape

    def to(self, device: str) -> "_FakeTensor":
        return self

    def __getitem__(self, index: Any) -> "_FakeTensor":
        return _FakeTensor(self.value[index])

    def numpy(self) -> np.ndarray:
        return self.value


class _FakeProbabilities:
    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size

    def max(self, dim: int):
        assert dim == 1
        height, width = self.size
        confidence = np.full((1, height, width), 0.8, dtype=np.float32)
        class_ids = np.zeros((1, height, width), dtype=np.int64)
        class_ids[:, :, -1] = 1
        return _FakeTensor(confidence), _FakeTensor(class_ids)


class _InferenceContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: Any) -> None:
        return None


def _install_fake_inference_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    channels: int,
    observed: dict[str, Any],
) -> None:
    torch_module = ModuleType("torch")
    functional_module = ModuleType("torch.nn.functional")
    nn_module = ModuleType("torch.nn")
    logits = _FakeTensor(np.zeros((1, channels, 1, 1), dtype=np.float32))

    def interpolate(value: _FakeTensor, *, size, mode: str, align_corners: bool):
        observed["resize"] = (size, mode, align_corners)
        observed["size"] = size
        return value

    torch_module.inference_mode = lambda: _InferenceContext()
    torch_module.softmax = lambda value, dim: _FakeProbabilities(observed["size"])
    functional_module.interpolate = interpolate
    nn_module.functional = functional_module
    torch_module.nn = nn_module
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch.nn", nn_module)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", functional_module)
    observed["logits"] = logits


def test_inference_resizes_logits_then_returns_confidence_at_original_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}
    _install_fake_inference_modules(monkeypatch, channels=2, observed=observed)

    class Processor:
        def __call__(self, **kwargs: Any):
            observed["processor"] = kwargs
            return {"pixel_values": _FakeTensor(np.zeros((1, 3, 1, 1)))}

    class Model:
        def __call__(self, **kwargs: Any):
            observed["model_inputs"] = kwargs
            return SimpleNamespace(logits=observed["logits"])

    class_ids, confidence = _run_transformers_inference(
        Model(), Processor(), Image.new("RGB", (5, 3)), "cpu", 2
    )

    assert observed["resize"] == ((3, 5), "bilinear", False)
    assert observed["processor"]["return_tensors"] == "pt"
    assert class_ids.shape == confidence.shape == (3, 5)
    assert np.allclose(confidence, 0.8)


def test_logits_channel_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, Any] = {}
    _install_fake_inference_modules(monkeypatch, channels=3, observed=observed)
    processor = lambda **kwargs: {"pixel_values": _FakeTensor(np.zeros((1, 3, 1, 1)))}
    model = lambda **kwargs: SimpleNamespace(logits=observed["logits"])

    with pytest.raises(SegFormerInferenceError, match="class contract"):
        _run_transformers_inference(
            model, processor, Image.new("RGB", (2, 2)), "cpu", 2
        )


def test_result_is_aligned_and_exposes_path_free_identity(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "private" / "checkpoint")
    client = SegFormerTransformersClient(
        settings,
        _CLASS_MAP,
        loader=lambda settings: (_FakeModel(), object(), "cpu"),
        inference_runner=lambda *args: _prediction(width=4, height=3),
    )

    result = client.predict(Image.new("L", (4, 3)))
    trace = result.trace_metadata()

    assert result.mask.shape == result.confidence_map.shape == (3, 4)
    assert (result.width, result.height) == (4, 3)
    assert result.model_revision == "revision-1"
    assert result.weights_sha256 == settings.weights_sha256
    assert str(settings.model_path) not in str(trace)
    assert set(trace) == {
        "logical_model_id",
        "revision",
        "sha256",
        "device",
        "cpu_fallback_used",
    }


def test_external_class_map_is_required_to_be_verified() -> None:
    settings = SegFormerSettings(device="cpu")

    with pytest.raises(SegFormerMetadataError, match="placeholder"):
        SegFormerTransformersClient(settings, {0: "LABEL_0"})
    with pytest.raises(SegFormerMetadataError, match="contiguous"):
        SegFormerTransformersClient(settings, {1: "vehicle"})


def test_loaded_model_channels_must_match_external_map(tmp_path: Path) -> None:
    client = SegFormerTransformersClient(
        _settings(tmp_path / "checkpoint"),
        _CLASS_MAP,
        loader=lambda settings: (_FakeModel(channels=3), object(), "cpu"),
    )

    with pytest.raises(SegFormerMetadataError, match="verified class map"):
        client.predict(Image.new("RGB", (2, 2)))


def test_missing_dependency_has_stable_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def blocked_import(name: str, *args: Any, **kwargs: Any):
        if name == "torch":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(SegFormerDependencyError, match="torch and transformers"):
        _load_transformers_runtime(_settings(tmp_path / "checkpoint"))


def test_missing_weights_and_hash_mismatch_fail_before_loader(tmp_path: Path) -> None:
    missing_settings = SegFormerSettings(
        model_path=tmp_path / "missing",
        logical_model_id="segformer-missing",
        weights_sha256="0" * 64,
        device="cpu",
    )
    missing = SegFormerTransformersClient(
        missing_settings,
        _CLASS_MAP,
        loader=lambda settings: (_FakeModel(), object(), "cpu"),
    )

    with pytest.raises(ModelAssetMissingError, match="model.safetensors"):
        missing.predict(Image.new("RGB", (2, 2)))

    mismatch = SegFormerTransformersClient(
        _settings(tmp_path / "mismatch", weights_sha256="0" * 64),
        _CLASS_MAP,
        loader=lambda settings: (_FakeModel(), object(), "cpu"),
    )
    with pytest.raises(ModelAssetHashMismatchError, match="digest mismatch"):
        mismatch.predict(Image.new("RGB", (2, 2)))


def test_cpu_fallback_requires_explicit_policy_and_is_traced(tmp_path: Path) -> None:
    denied = SegFormerTransformersClient(
        _settings(tmp_path / "denied", device="auto"),
        _CLASS_MAP,
        loader=lambda settings: (_FakeModel(), object(), "cpu"),
    )
    with pytest.raises(SegFormerDeviceError, match="fallback is not enabled"):
        denied.predict(Image.new("RGB", (3, 2)))

    allowed = SegFormerTransformersClient(
        _settings(
            tmp_path / "allowed",
            device="auto",
            allow_cpu_fallback=True,
        ),
        _CLASS_MAP,
        loader=lambda settings: (_FakeModel(), object(), "cpu"),
        inference_runner=lambda *args: _prediction(),
    )
    result = allowed.predict(Image.new("RGB", (3, 2)))

    assert result.device == "cpu"
    assert result.cpu_fallback_used is True
    assert result.trace_metadata()["cpu_fallback_used"] is True


def test_require_cuda_never_silently_accepts_cpu(tmp_path: Path) -> None:
    client = SegFormerTransformersClient(
        _settings(tmp_path / "checkpoint", device="auto", require_cuda=True),
        _CLASS_MAP,
        loader=lambda settings: (_FakeModel(), object(), "cpu"),
    )

    with pytest.raises(SegFormerDeviceError, match="requires CUDA"):
        client.predict(Image.new("RGB", (2, 2)))


def test_loader_error_does_not_leak_absolute_path(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "private-machine-path" / "checkpoint")

    def loader(_: SegFormerSettings):
        raise RuntimeError(f"failed at {settings.model_path.resolve()}")

    client = SegFormerTransformersClient(settings, _CLASS_MAP, loader=loader)

    with pytest.raises(SegFormerLoadError) as error:
        client.predict(Image.new("RGB", (2, 2)))

    assert str(settings.model_path.resolve()) not in str(error.value)
    assert str(error.value) == "SegFormer runtime could not be loaded: RuntimeError"
