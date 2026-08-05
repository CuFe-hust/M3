from __future__ import annotations

import sys
from types import ModuleType

import main as baseline_main
from models.qwen3vl import Qwen3VLBaseline, Qwen3VLSettings


def test_local_files_only_is_forwarded_to_model_and_processor(monkeypatch) -> None:
    model_calls: list[tuple[str, dict[str, object]]] = []
    processor_calls: list[tuple[str, dict[str, object]]] = []

    torch_module = ModuleType("torch")
    torch_module.float16 = object()
    torch_module.bfloat16 = object()
    torch_module.float32 = object()

    class FakeLoadedModel:
        def eval(self) -> None:
            return None

    class FakeConfigFactory:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> object:
            assert model_id == "/external/models/qwen3-vl"
            assert kwargs == {"local_files_only": True}
            return type("FakeConfig", (), {"model_type": "qwen3_vl"})()

    class FakeModelFactory:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> FakeLoadedModel:
            model_calls.append((model_id, kwargs))
            return FakeLoadedModel()

    class FakeProcessorFactory:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> object:
            processor_calls.append((model_id, kwargs))
            return object()

    transformers_module = ModuleType("transformers")
    transformers_module.AutoConfig = FakeConfigFactory
    transformers_module.Qwen3VLForConditionalGeneration = FakeModelFactory
    transformers_module.AutoProcessor = FakeProcessorFactory
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    Qwen3VLBaseline(
        Qwen3VLSettings(
            model_id="/external/models/qwen3-vl",
            dtype="bfloat16",
            local_files_only=True,
        )
    )

    assert model_calls == [
        (
            "/external/models/qwen3-vl",
            {
                "dtype": torch_module.bfloat16,
                "device_map": "auto",
                "local_files_only": True,
            },
        )
    ]
    assert processor_calls == [
        (
            "/external/models/qwen3-vl",
            {"local_files_only": True},
        )
    ]


def test_qwen35_uses_native_model_class(monkeypatch) -> None:
    model_calls: list[tuple[str, dict[str, object]]] = []

    torch_module = ModuleType("torch")
    torch_module.float16 = object()
    torch_module.bfloat16 = object()
    torch_module.float32 = object()

    class FakeConfigFactory:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> object:
            assert model_id == "/external/models/Qwen3.5-9B"
            assert kwargs == {"local_files_only": True}
            return type("FakeConfig", (), {"model_type": "qwen3_5"})()

    class FakeLoadedModel:
        config = type("FakeConfig", (), {"model_type": "qwen3_5"})()

        def eval(self) -> None:
            return None

    class FakeQwen35Factory:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> FakeLoadedModel:
            model_calls.append((model_id, kwargs))
            return FakeLoadedModel()

    class FakeProcessorFactory:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> object:
            return object()

    transformers_module = ModuleType("transformers")
    transformers_module.AutoConfig = FakeConfigFactory
    transformers_module.AutoProcessor = FakeProcessorFactory
    transformers_module.Qwen3_5ForConditionalGeneration = FakeQwen35Factory
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    Qwen3VLBaseline(
        Qwen3VLSettings(
            model_id="/external/models/Qwen3.5-9B",
            dtype="bfloat16",
            local_files_only=True,
        )
    )

    assert model_calls[0][0] == "/external/models/Qwen3.5-9B"
    assert model_calls[0][1]["dtype"] is torch_module.bfloat16


def test_baseline_config_enables_local_files_only(monkeypatch) -> None:
    captured: list[Qwen3VLSettings] = []

    class FakeBaseline:
        def __init__(self, settings: Qwen3VLSettings) -> None:
            captured.append(settings)

    monkeypatch.setattr("models.qwen3_vl.baseline.Qwen3VLBaseline", FakeBaseline)
    baseline_main._load_model(
        {
            "model": {
                "id": "/external/models/qwen3-vl",
                "local_files_only": True,
            }
        }
    )

    assert len(captured) == 1
    assert captured[0].model_id == "/external/models/qwen3-vl"
    assert captured[0].local_files_only is True
