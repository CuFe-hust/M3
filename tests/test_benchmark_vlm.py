"""Tests for the benchmark VLM wrapper and LoRA path resolution.
基准模型封装与 LoRA 路径解析测试。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

from PIL import Image

from data.schema import CanonicalSample
from models.benchmark_vlm import (
    _internvl_grounding_boxes,
    _internvl_prompt,
    resolve_device,
    resolve_load_paths,
)


def _fake_torch() -> ModuleType:
    torch_module = ModuleType("torch")
    torch_module.float16 = object()
    torch_module.bfloat16 = object()
    torch_module.float32 = object()

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeMps:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeBackends:
        mps = FakeMps()

    torch_module.cuda = FakeCuda()
    torch_module.backends = FakeBackends()
    return torch_module


def _make_peft_module(adapter_calls: list) -> ModuleType:
    class FakePeftModel:
        def __init__(self, model, adapter_path: str) -> None:
            self.model = model

        @classmethod
        def from_pretrained(cls, model, adapter_path: str) -> "FakePeftModel":
            adapter_calls.append(str(adapter_path))
            return cls(model, adapter_path)

        def to(self, device: str) -> "FakePeftModel":
            return self

        def eval(self) -> "FakePeftModel":
            return self

    peft_module = ModuleType("peft")
    peft_module.PeftModel = FakePeftModel
    return peft_module


def _install_fake_modules(monkeypatch, model_calls: list, processor_calls: list, adapter_calls: list) -> ModuleType:
    class FakeModel:
        def __init__(self) -> None:
            self.device = "cpu"

        def to(self, device: str) -> "FakeModel":
            return self

        def eval(self) -> "FakeModel":
            return self

    class FakeQwenFactory:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> FakeModel:
            model_calls.append((model_id, kwargs))
            return FakeModel()

    class FakeAutoProcessor:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> object:
            processor_calls.append((model_id, kwargs))
            return object()

    transformers_module = ModuleType("transformers")
    transformers_module.Qwen3VLForConditionalGeneration = FakeQwenFactory
    transformers_module.AutoProcessor = FakeAutoProcessor
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "peft", _make_peft_module(adapter_calls))
    return transformers_module


def test_resolve_load_paths_base_only(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    model_path, adapter = resolve_load_paths(base, None)
    assert model_path == base.resolve()
    assert adapter is None


def test_resolve_load_paths_adapter_dir(tmp_path: Path) -> None:
    base = tmp_path / "base"
    lora = tmp_path / "best_lora"
    base.mkdir()
    lora.mkdir()
    (lora / "adapter_config.json").write_text("{}", encoding="utf-8")
    model_path, adapter = resolve_load_paths(base, lora)
    assert model_path == base.resolve()
    assert adapter == lora.resolve()


def test_resolve_load_paths_merged_dir(tmp_path: Path) -> None:
    base = tmp_path / "base"
    merged = tmp_path / "merged"
    base.mkdir()
    merged.mkdir()
    (merged / "config.json").write_text("{}", encoding="utf-8")
    (merged / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    model_path, adapter = resolve_load_paths(base, merged)
    assert model_path == merged.resolve()
    assert adapter is None


def test_resolve_load_paths_invalid(tmp_path: Path) -> None:
    base = tmp_path / "base"
    lora = tmp_path / "lora"
    base.mkdir()
    lora.mkdir()
    try:
        resolve_load_paths(base, lora)
    except ValueError as error:
        assert "neither a PEFT adapter dir nor a merged full-model dir" in str(error)
    else:
        raise AssertionError("expected ValueError for invalid lora dir")


def test_resolve_device_explicit(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    assert resolve_device("cuda:0") == "cuda:0"
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_auto_cpu(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    assert resolve_device(None) == "cpu"


def test_qwen3_lora_loading_calls(monkeypatch, tmp_path: Path) -> None:
    base = tmp_path / "base"
    lora = tmp_path / "best_lora"
    base.mkdir()
    lora.mkdir()
    (lora / "adapter_config.json").write_text("{}", encoding="utf-8")
    model_calls: list = []
    processor_calls: list = []
    adapter_calls: list = []
    _install_fake_modules(monkeypatch, model_calls, processor_calls, adapter_calls)

    from models.benchmark_vlm import BenchmarkVLM

    model = BenchmarkVLM(
        model_type="qwen3-vl-4b",
        model_path=base,
        lora_path=lora,
        dtype="bfloat16",
        device="cpu",
        local_files_only=True,
    )
    assert model.effective_model_path == base.resolve()
    assert model.adapter_path == lora.resolve()
    assert model_calls[0][0] == str(base.resolve())
    assert model_calls[0][1]["local_files_only"] is True
    assert adapter_calls == [str(lora.resolve())]
    assert processor_calls[0][0] == str(base.resolve())


def test_internvl_loading_calls(monkeypatch, tmp_path: Path) -> None:
    base = tmp_path / "internvl"
    base.mkdir()
    model_calls: list = []
    tokenizer_calls: list = []
    adapter_calls: list = []

    class FakeModel:
        def to(self, device: str) -> "FakeModel":
            return self

        def eval(self) -> "FakeModel":
            return self

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> FakeModel:
            model_calls.append((model_id, kwargs))
            return FakeModel()

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> object:
            tokenizer_calls.append((model_id, kwargs))
            return object()

    transformers_module = ModuleType("transformers")
    transformers_module.AutoModel = FakeAutoModel
    transformers_module.AutoTokenizer = FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "peft", _make_peft_module(adapter_calls))

    from models.benchmark_vlm import BenchmarkVLM

    model = BenchmarkVLM(
        model_type="internvl3.5-8b",
        model_path=base,
        dtype="bfloat16",
        device="cpu",
        local_files_only=True,
    )
    assert model.effective_model_path == base.resolve()
    assert model_calls[0][0] == str(base.resolve())
    assert model_calls[0][1]["trust_remote_code"] is True
    assert tokenizer_calls[0][1]["trust_remote_code"] is True
    assert adapter_calls == []


def test_internvl_prompt_single_and_pair() -> None:
    image = Image.new("RGB", (16, 16))
    single = CanonicalSample(
        id="s1",
        task_type="vqa",
        images=[image],
        prompt="Answer the question.",
    )
    assert _internvl_prompt(single) == "<image>\nAnswer the question."
    pair = CanonicalSample(
        id="s2",
        task_type="change_caption",
        images=[image, image],
        prompt="Describe the change.",
    )
    prompt = _internvl_prompt(pair)
    assert prompt.startswith("Image-1 (before): <image>")
    assert "Image-2 (after): <image>" in prompt
    assert prompt.endswith("Describe the change.")


def test_internvl_grounding_boxes() -> None:
    boxes, status = _internvl_grounding_boxes([[10.0, 20.0, 30.0, 40.0]])
    assert boxes == [[10.0, 20.0, 30.0, 40.0]]
    assert status == "converted_prompt_native"
    boxes, _ = _internvl_grounding_boxes([[0.1, 0.2, 0.3, 0.4]])
    assert boxes == [[10.0, 20.0, 30.0, 40.0]]
    boxes, status = _internvl_grounding_boxes([])
    assert boxes == []
    assert status == "missing_box"
