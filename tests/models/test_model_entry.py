"""Contract tests for the unified model entry.

统一模型入口测试：注册/创建/列举、未知模型显式报错、惰性 builder
（import models.entry 不加载 transformers/torch）、qwen3_5 与 baseline
builder 可用且不复制 Agent 逻辑。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

from models.entry import create_model, list_models, register
from models.settings import QwenSettings


class _DummySettings(BaseModel):
    model: str = "fake"
    max_tokens: int = 8


def test_register_and_list_models() -> None:
    names = list_models()
    assert "qwen_transformers" in names
    assert "qwen3_vl_baseline" in names
    assert "qwen3_5_transformers" in names
    assert len(names) == len(set(names))


def test_register_duplicate_name_raises() -> None:
    with pytest.raises(ValueError, match="already registered"):

        @register("qwen_transformers")
        def _duplicate(**kwargs):  # pragma: no cover
            return None


def test_create_model_unknown_name_raises() -> None:
    with pytest.raises(KeyError, match="Unknown model entry"):
        create_model("no-such-model")


def test_import_entry_does_not_load_heavy_libraries() -> None:
    """Importing models.entry must not load transformers or torch.
    导入 models.entry 不得加载 transformers 或 torch。"""
    for heavy in ("transformers", "torch"):
        assert heavy not in sys.modules


def test_builders_are_lazy() -> None:
    """Builders must import concrete models lazily; constructing a client with
    injected fake model/processor must not touch transformers.
    builder 必须惰性导入具体模型；注入 fake 构造客户端不触碰 transformers。"""
    from models.qwen_transformers import QwenTransformersClient

    class _FakeProcessor:
        def apply_chat_template(self, *args, **kwargs):
            return "prompt"

        def __call__(self, *args, **kwargs):
            class _Inputs:
                shape = (1, 3)

                def to(self, device):
                    return self

            return {"input_ids": _Inputs()}

        def batch_decode(self, *args, **kwargs):
            return ["{}"]

    class _FakeModel:
        device = "cpu"

        def generate(self, **kwargs):
            class _Out:
                shape = (1, 3)
            return [_Out()]

    client = create_model(
        "qwen_transformers",
        settings=QwenSettings(model="fake", max_tokens=8),
        model=_FakeModel(),
        processor=_FakeProcessor(),
    )
    assert isinstance(client, QwenTransformersClient)
    assert "transformers" not in sys.modules


def test_qwen35_builder_returns_shared_client() -> None:
    from models.qwen_transformers import QwenTransformersClient

    client = create_model(
        "qwen3_5_transformers",
        settings=QwenSettings(model="fake"),
        model=object(),
        processor=object(),
    )
    assert isinstance(client, QwenTransformersClient)


def test_baseline_builder_with_injected_components() -> None:
    from models.qwen3_vl.baseline import Qwen3VLBaseline

    baseline = create_model(
        "qwen3_vl_baseline",
        settings=_DummySettings(),
        model=object(),
        processor=object(),
    )
    assert isinstance(baseline, Qwen3VLBaseline)


def test_baseline_and_client_contain_no_agent_logic(tmp_path: Path) -> None:
    """Baseline and client modules must not reference agents or router.
    基线与客户端模块不得引用 agents 或 router。"""
    for relative in (
        "models/qwen3_vl/baseline.py",
        "models/qwen_transformers.py",
        "models/entry.py",
    ):
        source = (Path(__file__).resolve().parents[2] / relative).read_text(encoding="utf-8")
        assert "spacers_agent" not in source, relative
        assert "Agent" not in source or "Agent" in source and "AgentName" not in source, relative
