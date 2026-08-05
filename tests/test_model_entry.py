"""Tests for the unified model entry point.
统一模型入口的测试。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from models.entry import create_model, list_models, register
from models.qwen_transformers import QwenTransformersClient
from spacers_agent.settings import QwenSettings


class _FakeModel:
    """Fake model object that never touches Transformers.
    不触碰 Transformers 的假模型对象。
    """


class _FakeProcessor:
    """Fake processor object that never touches Transformers.
    不触碰 Transformers 的假处理器对象。
    """


def test_qwen_transformers_entry_builds_client_without_loading() -> None:
    client = create_model(
        "qwen_transformers",
        settings=QwenSettings(model="local", max_tokens=32),
        model=_FakeModel(),
        processor=_FakeProcessor(),
    )

    assert isinstance(client, QwenTransformersClient)
    assert isinstance(client.model, _FakeModel)
    assert isinstance(client.processor, _FakeProcessor)


def test_qwen35_transformers_entry_builds_shared_client() -> None:
    client = create_model(
        "qwen3_5_transformers",
        settings=QwenSettings(model="local-qwen35", max_tokens=32),
        model=_FakeModel(),
        processor=_FakeProcessor(),
    )

    assert isinstance(client, QwenTransformersClient)
    assert isinstance(client.model, _FakeModel)


def test_qwen3_vl_baseline_entry_builds_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[object] = []

    class FakeBaseline:
        def __init__(self, settings: object) -> None:
            captured.append(settings)

    monkeypatch.setattr("models.qwen3_vl.baseline.Qwen3VLBaseline", FakeBaseline)

    result = create_model("qwen3_vl_baseline", settings=object())

    assert isinstance(result, FakeBaseline)
    assert len(captured) == 1


def test_unknown_model_name_raises_key_error_with_registered_names() -> None:
    with pytest.raises(KeyError) as exc_info:
        create_model("not-a-model")

    assert "Unknown model entry" in str(exc_info.value)
    assert "qwen_transformers" in str(exc_info.value)


def test_list_models_contains_main_flow_entries() -> None:
    names = list_models()

    assert "qwen_transformers" in names
    assert "qwen3_vl_baseline" in names
    assert "qwen3_5_transformers" in names


def test_register_rejects_duplicate_names() -> None:
    def duplicate_builder(**_: object) -> object:
        return object()

    with pytest.raises(ValueError):
        register("qwen_transformers")(duplicate_builder)


def test_import_models_does_not_enable_hf_offline() -> None:
    """Importing the models package must not set HF offline env vars.
    导入 models 包不得设置 HF 离线环境变量。
    """

    repo_root = Path(__file__).resolve().parents[1]
    code = (
        "import importlib, os; "
        "importlib.import_module('models'); importlib.import_module('models.entry'); "
        "assert 'HF_HUB_OFFLINE' not in os.environ; "
        "assert 'TRANSFORMERS_OFFLINE' not in os.environ"
    )
    env = dict(os.environ)
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_qwen_vllm_client_module_is_removed() -> None:
    with pytest.raises(ImportError):
        import spacers_agent.clients.qwen_vllm  # noqa: F401
