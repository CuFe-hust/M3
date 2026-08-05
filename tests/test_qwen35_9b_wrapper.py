"""Test the dedicated Qwen3.5-9B wrapper. / 测试独立 Qwen3.5-9B 封装。"""
import pytest
import models.qwen35_9b as wrapper

def test_nine_b_runtime_is_explicit(monkeypatch):
    captured = []
    monkeypatch.setattr(wrapper, "load_qwen35_runtime", lambda config: captured.append(config) or (object(), object()))
    model = wrapper.Qwen35NineBBaseline(wrapper.Qwen35NineBSettings(local_files_only=True))
    assert captured[0].expected_variant == "9b"
    assert captured[0].model_id == "Qwen/Qwen3.5-9B"
    assert model.settings.local_files_only is True

def test_nine_b_rejects_official_four_b():
    from models.qwen35_common import Qwen35RuntimeConfig, load_qwen35_runtime
    with pytest.raises(ValueError, match="9B wrapper"):
        load_qwen35_runtime(Qwen35RuntimeConfig("Qwen/Qwen3.5-4B", "9b", "auto", "auto", 1, None, None, True, 1000.0))
