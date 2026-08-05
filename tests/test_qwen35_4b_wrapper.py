"""Test the dedicated Qwen3.5-4B wrapper. / 测试独立 Qwen3.5-4B 封装。"""
import pytest
import models.qwen35_4b as wrapper

def test_four_b_runtime_is_explicit(monkeypatch):
    captured = []
    monkeypatch.setattr(wrapper, "load_qwen35_runtime", lambda config: captured.append(config) or (object(), object()))
    model = wrapper.Qwen35FourBBaseline(wrapper.Qwen35FourBSettings(local_files_only=True))
    assert captured[0].expected_variant == "4b"
    assert captured[0].model_id == "Qwen/Qwen3.5-4B"
    assert model.settings.local_files_only is True

def test_four_b_rejects_official_nine_b():
    from models.qwen35_common import Qwen35RuntimeConfig, load_qwen35_runtime
    with pytest.raises(ValueError, match="4B wrapper"):
        load_qwen35_runtime(Qwen35RuntimeConfig("Qwen/Qwen3.5-9B", "4b", "auto", "auto", 1, None, None, True, 1000.0))
