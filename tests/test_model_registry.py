"""Test explicit model routing. / 测试显式模型路由。"""
import pytest
import models.registry as registry

@pytest.mark.parametrize("model_type,class_name", [("qwen3vl", "Qwen3VLBaseline"), ("qwen35_4b", "Qwen35FourBBaseline"), ("qwen35_9b", "Qwen35NineBBaseline"), ("internvl35", "InternVL35Baseline"), ("minicpmv46", "MiniCPMV46Baseline"), ("ovis25", "Ovis25Baseline")])
def test_explicit_routes(monkeypatch, model_type, class_name):
    target = getattr(registry, class_name)
    monkeypatch.setattr(registry, class_name, lambda settings: (target, settings))
    loaded, settings = registry.load_model_from_config({"type": model_type})
    assert loaded is target
    assert settings.model_id

def test_qwen35_public_classes_and_defaults_are_distinct():
    assert registry.Qwen35FourBBaseline is not registry.Qwen35NineBBaseline
    assert registry.Qwen35FourBSettings().model_id == "Qwen/Qwen3.5-4B"
    assert registry.Qwen35NineBSettings().model_id == "Qwen/Qwen3.5-9B"

@pytest.mark.parametrize("model_id,expected", [("Qwen/Qwen3.5-4B", "Qwen35FourBBaseline"), ("Qwen/Qwen3.5-9B", "Qwen35NineBBaseline"), ("/local/Qwen3.5-9B", "Qwen3VLBaseline")])
def test_legacy_routes_are_deliberately_limited(monkeypatch, model_id, expected):
    monkeypatch.setattr(registry, expected, lambda settings: expected)
    assert registry.load_model_from_config({"id": model_id}) == expected

def test_unknown_type_lists_supported_values():
    with pytest.raises(ValueError, match="qwen35_9b.*internvl35"):
        registry.load_model_from_config({"type": "qwen"})

def test_string_boolean_is_rejected():
    with pytest.raises(TypeError, match="local_files_only"):
        registry.load_model_from_config({"type": "qwen3vl", "local_files_only": "false"})
