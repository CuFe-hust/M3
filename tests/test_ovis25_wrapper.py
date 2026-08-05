"""Test Ovis2.5 public settings. / 测试 Ovis2.5 公开配置。"""
import pytest
from models.ovis25 import Ovis25Baseline, Ovis25Settings

@pytest.mark.parametrize("kwargs", [{"min_pixels": 0}, {"min_pixels": 10, "max_pixels": 9}])
def test_invalid_pixel_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        Ovis25Baseline(Ovis25Settings(**kwargs))

def test_runtime_load_is_lazy(monkeypatch):
    monkeypatch.setattr(Ovis25Baseline, "_load", lambda self: object())
    model = Ovis25Baseline(Ovis25Settings(local_files_only=True))
    assert model.settings.model_id == "ATH-MaaS/Ovis2.5-2B"
