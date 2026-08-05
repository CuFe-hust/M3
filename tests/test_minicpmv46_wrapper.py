"""Test MiniCPM-V-4.6 public settings. / 测试 MiniCPM-V-4.6 公开配置。"""
import pytest
from models.minicpmv46 import MiniCPMV46Baseline, MiniCPMV46Settings

@pytest.mark.parametrize("kwargs", [{"downsample_mode": "8x"}, {"max_slice_nums": 0}])
def test_invalid_slice_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        MiniCPMV46Baseline(MiniCPMV46Settings(**kwargs))

def test_runtime_load_is_lazy(monkeypatch):
    monkeypatch.setattr(MiniCPMV46Baseline, "_load", lambda self: (object(), object()))
    model = MiniCPMV46Baseline(MiniCPMV46Settings(local_files_only=True))
    assert model.settings.model_id == "openbmb/MiniCPM-V-4.6"
