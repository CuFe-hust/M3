"""Test InternVL3.5 public settings. / 测试 InternVL3.5 公开配置。"""
import pytest
from models.internvl35 import InternVL35Baseline, InternVL35Settings

def test_invalid_tile_configuration_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        InternVL35Baseline(InternVL35Settings(max_num_tiles=0))

def test_runtime_load_is_lazy(monkeypatch):
    monkeypatch.setattr(InternVL35Baseline, "_load", lambda self: (object(), object()))
    model = InternVL35Baseline(InternVL35Settings(local_files_only=True))
    assert model.settings.model_id == "OpenGVLab/InternVL3_5-8B"
