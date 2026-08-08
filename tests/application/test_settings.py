"""Contract tests for application settings: YAML loading, environment
overrides, Windows paths, and secret exclusion.

应用配置契约测试：YAML 加载、环境变量覆盖、Windows 路径与密钥排除。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from application.settings import AppSettings, load_settings


def _yaml_path(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "settings.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_default_settings() -> None:
    settings = AppSettings()
    assert settings.runs.root == Path("outputs/runs")
    assert settings.paths.dataset_root == Path("dataset")
    assert settings.router.confidence_threshold == 0.7
    assert settings.models.qwen.allow_download is False
    assert settings.backend.yolo.enabled is False
    assert settings.agents.counting.default_backend == "auto"


def test_load_settings_from_yaml(tmp_path: Path) -> None:
    path = _yaml_path(
        tmp_path,
        """
models:
  qwen:
    model: qwen3-vl-4b-instruct
runs:
  root: outputs/custom
router:
  confidence_threshold: 0.8
""",
    )
    settings = load_settings(path, environ={})
    assert settings.models.qwen.model == "qwen3-vl-4b-instruct"
    assert settings.runs.root == Path("outputs/custom")
    assert settings.router.confidence_threshold == 0.8


def test_load_settings_env_overrides(tmp_path: Path) -> None:
    path = _yaml_path(tmp_path, "models:\n  qwen:\n    model: from-yaml\n")
    environ = {
        "QWEN_MODEL": "from-env",
        "DATASET_ROOT": r"C:\data\remote-sensing",
        "OUTPUT_ROOT": r"D:\runs\m3",
        "DEEPSEEK_MODEL": "deepseek-env",
    }
    settings = load_settings(path, environ=environ)
    assert settings.models.qwen.model == "from-env"  # env wins / 环境优先
    assert settings.paths.dataset_root == Path(r"C:\data\remote-sensing")
    assert settings.runs.root == Path(r"D:\runs\m3")
    assert settings.models.deepseek.model == "deepseek-env"


def test_windows_paths_are_coerced() -> None:
    settings = load_settings(
        None,
        environ={"DATASET_ROOT": "C:\\Users\\me\\datasets", "OUTPUT_ROOT": "C:\\runs"},
    )
    assert isinstance(settings.paths.dataset_root, Path)
    # The machine-portable snapshot always uses forward slashes, on every
    # platform (as_posix alone keeps backslashes on POSIX hosts).
    # 机器可移植快照在所有平台统一正斜杠（仅 as_posix 在 POSIX 主机会保留
    # 反斜杠）。
    payload = settings.to_config_payload()
    assert payload["paths"]["dataset_root"] == "C:/Users/me/datasets"
    assert payload["runs"]["root"] == "C:/runs"
    assert "\\" not in payload["paths"]["dataset_root"]


def test_invalid_yaml_fails_stable(tmp_path: Path) -> None:
    path = _yaml_path(tmp_path, "models: [unclosed")
    with pytest.raises(ValueError, match="settings YAML"):
        load_settings(path, environ={})


def test_safe_snapshot_is_json_safe_and_secret_free(tmp_path: Path) -> None:
    """Even with a secret-looking environment, snapshots never contain secret
    VALUES: settings only carry env var NAMES (api_key_env), never values.
    即使环境里存在疑似密钥值，快照也绝不包含密钥值：配置只携带环境变量名
    （api_key_env），绝不携带值。"""
    environ = {
        "DEEPSEEK_API_KEY": "sk-super-secret-value",
        "QWEN_MODEL": "qwen-model",
    }
    settings = load_settings(None, environ=environ)
    snapshot = settings.safe_snapshot()
    serialized = json.dumps(snapshot)
    assert "sk-super-secret-value" not in serialized
    assert "sk-" not in serialized
    # The env var NAME is declarative metadata, never a secret value.
    # 环境变量名是声明性元数据，绝非密钥值。
    assert snapshot["models"]["deepseek"]["api_key_env"] == "DEEPSEEK_API_KEY"
    # Path objects are serialized as POSIX strings. / Path 对象序列化为 POSIX 字符串。
    assert isinstance(snapshot["runs"]["root"], str)
    assert "\\" not in snapshot["runs"]["root"]
    # repr never leaks secret values either. / repr 同样不泄漏密钥值。
    assert "sk-" not in repr(settings)


def test_to_config_payload_matches_safe_snapshot() -> None:
    settings = AppSettings()
    assert settings.to_config_payload() == settings.safe_snapshot()
