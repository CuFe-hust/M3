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
    assert settings.models.segformer_isaid.allow_download is False
    assert settings.models.segformer_isaid.model_path == Path(
        "models/segformer_mitb2_isaid"
    )
    assert settings.models.segformer_oem.model_path == Path(
        "models/segformer_mitb2_oem"
    )
    assert settings.models.segformer_isaid.logical_model_id == (
        "SegFormer-MiT-B2:iSAID:local"
    )
    assert settings.models.segformer_isaid.classes_filename == "classes.json"
    assert settings.models.segformer_oem.classes_filename is None
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


def test_segformer_yaml_round_trip_keeps_physical_and_logical_identity_separate(
    tmp_path: Path,
) -> None:
    path = _yaml_path(
        tmp_path,
        """
models:
  segformer_isaid:
    model_path: C:\\checkpoints\\segformer-isaid
    logical_model_id: segformer-mitb2-isaid-test
    classes_filename: missing-classes.json
""",
    )
    settings = load_settings(path, environ={})
    assert settings.models.segformer_isaid.model_path == Path(
        r"C:\checkpoints\segformer-isaid"
    )
    assert settings.models.segformer_isaid.logical_model_id == (
        "segformer-mitb2-isaid-test"
    )
    snapshot = settings.safe_snapshot()
    assert snapshot["models"]["segformer_isaid"]["model_path"] == (
        "C:/checkpoints/segformer-isaid"
    )
    assert snapshot["models"]["segformer_isaid"]["logical_model_id"] == (
        "segformer-mitb2-isaid-test"
    )
    assert AppSettings.model_validate(snapshot).safe_snapshot() == snapshot


def test_load_settings_env_overrides(tmp_path: Path) -> None:
    path = _yaml_path(tmp_path, "models:\n  qwen:\n    model: from-yaml\n")
    environ = {
        "QWEN_MODEL": "from-env",
        "DATASET_ROOT": r"C:\data\remote-sensing",
        "OUTPUT_ROOT": r"D:\runs\m3",
        "DEEPSEEK_MODEL": "deepseek-env",
        "SEGFORMER_ISAID_MODEL": r"D:\models\isaid",
        "SEGFORMER_OEM_MODEL": r"D:\models\oem",
    }
    settings = load_settings(path, environ=environ)
    assert settings.models.qwen.model == "from-env"  # env wins / 环境优先
    assert settings.paths.dataset_root == Path(r"C:\data\remote-sensing")
    assert settings.runs.root == Path(r"D:\runs\m3")
    assert settings.models.deepseek.model == "deepseek-env"
    assert settings.models.segformer_isaid.model_path == Path(r"D:\models\isaid")
    assert settings.models.segformer_oem.model_path == Path(r"D:\models\oem")


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


def test_snapshot_preserves_host_paths_with_forward_slashes() -> None:
    """The snapshot is reproduction-oriented: configured host paths are
    preserved verbatim (separator-normalized), never rewritten to portable
    logical paths. 快照面向复现：配置的主机路径原样保留（仅归一化分隔符），
    绝不改写成可移植逻辑路径。"""
    from application.settings import PathSettings, RunSettings

    settings = AppSettings(
        runs=RunSettings(root=Path(r"C:\Users\me\runs")),
        paths=PathSettings(dataset_root=Path(r"D:\data")),
    )
    payload = settings.to_config_payload()
    assert payload["runs"]["root"] == "C:/Users/me/runs"
    assert payload["paths"]["dataset_root"] == "D:/data"
    # The documentation contract never claims machine portability.
    # 文档契约不再声称机器可移植。
    docstring = AppSettings.to_config_payload.__doc__ or ""
    assert "machine-portable" not in docstring
    assert "host" in docstring
