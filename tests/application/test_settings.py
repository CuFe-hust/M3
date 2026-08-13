"""Contract tests for application settings: YAML loading, environment
overrides, Windows paths, and secret exclusion.

应用配置契约测试：YAML 加载、环境变量覆盖、Windows 路径与密钥排除。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from application.settings import AppSettings, load_settings


REPO_ROOT = Path(__file__).resolve().parents[2]


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
    assert settings.models.segformer_experts == {}
    assert settings.backend.yolo.enabled is False
    assert settings.backend.yolo.detectors == []
    assert settings.agents.counting.default_backend == "auto"
    assert settings.counting.fallback_on_backend_unavailable is True
    assert settings.counting.verify_empty_detection is True
    assert settings.counting.verify_empty_semantic is False


def test_local_config_declares_detector_inventory() -> None:
    settings = load_settings(REPO_ROOT / "configs" / "local.yaml", environ={})

    assert settings.backend.yolo.enabled is True
    assert [item.name for item in settings.backend.yolo.detectors] == [
        "detector_obb_csl_001"
    ]
    assert settings.backend.yolo.detectors[0].enabled is True


def test_legacy_yolo_execution_policy_migrates_only_at_settings_boundary(
    tmp_path: Path,
) -> None:
    path = _yaml_path(
        tmp_path,
        """
backend:
  trust_empty_detection: true
  yolo:
    fallback_to_qwen_on_unavailable: false
    fallback_to_qwen_on_error: false
    verify_empty_with_qwen: false
""",
    )

    settings = load_settings(path, environ={})

    assert settings.counting.fallback_on_backend_unavailable is False
    assert settings.counting.fallback_on_backend_error is False
    assert settings.counting.verify_empty_detection is False
    assert settings.counting.trust_empty_detection is True
    assert "fallback_to_qwen" not in json.dumps(settings.safe_snapshot())


def test_legacy_and_generic_execution_policy_conflict_is_rejected(
    tmp_path: Path,
) -> None:
    path = _yaml_path(
        tmp_path,
        """
counting:
  fallback_on_backend_error: true
backend:
  yolo:
    fallback_to_qwen_on_error: false
""",
    )

    with pytest.raises(ValueError, match="both legacy key"):
        load_settings(path, environ={})


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


def test_additional_segformer_profile_round_trips_with_safe_paths(
    tmp_path: Path,
) -> None:
    path = _yaml_path(
        tmp_path,
        """
models:
  segformer_experts:
    segmenter_extra_001:
      model_path: D:\\models\\extra
      logical_model_id: segformer-extra-local
      classes_filename: labels.json
""",
    )
    settings = load_settings(path, environ={})
    profile = settings.models.segformer_profile(
        backend_name="segmenter_extra_001",
        logical_model_id="segformer-extra-local",
    )
    assert profile.model_path == Path(r"D:\models\extra")
    snapshot = settings.safe_snapshot()
    assert snapshot["models"]["segformer_experts"]["segmenter_extra_001"][
        "model_path"
    ] == "D:/models/extra"
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


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("legacy.yaml", {"enabled": False}),
        (
            "low_semantic.yaml",
            {"enabled": True, "feature_weight": 0.0, "semantic_weight": 0.5},
        ),
        (
            "low_feature.yaml",
            {"enabled": True, "feature_weight": 2 / 3, "semantic_weight": 0.0},
        ),
        (
            "three_source.yaml",
            {"enabled": True, "feature_weight": 0.5, "semantic_weight": 0.25},
        ),
        ("pif_robust.yaml", {"enabled": True, "threshold_mode": "pif_robust"}),
        ("local_match_r0.yaml", {"enabled": True, "radius": 0}),
        ("local_match_r1.yaml", {"enabled": True, "radius": 1}),
    ],
)
def test_change_ablation_presets_are_valid_partial_app_settings(
    filename: str,
    expected: dict[str, object],
) -> None:
    settings = load_settings(
        REPO_ROOT / "configs" / "change_ablations" / filename,
        environ={},
    )

    assert settings.agents.change.semantic.enabled is expected["enabled"]
    if "feature_weight" in expected:
        assert settings.agents.change.proposals.fusion_feature_weight == pytest.approx(
            expected["feature_weight"]
        )
    if "semantic_weight" in expected:
        assert settings.agents.change.proposals.fusion_semantic_weight == pytest.approx(
            expected["semantic_weight"]
        )
    if "threshold_mode" in expected:
        assert (
            settings.agents.change.proposals.threshold_mode
            == expected["threshold_mode"]
        )
    if "radius" in expected:
        assert settings.agents.change.semantic.local_match_radius == expected["radius"]


# ── visual planning group (C7, 14A2) / 视觉规划配置组 ─────────────────────


def test_visual_planning_defaults_to_frozen_disabled_state() -> None:
    """The whole group must default to the frozen disabled state: flag off,
    planner sane, no detector/segmenter policies, strict failure policy.
    整个配置组必须默认为冻结禁用状态：flag off、planner 合理默认、无
    detector/segmenter 策略、严格失败策略。"""
    settings = AppSettings()
    assert settings.visual_planning.enabled is False
    assert settings.visual_planning.planner.prompt_version == "v1"
    # The declared version must bind the same evidence catalog asset
    # (agents/evidence_catalog.json); the composition root verifies this.
    # 声明版本必须绑定同一证据目录资产（agents/evidence_catalog.json）；
    # 组合根校验该绑定。
    assert (
        settings.visual_planning.planner.catalog_version
        == "first-qwen-evidence-catalog-v2"
    )
    assert settings.visual_planning.planner.max_rois == 3
    assert settings.visual_planning.planner.halo_ratio == pytest.approx(0.10)
    assert settings.visual_planning.planner.confidence_threshold == pytest.approx(0.70)
    assert settings.visual_planning.planner.failure.on_low_confidence == "fail"
    assert settings.visual_planning.planner.failure.on_planner_error == "fail"
    assert settings.visual_planning.detectors == {}
    assert settings.visual_planning.segmenters == {}
    assert settings.visual_planning.roi_partial_failure == "continue"


def test_visual_planning_detector_policy_accepts_calibrated_values() -> None:
    """Range validation applies once calibrated values are provided.
    一旦提供已校准值即进行范围校验。"""
    settings = AppSettings(
        visual_planning={
            "detectors": {
                "small_vehicle": {
                    "confidence_threshold": 0.5,
                    "nms_iou_threshold": 0.45,
                    "max_detections": 5,
                }
            }
        }
    )
    detector = settings.visual_planning.detectors["small_vehicle"]
    assert detector.confidence_threshold == 0.5
    assert detector.nms_iou_threshold == 0.45
    assert detector.max_detections == 5


def test_visual_planning_detector_out_of_range_rejected_when_set() -> None:
    """Out-of-range calibrated values fail at parse time, never silently.
    越界的已校准值在解析时失败，绝不静默接受。"""
    with pytest.raises(ValueError):
        AppSettings(
            visual_planning={
                "detectors": {"small_vehicle": {"confidence_threshold": 1.5}}
            }
        )
    with pytest.raises(ValueError):
        AppSettings(
            visual_planning={
                "detectors": {"small_vehicle": {"max_detections": 0}}
            }
        )
    with pytest.raises(ValueError):
        AppSettings(
            visual_planning={
                "detectors": {"small_vehicle": {"nms_iou_threshold": -0.1}}
            }
        )


def test_visual_planning_unset_detector_values_remain_none() -> None:
    """Uncalibrated means capability off: every value stays None and no
    arbitrary default is invented (approved gate). 未校准即能力关闭：所有值
    保持 None，绝不杜撰任意默认值（已批准门禁）。"""
    detector = AppSettings(visual_planning={}).visual_planning.detectors.get(
        "small_vehicle"
    )
    assert detector is None


def test_visual_planning_segmenter_requires_class_map_when_enabled() -> None:
    """An enabled segmenter must declare an approved class map version.
    启用的分割器必须声明已批准 class map 版本。"""
    with pytest.raises(ValueError):
        AppSettings(
            visual_planning={"segmenters": {"building": {"enabled": True}}}
        )
    settings = AppSettings(
        visual_planning={
            "segmenters": {"building": {"enabled": True, "class_map_version": "isaid-v2"}}
        }
    )
    assert settings.visual_planning.segmenters["building"].class_map_version == "isaid-v2"


def test_visual_planning_unknown_fields_rejected() -> None:
    """The strict extra=forbid semantics must not weaken for the new group.
    新配置组不得弱化严格 extra=forbid 语义。"""
    with pytest.raises(ValueError):
        AppSettings(visual_planning={"unknown_field": 1})
    with pytest.raises(ValueError):
        AppSettings(
            visual_planning={"roi_partial_failure": "abort"}
        )


def test_visual_planning_enabled_does_not_imply_calibration() -> None:
    """Even with the feature flag on, empty detector/segmenter policies keep
    every capability disabled. 即使特性 flag 开启，空的 detector/segmenter
    策略仍保持全部能力禁用。"""
    settings = AppSettings(visual_planning={"enabled": True})
    assert settings.visual_planning.enabled is True
    assert settings.visual_planning.detectors == {}
    assert settings.visual_planning.segmenters == {}
