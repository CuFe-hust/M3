"""Contract tests for application settings: YAML loading, environment
overrides, Windows paths, and secret exclusion.

应用配置契约测试：YAML 加载、环境变量覆盖、Windows 路径与密钥排除。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from application.settings import AppSettings, load_dotenv, load_settings


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
    assert settings.models.qwen_adapters == {}
    assert set(settings.models.qwen_adapter_bindings.as_dict().values()) == {"base"}
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
    assert settings.models.segformer_oem.classes_filename == "classes.json"
    assert settings.models.segformer_experts == {}
    assert settings.backend.yolo.enabled is False
    assert settings.backend.yolo.detectors == []
    assert settings.agents.counting.default_backend == "auto"
    assert settings.counting.fallback_on_backend_unavailable is True
    assert settings.counting.verify_empty_detection is True
    assert settings.counting.verify_empty_semantic is False
    assert settings.visual_planning.planner.task_prompt_version == "v5"
    assert settings.visual_planning.planner.planning_mode == "visual-task-plan-v5"
    assert settings.visual_planning.planner.roi_coordinate_frame == (
        "normalized_0_999_top_left"
    )
    assert settings.visual_planning.planner.roi_quantum == 1024
    assert settings.visual_planning.planner.roi_materialization_policy == (
        "longest-side-ceil-quantum-center-clip"
    )


def test_visual_planner_rejects_unapproved_roi_quantum() -> None:
    with pytest.raises(ValueError, match="roi_quantum"):
        AppSettings.model_validate(
            {"visual_planning": {"planner": {"roi_quantum": 2048}}}
        )


def test_local_config_declares_detector_inventory() -> None:
    settings = load_settings(REPO_ROOT / "configs" / "local.yaml", environ={})

    assert settings.backend.yolo.enabled is True
    assert [item.name for item in settings.backend.yolo.detectors] == [
        "detector_obb_csl_001", "detector_obb_ultralytics_001"
    ]
    assert all(item.enabled for item in settings.backend.yolo.detectors)
    assert all(item.device == "cpu" for item in settings.backend.yolo.detectors)
    assert all(item.require_cuda is False for item in settings.backend.yolo.detectors)
    assert all(item.allow_cpu_fallback is False for item in settings.backend.yolo.detectors)
    vrsbench = settings.backend.yolo.detectors[1]
    assert vrsbench.task == "obb"
    assert vrsbench.model_id == "YOLO11m-OBB:VRSBench-QA1024:best"
    assert vrsbench.classes[0] == "plane"
    assert vrsbench.classes[-1] == "helipad"
    assert settings.models.qwen.model == "models/Qwen3.5-9B"
    assert settings.models.qwen.cache_model_id == "Qwen/Qwen3.5-9B:local"
    assert set(settings.visual_planning.detectors) == {"detector_obb_csl_001"}
    assert set(settings.visual_planning.segmenters) == {
        "segmenter_mitb2_001", "segmenter_oem_001"
    }
    assert settings.counting.max_selected_detector_experts == 2
    assert settings.counting.ensemble_singleton_high_confidence == 0.65
    assert settings.agents.change.semantic.max_experts == 2
    assert settings.agents.change.building_rescue.enabled is True
    assert settings.agents.change.learned_change.enabled is False
    adapter = settings.models.qwen_adapters["visual-planner-supplement"]
    assert adapter.logical_id == (
        "qwen35-9b-visual-planner-supplement-20260824"
    )
    assert set(settings.models.qwen_adapter_bindings.as_dict().values()) == {
        "visual-planner-supplement"
    }
    snapshot = settings.safe_snapshot()
    assert snapshot["models"]["qwen_adapters"]["visual-planner-supplement"][
        "path"
    ].endswith("/final_adapter")


def test_qwen_adapter_settings_fail_closed() -> None:
    base = {
        "path": "~/adapters/a",
        "logical_id": "adapter-a",
        "revision": "a" * 64,
    }
    with pytest.raises(ValueError, match="extra"):
        AppSettings.model_validate(
            {"models": {"qwen_adapters": {"a": {**base, "unknown": True}}}}
        )
    with pytest.raises(ValueError, match="local path"):
        AppSettings.model_validate(
            {
                "models": {
                    "qwen_adapters": {
                        "a": {**base, "logical_id": "/private/adapter-a"}
                    }
                }
            }
        )
    with pytest.raises(ValueError, match="SHA-256"):
        AppSettings.model_validate(
            {
                "models": {
                    "qwen_adapters": {"a": {**base, "revision": "latest"}}
                }
            }
        )
    with pytest.raises(ValueError, match="unknown adapter"):
        AppSettings.model_validate(
            {"models": {"qwen_adapter_bindings": {"planner": "missing"}}}
        )


def test_qwen_adapter_path_is_not_expanded_by_settings(tmp_path: Path) -> None:
    path = _yaml_path(
        tmp_path,
        """
models:
  qwen_adapters:
    adapter-a:
      path: ~/private/adapter-a
      logical_id: adapter-a-v1
      revision: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
""",
    )
    settings = load_settings(path, environ={})
    assert settings.models.qwen_adapters["adapter-a"].path == Path(
        "~/private/adapter-a"
    )


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


def test_qwen_model_override_clears_profile_cache_identity(tmp_path: Path) -> None:
    path = _yaml_path(
        tmp_path,
        "models:\n  qwen:\n    model: /profile/model\n    cache_model_id: profile-id\n",
    )
    settings = load_settings(path, environ={"QWEN_MODEL": "qwen3-vl-4b-instruct"})
    assert settings.models.qwen.model == "qwen3-vl-4b-instruct"
    assert settings.models.qwen.cache_model_id is None


def test_qwen_local_model_override_requires_explicit_cache_identity(tmp_path: Path) -> None:
    path = _yaml_path(
        tmp_path,
        "models:\n  qwen:\n    model: models/qwen3_vl_8b/merged\n    cache_model_id: profile-id\n",
    )
    with pytest.raises(ValueError, match="cache_model_id"):
        load_settings(path, environ={"QWEN_MODEL": "/tmp/model"})


def test_qwen_local_model_override_accepts_explicit_cache_identity(tmp_path: Path) -> None:
    path = _yaml_path(tmp_path, "models:\n  qwen:\n    model: qwen3-vl-4b-instruct\n")
    settings = load_settings(
        path,
        environ={
            "QWEN_MODEL": "/tmp/model",
            "QWEN_CACHE_MODEL_ID": "test-local-qwen-v1",
        },
    )
    assert settings.models.qwen.model == "/tmp/model"
    assert settings.models.qwen.cache_model_id == "test-local-qwen-v1"


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


def test_load_dotenv_fills_missing_environment_without_overriding_existing_values(
    tmp_path: Path,
) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "# comment\n"
        "export QWEN_MODEL=from-dotenv\n"
        "DEEPSEEK_BASE_URL=\"https://example.test/v1\"\n"
        "DEEPSEEK_API_KEY='secret-value'\n"
        "IGNORED_LINE\n",
        encoding="utf-8",
    )
    environ = {"QWEN_MODEL": "from-process"}

    loaded = load_dotenv(dotenv, environ=environ)

    assert loaded == dotenv
    assert environ == {
        "QWEN_MODEL": "from-process",
        "DEEPSEEK_BASE_URL": "https://example.test/v1",
        "DEEPSEEK_API_KEY": "secret-value",
    }


def test_load_dotenv_secret_does_not_enter_settings_snapshot(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("DEEPSEEK_API_KEY=secret-value\n", encoding="utf-8")
    environ: dict[str, str] = {}

    load_dotenv(dotenv, environ=environ)
    settings = load_settings(environ=environ)

    assert environ["DEEPSEEK_API_KEY"] == "secret-value"
    assert "secret-value" not in json.dumps(settings.safe_snapshot())


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
    # The v5 planner identity legitimately contains the substring "task-";
    # only the secret value itself is forbidden here.
    # v5 规划器身份合法包含 "task-" 子串；这里只禁止实际 secret value。
    assert "sk-super" not in serialized
    # The env var NAME is declarative metadata, never a secret value.
    # 环境变量名是声明性元数据，绝非密钥值。
    assert snapshot["models"]["deepseek"]["api_key_env"] == "DEEPSEEK_API_KEY"
    # Path objects are serialized as POSIX strings. / Path 对象序列化为 POSIX 字符串。
    assert isinstance(snapshot["runs"]["root"], str)
    assert "\\" not in snapshot["runs"]["root"]
    # repr never leaks secret values either. / repr 同样不泄漏密钥值。
    assert "sk-super" not in repr(settings)


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
        ("legacy.yaml", {"enabled": False, "registration": False}),
        ("registration_only.yaml", {"enabled": False, "registration": True}),
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
        (
            "registered_three_source.yaml",
            {"enabled": True, "registration": True, "feature_weight": 0.5, "semantic_weight": 0.25},
        ),
        (
            "multiscale_registered.yaml",
            {"enabled": True, "registration": True, "feature_stages": (1, 2, 3)},
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
    if "registration" in expected:
        assert settings.agents.change.registration.enabled is expected["registration"]
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
    if "feature_stages" in expected:
        assert settings.agents.change.semantic.feature_stages == expected["feature_stages"]


# ── visual planning group (C7, 14A2) / 视觉规划配置组 ─────────────────────


def test_visual_planning_defaults_to_v5_planner_state() -> None:
    """Fresh execution defaults to the canonical v5 planner configuration.
    新鲜执行默认使用规范 v5 规划器配置。"""
    settings = AppSettings()
    planner = settings.visual_planning.planner
    assert not hasattr(settings.visual_planning, "enabled")
    assert planner.planning_mode == "visual-task-plan-v5"
    assert planner.task_prompt_version == "v5"
    assert (
        planner.catalog_version
        == "visual-evidence-catalog-v4"
    )
    assert not hasattr(planner, "confidence_threshold")
    assert planner.preview_max_side == 1080
    assert planner.roi_coordinate_frame == "normalized_0_999_top_left"
    assert planner.roi_quantum == 1024
    assert planner.roi_materialization_policy == "longest-side-ceil-quantum-center-clip"
    assert settings.visual_planning.detectors == {}
    assert settings.visual_planning.segmenters == {}


def test_evidence_preprocessing_defaults_to_fresh_v2_identity() -> None:
    """The preprocessing identity is typed and frozen, never free dict
    fields; fresh runs default to the combined yolo-v1-segformer-pad-v1
    identity with backend-specific versions. 预处理身份是类型化且冻结的，绝不
    放进自由 dict；新鲜运行默认使用 yolo-v1-segformer-pad-v1 组合身份与
    backend-specific 版本。"""
    preprocessing = AppSettings().visual_planning.preprocessing
    assert preprocessing.version == "yolo-v1-segformer-pad-v1"
    assert preprocessing.tile_size == 1024
    assert preprocessing.partition_policy == "greedy-row-major-no-overlap"
    assert preprocessing.remainder_resize == "stretch"
    assert preprocessing.rgb_interpolation == "lanczos"
    assert preprocessing.mask_inverse_interpolation == "nearest"
    assert preprocessing.max_tile_concurrency == 4
    assert preprocessing.yolo_version == "greedy-1024-stretch-v1"
    assert preprocessing.segformer_version == "pad-multiple-1024-resize-square-v1"
    assert preprocessing.segformer_padding_mode == "constant-black-right-bottom"
    assert preprocessing.segformer_rgb_interpolation == "lanczos"
    assert preprocessing.segformer_mask_inverse_interpolation == "nearest"


def test_evidence_preprocessing_accepts_explicit_legacy_v1_identity() -> None:
    """The legacy v1 combined identity stays expressible as an explicit
    configuration for historical interpretation; it never silently upgrades
    to the pad protocol. 旧 v1 组合身份仍可作为显式配置表达历史解释；绝不静默
    升级为 pad 协议。"""
    settings = AppSettings(
        visual_planning={"preprocessing": {"version": "greedy-1024-stretch-v1"}}
    )
    preprocessing = settings.visual_planning.preprocessing
    assert preprocessing.version == "greedy-1024-stretch-v1"
    assert preprocessing.segformer_version == "pad-multiple-1024-resize-square-v1"


def test_evidence_preprocessing_rejects_unknown_policies_and_extra_fields() -> None:
    with pytest.raises(ValueError):
        AppSettings(
            visual_planning={
                "preprocessing": {"tile_size": 512},
            }
        )
    with pytest.raises(ValueError):
        AppSettings(
            visual_planning={
                "preprocessing": {"remainder_resize": "crop"},
            }
        )
    with pytest.raises(ValueError):
        AppSettings(
            visual_planning={
                "preprocessing": {"max_tile_concurrency": 0},
            }
        )
    with pytest.raises(ValueError):
        AppSettings(
            visual_planning={
                "preprocessing": {"surprise": True},
            }
        )
    with pytest.raises(ValueError):
        AppSettings(
            visual_planning={
                "preprocessing": {"version": "pad-v9"},
            }
        )
    with pytest.raises(ValueError):
        AppSettings(
            visual_planning={
                "preprocessing": {"segformer_version": "stretch-v1"},
            }
        )
    with pytest.raises(ValueError):
        AppSettings(
            visual_planning={
                "preprocessing": {"segformer_padding_mode": "constant-black-all"},
            }
        )


def test_segmenter_binding_keys_are_stable_logical_identifiers() -> None:
    for key in ("segmenter_mitb2_001", "segmenter_oem_001"):
        settings = AppSettings(
            visual_planning={
                "segmenters": {key: {"enabled": False}},
            }
        )
        assert key in settings.visual_planning.segmenters
    for key in ("../escape", "/etc/passwd", "SegFormer!Path", "", "a b"):
        with pytest.raises(ValueError):
            AppSettings(
                visual_planning={
                    "segmenters": {key: {"enabled": False}},
                }
            )


def test_enabled_segmenter_requires_version_not_path() -> None:
    for version in (None, "models/segformer_mitb2_oem", "/opt/weights/v1"):
        with pytest.raises(ValueError):
            AppSettings(
                visual_planning={
                    "segmenters": {
                        "segmenter_oem_001": {
                            "enabled": True,
                            "class_map_version": version,
                        }
                    }
                }
            )


def test_visual_planner_rejects_removed_confidence_threshold() -> None:
    with pytest.raises(ValueError):
        AppSettings(
            visual_planning={
                "planner": {"confidence_threshold": 0.7},
            }
        )


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
        AppSettings(visual_planning={"enabled": True})
    with pytest.raises(ValueError):
        AppSettings(visual_planning={"roi_partial_failure": "abort"})


def test_visual_planning_empty_evidence_policies_stay_closed() -> None:
    """Empty evidence policies keep optional capabilities closed.
    空的证据策略保持可选能力关闭。"""
    settings = AppSettings(visual_planning={})
    assert settings.visual_planning.detectors == {}
    assert settings.visual_planning.segmenters == {}
