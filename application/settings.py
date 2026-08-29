"""Application settings: YAML loading, environment references, and safe
snapshots. 应用配置：YAML 加载、环境变量引用与安全快照。

Secret values never enter settings, snapshots, repr, or artifacts: only the
environment variable NAME (api_key_env) is declared, the value is read once by
the composition root and injected directly into the judge client.
密钥值绝不进入配置、快照、repr 或产物：只声明环境变量名（api_key_env），
值由组合根读取一次并直接注入 judge 客户端。
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any, Literal, Mapping, MutableMapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agents.change.settings import AgentChangeSettings
from agents.counting.settings import (
    AgentCountingSettings,
    CountingSettings,
    YoloCountingSettings,
)
from models.settings import ModelSettings

_BINDING_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$")


def _looks_path_like(value: str) -> bool:
    """A version string must never smuggle a physical path.
    版本字符串绝不携带物理路径。"""
    return (
        value.startswith(("/", ".", "~"))
        or "\\" in value
        or "/" in value
        or ":" in value
        or value in {".", ".."}
    )


class RunSettings(BaseModel):
    """Run output directory and artifact-saving switches.
    运行输出目录与产物保存开关。"""

    model_config = ConfigDict(extra="forbid")

    root: Path = Path("outputs/runs")
    save_tiles: bool = False
    save_annotated_images: bool = True
    save_raw_responses: bool = True


class ReportingSettings(BaseModel):
    """Native Report V2 output policy. / 原生 Report V2 输出策略。"""

    model_config = ConfigDict(extra="forbid")

    native_html: bool = True
    max_visual_samples: int | None = Field(default=None, ge=0)


class RouterSettings(BaseModel):
    """Task resolution and per-sample budget defaults.
    任务解析与逐样本预算默认值。"""

    model_config = ConfigDict(extra="forbid")

    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    default_qwen_calls: int = Field(default=50, ge=1)
    default_deepseek_calls: int = Field(default=10, ge=0)
    fallback_on_partial: bool = False


class PathSettings(BaseModel):
    """Dataset resolution root. / 数据集解析根。"""

    model_config = ConfigDict(extra="forbid")

    dataset_root: Path = Path("dataset")


class BackendSettings(BaseModel):
    """Counting backend configuration group. / 计数后端配置组。"""

    model_config = ConfigDict(extra="forbid")

    yolo: YoloCountingSettings = Field(default_factory=YoloCountingSettings)


class AgentsSettings(BaseModel):
    """Agent-level configuration group. / Agent 级配置组。"""

    model_config = ConfigDict(extra="forbid")

    counting: AgentCountingSettings = Field(default_factory=AgentCountingSettings)
    change: AgentChangeSettings = Field(default_factory=AgentChangeSettings)


class VisualPlannerSettings(BaseModel):
    """Canonical v5 visual-only planner parameters.
    规范 v5 纯视觉规划器参数。"""

    model_config = ConfigDict(extra="forbid")

    # Must equal the version declared by agents/evidence_catalog.json; the
    # composition root verifies this binding for every fresh runtime.
    # 必须等于 agents/evidence_catalog.json 声明的版本；每次新鲜运行均由组合根校验。
    catalog_version: str = "visual-evidence-catalog-v4"
    task_prompt_version: str = "v5"
    planning_mode: Literal["visual-task-plan-v5"] = "visual-task-plan-v5"
    preview_max_side: int = Field(default=1080, gt=0)
    roi_coordinate_frame: Literal["normalized_0_999_top_left"] = (
        "normalized_0_999_top_left"
    )
    roi_quantum: int = Field(default=1024, gt=0)
    roi_materialization_policy: Literal[
        "longest-side-ceil-quantum-center-clip"
    ] = "longest-side-ceil-quantum-center-clip"
    large_image_policy: Literal["both-dimensions-strictly-greater-than-1024"] = (
        "both-dimensions-strictly-greater-than-1024"
    )

    @model_validator(mode="after")
    def validate_v5_roi_identity(self) -> "VisualPlannerSettings":
        """Keep the v5 geometry identity frozen at the approved quantum.
        将 v5 几何身份冻结为已批准的量化单位。"""
        if self.roi_quantum != 1024:
            raise ValueError("roi_quantum is frozen at 1024")
        return self


class VisualDetectorSettings(BaseModel):
    """Per-binding detector calibration policy (C7, 14A2): every value defaults to None
    meaning "not calibrated" and disabling the capability (approved gate:
    uncalibrated = capability off). Values are range-validated when set; no
    arbitrary production default is ever invented. 逐 binding 检测策略（C7，14A2）：
    每个值默认 None 表示“未校准”并关闭能力（已批准门禁：未校准=能力关闭）。
    设置值时校验范围；绝不杜撰任意生产默认值。"""

    model_config = ConfigDict(extra="forbid")

    confidence_threshold: float | None = None
    nms_iou_threshold: float | None = None
    max_detections: int | None = None

    @model_validator(mode="after")
    def validate_calibrated_values(self) -> "VisualDetectorSettings":
        """Validate ranges only for calibrated (non-None) values.
        只对已校准（非 None）值校验范围。"""

        if self.confidence_threshold is not None:
            if not math.isfinite(self.confidence_threshold):
                raise ValueError("confidence_threshold must be finite")
            if not 0.0 <= self.confidence_threshold <= 1.0:
                raise ValueError("confidence_threshold must be within [0.0, 1.0]")
        if self.nms_iou_threshold is not None:
            if not math.isfinite(self.nms_iou_threshold):
                raise ValueError("nms_iou_threshold must be finite")
            if not 0.0 <= self.nms_iou_threshold <= 1.0:
                raise ValueError("nms_iou_threshold must be within [0.0, 1.0]")
        if self.max_detections is not None and self.max_detections < 1:
            raise ValueError("max_detections must be at least 1 when set")
        return self


class VisualSegmenterSettings(BaseModel):
    """Per-label segmenter policy (C7, 14A2): disabled until explicitly
    calibrated with an approved class map version. 逐标签分割器策略（C7，
    14A2）：显式以已批准 class map 版本校准前保持禁用。

    The dict key in ``visual_planning.segmenters`` is the stable logical
    binding (e.g. ``segmenter_mitb2_001``) that the composition root maps to
    one verified logical client; it is never a checkpoint path or device.
    ``visual_planning.segmenters`` 的 dict key 是稳定逻辑 binding（如
    ``segmenter_mitb2_001``），由组合根映射到一个已验证逻辑客户端；绝不
    是 checkpoint 路径或设备。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    class_map_version: str | None = None

    @model_validator(mode="after")
    def require_class_map_when_enabled(self) -> "VisualSegmenterSettings":
        """An enabled segmenter must declare the approved class map version —
        never a guessed label mapping. 启用的分割器必须声明已批准 class map
        版本——绝不猜测标签映射。"""

        if self.enabled and not self.class_map_version:
            raise ValueError("enabled segmenter requires a class_map_version")
        if self.class_map_version is not None and _looks_path_like(self.class_map_version):
            raise ValueError("class_map_version must be a version, not a path")
        return self


class EvidenceGpuWorkerLimitSettings(BaseModel):
    """One restartable evidence worker memory policy.
    一个可重启 evidence worker 的显存策略。
    """

    model_config = ConfigDict(extra="forbid")

    soft_limit_gib: float = Field(gt=0)
    hard_limit_gib: float = Field(gt=0)

    @model_validator(mode="after")
    def require_ordered_limits(self) -> "EvidenceGpuWorkerLimitSettings":
        if self.soft_limit_gib >= self.hard_limit_gib:
            raise ValueError("evidence worker requires soft_limit_gib < hard_limit_gib")
        return self


class EvidenceGpuWorkersSettings(BaseModel):
    """Isolated YOLO/SegFormer GPU worker guard. / 隔离的 GPU worker 保护。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    yolo: EvidenceGpuWorkerLimitSettings = Field(
        default_factory=lambda: EvidenceGpuWorkerLimitSettings(
            soft_limit_gib=6, hard_limit_gib=8
        )
    )
    segformer: EvidenceGpuWorkerLimitSettings = Field(
        default_factory=lambda: EvidenceGpuWorkerLimitSettings(
            soft_limit_gib=10, hard_limit_gib=12
        )
    )
    device_free_floor_gib: float = Field(default=8, gt=0)
    poll_interval_seconds: float = Field(default=1, gt=0)
    max_retries: Literal[1] = 1


class VisualEvidencePreprocessSettings(BaseModel):
    """Frozen evidence preprocessing identity shared by VQA evidence phases.
    One deterministic identity for every model call made inside the planner
    ROI pipeline. version names the complete algorithm combination: fresh
    runs default to ``yolo-v1-segformer-pad-v1`` (YOLO on greedy tiles,
    SegFormer on the pad-multiple-1024-resize-square protocol); the legacy
    ``greedy-1024-stretch-v1`` remains expressible only as an explicit
    configuration for historical interpretation. The backend-specific
    yolo_version/segformer_version fields freeze each phase's own protocol.
    冻结的 evidence 预处理身份，由 VQA evidence 各阶段共享；planner ROI 管线
    内的每次模型调用使用同一个确定性身份。version 标识完整算法组合：新鲜运行
    默认 ``yolo-v1-segformer-pad-v1``（YOLO 走 greedy tiles，SegFormer 走
    pad-multiple-1024-resize-square 协议）；旧 ``greedy-1024-stretch-v1``
    只允许作为显式配置表达历史解释。backend-specific 的
    yolo_version/segformer_version 字段分别冻结各阶段的协议。

    Identifiers of the form ``*_v1`` are a typed capability contract: when the
    pipeline semantics change, a new version must be declared instead of
    silently mutating this one. 形如 ``*_v1`` 的标识是类型化能力契约：管线
    语义变化时必须声明新版本，而不是悄悄改动本版本。"""

    model_config = ConfigDict(extra="forbid")

    version: Literal["greedy-1024-stretch-v1", "yolo-v1-segformer-pad-v1"] = (
        "yolo-v1-segformer-pad-v1"
    )
    tile_size: Literal[1024] = 1024
    partition_policy: Literal["greedy-row-major-no-overlap"] = (
        "greedy-row-major-no-overlap"
    )
    remainder_resize: Literal["stretch"] = "stretch"
    rgb_interpolation: Literal["lanczos"] = "lanczos"
    mask_inverse_interpolation: Literal["nearest"] = "nearest"
    max_tile_concurrency: int = Field(default=4, ge=1, le=32)
    # Backend-specific frozen identities: YOLO stays on the v1 tile protocol
    # under both combined versions; SegFormer defaults to the fresh pad
    # protocol. 后端特定冻结身份：两种组合版本下 YOLO 都保持 v1 tile 协议；
    # SegFormer 默认使用新鲜 pad 协议。
    yolo_version: Literal["greedy-1024-stretch-v1"] = "greedy-1024-stretch-v1"
    segformer_version: Literal["pad-multiple-1024-resize-square-v1"] = (
        "pad-multiple-1024-resize-square-v1"
    )
    segformer_padding_mode: Literal["constant-black-right-bottom"] = (
        "constant-black-right-bottom"
    )
    segformer_rgb_interpolation: Literal["lanczos"] = "lanczos"
    segformer_mask_inverse_interpolation: Literal["nearest"] = "nearest"


class VisualPlanningSettings(BaseModel):
    """Visual-only planner and evidence capability configuration.
    纯视觉规划器与视觉证据能力配置。

    Fresh execution is always planner-first. / 新鲜执行始终先规划。
    """

    model_config = ConfigDict(extra="forbid")

    planner: VisualPlannerSettings = Field(default_factory=VisualPlannerSettings)
    detectors: dict[str, VisualDetectorSettings] = Field(default_factory=dict)
    segmenters: dict[str, VisualSegmenterSettings] = Field(default_factory=dict)
    gpu_workers: EvidenceGpuWorkersSettings = Field(
        default_factory=EvidenceGpuWorkersSettings
    )
    preprocessing: VisualEvidencePreprocessSettings = Field(
        default_factory=VisualEvidencePreprocessSettings
    )

    @model_validator(mode="after")
    def validate_visual_binding_keys(self) -> "VisualPlanningSettings":
        """Detector and segmenter keys are stable logical bindings.
        检测器与分割器 key 都必须是稳定逻辑 binding。"""
        for kind, bindings in (
            ("detector", self.detectors),
            ("segmenter", self.segmenters),
        ):
            for key in bindings:
                if re.fullmatch(_BINDING_KEY_PATTERN, key) is None:
                    raise ValueError(
                        f"invalid {kind} binding key {key!r}: must be a stable "
                        "logical binding identifier, not a path or free text"
                    )
        return self


class AppSettings(BaseModel):
    """Full application settings; no secret values by construction.
    完整应用配置；构造上不含任何密钥值。"""

    model_config = ConfigDict(extra="forbid")

    models: ModelSettings = Field(default_factory=ModelSettings)
    counting: CountingSettings = Field(default_factory=CountingSettings)
    runs: RunSettings = Field(default_factory=RunSettings)
    reporting: ReportingSettings = Field(default_factory=ReportingSettings)
    router: RouterSettings = Field(default_factory=RouterSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    backend: BackendSettings = Field(default_factory=BackendSettings)
    agents: AgentsSettings = Field(default_factory=AgentsSettings)
    visual_planning: VisualPlanningSettings = Field(
        default_factory=VisualPlanningSettings
    )

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_counting_execution_policy(cls, data: Any) -> Any:
        """Move legacy YOLO policy only at the settings boundary.
        仅在 settings 边界迁移旧 YOLO 执行策略。"""

        if not isinstance(data, Mapping):
            return data
        payload = dict(data)
        counting = dict(payload.get("counting", {}))
        backend = dict(payload.get("backend", {}))
        yolo = dict(backend.get("yolo", {}))
        aliases = {
            "fallback_to_qwen_on_unavailable": "fallback_on_backend_unavailable",
            "fallback_to_qwen_on_error": "fallback_on_backend_error",
            "verify_empty_with_qwen": "verify_empty_detection",
        }
        for legacy, current in aliases.items():
            if legacy not in yolo:
                continue
            if current in counting:
                raise ValueError(
                    f"cannot configure both legacy key {legacy!r} and {current!r}"
                )
            counting[current] = yolo.pop(legacy)
        if "trust_empty_detection" in backend:
            if "trust_empty_detection" in counting:
                raise ValueError(
                    "cannot configure both legacy backend trust_empty_detection "
                    "and counting trust_empty_detection"
                )
            counting["trust_empty_detection"] = backend.pop("trust_empty_detection")
        backend["yolo"] = yolo
        payload["backend"] = backend
        payload["counting"] = counting
        return payload

    def to_config_payload(self) -> dict[str, Any]:
        """JSON-safe reproduction snapshot for run manifests: no secret
        values, no Path objects, paths serialized as strings with forward
        slashes on every platform. Configured host paths are preserved
        verbatim — this snapshot is reproduction/debug oriented and carries
        the host's own path semantics. 供 run manifest 使用的 JSON 安全复现
        快照：无密钥值、无 Path 对象、路径在所有平台统一正斜杠字符串。
        配置的主机路径原样保留——本快照面向复现/调试，承载主机自身的路径
        语义。"""

        payload = self.model_dump(mode="json")
        payload["runs"]["root"] = _posix(self.runs.root)
        payload["paths"]["dataset_root"] = _posix(self.paths.dataset_root)
        for profile_name, profile in (
            ("segformer_isaid", self.models.segformer_isaid),
            ("segformer_oem", self.models.segformer_oem),
        ):
            payload["models"][profile_name]["model_path"] = _posix(
                profile.model_path
            )
            if profile.processor_path is not None:
                payload["models"][profile_name]["processor_path"] = _posix(
                    profile.processor_path
                )
        for backend_name, profile in self.models.segformer_experts.items():
            serialized = payload["models"]["segformer_experts"][backend_name]
            serialized["model_path"] = _posix(profile.model_path)
            if profile.processor_path is not None:
                serialized["processor_path"] = _posix(profile.processor_path)
        for adapter_name, adapter in self.models.qwen_adapters.items():
            payload["models"]["qwen_adapters"][adapter_name]["path"] = _posix(
                adapter.path
            )
        return payload

    def safe_snapshot(self) -> dict[str, Any]:
        """Alias of to_config_payload: the only allowed serialization of
        settings. 与 to_config_payload 相同的安全快照：配置唯一允许的序列化
        形式。"""

        return self.to_config_payload()


_ENV_OVERRIDES = {
    "QWEN_MODEL": ("models", "qwen", "model"),
    "QWEN_CACHE_MODEL_ID": ("models", "qwen", "cache_model_id"),
    "SEGFORMER_ISAID_MODEL": ("models", "segformer_isaid", "model_path"),
    "SEGFORMER_OEM_MODEL": ("models", "segformer_oem", "model_path"),
    "DEEPSEEK_BASE_URL": ("models", "deepseek", "base_url"),
    "DEEPSEEK_MODEL": ("models", "deepseek", "model"),
    "DATASET_ROOT": ("paths", "dataset_root"),
    "OUTPUT_ROOT": ("runs", "root"),
}

_DOTENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_dotenv(
    path: Path | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> Path | None:
    """Load a project ``.env`` file without overriding exported variables.

    The project deliberately avoids a runtime dependency on ``python-dotenv``.
    This small parser supports the dotenv forms used by the application,
    including comments, ``export KEY=value``, and quoted values.  Existing
    process environment variables win over values from the file.

    Secret values are placed only in the process environment; they are not
    copied into :class:`AppSettings` or any settings snapshot.
    """

    target = environ if environ is not None else os.environ
    dotenv_path = path or (Path(__file__).resolve().parents[1] / ".env")
    if not dotenv_path.is_file():
        return None

    try:
        lines = dotenv_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not _DOTENV_KEY.fullmatch(key) or key in target:
            continue
        target[key] = _parse_dotenv_value(raw_value.strip())
    return dotenv_path


def _parse_dotenv_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
        return value.replace(r"\n", "\n").replace(r"\r", "\r").replace(
            r"\t", "\t"
        ).replace(r'\"', '"').replace(r"\\", "\\")
    return value


def load_settings(
    path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> AppSettings:
    """Load settings from defaults, an optional YAML file, and environment
    overrides (environment wins). Secret VALUES are never read — only declared
    env var names travel through settings. 从默认值、可选 YAML 文件与环境变量
    覆盖加载配置（环境变量优先）。密钥值绝不读取——只有声明的环境变量名经过
    配置。"""

    settings = AppSettings()
    if path is not None and path.is_file():
        merged = settings.model_dump()
        raw = AppSettings.migrate_legacy_counting_execution_policy(_load_yaml(path))
        _deep_merge(merged, raw)
        settings = AppSettings.model_validate(merged)
    environ = environ if environ is not None else os.environ
    overrides: dict[str, Any] = {}
    for env_name, target in _ENV_OVERRIDES.items():
        value = environ.get(env_name)
        if value is None or value == "":
            continue
        node = overrides
        for key in target[:-1]:
            node = node.setdefault(key, {})
        node[target[-1]] = value
    # A physical/logical QWEN_MODEL override replaces the YAML model identity.
    # Never retain a stale cache_model_id from the selected profile. A local
    # checkpoint must then provide QWEN_CACHE_MODEL_ID explicitly and fail
    # validation if it does not.
    qwen_model = environ.get("QWEN_MODEL")
    qwen_cache_model_id = environ.get("QWEN_CACHE_MODEL_ID")
    if qwen_model:
        qwen = overrides.setdefault("models", {}).setdefault("qwen", {})
        qwen["cache_model_id"] = qwen_cache_model_id or None
    if overrides:
        merged = settings.model_dump()
        _deep_merge(merged, overrides)
        settings = AppSettings.model_validate(merged)
    return settings


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid settings YAML: {type(exc).__name__}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("settings YAML must map to an object")
    return raw


def _posix(path: Path) -> str:
    """POSIX serialization with forward-slash separators on every platform
    (as_posix alone keeps backslashes on POSIX hosts). 所有平台统一正斜杠
    的 POSIX 序列化（仅 as_posix 在 POSIX 主机上会保留反斜杠）。"""

    return path.as_posix().replace("\\", "/")


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
