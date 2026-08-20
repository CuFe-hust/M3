"""Model settings declarations; no environment reading, no secret values.

模型配置声明；不读取环境变量、不保存密钥值。环境变量解析由
application/settings.py 负责。
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from models.base import is_local_model_path, validate_logical_model_id


class QwenSettings(BaseModel):
    """Settings for the local Transformers Qwen backend.
    本地 Transformers Qwen 后端的配置。"""

    model_config = ConfigDict(extra="forbid")

    # Physical checkpoint path / name passed to from_pretrained.
    # 传给 from_pretrained 的物理 checkpoint 路径/名称。
    model: str = "qwen3-vl-4b-instruct"
    # Logical, machine-independent model identity used for hashes and traces.
    # 用于哈希与 trace 的逻辑、与机器无关的模型身份。
    cache_model_id: str | None = None
    max_tokens: int = Field(default=4096, gt=0)
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    device_map: str = "auto"
    use_kernels: bool = False
    # Single source of truth for online behaviour: offline by default.
    # 联网行为的唯一来源：默认离线。
    allow_download: bool = False
    min_pixels: int | None = Field(default=None, gt=0)
    max_pixels: int | None = Field(default=None, gt=0)
    revision: str | None = None

    @model_validator(mode="after")
    def validate_model_identity(self) -> "QwenSettings":
        """A local checkpoint path (POSIX, Windows drive, UNC, or file URI)
        must carry an explicit logical cache model id, which itself must be a
        logical identifier — never a local path — so hashes and traces never
        leak machine paths. 本地 checkpoint 路径（POSIX、Windows drive、UNC
        或 file URI）必须携带显式逻辑缓存模型 ID；该 ID 本身必须是逻辑标识
        符而非本地路径，使哈希与 trace 永不泄漏机器路径。"""
        if is_local_model_path(self.model) and not self.cache_model_id:
            raise ValueError("cache_model_id is required when model is a local path")
        if self.cache_model_id is not None:
            self.cache_model_id = validate_logical_model_id(
                self.cache_model_id,
                where="cache_model_id",
            )
        return self

    @property
    def effective_cache_model_id(self) -> str:
        """Logical model identity for hashes and traces; the returned value is
        always validated — absolute local checkpoint paths are rejected without
        an explicit cache_model_id, and the declared model name is safe.
        用于哈希与 trace 的逻辑模型身份；返回值始终经过校验——无显式
        cache_model_id 的绝对本地 checkpoint 路径已被拒绝，声明模型名安全。"""
        if self.cache_model_id is not None:
            return self.cache_model_id
        return validate_logical_model_id(self.model, where="model")


class DeepSeekSettings(BaseModel):
    """Settings for a DeepSeek structured judge client (declaration only).
    未来 DeepSeek 结构化评估客户端的配置（仅声明）。"""

    model_config = ConfigDict(extra="forbid")

    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    api_key_env: str = "DEEPSEEK_API_KEY"
    timeout_seconds: int = Field(default=120, gt=0)
    max_retries: int = Field(default=3, ge=0)


class SegFormerSettings(BaseModel):
    """Settings for one local fine-tuned SegFormer checkpoint.
    单个本地微调 SegFormer checkpoint 的配置。"""

    model_config = ConfigDict(extra="forbid")

    # Physical checkpoint directory; never used as the logical model identity.
    # 物理 checkpoint 目录；绝不作为逻辑模型身份。
    model_path: Path = Path("models/segformer_mitb2_isaid")
    logical_model_id: str = "SegFormer-MiT-B2:iSAID:local"
    weights_filename: str = "model.safetensors"
    weights_sha256: str | None = (
        "f8e60686ec41160b5cbc494e8a3c1d28a92f7afdd41708c7b77e3d5793908b9a"
    )
    classes_filename: str | None = "classes.json"
    processor_path: Path | None = None
    device: str = "auto"
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    require_cuda: bool = False
    allow_cpu_fallback: bool = False
    allow_download: Literal[False] = False
    revision: str | None = None

    @model_validator(mode="after")
    def validate_runtime_declaration(self) -> "SegFormerSettings":
        """Validate identity, filenames, digest, and device without touching
        the filesystem. 在不访问文件系统的前提下校验身份、文件名、摘要与设备。"""

        self.logical_model_id = validate_logical_model_id(
            self.logical_model_id,
            where="logical_model_id",
        )
        self.weights_filename = _plain_filename(
            self.weights_filename,
            where="weights_filename",
        )
        if self.classes_filename is not None:
            self.classes_filename = _plain_filename(
                self.classes_filename,
                where="classes_filename",
            )
        if self.weights_sha256 is not None:
            digest = self.weights_sha256.strip().casefold()
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(
                    "weights_sha256 must be a 64-character hexadecimal digest"
                )
            self.weights_sha256 = digest
        if self.device not in {"auto", "cpu", "cuda"} and not (
            self.device.startswith("cuda:") and self.device[5:].isdigit()
        ):
            raise ValueError("device must be auto, cpu, cuda, or cuda:<index>")
        if self.require_cuda and self.device == "cpu":
            raise ValueError("require_cuda is incompatible with device='cpu'")
        if self.require_cuda and self.allow_cpu_fallback:
            raise ValueError("require_cuda is incompatible with CPU fallback")
        if self.device == "cpu" and self.allow_cpu_fallback:
            raise ValueError("device='cpu' does not use CPU fallback")
        if self.revision is not None:
            self.revision = self.revision.strip()
            if not self.revision or any(
                character in self.revision for character in ("\x00", "\n", "\r")
            ):
                raise ValueError("revision contains forbidden characters")
        return self


class ModelSettings(BaseModel):
    """Group model settings without storing secret values.
    聚合模型配置且不保存密钥值。"""

    model_config = ConfigDict(extra="forbid")

    qwen: QwenSettings = Field(default_factory=QwenSettings)
    deepseek: DeepSeekSettings = Field(default_factory=DeepSeekSettings)
    segformer_isaid: SegFormerSettings = Field(default_factory=SegFormerSettings)
    segformer_oem: SegFormerSettings = Field(
        default_factory=lambda: SegFormerSettings(
            model_path=Path("models/segformer_mitb2_oem"),
            logical_model_id="SegFormer-MiT-B2:OpenEarthMap:local",
            weights_sha256=(
                "d2141c79b2fc27ea5505db378b48e90e75e5ee06751df1c5b4028ef662fb2fab"
            ),
            # The checkpoint-specific channel map is not verified.  Keep the
            # profile unable to construct a semantic runtime until evidence is
            # recorded and injected explicitly.
            classes_filename=None,
        )
    )
    # Additional runtime-only profiles are keyed by the stable catalog
    # backend name.  Their asset identity is replaced by the catalog at the
    # composition boundary; these values only carry provider/device policy.
    segformer_experts: dict[str, SegFormerSettings] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_segformer_expert_profiles(self) -> "ModelSettings":
        normalized = [name.strip() for name in self.segformer_experts]
        if any(not name for name in normalized) or len(normalized) != len(set(normalized)):
            raise ValueError("SegFormer expert profile names must be non-empty and unique")
        if tuple(normalized) != tuple(self.segformer_experts):
            raise ValueError("SegFormer expert profile names must be trimmed")
        return self

    def segformer_profile(
        self,
        *,
        backend_name: str,
        logical_model_id: str,
    ) -> SegFormerSettings:
        """Resolve provider policy without making bootstrap expert-specific."""

        explicit = self.segformer_experts.get(backend_name)
        if explicit is not None:
            if explicit.logical_model_id != logical_model_id:
                raise ValueError("SegFormer profile logical model id differs from catalog")
            return explicit
        matches = [
            profile
            for profile in (self.segformer_isaid, self.segformer_oem)
            if profile.logical_model_id == logical_model_id
        ]
        if len(matches) != 1:
            raise ValueError("SegFormer catalog expert has no unique runtime profile")
        return matches[0]


def _plain_filename(value: str, *, where: str) -> str:
    """Require one cross-platform plain filename with no directory parts.
    要求跨平台安全、且不含目录部分的纯文件名。"""

    normalized = value.strip()
    if not normalized or normalized in {".", ".."}:
        raise ValueError(f"{where} must be a plain filename")
    if (
        PurePosixPath(normalized).name != normalized
        or PureWindowsPath(normalized).name != normalized
    ):
        raise ValueError(f"{where} must be a plain filename")
    return normalized
