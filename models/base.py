"""Shared model-client contracts, request hashing, and sanitization.

模型客户端共用协议、请求哈希与脱敏。本模块不依赖 data / agents /
application，不读取任何 API key，不触发 transformers / torch 导入。
缓存与图像工具分别位于 models/cache.py 与 models/images.py。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Protocol, TypeVar
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict

ModelT = TypeVar("ModelT", bound=BaseModel)

# Sensitive key names and high-risk value prefixes that must never appear in a
# cache identity. Keys are matched after normalization; values after
# lstrip().lower(). / 缓存身份中绝不能出现的敏感键名与高风险值前缀；键在
# 归一化后精确匹配，值在 lstrip().lower() 后检查前缀。
_SENSITIVE_IDENTITY_KEYS = frozenset({
    "api_key", "apikey", "authorization", "access_token", "refresh_token",
    "private_key", "password", "credential",
})
_IDENTITY_VALUE_PREFIXES = (
    "sk-",
    "bearer ",
    "data:image/",
    "-----begin private key-----",
)


def is_local_model_path(value: str) -> bool:
    """Cross-platform detection of local filesystem paths: POSIX absolute,
    Windows drive (backslash or slash), Windows UNC (backslash or slash form),
    and file URIs. 跨平台识别本地文件系统路径：POSIX 绝对、Windows drive
    （反斜杠或斜杠）、Windows UNC（反斜杠或斜杠形式）与 file URI。"""
    stripped = value.strip()
    if not stripped:
        return False
    parsed = urlparse(stripped)
    if parsed.scheme.lower() == "file":
        return True
    return (
        PurePosixPath(stripped).is_absolute()
        or PureWindowsPath(stripped).is_absolute()
        or stripped.startswith("\\")
        or stripped.startswith("//")
    )


def validate_logical_model_id(value: str, *, where: str) -> str:
    """Validate a logical model identifier: a non-empty string without control
    characters that is never a local filesystem path. Remote model names such
    as "Qwen/Qwen3-VL-4B-Instruct" or "org:model@rev" remain allowed.
    校验逻辑模型标识符：非空、无控制字符、且绝不是本地文件系统路径的字符串。
    远程模型名（如 "Qwen/Qwen3-VL-4B-Instruct" 或 "org:model@rev"）仍被允许。"""
    if not isinstance(value, str):
        raise TypeError(f"{where} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{where} must not be empty")
    if "\x00" in normalized or "\n" in normalized or "\r" in normalized:
        raise ValueError(f"{where} contains forbidden control characters")
    if is_local_model_path(normalized):
        raise ValueError(f"{where} must be a logical identifier, not a local path")
    return normalized


@dataclass(frozen=True)
class ModelCacheIdentity:
    """Stable, JSON-safe identity of one model client for cache keying.

    The visual agent builds its request hash exclusively from this object so
    the hashed model name, generation parameters, client version, and revision
    can never drift from the client that actually runs the call. generation is
    deep-frozen at construction: nested dicts/lists are canonicalized into
    immutable tuples, string keys are mandatory, and later mutation of the
    caller's source mapping cannot alter this identity.
    单个模型客户端用于缓存键的稳定、JSON 安全身份。视觉 Agent 只从该对象
    构建请求哈希，使参与哈希的模型名、生成参数、客户端版本与 revision 永远
    不会与实际执行调用的客户端漂移。generation 在构造时深度冻结：嵌套
    dict/list 规范化为不可变 tuple，键必须是字符串，调用方后续修改源映射
    不会改变本身份。"""

    model: str
    generation: Mapping[str, Any]
    client_version: str
    revision: str | None = None
    _frozen_generation: tuple[Any, ...] = field(init=False, repr=False, compare=True)

    def __post_init__(self) -> None:
        # The logical model id is validated independently of any settings
        # class so even a custom client cannot inject a machine path.
        # 逻辑模型 ID 独立于任何配置类校验，自定义客户端也无法注入机器路径。
        object.__setattr__(
            self,
            "model",
            validate_logical_model_id(self.model, where="ModelCacheIdentity.model"),
        )
        client_version = self.client_version.strip()
        if not client_version:
            raise ValueError("client_version must not be empty")
        object.__setattr__(self, "client_version", client_version)
        if self.revision is not None:
            revision = self.revision.strip()
            if not revision or any(
                character in revision for character in ("\x00", "\n", "\r")
            ):
                raise ValueError("revision contains forbidden characters")
            object.__setattr__(self, "revision", revision)
        _validate_identity_value(self.generation, "generation")
        object.__setattr__(
            self,
            "_frozen_generation",
            _canonical_generation(self.generation, "generation"),
        )
        object.__setattr__(
            self,
            "generation",
            _FrozenJsonMapping(self.generation, "generation"),
        )

    def generation_payload(self) -> dict[str, Any]:
        """Return a fresh plain-JSON copy of the generation payload; the
        internal frozen structure is never exposed.
        返回生成载荷的全新普通 JSON 副本；绝不暴露内部冻结结构。"""
        return _unfreeze_identity_value(self._frozen_generation)


class CacheIdentifiedClient(Protocol):
    """A model client exposing its stable cache identity.
    暴露其稳定缓存身份的模型客户端。"""

    @property
    def cache_identity(self) -> ModelCacheIdentity: ...


class MissingModelCacheIdentityError(RuntimeError):
    """Raised when a client does not expose a valid cache identity; model calls
    never fall back to fabricated identities.
    客户端未暴露有效缓存身份时抛出；模型调用绝不使用伪造身份回退。"""


class ModelAssetError(RuntimeError):
    """Base error for an unusable local model asset.
    本地模型资产不可用时的基础错误。"""


class ModelAssetMissingError(ModelAssetError, FileNotFoundError):
    """Raised when a configured local model asset does not exist.
    配置的本地模型资产不存在时抛出。"""


class ModelAssetPointerError(ModelAssetError):
    """Raised when a Git LFS pointer is supplied instead of model bytes.
    将 Git LFS 指针误作模型二进制时抛出。"""


class ModelAssetHashMismatchError(ModelAssetError):
    """Raised when a local model asset fails its declared SHA-256 check.
    本地模型资产未通过声明的 SHA-256 校验时抛出。"""


_GIT_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def validate_local_model_asset(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> str:
    """Validate one local model file and return its SHA-256 digest.

    Validation is deliberately performed before any optional runtime sees the
    file, producing stable errors for a missing file, a Git LFS pointer, or a
    digest mismatch. The error text only contains the basename so persisted
    public errors cannot leak a machine-specific absolute path.
    在任何可选运行时读取文件前校验本地模型文件，并返回 SHA-256。对于文件
    缺失、Git LFS 指针和摘要不匹配给出稳定错误；错误文本只包含 basename，
    避免公共持久化错误泄漏机器绝对路径。
    """

    if expected_sha256 is not None:
        expected_sha256 = expected_sha256.strip().casefold()
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha256
        ):
            raise ValueError("expected_sha256 must be a 64-character hexadecimal digest")
    if not path.is_file():
        raise ModelAssetMissingError(f"model asset is missing: {path.name}")
    try:
        with path.open("rb") as file:
            prefix = file.read(len(_GIT_LFS_POINTER_PREFIX))
            if prefix == _GIT_LFS_POINTER_PREFIX:
                raise ModelAssetPointerError(
                    "model weight is a Git LFS pointer; actual binary has not "
                    f"been downloaded: {path.name}"
                )
            digest = hashlib.sha256()
            digest.update(prefix)
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ModelAssetError(
            f"model asset could not be read: {path.name} ({type(error).__name__})"
        ) from error
    actual = digest.hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        raise ModelAssetHashMismatchError(
            f"model asset digest mismatch for {path.name}: expected "
            f"{expected_sha256}, got {actual}"
        )
    return actual


def require_model_cache_identity(
    client: object,
    *,
    component: str,
) -> ModelCacheIdentity:
    """Require a real ModelCacheIdentity instance — duck-typed stand-ins are
    rejected before any model call. This is the single authoritative helper for
    every model client in the runtime. 要求真实 ModelCacheIdentity 实例——鸭子
    类型替代品在任何模型调用前被拒绝。这是整个运行时所有模型客户端的唯一
    权威 helper。"""
    identity = getattr(client, "cache_identity", None)
    if not isinstance(identity, ModelCacheIdentity):
        raise MissingModelCacheIdentityError(
            f"{component} requires a valid ModelCacheIdentity"
        )
    return identity


class _FrozenJsonMapping(Mapping[str, Any]):
    """Read-only, deep-frozen Mapping over JSON-safe values. Nested dicts
    become further _FrozenJsonMapping instances and nested lists become
    tuples, so no part of the payload can be mutated in place.
    覆盖 JSON 安全值的只读、深度冻结 Mapping。嵌套 dict 变为新的
    _FrozenJsonMapping，嵌套 list 变为 tuple，载荷任何部分都无法原地修改。"""

    __slots__ = ("_items",)

    def __init__(self, items: Mapping[str, Any], where: str) -> None:
        object.__setattr__(
            self,
            "_items",
            tuple(
                (
                    key,
                    _freeze_identity_value(item, f"{where}.{key}"),
                )
                for key, item in sorted(items.items())
            ),
        )

    def __getitem__(self, key: str) -> Any:
        for item_key, item_value in self._items:
            if item_key == key:
                return item_value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return repr(dict(self))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, _FrozenJsonMapping):
            return self._items == other._items
        if isinstance(other, Mapping):
            return dict(self) == dict(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._items)


# Tag used by the canonical frozen generation representation; it cannot
# collide with user data because tuples are rejected by identity validation.
# 规范冻结生成表示使用的类型标记；由于 tuple 会被身份校验拒绝，它不会与
# 用户数据冲突。
_DICT_TAG = "_dict"


def _canonical_generation(
    mapping: Mapping[str, Any],
    where: str,
) -> tuple[Any, ...]:
    """Canonicalize a mapping into a tagged, sorted, deep-frozen structure
    used for equality and hashing. 将映射规范化为用于相等性与哈希的带标记、
    排序、深度冻结结构。"""
    return (
        _DICT_TAG,
        tuple(
            (
                key,
                _freeze_identity_value(item, f"{where}.{key}"),
            )
            for key, item in sorted(mapping.items())
        ),
    )


def _freeze_identity_value(value: Any, where: str) -> Any:
    """Recursively freeze dicts as read-only mappings and lists as tuples;
    other JSON-safe values pass through. 递归将 dict 冻结为只读映射、list
    冻结为 tuple；其他 JSON 安全值原样通过。"""
    if isinstance(value, dict):
        return _FrozenJsonMapping(value, where)
    if isinstance(value, list):
        return tuple(
            _freeze_identity_value(item, f"{where}[{index}]")
            for index, item in enumerate(value)
        )
    return value


def _unfreeze_identity_value(value: Any) -> Any:
    """Convert a frozen identity payload back into a plain JSON structure.
    将冻结的身份载荷还原为普通 JSON 结构。"""
    if isinstance(value, tuple) and len(value) == 2 and value[0] == _DICT_TAG:
        return {
            key: _unfreeze_identity_value(item)
            for key, item in value[1]
        }
    if isinstance(value, _FrozenJsonMapping):
        return {
            key: _unfreeze_identity_value(item)
            for key, item in value._items
        }
    if isinstance(value, tuple):
        return [_unfreeze_identity_value(item) for item in value]
    return value


def _validate_identity_value(value: Any, where: str) -> None:
    """Require JSON-safe, finite, secret-free identity content with string
    mapping keys only. 要求身份内容 JSON 安全、数值有限、不含密钥，且映射键
    全部为字符串。"""
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, str):
        normalized = value.lstrip().lower()
        for prefix in _IDENTITY_VALUE_PREFIXES:
            if normalized.startswith(prefix):
                raise ValueError(f"{where} contains a sensitive value prefix")
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{where} contains a non-finite number")
        return
    if isinstance(value, Path):
        raise ValueError(f"{where} contains a Path object")
    if isinstance(value, (set, bytes, bytearray)):
        raise ValueError(f"{where} contains a {type(value).__name__}")
    if callable(value):
        raise ValueError(f"{where} contains a callable")
    if isinstance(value, list):
        for item in value:
            _validate_identity_value(item, where)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{where} contains a non-string key {key!r}")
            normalized_key = key.lower().replace("-", "_").replace(" ", "_")
            if normalized_key in _SENSITIVE_IDENTITY_KEYS:
                raise ValueError(f"{where} contains a sensitive key {key!r}")
            _validate_identity_value(item, where)
        return
    raise ValueError(f"{where} contains unsupported type {type(value).__name__}")


class RequestMeta(BaseModel):
    """Traceable request metadata that deliberately excludes credentials and Base64.
    可追踪的请求元数据，刻意排除凭据和 Base64。
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str
    request_hash: str
    prompt_version: str
    sample_id: str | None = None
    tile_id: str | None = None
    image_sha256: str | None = None
    artifact_dir: Path | None = None


class VisionLanguageClient(Protocol):
    """Protocol shared by live and offline structured vision-language clients.
    线上与离线结构化视觉语言客户端共用的协议。
    """

    async def complete_json(
        self,
        *,
        messages: list[dict[str, Any]],
        response_model: type[ModelT],
        request_meta: RequestMeta,
        max_tokens: int | None = None,
    ) -> ModelT:
        """Return one schema-validated JSON response.
        返回一条经 Schema 校验的 JSON 响应。
        """


@dataclass(frozen=True)
class DenseSemanticOutput:
    """CPU dense semantic tensors and their actual image-relative grids.

    Arrays intentionally remain ``Any`` so the shared contract imports neither
    NumPy nor torch. Concrete backends must return finite float32 arrays with
    ``probabilities`` shaped ``[C,Hs,Ws]`` and ``features`` shaped
    ``[D,Hf,Wf]``. 稠密语义 CPU 张量及其相对原图的实际网格；共享契约不导入
    NumPy/torch，具体 backend 负责保证 float32、有限值与形状约定。
    """

    probabilities: Any
    features: Any
    semantic_stride: tuple[float, float]
    feature_stride: tuple[float, float]
    original_size: tuple[int, int]
    class_names: tuple[str, ...]
    diagnostics: Mapping[str, Any]


class DenseSemanticClient(Protocol):
    """Abstract image-to-dense-semantics model contract."""

    @property
    def cache_identity(self) -> ModelCacheIdentity: ...

    def infer(
        self,
        image: Any,
        *,
        tile_size: int,
        tile_overlap: int,
        feature_stage: int,
    ) -> DenseSemanticOutput: ...


class SemanticSegmentationOutput(Protocol):
    """In-memory semantic prediction; dense maps are deliberately not JSON models.
    内存中的语义预测；稠密像素图刻意不定义为 JSON 模型。"""

    width: int
    height: int
    mask: Any
    confidence_map: Any
    id_to_label: Mapping[int, str]
    logical_model_id: str
    model_revision: str | None
    weights_sha256: str

    @property
    def revision(self) -> str | None: ...

    @property
    def sha256(self) -> str: ...


class SemanticSegmentationClient(Protocol):
    """Model-layer protocol for image-aligned semantic inference.
    与输入图像像素对齐的模型层语义推理协议。"""

    def predict(self, image: Any) -> SemanticSegmentationOutput:
        """Return class IDs and confidence at the source image resolution.
        返回源图像分辨率下的类别 ID 与置信度。"""


def build_request_hash(
    *,
    model: str,
    generation: Mapping[str, Any],
    prompt_version: str,
    messages: Sequence[Mapping[str, Any]],
    image_sha256: str | None,
    tile_geometry: Mapping[str, Any] | None = None,
    target_spec: Mapping[str, Any] | None = None,
    response_schema: Mapping[str, Any] | None = None,
    client_version: str | None = None,
    model_revision: str | None = None,
) -> str:
    """Hash cache inputs while replacing data URLs with their digest and size.
    The sanitized payload makes the hash stable across machines and runs.
    response_schema / client_version / model_revision are included so cache
    keys cover the full inference semantics.
    对缓存输入计算哈希，同时以摘要和大小替换数据 URL；脱敏载荷使哈希
    跨机器、跨运行稳定。response_schema / client_version / model_revision
    参与哈希，使缓存键覆盖完整推理语义。"""

    payload = {
        "model": model,
        "generation": generation,
        "prompt_version": prompt_version,
        "messages": sanitize_messages(messages),
        "image_sha256": image_sha256,
        "tile_geometry": tile_geometry,
        "target_spec": target_spec,
        "response_schema": response_schema,
        "client_version": client_version,
        "model_revision": model_revision,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sanitize_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Remove raw Base64 data URLs before logs, hashes, or artifacts are written.
    在写入日志、哈希或产物前移除原始 Base64 数据 URL。"""

    return [_sanitize_value(message) for message in messages]


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("data:image/") and ";base64," in value:
        return {
            "redacted_data_url_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
            "encoded_bytes": len(value.encode("utf-8")),
        }
    if isinstance(value, Mapping):
        return {str(key): _sanitize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    return value
