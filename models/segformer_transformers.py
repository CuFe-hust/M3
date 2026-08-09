"""Offline-first Transformers runtime for local SegFormer checkpoints.

本地 SegFormer checkpoint 的离线优先 Transformers 运行时。模型、processor
与重依赖都在首次 ``load``/``predict`` 时惰性加载；模块导入不触发 torch、
transformers 或权重读取。
"""

from __future__ import annotations

import json
import math
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from models.base import validate_local_model_asset
from models.images import read_normalized_image
from models.settings import SegFormerSettings


class SegFormerError(RuntimeError):
    """Base error for SegFormer loading or inference.
    SegFormer 加载或推理错误的基础类型。"""


class SegFormerMetadataError(SegFormerError):
    """Raised when checkpoint metadata is missing or inconsistent.
    checkpoint 元数据缺失或不一致时抛出。"""


class SegFormerDependencyError(SegFormerError, ImportError):
    """Raised when the optional SegFormer runtime is not installed.
    可选 SegFormer 运行依赖未安装时抛出。"""


class SegFormerLoadError(SegFormerError):
    """Raised when a local Transformers runtime cannot be constructed.
    本地 Transformers 运行时无法构造时抛出。"""


class SegFormerInferenceError(SegFormerError):
    """Raised when logits cannot produce a valid class mask.
    logits 无法生成合法类别 mask 时抛出。"""


class SegFormerDeviceError(SegFormerError):
    """Raised when the configured execution provider cannot be honored.
    配置的执行设备策略无法满足时抛出。"""


@dataclass(frozen=True)
class ClassInfo:
    """One checkpoint output channel and its authoritative label.
    一个 checkpoint 输出 channel 及其权威标签。"""

    class_id: int
    name: str


@dataclass(frozen=True)
class SegmentationResult:
    """Model-independent semantic mask returned by SegFormer inference.
    SegFormer 推理返回的模型无关语义 mask。"""

    mask: Any
    confidence_map: Any
    width: int
    height: int
    classes: tuple[ClassInfo, ...]
    logical_model_id: str
    model_revision: str | None
    weights_sha256: str
    device: str
    dtype: str
    cpu_fallback_used: bool

    @property
    def class_ids(self) -> tuple[int, ...]:
        """Class IDs present in the mask, in ascending order.
        mask 中出现的类别 ID，按升序排列。"""

        return tuple(item.class_id for item in self.classes)

    def class_name(self, class_id: int) -> str:
        """Resolve a present class ID to its checkpoint label.
        将 mask 中出现的类别 ID 解析为 checkpoint 标签。"""

        for item in self.classes:
            if item.class_id == class_id:
                return item.name
        raise KeyError(class_id)

    @property
    def class_id_map(self) -> Any:
        """Dense class-ID map aligned to the source image.
        与源图像对齐的稠密类别 ID 图。"""

        return self.mask

    @property
    def revision(self) -> str | None:
        """Public model revision alias."""

        return self.model_revision

    @property
    def sha256(self) -> str:
        """Verified digest of the loaded weight file."""

        return self.weights_sha256

    def trace_metadata(self) -> dict[str, object]:
        """Return path-free public identity and provider metadata.
        返回不含物理路径的公共身份与执行设备元数据。"""

        return {
            "logical_model_id": self.logical_model_id,
            "revision": self.revision,
            "sha256": self.sha256,
            "device": self.device,
            "cpu_fallback_used": self.cpu_fallback_used,
        }


@dataclass(frozen=True)
class _CheckpointMetadata:
    labels: tuple[str, ...]
    config: Mapping[str, Any]


RuntimeLoader = Callable[[SegFormerSettings], tuple[Any, Any, str]]
InferenceRunner = Callable[[Any, Any, Image.Image, str, int], Any]

_PLACEHOLDER_LABEL = re.compile(r"^LABEL_\d+$", re.IGNORECASE)
_SHARED_RUNTIMES: dict[tuple[object, ...], tuple[Any, Any, str, bool]] = {}
_SHARED_RUNTIME_LOCK = threading.Lock()


class SegFormerRuntime:
    """Load one verified local SegFormer model once and return class masks.

    The runtime owns checkpoint validation, processor/model loading, device
    placement, preprocessing, logits upsampling, and argmax. Agents never need
    to handle raw Transformers objects.
    运行时负责 checkpoint 校验、processor/model 加载、设备放置、预处理、
    logits 上采样与 argmax；Agent 无需处理 Transformers 原始对象。
    """

    def __init__(
        self,
        settings: SegFormerSettings,
        *,
        id_to_label: Mapping[int, str] | None = None,
        model: Any | None = None,
        processor: Any | None = None,
        loader: RuntimeLoader | None = None,
        inference_runner: InferenceRunner | None = None,
    ) -> None:
        if (model is None) != (processor is None):
            raise ValueError("model and processor must be supplied together")
        self.settings = settings
        self._external_labels = (
            _verified_external_labels(id_to_label) if id_to_label is not None else None
        )
        self._model = model
        self._processor = processor
        self._loader = loader
        self._inference_runner = inference_runner
        self._metadata: _CheckpointMetadata | None = None
        self._resolved_device: str | None = None
        self._cpu_fallback_used = False
        self._actual_weights_sha256: str | None = None
        self._load_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        """Whether validation and one successful load have completed.
        是否已完成校验并成功加载一次。"""

        return self._metadata is not None and self._resolved_device is not None

    @property
    def labels(self) -> tuple[str, ...]:
        """Return authoritative checkpoint labels, loading metadata if needed.
        返回 checkpoint 权威标签；必要时加载元数据。"""

        self.load()
        assert self._metadata is not None
        return self._metadata.labels

    def load(self) -> "SegFormerRuntime":
        """Validate and load exactly once; failed loads never poison state.
        校验并只加载一次；失败加载绝不污染状态。"""

        if self.loaded:
            return self
        with self._load_lock:
            if self.loaded:
                return self
            metadata = (
                _CheckpointMetadata(labels=self._external_labels, config={})
                if self._external_labels is not None
                else _load_checkpoint_metadata(self.settings)
            )
            weights_path = self.settings.model_path / self.settings.weights_filename
            actual_weights_sha256 = validate_local_model_asset(
                weights_path,
                expected_sha256=self.settings.weights_sha256,
            )
            if self._model is not None:
                model = self._model
                processor = self._processor
                resolved_device = self.settings.device
                cpu_fallback_used = False
            else:
                loader = self._loader or _load_transformers_runtime
                cache_key = _runtime_cache_key(
                    self.settings,
                    metadata.labels,
                    actual_weights_sha256,
                    loader,
                )
                with _SHARED_RUNTIME_LOCK:
                    cached = _SHARED_RUNTIMES.get(cache_key)
                    if cached is None:
                        try:
                            model, processor, resolved_device = loader(self.settings)
                        except SegFormerError:
                            raise
                        except Exception as error:
                            if self._external_labels is None:
                                raise
                            raise SegFormerLoadError(
                                "SegFormer runtime could not be loaded: "
                                f"{type(error).__name__}"
                            ) from None
                        cpu_fallback_used = (
                            resolved_device == "cpu" and self.settings.device != "cpu"
                        )
                        _validate_device_contract(
                            self.settings,
                            resolved_device,
                            cpu_fallback_used=cpu_fallback_used,
                        )
                        _validate_model_channel_count(model, len(metadata.labels))
                        cached = (
                            model,
                            processor,
                            resolved_device,
                            cpu_fallback_used,
                        )
                        _SHARED_RUNTIMES[cache_key] = cached
                model, processor, resolved_device, cpu_fallback_used = cached
            _validate_device_contract(
                self.settings,
                resolved_device,
                cpu_fallback_used=cpu_fallback_used,
            )
            _validate_model_channel_count(model, len(metadata.labels))
            # Assign state only after every validation succeeds.
            # 仅在全部校验成功后写入状态。
            self._model = model
            self._processor = processor
            self._metadata = metadata
            self._resolved_device = resolved_device
            self._cpu_fallback_used = cpu_fallback_used
            self._actual_weights_sha256 = actual_weights_sha256
        return self

    def predict(self, image: Image.Image | Path) -> SegmentationResult:
        """Run preprocessing -> model -> upsample -> argmax for one image.
        对一张图执行预处理 -> 模型 -> 上采样 -> argmax。"""

        self.load()
        assert self._model is not None
        assert self._processor is not None
        assert self._metadata is not None
        assert self._resolved_device is not None
        assert self._actual_weights_sha256 is not None
        normalized = (
            read_normalized_image(image)
            if isinstance(image, Path)
            else image.convert("RGB")
        )
        runner = self._inference_runner or _run_transformers_inference
        try:
            prediction = runner(
                self._model,
                self._processor,
                normalized,
                self._resolved_device,
                len(self._metadata.labels),
            )
            if isinstance(prediction, tuple) and len(prediction) == 2:
                mask, confidence_map = prediction
            else:
                mask = prediction
                confidence_map = _unit_confidence_map(mask)
            height, width, class_ids = _inspect_mask(mask)
            confidence_height, confidence_width = _inspect_confidence_map(confidence_map)
        except SegFormerError:
            raise
        except Exception as error:
            raise SegFormerInferenceError(
                f"SegFormer inference failed: {type(error).__name__}"
            ) from None
        if (width, height) != normalized.size:
            raise SegFormerInferenceError(
                "SegFormer mask dimensions differ from the source image"
            )
        if (confidence_width, confidence_height) != normalized.size:
            raise SegFormerInferenceError(
                "SegFormer confidence dimensions differ from the source image"
            )
        invalid = [
            class_id
            for class_id in class_ids
            if class_id < 0 or class_id >= len(self._metadata.labels)
        ]
        if invalid:
            raise SegFormerInferenceError(
                "SegFormer mask contains a class ID outside checkpoint metadata"
            )
        return SegmentationResult(
            mask=mask,
            confidence_map=confidence_map,
            width=width,
            height=height,
            classes=tuple(
                ClassInfo(class_id=class_id, name=self._metadata.labels[class_id])
                for class_id in class_ids
            ),
            logical_model_id=self.settings.logical_model_id,
            model_revision=self.settings.revision,
            weights_sha256=self._actual_weights_sha256,
            device=self._resolved_device,
            dtype=self.settings.dtype,
            cpu_fallback_used=self._cpu_fallback_used,
        )


def _verified_external_labels(id_to_label: Mapping[int, str]) -> tuple[str, ...]:
    """Validate a caller-supplied authoritative, contiguous class map.
    校验调用方提供的权威、连续类别映射。"""

    if not isinstance(id_to_label, Mapping) or not id_to_label:
        raise SegFormerMetadataError("verified class map must be non-empty")
    if any(
        not isinstance(class_id, int) or isinstance(class_id, bool)
        for class_id in id_to_label
    ):
        raise SegFormerMetadataError("verified class map IDs must be integers")
    expected_ids = set(range(len(id_to_label)))
    if set(id_to_label) != expected_ids:
        raise SegFormerMetadataError("verified class map IDs must be contiguous from zero")
    if any(not isinstance(label, str) for label in id_to_label.values()):
        raise SegFormerMetadataError("verified class labels must be strings")
    labels = tuple(
        id_to_label[class_id].strip() for class_id in range(len(id_to_label))
    )
    if any(not label for label in labels) or len(set(labels)) != len(labels):
        raise SegFormerMetadataError("verified class labels must be non-empty and unique")
    if any(_PLACEHOLDER_LABEL.fullmatch(label) for label in labels):
        raise SegFormerMetadataError("placeholder labels are not a verified class map")
    return labels


def _runtime_cache_key(
    settings: SegFormerSettings,
    labels: tuple[str, ...],
    weights_sha256: str,
    loader: RuntimeLoader,
) -> tuple[object, ...]:
    """Build a private path-free key for reusable loaded objects."""

    return (
        settings.logical_model_id,
        settings.revision,
        weights_sha256,
        settings.device,
        settings.dtype,
        settings.require_cuda,
        settings.allow_cpu_fallback,
        labels,
        loader,
    )


def _validate_model_channel_count(model: Any, label_count: int) -> None:
    """Fail closed when model metadata disagrees with the verified map."""

    model_num_labels = getattr(getattr(model, "config", None), "num_labels", None)
    if isinstance(model_num_labels, int) and model_num_labels != label_count:
        raise SegFormerMetadataError(
            "loaded SegFormer output channels differ from verified class map"
        )


def _validate_device_contract(
    settings: SegFormerSettings,
    resolved_device: str,
    *,
    cpu_fallback_used: bool,
) -> None:
    """Reject provider results that weaken the declared device policy."""

    if not isinstance(resolved_device, str) or not resolved_device:
        raise SegFormerDeviceError("SegFormer loader returned an invalid device")
    if settings.require_cuda and not resolved_device.startswith("cuda"):
        raise SegFormerDeviceError("SegFormer requires CUDA but CUDA is unavailable")
    if settings.device == "cpu" and resolved_device != "cpu":
        raise SegFormerDeviceError("SegFormer loader violated the configured CPU device")
    if settings.device.startswith("cuda") and not resolved_device.startswith("cuda"):
        if not (settings.allow_cpu_fallback and cpu_fallback_used):
            raise SegFormerDeviceError("configured SegFormer CUDA device is unavailable")
    if settings.device == "auto" and resolved_device == "cpu":
        if not (settings.allow_cpu_fallback and cpu_fallback_used):
            raise SegFormerDeviceError("SegFormer CPU fallback is not enabled")
    if cpu_fallback_used and not settings.allow_cpu_fallback:
        raise SegFormerDeviceError("SegFormer CPU fallback is not enabled")


def _load_checkpoint_metadata(settings: SegFormerSettings) -> _CheckpointMetadata:
    """Load and validate config/classes without importing Transformers.
    在不导入 Transformers 的情况下加载并校验 config/classes。"""

    if not settings.model_path.is_dir():
        raise SegFormerMetadataError("SegFormer checkpoint directory is missing")
    config = _read_json_object(settings.model_path / "config.json", "config.json")
    if config.get("model_type") != "segformer":
        raise SegFormerMetadataError("checkpoint config model_type is not segformer")
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or (
        "SegformerForSemanticSegmentation" not in architectures
    ):
        raise SegFormerMetadataError(
            "checkpoint config does not declare SegformerForSemanticSegmentation"
        )
    config_labels = _indexed_labels(config.get("id2label"), where="config.id2label")
    labels = config_labels
    if settings.classes_filename is not None:
        classes = _read_json_object(
            settings.model_path / settings.classes_filename,
            settings.classes_filename,
        )
        labels = _indexed_labels(classes.get("id2name"), where="classes.id2name")
        if classes.get("num_classes") != len(labels):
            raise SegFormerMetadataError("classes num_classes differs from id2name")
        name2id = classes.get("name2id")
        if not isinstance(name2id, dict) or {
            str(name): class_id for class_id, name in enumerate(labels)
        } != name2id:
            raise SegFormerMetadataError("classes name2id differs from id2name")
    if len(labels) != len(config_labels):
        raise SegFormerMetadataError(
            "checkpoint class mapping differs from configured output channels"
        )
    return _CheckpointMetadata(labels=labels, config=config)


def _read_json_object(path: Path, display_name: str) -> dict[str, Any]:
    """Read one UTF-8 JSON object with a stable, path-free error.
    读取一个 UTF-8 JSON 对象，并提供稳定且不含主机路径的错误。"""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SegFormerMetadataError(
            f"invalid or missing SegFormer metadata {display_name}: "
            f"{type(error).__name__}"
        ) from error
    if not isinstance(value, dict):
        raise SegFormerMetadataError(f"SegFormer metadata {display_name} must be an object")
    return value


def _indexed_labels(value: Any, *, where: str) -> tuple[str, ...]:
    """Require a contiguous string-keyed 0..N-1 label mapping.
    要求键为字符串且从 0 到 N-1 连续的标签映射。"""

    if not isinstance(value, dict) or not value:
        raise SegFormerMetadataError(f"{where} must be a non-empty object")
    expected = {str(index) for index in range(len(value))}
    if set(value) != expected:
        raise SegFormerMetadataError(f"{where} class IDs must be contiguous from zero")
    labels = tuple(str(value[str(index)]).strip() for index in range(len(value)))
    if any(not label for label in labels) or len(set(labels)) != len(labels):
        raise SegFormerMetadataError(f"{where} labels must be non-empty and unique")
    return labels


def _load_transformers_runtime(settings: SegFormerSettings) -> tuple[Any, Any, str]:
    """Load the local checkpoint and processor without implicit download.
    在不隐式下载的情况下加载本地 checkpoint 与 processor。"""

    try:
        import torch
        from transformers import (
            AutoImageProcessor,
            AutoModelForSemanticSegmentation,
        )
    except ImportError as error:
        raise SegFormerDependencyError(
            "SegFormer runtime requires torch and transformers"
        ) from error
    device = _resolve_device(torch, settings)
    model_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "revision": settings.revision,
    }
    if settings.dtype != "auto":
        model_kwargs["dtype"] = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[settings.dtype]
    try:
        model = AutoModelForSemanticSegmentation.from_pretrained(
            str(settings.model_path),
            **model_kwargs,
        )
        processor = AutoImageProcessor.from_pretrained(
            str(settings.processor_path or settings.model_path),
            local_files_only=True,
            revision=settings.revision,
        )
        model.to(device)
        model.eval()
    except SegFormerError:
        raise
    except Exception as error:
        raise SegFormerLoadError(
            f"SegFormer runtime could not be loaded: {type(error).__name__}"
        ) from None
    return model, processor, device


def _resolve_device(torch: Any, settings: SegFormerSettings) -> str:
    """Resolve the explicit CUDA/CPU policy without silent fallback."""

    requested = settings.device
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda" if requested == "auto" else requested
    if settings.require_cuda:
        raise SegFormerDeviceError("SegFormer requires CUDA but CUDA is unavailable")
    if settings.allow_cpu_fallback:
        return "cpu"
    raise SegFormerDeviceError("configured SegFormer CUDA device is unavailable")


def _run_transformers_inference(
    model: Any,
    processor: Any,
    image: Image.Image,
    device: str,
    label_count: int,
) -> Any:
    """Execute the standard local SegFormer inference path.
    执行标准本地 SegFormer 推理路径。"""

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as error:
        raise SegFormerDependencyError("SegFormer inference requires torch") from error
    inputs = processor(images=image, return_tensors="pt")
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    elif isinstance(inputs, Mapping):
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
    with torch.inference_mode():
        outputs = model(**inputs)
        logits = getattr(outputs, "logits", None)
        shape = getattr(logits, "shape", ())
        if len(shape) != 4 or int(shape[0]) != 1 or int(shape[1]) != label_count:
            raise SegFormerInferenceError(
                "SegFormer logits do not match the checkpoint class contract"
            )
        upsampled = functional.interpolate(
            logits,
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )
        probabilities = torch.softmax(upsampled, dim=1)
        confidence, class_ids = probabilities.max(dim=1)
        return (
            class_ids[0].to("cpu").numpy(),
            confidence[0].to("cpu").numpy(),
        )


def _inspect_mask(mask: Any) -> tuple[int, int, tuple[int, ...]]:
    """Return 2-D mask dimensions and unique integer IDs without importing NumPy.
    在不导入 NumPy 的情况下返回二维 mask 尺寸与唯一整数 ID。"""

    shape = getattr(mask, "shape", None)
    if shape is not None:
        if len(shape) != 2:
            raise SegFormerInferenceError("SegFormer mask must be two-dimensional")
        height, width = int(shape[0]), int(shape[1])
        flattened = mask.reshape(-1)
        values = flattened.tolist() if hasattr(flattened, "tolist") else list(flattened)
    else:
        if not isinstance(mask, (list, tuple)) or not mask:
            raise SegFormerInferenceError("SegFormer mask must be a non-empty 2-D array")
        height = len(mask)
        width = len(mask[0]) if isinstance(mask[0], (list, tuple)) else 0
        if width == 0 or any(
            not isinstance(row, (list, tuple)) or len(row) != width for row in mask
        ):
            raise SegFormerInferenceError("SegFormer mask rows must have equal width")
        values = [value for row in mask for value in row]
    class_ids: set[int] = set()
    for value in values:
        integer = int(value)
        if integer != value:
            raise SegFormerInferenceError("SegFormer mask contains a non-integer class ID")
        class_ids.add(integer)
    return height, width, tuple(sorted(class_ids))


def _unit_confidence_map(mask: Any) -> list[list[float]]:
    """Create a deterministic compatibility confidence map for injected runners."""

    height, width, _ = _inspect_mask(mask)
    return [[1.0 for _ in range(width)] for _ in range(height)]


def _inspect_confidence_map(confidence_map: Any) -> tuple[int, int]:
    """Validate a finite two-dimensional probability map."""

    shape = getattr(confidence_map, "shape", None)
    if shape is not None:
        if len(shape) != 2:
            raise SegFormerInferenceError(
                "SegFormer confidence map must be two-dimensional"
            )
        height, width = int(shape[0]), int(shape[1])
        flattened = confidence_map.reshape(-1)
        values = flattened.tolist() if hasattr(flattened, "tolist") else list(flattened)
    else:
        if not isinstance(confidence_map, (list, tuple)) or not confidence_map:
            raise SegFormerInferenceError(
                "SegFormer confidence map must be a non-empty 2-D array"
            )
        height = len(confidence_map)
        width = (
            len(confidence_map[0])
            if isinstance(confidence_map[0], (list, tuple))
            else 0
        )
        if width == 0 or any(
            not isinstance(row, (list, tuple)) or len(row) != width
            for row in confidence_map
        ):
            raise SegFormerInferenceError(
                "SegFormer confidence map rows must have equal width"
            )
        values = [value for row in confidence_map for value in row]
    for value in values:
        probability = float(value)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise SegFormerInferenceError(
                "SegFormer confidence map contains an invalid probability"
            )
    return height, width


class SegFormerTransformersClient(SegFormerRuntime):
    """Strict public client requiring an externally verified class map."""

    def __init__(
        self,
        settings: SegFormerSettings,
        id_to_label: Mapping[int, str],
        *,
        loader: RuntimeLoader | None = None,
        inference_runner: InferenceRunner | None = None,
    ) -> None:
        super().__init__(
            settings,
            id_to_label=id_to_label,
            loader=loader,
            inference_runner=inference_runner,
        )


__all__ = [
    "ClassInfo",
    "SegFormerDependencyError",
    "SegFormerDeviceError",
    "SegFormerError",
    "SegFormerInferenceError",
    "SegFormerLoadError",
    "SegFormerMetadataError",
    "SegFormerRuntime",
    "SegFormerTransformersClient",
    "SegmentationResult",
]
