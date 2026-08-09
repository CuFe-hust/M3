"""Offline-first Transformers runtime for local SegFormer checkpoints.

本地 SegFormer checkpoint 的离线优先 Transformers 运行时。模型、processor
与重依赖都在首次 ``load``/``predict`` 时惰性加载；模块导入不触发 torch、
transformers 或权重读取。
"""

from __future__ import annotations

import json
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


class SegFormerInferenceError(SegFormerError):
    """Raised when logits cannot produce a valid class mask.
    logits 无法生成合法类别 mask 时抛出。"""


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
    width: int
    height: int
    classes: tuple[ClassInfo, ...]
    logical_model_id: str
    device: str
    dtype: str

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


@dataclass(frozen=True)
class _CheckpointMetadata:
    labels: tuple[str, ...]
    config: Mapping[str, Any]


RuntimeLoader = Callable[[SegFormerSettings], tuple[Any, Any, str]]
InferenceRunner = Callable[[Any, Any, Image.Image, str, int], Any]


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
        model: Any | None = None,
        processor: Any | None = None,
        loader: RuntimeLoader | None = None,
        inference_runner: InferenceRunner | None = None,
    ) -> None:
        if (model is None) != (processor is None):
            raise ValueError("model and processor must be supplied together")
        self.settings = settings
        self._model = model
        self._processor = processor
        self._loader = loader
        self._inference_runner = inference_runner
        self._metadata: _CheckpointMetadata | None = None
        self._resolved_device: str | None = None
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
            metadata = _load_checkpoint_metadata(self.settings)
            weights_path = self.settings.model_path / self.settings.weights_filename
            validate_local_model_asset(
                weights_path,
                expected_sha256=self.settings.weights_sha256,
            )
            if self._model is not None:
                model = self._model
                processor = self._processor
                resolved_device = self.settings.device
            else:
                loader = self._loader or _load_transformers_runtime
                model, processor, resolved_device = loader(self.settings)
            model_num_labels = getattr(getattr(model, "config", None), "num_labels", None)
            if isinstance(model_num_labels, int) and model_num_labels != len(metadata.labels):
                raise SegFormerMetadataError(
                    "loaded SegFormer output channels differ from checkpoint metadata"
                )
            # Assign state only after every validation succeeds.
            # 仅在全部校验成功后写入状态。
            self._model = model
            self._processor = processor
            self._metadata = metadata
            self._resolved_device = resolved_device
        return self

    def predict(self, image: Image.Image | Path) -> SegmentationResult:
        """Run preprocessing -> model -> upsample -> argmax for one image.
        对一张图执行预处理 -> 模型 -> 上采样 -> argmax。"""

        self.load()
        assert self._model is not None
        assert self._processor is not None
        assert self._metadata is not None
        assert self._resolved_device is not None
        normalized = (
            read_normalized_image(image)
            if isinstance(image, Path)
            else image.convert("RGB")
        )
        runner = self._inference_runner or _run_transformers_inference
        try:
            mask = runner(
                self._model,
                self._processor,
                normalized,
                self._resolved_device,
                len(self._metadata.labels),
            )
            height, width, class_ids = _inspect_mask(mask)
        except SegFormerError:
            raise
        except Exception as error:
            raise SegFormerInferenceError(
                f"SegFormer inference failed: {type(error).__name__}"
            ) from error
        if (width, height) != normalized.size:
            raise SegFormerInferenceError(
                "SegFormer mask dimensions differ from the source image"
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
            width=width,
            height=height,
            classes=tuple(
                ClassInfo(class_id=class_id, name=self._metadata.labels[class_id])
                for class_id in class_ids
            ),
            logical_model_id=self.settings.logical_model_id,
            device=self._resolved_device,
            dtype=self.settings.dtype,
        )


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
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )
    except ImportError as error:
        raise SegFormerDependencyError(
            "SegFormer runtime requires torch and transformers"
        ) from error
    device = _resolve_device(torch, settings.device)
    model_kwargs: dict[str, Any] = {
        "local_files_only": not settings.allow_download,
        "revision": settings.revision,
    }
    if settings.dtype != "auto":
        model_kwargs["dtype"] = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[settings.dtype]
    model = SegformerForSemanticSegmentation.from_pretrained(
        str(settings.model_path),
        **model_kwargs,
    )
    processor_source = settings.processor_path
    if processor_source is not None or (
        settings.model_path / "preprocessor_config.json"
    ).is_file():
        processor = SegformerImageProcessor.from_pretrained(
            str(processor_source or settings.model_path),
            local_files_only=not settings.allow_download,
            revision=settings.revision,
        )
    else:
        # These migrated fine-tuned directories contain config + weights only;
        # use the library's deterministic SegFormer image processor defaults.
        # 迁移的微调目录只有 config + 权重；使用库内确定性的 SegFormer
        # 图像 processor 默认值。
        processor = SegformerImageProcessor()
    model.to(device)
    model.eval()
    return model, processor, device


def _resolve_device(torch: Any, requested: str) -> str:
    """Resolve auto to CUDA when available, otherwise CPU.
    auto 在 CUDA 可用时解析为 CUDA，否则解析为 CPU。"""

    if requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise SegFormerError("configured SegFormer CUDA device is unavailable")
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


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
        return upsampled.argmax(dim=1)[0].to("cpu").numpy()


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
