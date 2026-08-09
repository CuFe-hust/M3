"""Model-layer contracts, caching, image tools, and settings declarations.
模型层协议、缓存、图像工具与配置声明。

Importing this package must not import transformers or torch.
导入本包不得触发 transformers 或 torch 导入。
"""

from models.base import (
    CacheIdentifiedClient,
    ModelCacheIdentity,
    ModelT,
    RequestMeta,
    SemanticSegmentationClient,
    SemanticSegmentationOutput,
    VisionLanguageClient,
    build_request_hash,
    sanitize_messages,
)
from models.cache import CacheEntry, CacheWriteError, JsonResponseCache, ModelCacheError
from models.images import (
    UnsupportedImageFormatError,
    detect_image_mime,
    image_sha256,
    image_to_data_url,
    read_normalized_image,
)
from models.segformer_transformers import SegFormerTransformersClient
from models.settings import DeepSeekSettings, ModelSettings, QwenSettings, SegFormerSettings

__all__ = [
    "CacheEntry",
    "CacheIdentifiedClient",
    "CacheWriteError",
    "DeepSeekSettings",
    "JsonResponseCache",
    "ModelCacheError",
    "ModelCacheIdentity",
    "ModelSettings",
    "ModelT",
    "QwenSettings",
    "RequestMeta",
    "SegFormerSettings",
    "SegFormerTransformersClient",
    "SemanticSegmentationClient",
    "SemanticSegmentationOutput",
    "UnsupportedImageFormatError",
    "VisionLanguageClient",
    "build_request_hash",
    "detect_image_mime",
    "image_sha256",
    "image_to_data_url",
    "read_normalized_image",
    "sanitize_messages",
]
