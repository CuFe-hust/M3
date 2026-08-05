"""Explicit canonical model registry. / 显式统一模型注册表。"""
from typing import Any, Protocol
from data.schema import CanonicalPrediction, CanonicalSample
from models.internvl35 import InternVL35Baseline, InternVL35Settings
from models.minicpmv46 import MiniCPMV46Baseline, MiniCPMV46Settings
from models.ovis25 import Ovis25Baseline, Ovis25Settings
from models.qwen35_4b import Qwen35FourBBaseline, Qwen35FourBSettings
from models.qwen35_9b import Qwen35NineBBaseline, Qwen35NineBSettings
from models.qwen3vl import Qwen3VLBaseline, Qwen3VLSettings

class CanonicalVisionModel(Protocol):
    """Predict canonical records. / 生成统一预测记录。"""
    def predict(self, sample: CanonicalSample) -> CanonicalPrediction: ...

SUPPORTED_MODEL_TYPES = ("qwen3vl", "qwen35_4b", "qwen35_9b", "internvl35", "minicpmv46", "ovis25")

def _value(config: dict[str, Any], key: str, default: Any, expected: type | tuple[type, ...], *, optional: bool = False) -> Any:
    value = config.get(key, default)
    if optional and value is None: return None
    if expected is float and isinstance(value, int) and not isinstance(value, bool): return float(value)
    if not isinstance(value, expected) or isinstance(value, bool) and expected is not bool: raise TypeError(f"model.{key} must be {getattr(expected, '__name__', expected)}, got {type(value).__name__}.")
    return value

def load_model_from_config(model_config: dict[str, Any]) -> CanonicalVisionModel:
    if not isinstance(model_config, dict): raise TypeError("model configuration must be an object.")
    configured_id = _value(model_config, "id", "Qwen/Qwen3-VL-4B-Instruct", str)
    model_type = _value(model_config, "type", _legacy_model_type(configured_id), str)
    if model_type not in SUPPORTED_MODEL_TYPES: raise ValueError(f"Unsupported model type {model_type!r}; expected one of {', '.join(SUPPORTED_MODEL_TYPES)}.")
    default_ids = {"qwen3vl": "Qwen/Qwen3-VL-4B-Instruct", "qwen35_4b": "Qwen/Qwen3.5-4B", "qwen35_9b": "Qwen/Qwen3.5-9B", "internvl35": "OpenGVLab/InternVL3_5-8B", "minicpmv46": "openbmb/MiniCPM-V-4.6", "ovis25": "ATH-MaaS/Ovis2.5-2B"}
    model_id = _value(model_config, "id", default_ids[model_type], str)
    common = dict(model_id=model_id, dtype=_value(model_config, "dtype", "auto", str), device_map=_value(model_config, "device_map", "auto", str), max_new_tokens=_value(model_config, "max_new_tokens", 256, int), local_files_only=_value(model_config, "local_files_only", False, bool))
    scale_default = 100.0 if model_type in {"minicpmv46", "ovis25"} else 1000.0
    common["grounding_source_scale"] = _value(model_config, "grounding_source_scale", scale_default, float)
    if model_type == "qwen3vl": return Qwen3VLBaseline(Qwen3VLSettings(**common, adapter_id=_value(model_config, "adapter_id", None, str, optional=True), adapter_revision=_value(model_config, "adapter_revision", None, str, optional=True), merge_adapter=_value(model_config, "merge_adapter", False, bool), min_pixels=_value(model_config, "min_pixels", None, int, optional=True), max_pixels=_value(model_config, "max_pixels", None, int, optional=True)))
    if model_type == "qwen35_4b": return Qwen35FourBBaseline(Qwen35FourBSettings(**common, min_pixels=_value(model_config, "min_pixels", None, int, optional=True), max_pixels=_value(model_config, "max_pixels", None, int, optional=True)))
    if model_type == "qwen35_9b": return Qwen35NineBBaseline(Qwen35NineBSettings(**common, min_pixels=_value(model_config, "min_pixels", None, int, optional=True), max_pixels=_value(model_config, "max_pixels", None, int, optional=True)))
    if model_type == "internvl35": return InternVL35Baseline(InternVL35Settings(**common, adapter_id=_value(model_config, "adapter_id", None, str, optional=True), adapter_revision=_value(model_config, "adapter_revision", None, str, optional=True), merge_adapter=_value(model_config, "merge_adapter", False, bool), input_size=_value(model_config, "input_size", 448, int), max_num_tiles=_value(model_config, "max_num_tiles", 12, int), use_thumbnail=_value(model_config, "use_thumbnail", True, bool), low_cpu_mem_usage=_value(model_config, "low_cpu_mem_usage", True, bool), use_flash_attn=_value(model_config, "use_flash_attn", False, bool)))
    if model_type == "minicpmv46": return MiniCPMV46Baseline(MiniCPMV46Settings(**common, downsample_mode=_value(model_config, "downsample_mode", "16x", str), max_slice_nums=_value(model_config, "max_slice_nums", 36, int)))
    return Ovis25Baseline(Ovis25Settings(**common, min_pixels=_value(model_config, "min_pixels", 448 * 448, int), max_pixels=_value(model_config, "max_pixels", 1792 * 1792, int)))

def _legacy_model_type(model_id: str) -> str:
    return {"Qwen/Qwen3.5-4B": "qwen35_4b", "Qwen/Qwen3.5-9B": "qwen35_9b"}.get(model_id, "qwen3vl")
