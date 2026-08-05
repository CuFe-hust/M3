"""Dedicated Qwen3.5-4B wrapper. / 独立 Qwen3.5-4B 封装。"""
from dataclasses import dataclass
from data.schema import CanonicalPrediction, CanonicalSample
from models.qwen35_common import Qwen35RuntimeConfig, load_qwen35_runtime, predict_qwen35

@dataclass
class Qwen35FourBSettings:
    model_id: str = "Qwen/Qwen3.5-4B"
    dtype: str = "auto"
    device_map: str = "auto"
    max_new_tokens: int = 256
    min_pixels: int | None = None
    max_pixels: int | None = None
    local_files_only: bool = False
    grounding_source_scale: float = 1000.0

class Qwen35FourBBaseline:
    """Run the dedicated Qwen3.5-4B wrapper. / 运行独立 Qwen3.5-4B 封装。"""
    def __init__(self, settings: Qwen35FourBSettings) -> None:
        self.settings = settings
        self.runtime_config = Qwen35RuntimeConfig(settings.model_id, "4b", settings.dtype, settings.device_map, settings.max_new_tokens, settings.min_pixels, settings.max_pixels, settings.local_files_only, settings.grounding_source_scale)
        self.model, self.processor = load_qwen35_runtime(self.runtime_config)
    def predict(self, sample: CanonicalSample) -> CanonicalPrediction:
        return predict_qwen35(model=self.model, processor=self.processor, sample=sample, config=self.runtime_config, public_model_type="qwen35_4b")
