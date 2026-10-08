"""Replace RealModel with your own VLM. No model framework is required by the demo.

部署修改（挑战杯评测桥接）：RealModel 解析到 m3_bridge 的 M3 实现；
同时向 HTTP 外壳暴露稳定的 AdapterError（任何模式下可用）。
"""

import os
import sys
from pathlib import Path
from typing import Any

# Deployment glue: make the m3_bridge package (next to this deployment's
# repository checkout) importable in every mode, including demo/test runs.
# 部署胶水：让部署根目录下的 m3_bridge 在任何模式（含 demo/测试）下可导入。
_EVAL_ROOT = Path(__file__).resolve().parent.parent
if str(_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_ROOT))
try:
    from m3_bridge.errors import AdapterError
except ImportError:  # standalone copies without the bridge deployment
    class AdapterError(Exception):
        """Stable error carrying an HTTP status and a code.
        携带 HTTP 状态与错误码的稳定错误。"""

        def __init__(self, status: int, code: str) -> None:
            super().__init__(code)
            self.status = status
            self.code = code


class DemoModel:
    """Protocol-only placeholder: it does not inspect images or answer questions."""

    def predict(self, request: dict[str, Any], image_paths: list[Path]) -> str | list[int]:
        constraint = request["response_constraint"]
        kind = constraint["type"]
        if kind == "bbox":
            width, height = request["image_width"], request["image_height"]
            return [0, 0, width, height]
        if kind in {"enum", "single_choice"}:
            return constraint["values"][0]
        if kind == "integer":
            return "0"
        return "demo"


class RealModel:
    """Team-owned adapter. Load weights ONCE in __init__, infer for each predict call.
    队伍适配器：启动时恰好构建一次 M3 运行时，逐题推理。"""

    def __init__(self) -> None:
        from m3_bridge.real_model import M3RealModel

        self._impl = M3RealModel()

    def predict(self, request: dict[str, Any], image_paths: list[Path]) -> str | list[float]:
        # image_paths are verified local files, in the same order as request["images"].
        # 已校验的本地图片路径，顺序与请求一致。
        return self._impl.predict(request, image_paths)


def build_model() -> Any:
    mode = os.environ.get("MODEL_MODE", "demo")
    if mode == "demo":
        return DemoModel()
    if mode == "real":
        return RealModel()
    if mode == "qwen":
        from qwen_adapter import QwenModelAdapter

        return QwenModelAdapter()
    raise ValueError("MODEL_MODE must be demo, qwen, or real")
