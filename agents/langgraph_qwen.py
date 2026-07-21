"""Fixed LangGraph workflow around the local Qwen3-VL baseline.
围绕本地 Qwen3-VL 基线构建的固定 LangGraph 工作流。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from data.schema import CanonicalPrediction, CanonicalSample
from models.qwen3vl import Qwen3VLBaseline


SaveCallback = Callable[[CanonicalSample, CanonicalPrediction, float], None]


class AgentState(TypedDict, total=False):
    """State passed through the fixed four-node workflow.
    在固定四节点工作流中传递的状态。
    """

    sample: CanonicalSample
    prediction: CanonicalPrediction
    model_elapsed_seconds: float
    save_callback: SaveCallback


class LangGraphQwenAgent:
    """Run read, model-call, validation, and save nodes in order.
    依次运行读取、模型调用、校验和保存节点。
    """

    def __init__(self, model: Qwen3VLBaseline) -> None:
        self.model = model
        builder = StateGraph(AgentState)
        builder.add_node("read_sample", self._read_sample)
        builder.add_node("call_qwen", self._call_qwen)
        builder.add_node("validate_prediction", self._validate_prediction)
        builder.add_node("save_result", self._save_result)
        builder.add_edge(START, "read_sample")
        builder.add_edge("read_sample", "call_qwen")
        builder.add_edge("call_qwen", "validate_prediction")
        builder.add_edge("validate_prediction", "save_result")
        builder.add_edge("save_result", END)
        self.graph = builder.compile()

    def run(self, sample: CanonicalSample, save_callback: SaveCallback) -> CanonicalPrediction:
        """Run one sample and persist it inside the graph's save node.
        运行一条样本，并在图的保存节点中持久化结果。
        """

        result = self.graph.invoke({"sample": sample, "save_callback": save_callback})
        return result["prediction"]

    @staticmethod
    def _read_sample(state: AgentState) -> AgentState:
        state["sample"].validate()
        return {}

    def _call_qwen(self, state: AgentState) -> AgentState:
        started = time.perf_counter()
        prediction = self.model.predict(state["sample"])
        elapsed = time.perf_counter() - started
        return {"prediction": prediction, "model_elapsed_seconds": elapsed}

    @staticmethod
    def _validate_prediction(state: AgentState) -> AgentState:
        sample = state["sample"]
        prediction = state["prediction"]
        prediction.validate()
        if prediction.id != sample.id:
            raise ValueError("Prediction id does not match the sample id.")
        if prediction.task_type != sample.task_type:
            raise ValueError("Prediction task type does not match the sample task type.")
        return {}

    @staticmethod
    def _save_result(state: AgentState) -> AgentState:
        state["save_callback"](
            state["sample"], state["prediction"], state["model_elapsed_seconds"]
        )
        return {}
