"""
Minimal Agent Pipeline using LangGraph.
基于 LangGraph 的最小 Agent 流程：读取样本 → 调用模型 → 校验 → 保存。

Usage:
    python agent_pipeline.py --config config/local.server.json --dataset levir_cc --limit 2
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

# ---- Project imports ----
sys.path.insert(0, str(Path(__file__).parent))
from data.loaders import load_samples
from data.schema import CanonicalSample, CanonicalPrediction
from models.qwen3vl import Qwen3VLBaseline, Qwen3VLSettings


# ---------- State definition / 状态定义 ----------

class PipelineState(TypedDict):
    """State passed between graph nodes. / 图节点间传递的状态。"""
    samples: list[CanonicalSample]          # all samples to process
    current_sample: CanonicalSample | None  # sample being processed
    current_prediction: CanonicalPrediction | None
    results: list[dict]                     # accumulated JSON-serialized results
    errors: list[dict]                      # accumulated error records
    output_path: str                        # where to save JSONL
    idx: int                                # current sample index


# ---------- Graph nodes / 图节点 ----------

def node_read(state: PipelineState) -> dict:
    """Node 1: Read next sample. / 读取下一条样本。"""
    idx = state["idx"]
    if idx >= len(state["samples"]):
        raise StopIteration("No more samples.")
    sample = state["samples"][idx]
    print(f"[Agent] 样本 {sample.id} ({idx+1}/{len(state['samples'])})")
    return {"current_sample": sample}


def node_predict(state: PipelineState, model: Qwen3VLBaseline) -> dict:
    """Node 2: Call the existing Qwen baseline predictor.
    节点 2：调用已有的 Qwen 基线预测器。"""
    sample = state["current_sample"]
    prediction = model.predict(sample)
    return {"current_prediction": prediction}


def node_validate(state: PipelineState) -> dict:
    """Node 3: Validate the prediction. / 节点 3：校验预测结果。"""
    pred = state["current_prediction"]
    try:
        pred.validate()
    except Exception as e:
        err = {"sample_id": state["current_sample"].id, "node": "validate", "error": str(e)}
        return {"errors": state["errors"] + [err]}
    return {}


def node_save(state: PipelineState) -> dict:
    """Node 4: Serialize and accumulate result. / 节点 4：序列化并累计结果。"""
    sample = state["current_sample"]
    pred = state["current_prediction"]
    record = {
        "sample": sample.serializable(),
        "prediction": pred.serializable(),
    }
    new_results = state["results"] + [record]
    idx = state["idx"] + 1
    return {"results": new_results, "idx": idx}


def node_write(state: PipelineState) -> dict:
    """Final node: Write all results to JSONL. / 最终节点：写入 JSONL 文件。"""
    path = Path(state["output_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in state["results"]:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[Agent] 已保存 {len(state['results'])} 条到 {path}")
    if state["errors"]:
        print(f"[Agent] 失败 {len(state['errors'])} 条:")
        for e in state["errors"]:
            print(f"      样本 {e['sample_id']}: {e['error']}")
    return {}


# ---------- Conditional edge / 条件边 ----------

def should_continue(state: PipelineState) -> str:
    """After saving, decide: next sample or finish. / 判断继续还是结束。"""
    if state["idx"] >= len(state["samples"]):
        return "write"
    return "read"


# ---------- Build graph / 构建图 ----------

def build_agent(model: Qwen3VLBaseline) -> StateGraph:
    """Build and compile the LangGraph agent.
    构建并编译 LangGraph Agent。"""
    builder = StateGraph(PipelineState)

    builder.add_node("read", node_read)
    builder.add_node("predict", lambda s: node_predict(s, model))
    builder.add_node("validate", node_validate)
    builder.add_node("save", node_save)
    builder.add_node("write", node_write)

    builder.add_edge("read", "predict")
    builder.add_edge("predict", "validate")
    builder.add_edge("validate", "save")
    builder.add_conditional_edges("save", should_continue, {"read": "read", "write": "write"})
    builder.add_edge("write", END)

    builder.set_entry_point("read")
    return builder.compile()


# ---------- Main entry / 主入口 ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Load config / 加载配置
    with open(args.config) as f:
        config = json.load(f)
    data_root = Path(config["paths"]["data_root"])
    output_root = Path(config["paths"]["output_root"])
    model_cfg = config["model"]

    # Load model / 加载模型
    print("[Agent] 加载模型...")
    settings = Qwen3VLSettings(
        model_id=model_cfg["id"],
        dtype=model_cfg.get("dtype", "auto"),
        max_new_tokens=model_cfg.get("max_new_tokens", 256),
        device_map=model_cfg.get("device_map", "auto"),
    )
    model = Qwen3VLBaseline(settings)

    # Load samples / 加载样本
    print(f"[Agent] 加载数据集: {args.dataset}")
    samples = list(load_samples(args.dataset, data_root))
    if args.limit:
        samples = samples[:args.limit]
    print(f"[Agent] 共 {len(samples)} 条")

    # Prepare output path / 准备输出路径
    out_name = args.output or f"{args.dataset}_agent.jsonl"
    output_path = str(output_root / out_name)

    # Build & run graph / 构建并运行图
    agent = build_agent(model)
    initial: PipelineState = {
        "samples": samples,
        "current_sample": None,
        "current_prediction": None,
        "results": [],
        "errors": [],
        "output_path": output_path,
        "idx": 0,
    }
    final = agent.invoke(initial)
    print("[Agent] 流程完成。")
