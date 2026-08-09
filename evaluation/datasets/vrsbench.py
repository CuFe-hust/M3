"""VRSBench dataset-specific official evaluation seam.

VRSBench 数据集专属官方评估 seam（已批准的未来路径）。只处理数据集特定
官方评估/导出归一化：答案归一化、当前记录 → 官方评估器输入映射、封闭词汇
元数据。绝不选择 Agent、绝不调用模型、绝不修改任务、绝不重复通用指标。
问题子类型分类与任务归一化保留在 data.adapters.vrsbench.task_normalizer
（数据层权威；本模块不重复其判定逻辑，只声明评估侧使用的封闭词汇元数据）。
"""

from __future__ import annotations

from typing import Any, Sequence

# Closed-vocabulary metadata used by official answer normalization. This is
# evaluation-side metadata only; question-subtype classification stays in the
# data layer normalizer. 官方答案归一化使用的封闭词汇元数据。仅评估侧元
# 数据；问题子类型分类保留在数据层 normalizer。
VRSBENCH_CLOSED_VOCABULARY: dict[str, list[str]] = {
    "existence": ["yes", "no"],
    "proximity": ["yes", "no"],
    "extreme_category": ["small-vehicle", "large-vehicle"],
    "grid_position": [
        "top-left",
        "top-middle",
        "top-right",
        "middle-left",
        "middle-middle",
        "middle-right",
        "bottom-left",
        "bottom-middle",
        "bottom-right",
    ],
    "orientation": ["north-south", "east-west"],
    "arrangement": ["in rows", "clustered", "scattered"],
    "color": [
        "black",
        "blue",
        "brown",
        "gray",
        "green",
        "orange",
        "red",
        "white",
        "yellow",
    ],
}

_OFFICIAL_EVALUATOR_VERSION = "vrsbench-official-eval-v1"


def normalize_answer(answer: str, vocabulary: Sequence[str]) -> str:
    """Normalize one free answer against a closed vocabulary; unmatched
    answers are kept verbatim — never guessed. 按封闭词汇归一化单个自由
    答案；未匹配答案原样保留——绝不猜测。"""

    stripped = answer.strip()
    if not stripped:
        return answer
    lowered = {value.casefold(): value for value in vocabulary}
    return lowered.get(stripped.casefold(), stripped)


def to_official_evaluator_input(
    *,
    question: str,
    references: Sequence[str],
    candidate_answer: str,
    question_id: str | None = None,
) -> dict[str, Any]:
    """Map one current evaluation record to the official evaluator input row.
    Deterministic and pure: no model calls, no task changes.
    将一条当前评估记录映射为官方评估器输入行。确定性纯函数：无模型调用、
    不修改任务。"""

    return {
        "version": _OFFICIAL_EVALUATOR_VERSION,
        "question_id": question_id,
        "question": question,
        "references": list(references),
        "candidate_answer": candidate_answer,
    }


def export_official_input(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic export of official evaluator input rows: input order is
    preserved, each row is copied, and no extra fields are invented.
    官方评估器输入行的确定性导出：保留输入顺序、逐行复制、不发明额外字段。"""

    return [dict(row) for row in rows]
