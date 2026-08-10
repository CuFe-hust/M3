"""Dataset-specific official evaluation seams (approved future paths).

数据集专属官方评估 seam（已批准的未来路径）。只做数据集特定官方评估/导出
归一化；绝不选择 Agent、绝不调用模型、绝不修改任务。任务语义保留在数据层
adapter normalizer。本包只做导出（AGENTS.md：__init__.py 仅导出）。
"""

from evaluation.datasets.vrsbench import (
    VRSBENCH_CLOSED_VOCABULARY,
    export_official_input,
    normalize_answer,
    to_official_evaluator_input,
)

__all__ = [
    "VRSBENCH_CLOSED_VOCABULARY",
    "export_official_input",
    "normalize_answer",
    "to_official_evaluator_input",
]
