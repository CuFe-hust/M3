"""Explicit external standard-evaluator seam (approved future path).

显式外部标准评估器 seam（已批准的未来路径）。只导出 adapter 的两个公共
函数；本包不定义任何业务逻辑（AGENTS.md：__init__.py 仅导出）。
"""

from evaluation.standard.adapter import (
    default_standard_report_path,
    run_standard_evaluation,
)

__all__ = ["default_standard_report_path", "run_standard_evaluation"]
