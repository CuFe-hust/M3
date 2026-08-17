"""Optional pycocoevalcap caption metrics with lazy import.

可选的 pycocoevalcap 描述指标（惰性导入）。pycocoevalcap 不是声明依赖：
缺少时调用方得到明确的 RuntimeError；导入只在真正计算时发生，模块导入
本身无副作用、无网络访问。
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from typing import Any

from evaluation.records import CaptionDeterministicMetrics, EvaluationRecord


class CaptionMetricDependencyError(RuntimeError):
    """The optional caption-metric runtime is not installed.
    可选 caption 指标运行时未安装。"""


def evaluate_caption(
    references: Mapping[str, Sequence[str]],
    candidates: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Compute BLEU/METEOR/ROUGE_L/CIDEr over a homogeneous caption record
    set. Raises CaptionMetricDependencyError when the optional
    pycocoevalcap dependency is missing. 对同质描述记录集计算
    BLEU/METEOR/ROUGE_L/CIDEr。缺少可选依赖 pycocoevalcap 时抛稳定异常。"""

    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
    except ImportError as error:
        raise CaptionMetricDependencyError(
            "Install pycocoevalcap to compute caption metrics."
        ) from error

    results: dict[str, Any] = {"total": len(references)}
    bleu, _ = Bleu(4).compute_score(references, candidates)
    for index, score in enumerate(bleu, start=1):
        results[f"BLEU_{index}"] = score

    not_computed: list[str] = []
    if shutil.which("java") is None:
        not_computed.append("METEOR")
    else:
        try:
            score, _ = Meteor().compute_score(references, candidates)
            results["METEOR"] = score
        except OSError:
            not_computed.append("METEOR")

    for name, scorer in (("ROUGE_L", Rouge()), ("CIDEr", Cider())):
        score, _ = scorer.compute_score(references, candidates)
        results[name] = score
    if not_computed:
        # Keep independent scorers useful when one optional external runtime is
        # absent; never substitute or estimate the missing metric.
        # 一个可选外部运行时缺失时仍保留独立 scorer 的真实结果；绝不替换或
        # 估算缺失指标。
        results["metric_status"] = "partial"
        results["not_computed"] = not_computed
    return results


def merge_caption_evaluation(
    *,
    sample_id: str,
    references: Sequence[str],
    candidate: str,
    judge_parsed: Any | None = None,
    judge_error: str | None = None,
) -> EvaluationRecord:
    """Merge deterministic caption evidence with an optional text judge.

    The judge is auxiliary and never replaces the caption metrics.
    将 caption 的确定性指标与可选文本 judge 合并；judge 仅为辅助证据，
    绝不替换 caption 确定性指标。
    """

    metrics = CaptionDeterministicMetrics(
        candidate=str(candidate),
        references=list(references),
    )
    if judge_error is not None:
        status = "failed"
    elif judge_parsed is None:
        status = "not_requested"
    else:
        status = "succeeded"
    return EvaluationRecord(
        sample_id=sample_id,
        task="caption",
        deterministic_metrics=metrics,
        judge_status=status,
        judge_parsed=judge_parsed,
        judge_error=judge_error,
    )


def aggregate_caption(records: Sequence[EvaluationRecord]) -> dict[str, Any]:
    """Collect per-sample candidates and references from unified caption
    records and compute the corpus-level caption metrics.
    从统一 caption 记录收集逐样本候选与参考答案，计算语料级描述指标。"""

    references: dict[str, list[str]] = {}
    candidates: dict[str, list[str]] = {}
    for record in records:
        metrics = record.deterministic_metrics
        if not isinstance(metrics, CaptionDeterministicMetrics):
            raise ValueError(
                f"caption record {record.sample_id!r} lacks CaptionDeterministicMetrics"
            )
        references[record.sample_id] = list(metrics.references)
        candidates[record.sample_id] = [metrics.candidate]
    return evaluate_caption(references, candidates)
