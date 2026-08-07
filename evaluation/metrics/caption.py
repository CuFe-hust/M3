"""Optional pycocoevalcap caption metrics with lazy import.

可选的 pycocoevalcap 描述指标（惰性导入）。pycocoevalcap 不是声明依赖：
缺少时调用方得到明确的 RuntimeError；导入只在真正计算时发生，模块导入
本身无副作用、无网络访问。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def evaluate_caption(
    references: Mapping[str, Sequence[str]],
    candidates: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Compute BLEU/METEOR/ROUGE_L/CIDEr over a homogeneous caption record
    set. Raises RuntimeError when the optional pycocoevalcap dependency is
    missing. 对同质描述记录集计算 BLEU/METEOR/ROUGE_L/CIDEr。缺少可选依赖
    pycocoevalcap 时抛 RuntimeError。"""

    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
    except ImportError as error:
        raise RuntimeError("Install pycocoevalcap to compute caption metrics.") from error

    results: dict[str, Any] = {"total": len(references)}
    bleu, _ = Bleu(4).compute_score(references, candidates)
    for index, score in enumerate(bleu, start=1):
        results[f"BLEU_{index}"] = score
    for name, scorer in (("METEOR", Meteor()), ("ROUGE_L", Rouge()), ("CIDEr", Cider())):
        score, _ = scorer.compute_score(references, candidates)
        results[name] = score
    return results
