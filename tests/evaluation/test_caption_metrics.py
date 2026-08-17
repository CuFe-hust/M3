"""Contract tests for optional caption metrics with lazy import.

可选 caption 指标契约测试：缺少 pycocoevalcap 时给出明确 RuntimeError；
注入 fake scorer 时返回 BLEU/METEOR/ROUGE_L/CIDEr；模块导入本身无网络
副作用。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from evaluation.metrics.caption import (
    CaptionMetricDependencyError,
    aggregate_caption,
    evaluate_caption,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

_REFERENCES = {"a": ["a car on the road"], "b": ["two buildings"]}
_CANDIDATES = {"a": ["a car on the road"], "b": ["two buildings"]}


def test_missing_caption_dependency_raises_stable_error(monkeypatch) -> None:
    """Blocking the optional import yields a stable public error without
    exposing the raw import message. 拦截可选导入时给出稳定公共错误，且不暴露
    原始导入消息。"""
    real_import = __import__

    def _blocked(name, *args, **kwargs):
        if name == "pycocoevalcap" or name.startswith("pycocoevalcap."):
            raise ImportError("blocked optional dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _blocked)
    with pytest.raises(CaptionMetricDependencyError) as captured:
        evaluate_caption(_REFERENCES, _CANDIDATES)
    assert str(captured.value) == "Install pycocoevalcap to compute caption metrics."
    assert "blocked optional dependency" not in str(captured.value)


class _FakeScorer:
    """Stands in for a pycocoevalcap scorer. / pycocoevalcap scorer 替身。"""

    def __init__(self, *args, **kwargs) -> None:
        self.scores = [0.5, 0.4, 0.3, 0.2] if args and args[0] == 4 else 0.42

    def compute_score(self, references, candidates):
        return self.scores, None


def _inject_fake_caption_modules(monkeypatch) -> None:
    """Inject fake pycocoevalcap modules into sys.modules so the lazy import
    inside evaluate_caption resolves to them.
    向 sys.modules 注入 fake pycocoevalcap 模块，使 evaluate_caption 内的
    惰性导入解析到它们。"""
    monkeypatch.setattr(
        "evaluation.metrics.caption.shutil.which", lambda _name: "/usr/bin/java"
    )
    package = types.ModuleType("pycocoevalcap")
    monkeypatch.setitem(sys.modules, "pycocoevalcap", package)
    for parent, child in (("pycocoevalcap", "bleu"), ("pycocoevalcap", "cider"),
                          ("pycocoevalcap", "meteor"), ("pycocoevalcap", "rouge")):
        module = types.ModuleType(f"{parent}.{child}")
        monkeypatch.setitem(sys.modules, f"{parent}.{child}", module)
    for full, scorer_name in (
        ("pycocoevalcap.bleu.bleu", "Bleu"),
        ("pycocoevalcap.cider.cider", "Cider"),
        ("pycocoevalcap.meteor.meteor", "Meteor"),
        ("pycocoevalcap.rouge.rouge", "Rouge"),
    ):
        module = types.ModuleType(full)
        module.__dict__[scorer_name] = _FakeScorer
        monkeypatch.setitem(sys.modules, full, module)


def test_evaluate_caption_with_fake_scorers(monkeypatch) -> None:
    _inject_fake_caption_modules(monkeypatch)
    results = evaluate_caption(_REFERENCES, _CANDIDATES)
    assert results["total"] == 2
    assert results["BLEU_1"] == 0.5
    assert results["BLEU_2"] == 0.4
    assert results["BLEU_3"] == 0.3
    assert results["BLEU_4"] == 0.2
    assert results["METEOR"] == 0.42
    assert results["ROUGE_L"] == 0.42
    assert results["CIDEr"] == 0.42


def test_missing_caption_runtime_is_scoped_to_meteor(monkeypatch) -> None:
    _inject_fake_caption_modules(monkeypatch)

    class _MissingMeteor:
        def __init__(self) -> None:
            raise FileNotFoundError("java")

    sys.modules["pycocoevalcap.meteor.meteor"].Meteor = _MissingMeteor
    results = evaluate_caption(_REFERENCES, _CANDIDATES)
    assert results["metric_status"] == "partial"
    assert results["not_computed"] == ["METEOR"]
    assert results["BLEU_1"] == 0.5
    assert results["ROUGE_L"] == 0.42
    assert results["CIDEr"] == 0.42


def test_caption_module_import_has_no_network_side_effects() -> None:
    """Importing the module itself never imports pycocoevalcap or touches the
    network. 模块导入本身不导入 pycocoevalcap、不触碰网络。"""
    import evaluation.metrics.caption as caption_module

    assert "pycocoevalcap" not in sys.modules
    source = (REPO_ROOT / "evaluation/metrics/caption.py").read_text(encoding="utf-8")
    for token in ("urlopen", "requests", "socket", "httpx", "api.deepseek", "http://", "https://"):
        assert token not in source, token
    assert "from pycocoevalcap" in source or "import pycocoevalcap" in source


def test_aggregate_caption_uses_unified_records(monkeypatch) -> None:
    """Corpus-level aggregate collects candidates/references from unified
    caption records and routes them through evaluate_caption.
    语料级汇总从统一 caption 记录收集候选/参考答案并交给 evaluate_caption。"""
    _inject_fake_caption_modules(monkeypatch)
    from evaluation.records import CaptionDeterministicMetrics, EvaluationRecord

    records = [
        EvaluationRecord(
            sample_id="a",
            task="caption",
            deterministic_metrics=CaptionDeterministicMetrics(
                candidate="a car on the road", references=["a car on the road"]
            ),
            judge_status="not_requested",
        ),
        EvaluationRecord(
            sample_id="b",
            task="caption",
            deterministic_metrics=CaptionDeterministicMetrics(
                candidate="two buildings", references=["two buildings"]
            ),
            judge_status="not_requested",
        ),
    ]
    results = aggregate_caption(records)
    assert results["total"] == 2
    assert results["BLEU_1"] == 0.5
    assert results["CIDEr"] == 0.42


def test_aggregate_caption_rejects_records_without_caption_metrics(monkeypatch) -> None:
    _inject_fake_caption_modules(monkeypatch)
    from evaluation.records import EvaluationRecord

    # A caption record without metrics is rejected by aggregate fail-closed.
    # 无指标的 caption 记录被聚合器显式拒绝。
    record = EvaluationRecord(
        sample_id="x",
        task="caption",
        deterministic_metrics=None,
        judge_status="not_requested",
    )
    with pytest.raises(ValueError, match="CaptionDeterministicMetrics"):
        aggregate_caption([record])
