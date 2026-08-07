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

from evaluation.metrics.caption import evaluate_caption

REPO_ROOT = Path(__file__).resolve().parents[2]

_REFERENCES = {"a": ["a car on the road"], "b": ["two buildings"]}
_CANDIDATES = {"a": ["a car on the road"], "b": ["two buildings"]}


def test_missing_caption_dependency_raises_runtime_error(monkeypatch) -> None:
    """Blocking the optional import yields a clear RuntimeError.
    拦截可选导入时给出明确的 RuntimeError。"""
    real_import = __import__

    def _blocked(name, *args, **kwargs):
        if name == "pycocoevalcap" or name.startswith("pycocoevalcap."):
            raise ImportError("blocked optional dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _blocked)
    with pytest.raises(RuntimeError, match="pycocoevalcap"):
        evaluate_caption(_REFERENCES, _CANDIDATES)


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


def test_caption_module_import_has_no_network_side_effects() -> None:
    """Importing the module itself never imports pycocoevalcap or touches the
    network. 模块导入本身不导入 pycocoevalcap、不触碰网络。"""
    import evaluation.metrics.caption as caption_module

    assert "pycocoevalcap" not in sys.modules
    source = (REPO_ROOT / "evaluation/metrics/caption.py").read_text(encoding="utf-8")
    for token in ("urlopen", "requests", "socket", "httpx", "api.deepseek", "http://", "https://"):
        assert token not in source, token
    assert "from pycocoevalcap" in source or "import pycocoevalcap" in source
