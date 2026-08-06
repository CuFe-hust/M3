"""Tests for deterministic report context building and markdown rendering."""

from m3rs_eval.reporting import build_report_context, render_report_context_markdown


def _find_delta(context, metric_id: str, baseline_run_id: str):
    """Extract a delta entry for assertions."""
    deltas = context.get("deltas", {})
    return deltas.get(metric_id, {})


class TestReportContext:
    """Step 5: report context building with history fixture."""

    def test_report_context_compares_every_compatible_prior_run(self, history_fixture, registry):
        context = build_report_context("candidate", history_fixture, registry)
        assert [row["run_id"] for row in context["compatible_history"]] == ["r1", "r2", "r3"]
        assert context["latest_baseline"]["run_id"] == "r3"
        assert context["best_by_metric"]["mme_rs.avg"]["run_id"] == "r2"

    def test_lower_is_better_metric_has_positive_improvement(self, history_fixture, registry):
        context = build_report_context("candidate", history_fixture, registry)
        row = _find_delta(context, "system.latency.e2e_p50_ms", "r3")
        assert row["raw_delta"] < 0
        assert row["improvement"] > 0

    def test_candidate_not_in_compatible_history(self, history_fixture, registry):
        context = build_report_context("candidate", history_fixture, registry)
        compat_ids = [row["run_id"] for row in context["compatible_history"]]
        assert "candidate" not in compat_ids

    def test_incompatible_runs_appear_in_appendix(self, history_fixture, registry):
        context = build_report_context("candidate", history_fixture, registry)
        incompatible = context.get("incompatible_runs", [])
        assert len(incompatible) >= 1
        incompatible_ids = [r["run_id"] for r in incompatible]
        assert "r4-incompat" in incompatible_ids

    def test_missing_run_yields_error(self, history_fixture, registry):
        context = build_report_context("nonexistent", history_fixture, registry)
        assert "error" in context


class TestMarkdownRendering:
    """Markdown output contains key identifiers."""

    def test_markdown_contains_run_id(self, history_fixture, registry):
        context = build_report_context("candidate", history_fixture, registry)
        md = render_report_context_markdown(context)
        assert "candidate" in md
        assert "r1" in md
        assert "r2" in md
        assert "r3" in md

    def test_markdown_contains_metric_id(self, history_fixture, registry):
        context = build_report_context("candidate", history_fixture, registry)
        md = render_report_context_markdown(context)
        assert "mme_rs.avg" in md
        assert "system.latency.e2e_p50_ms" in md

    def test_markdown_has_chinese_section_labels(self, history_fixture, registry):
        context = build_report_context("candidate", history_fixture, registry)
        md = render_report_context_markdown(context)
        assert "当前运行摘要" in md
        assert "最近兼容基线" in md
        assert "全部兼容历史运行" in md
        assert "历史最佳" in md
        assert "不可比运行附录" in md

    def test_error_context_renders(self):
        md = render_report_context_markdown({"error": "run not found"})
        assert "错误" in md
        assert "run not found" in md
