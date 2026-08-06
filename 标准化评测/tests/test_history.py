"""Tests for deterministic history rebuild."""

import json

from m3rs_eval.history import HistoryIndex, rebuild_history


class TestHistoryRanking:
    """Step 1: ranking excludes smoke/incomplete runs."""

    def test_history_excludes_smoke_and_incomplete_from_ranking(
        self, run_factory, tmp_path
    ):
        run_factory("full-ok", mode="full", status="complete", eligible=True)
        run_factory("smoke", mode="smoke", status="complete", eligible=False)
        run_factory("full-bad", mode="full", status="incomplete", eligible=False)
        history = rebuild_history(run_factory.runs_root, tmp_path / "history")
        assert history.ranked_run_ids == ["full-ok"]

    def test_history_orders_by_created_at_then_run_id(self, run_factory, tmp_path):
        run_factory("a", created_at="2025-03-01T00:00:00+08:00")
        run_factory("b", created_at="2025-01-01T00:00:00+08:00")
        run_factory("c", created_at="2025-02-01T00:00:00+08:00")
        history = rebuild_history(run_factory.runs_root, tmp_path / "history")
        assert history.ranked_run_ids == ["b", "c", "a"]


class TestHistoryIdempotency:
    """Step 1: rebuild must be idempotent."""

    def test_history_rebuild_is_idempotent(self, run_factory, tmp_path):
        run_factory("r1", mode="full", status="complete", eligible=True)
        first = rebuild_history(run_factory.runs_root, tmp_path / "history1")
        second = rebuild_history(run_factory.runs_root, tmp_path / "history2")
        assert first.file_hashes == second.file_hashes
        assert first.ranked_run_ids == second.ranked_run_ids

    def test_same_input_different_output_dir_produces_same_hashes(
        self, run_factory, tmp_path
    ):
        run_factory("r1", mode="full", status="complete", eligible=True)
        run_factory("r2", mode="full", status="complete", eligible=True)
        first = rebuild_history(run_factory.runs_root, tmp_path / "history_a")
        second = rebuild_history(run_factory.runs_root, tmp_path / "history_b")
        assert first.file_hashes == second.file_hashes


class TestCsvOutput:
    """CSV output format checks."""

    def test_csv_has_utf8_bom(self, run_factory, tmp_path):
        run_factory("r1", mode="full", status="complete", eligible=True)
        history = rebuild_history(run_factory.runs_root, tmp_path / "history")

        runs_csv = history.history_root / "runs.csv"
        content = runs_csv.read_bytes()
        assert content[:3] == b"\xef\xbb\xbf", "CSV must start with UTF-8 BOM"

    def test_csv_columns_are_stable(self, run_factory, tmp_path):
        run_factory("r1", mode="full", status="complete", eligible=True)
        rebuild_history(run_factory.runs_root, tmp_path / "history1")

        # Rebuild again and check
        history2 = rebuild_history(run_factory.runs_root, tmp_path / "history2")

        import csv, io
        runs_csv = history2.history_root / "runs.csv"
        content = runs_csv.read_bytes()
        # Strip BOM
        content_str = content.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(content_str))
        rows = list(reader)
        assert len(rows) == 1
        # Spot-check stable columns
        assert rows[0]["run_id"] == "r1"
        assert rows[0]["status"] == "complete"
        assert rows[0]["mode"] == "full"

    def test_metrics_long_csv_contains_all_records(self, run_factory, tmp_path):
        run_factory(
            "r1",
            mode="full",
            status="complete",
            eligible=True,
            metrics={"mme_rs.avg": 0.85, "system.latency.e2e_p50_ms": 100.0},
        )
        history = rebuild_history(run_factory.runs_root, tmp_path / "history")

        import csv, io
        metrics_csv = history.history_root / "metrics_long.csv"
        content = metrics_csv.read_bytes().decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(content))
        rows = list(reader)
        assert len(rows) == 2
        metric_ids = {r["metric_id"] for r in rows}
        assert "mme_rs.avg" in metric_ids
        assert "system.latency.e2e_p50_ms" in metric_ids
        assert all(r["run_id"] == "r1" for r in rows)


class TestInvalidRuns:
    """Broken run packages are recorded, not silently dropped."""

    def test_invalid_run_recorded_in_manifest(self, run_factory, tmp_path):
        run_factory("r1", mode="full", status="complete", eligible=True)

        # Create a broken run dir with no manifest
        broken_dir = run_factory.runs_root / "broken-run"
        broken_dir.mkdir()
        (broken_dir / "some_file.txt").write_text("not a manifest", encoding="utf-8")

        history = rebuild_history(run_factory.runs_root, tmp_path / "history")
        assert len(history.invalid_runs) >= 1
        invalid_ids = [r["run_id"] for r in history.invalid_runs]
        assert "broken-run" in invalid_ids

    def test_broken_manifest_gives_invalid(self, run_factory, tmp_path):
        run_factory("r1")

        # Corrupt an existing manifest
        manifest_path = run_factory.runs_root / "r1" / "run_manifest.json"
        manifest_path.write_text("this is not json", encoding="utf-8")

        history = rebuild_history(run_factory.runs_root, tmp_path / "history")
        assert len(history.invalid_runs) >= 1
        # Valid run should be empty in ranked; corrupted one goes to invalid
        assert "r1" not in history.ranked_run_ids
