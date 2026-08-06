"""Workbook input reconciliation tests (Task 8)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from m3rs_eval.registry import MetricRegistry


class TestHistoryMetricIdsInRegistry:
    """Every metric_id appearing in history CSV must be registered."""

    def test_all_history_metrics_exist_in_registry(
        self, project_root: Path, registry: MetricRegistry
    ):
        history_dir = project_root / "test_fixtures" / "workbook"
        metrics_csv = history_dir / "metrics_long.csv"
        assert metrics_csv.is_file(), f"missing {metrics_csv}"

        with metrics_csv.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        metric_ids_in_csv = {row["metric_id"] for row in rows}
        assert metric_ids_in_csv, "metrics_long.csv is empty"

        registry_ids = set(registry.ids)
        missing = sorted(metric_ids_in_csv - registry_ids)
        assert not missing, (
            f"metrics_long.csv contains {len(missing)} metric_id(s) not in registry: {missing}"
        )


class TestEligibleRunCount:
    """runs.csv eligible_for_history=true row count must match expected."""

    def test_core_run_rows_equal_eligible_history_runs(self, project_root: Path):
        history_dir = project_root / "test_fixtures" / "workbook"
        runs_csv = history_dir / "runs.csv"
        assert runs_csv.is_file(), f"missing {runs_csv}"

        with runs_csv.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        eligible = sum(
            1 for row in rows if row.get("eligible_for_history", "").strip().lower() == "true"
        )
        total = len(rows)
        assert total >= 2, f"Expected at least 2 runs, got {total}"
        assert eligible >= 1, f"Expected at least 1 eligible run, got {eligible}"
        assert eligible <= total, "eligible runs cannot exceed total runs"
        # The fixture has exactly 2 eligible runs out of 3
        assert eligible == 2, (
            f"Expected exactly 2 eligible_for_history=true runs, got {eligible} out of {total}"
        )


class TestCoverageCsvShape:
    """Coverage CSV must be readable with stable column headers."""

    REQUIRED_COLUMNS = ["dataset", "task", "expected", "requested", "predicted", "failed", "status"]

    def test_coverage_csv_has_stable_columns(self, project_root: Path):
        coverage_csv = project_root / "test_fixtures" / "workbook" / "coverage.csv"
        assert coverage_csv.is_file(), f"missing {coverage_csv}"

        with coverage_csv.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            headers = reader.fieldnames or []
            rows = list(reader)

        assert headers, "coverage.csv has no headers"
        for col in self.REQUIRED_COLUMNS:
            assert col in headers, f"coverage.csv missing required column: {col}"

        assert len(rows) >= 1, "coverage.csv is empty"


class TestCsvReadability:
    """All history CSVs must be parseable."""

    def test_runs_csv_is_readable(self, project_root: Path):
        runs_csv = project_root / "test_fixtures" / "workbook" / "runs.csv"
        assert runs_csv.is_file(), f"missing {runs_csv}"
        with runs_csv.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert rows, "runs.csv has no data rows"
        for row in rows:
            assert "run_id" in row, f"runs.csv row missing run_id"
            assert "eligible_for_history" in row, f"runs.csv row missing eligible_for_history"

    def test_metrics_long_csv_is_readable(self, project_root: Path):
        metrics_csv = project_root / "test_fixtures" / "workbook" / "metrics_long.csv"
        assert metrics_csv.is_file(), f"missing {metrics_csv}"
        with metrics_csv.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert rows, "metrics_long.csv has no data rows"
        for row in rows:
            assert "metric_id" in row, f"metrics_long.csv row missing metric_id"
            assert "run_id" in row, f"metrics_long.csv row missing run_id"

    def test_coverage_csv_datasets_are_known(self, project_root: Path):
        known_datasets = {"VRSBench", "MME-RealWorld-RS", "XLRS-Bench", "LEVIR-CC"}
        coverage_csv = project_root / "test_fixtures" / "workbook" / "coverage.csv"
        with coverage_csv.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                assert row["dataset"] in known_datasets, (
                    f"Unknown dataset in coverage.csv: {row['dataset']}"
                )
