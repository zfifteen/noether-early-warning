from __future__ import annotations

import csv
from pathlib import Path

import pytest

from early_warning_research.benchmark import run_benchmark1


@pytest.fixture(scope="module")
def smoke_benchmark(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    output_root = tmp_path_factory.mktemp("benchmark1")
    return run_benchmark1(output_root=output_root, smoke=True, quiet=True)


def test_smoke_benchmark_writes_expected_artifacts(smoke_benchmark: dict[str, object]) -> None:
    artifact_files = smoke_benchmark["artifact_files"]
    for key in ("summary_json", "runs_csv", "representative_timeseries_png", "onset_ordering_png"):
        assert Path(artifact_files[key]).exists()


def test_smoke_benchmark_reports_only_benchmark1_fields(smoke_benchmark: dict[str, object]) -> None:
    assert smoke_benchmark["benchmark_name"] == "benchmark1"
    assert "benchmark1_verdict" in smoke_benchmark
    assert smoke_benchmark["direct_symmetry_detector"] == "covariance_mismatch"
    assert "recommended_stochastic_direct_detector" not in smoke_benchmark


def test_smoke_benchmark_runs_only_main_suite(smoke_benchmark: dict[str, object]) -> None:
    assert smoke_benchmark["parameters"]["name"] == "main_paired_mlp"
    assert smoke_benchmark["benchmark_horizon_steps"] == 9600


def test_smoke_benchmark_summarizes_leads_from_both_onsets(smoke_benchmark: dict[str, object]) -> None:
    runs = list(csv.DictReader(Path(smoke_benchmark["artifact_files"]["runs_csv"]).open()))
    assert runs
    for row in runs:
        if row["drift_onset_step"] and row["symmetry_onset_step"]:
            assert row["lead_steps"] != ""


def test_smoke_benchmark_verdict_is_in_expected_set(smoke_benchmark: dict[str, object]) -> None:
    assert smoke_benchmark["benchmark1_verdict"] in {"SUPPORTED", "FALSIFIED", "INCONCLUSIVE"}
