from __future__ import annotations

from pathlib import Path

import pytest

from early_warning_research.benchmark_suite import run_benchmark_suite


@pytest.fixture(scope="module")
def smoke_suite(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    output_root = tmp_path_factory.mktemp("benchmark_suite")
    return run_benchmark_suite(output_root=output_root, smoke=True, quiet=True)


def test_smoke_suite_writes_expected_artifacts(smoke_suite: dict[str, object]) -> None:
    artifact_files = smoke_suite["artifact_files"]
    for key in ("summary_json", "benchmark_rows_csv", "report_md", "suite_verdicts_png"):
        assert Path(artifact_files[key]).exists()


def test_smoke_suite_reports_all_four_benchmarks(smoke_suite: dict[str, object]) -> None:
    rows = smoke_suite["benchmark_rows"]
    assert [row["benchmark"] for row in rows] == ["B1", "B2", "B3", "B4"]


def test_smoke_suite_links_to_benchmark_output_dirs(smoke_suite: dict[str, object]) -> None:
    outputs = smoke_suite["benchmark_outputs"]
    assert set(outputs.keys()) == {"benchmark1", "benchmark2", "benchmark3", "benchmark4"}
    for output_dir in outputs.values():
        assert Path(output_dir).exists()


def test_smoke_suite_verdict_is_expected(smoke_suite: dict[str, object]) -> None:
    assert smoke_suite["suite_verdict"] in {"SUPPORTED", "INCONCLUSIVE"}
