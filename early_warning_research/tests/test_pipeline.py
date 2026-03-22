from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from early_warning_research.experiments import run_named_suite


def test_smoke_cli_writes_artifacts_for_each_suite(tmp_path: Path) -> None:
    for suite_name in ("main_paired_mlp", "instant_break_control", "toy_sanity"):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "early_warning_research.run",
                "--suite",
                suite_name,
                "--smoke",
                "--output-dir",
                str(tmp_path),
                "--quiet",
            ],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
        )

    summaries = list(tmp_path.glob("**/summary.json"))
    assert summaries
    for summary_path in summaries:
        payload = json.loads(summary_path.read_text())
        if payload.get("suite_name") == "all":
            continue
        assert Path(payload["artifact_files"]["runs_csv"]).exists()
        assert Path(payload["artifact_files"]["probe_metrics_csv"]).exists()
        assert Path(payload["artifact_files"]["representative_timeseries_png"]).exists()


def test_main_suite_has_comparable_runs_and_censored_runs_do_not_auto_support(tmp_path: Path) -> None:
    summary = run_named_suite("main_paired_mlp", output_root=tmp_path, smoke=True, quiet=True)
    stats = summary["verdict_stats"]
    assert stats["comparable_runs"] > 0
    assert stats["supportive_runs"] <= stats["comparable_runs"]


def test_instant_break_control_falsifies(tmp_path: Path) -> None:
    summary = run_named_suite("instant_break_control", output_root=tmp_path, smoke=True, quiet=True)
    assert summary["verdict"] == "FALSIFIED"


def test_repeated_main_runs_are_reproducible(tmp_path: Path) -> None:
    first = run_named_suite("main_paired_mlp", output_root=tmp_path / "first", smoke=True, quiet=True)
    second = run_named_suite("main_paired_mlp", output_root=tmp_path / "second", smoke=True, quiet=True)

    assert first["verdict"] == second["verdict"]
    first_runs = {row["run_id"]: row for row in first["run_summaries"]}
    second_runs = {row["run_id"]: row for row in second["run_summaries"]}
    assert first_runs.keys() == second_runs.keys()
    for run_id in first_runs:
        assert first_runs[run_id]["drift_onset_step"] == second_runs[run_id]["drift_onset_step"]
        assert first_runs[run_id]["symmetry_onset_step"] == second_runs[run_id]["symmetry_onset_step"]
