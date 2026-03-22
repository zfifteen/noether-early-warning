from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .experiments import execute_run, library_versions
from .plotting import plot_onset_ordering, plot_representative_timeseries
from .suites import get_suite, suite_to_dict


BENCHMARK_NAME = "benchmark2"
DIRECT_SYMMETRY_DETECTOR = "covariance_mismatch"
BENCHMARK_HORIZON = 9600


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def default_benchmark_output_root() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "artifacts" / BENCHMARK_NAME


def _benchmark_suite(smoke: bool) -> object:
    suite = get_suite("instant_break_control", smoke=smoke)
    return replace(suite, training=replace(suite.training, total_steps=BENCHMARK_HORIZON))


def _serialize_record(record: object) -> dict[str, object]:
    drift_onset = record.drift_onset_step
    symmetry_onset = record.detector_onsets[DIRECT_SYMMETRY_DETECTOR]
    lead = None if drift_onset is None or symmetry_onset is None else int(symmetry_onset) - int(drift_onset)
    comparable = lead is not None
    if drift_onset is None:
        verdict_label = "drift_miss"
    elif symmetry_onset is None:
        verdict_label = "symmetry_miss"
    elif int(symmetry_onset) <= int(drift_onset):
        verdict_label = "supportive"
    else:
        verdict_label = "falsifying"

    return {
        "run_id": record.run_id,
        "seed": record.seed,
        "learning_rate": record.learning_rate,
        "input_scale": record.input_scale,
        "total_steps": record.total_steps,
        "drift_onset_step": drift_onset,
        "symmetry_onset_step": symmetry_onset,
        "lead_steps": lead,
        "comparable": comparable,
        "drift_detected": drift_onset is not None,
        "symmetry_detected": symmetry_onset is not None,
        "verdict_label": verdict_label,
        "final_loss": record.final_loss,
        "early_curvature": record.early_curvature,
        "max_update_norm": record.max_update_norm,
        "max_symmetry_score": record.detector_max_scores[DIRECT_SYMMETRY_DETECTOR],
    }


def _choose_representative_run(run_rows: list[dict[str, object]]) -> str:
    comparable = [row for row in run_rows if row["lead_steps"] is not None]
    if comparable:
        target = float(np.median([float(row["lead_steps"]) for row in comparable]))
        chosen = min(comparable, key=lambda row: (abs(float(row["lead_steps"]) - target), row["run_id"]))
        return str(chosen["run_id"])
    symmetry_detected = [row for row in run_rows if row["symmetry_detected"]]
    if symmetry_detected:
        return str(symmetry_detected[0]["run_id"])
    return str(run_rows[0]["run_id"])


def _benchmark_verdict(run_rows: list[dict[str, object]]) -> tuple[str, dict[str, object]]:
    total_runs = len(run_rows)
    comparable_rows = [row for row in run_rows if row["comparable"]]
    supportive_rows = [row for row in run_rows if row["verdict_label"] == "supportive"]
    symmetry_detected_rows = [row for row in run_rows if row["symmetry_detected"]]

    comparable_count = len(comparable_rows)
    supportive_count = len(supportive_rows)
    comparable_fraction = 0.0 if total_runs == 0 else comparable_count / total_runs
    supportive_fraction = 0.0 if total_runs == 0 else supportive_count / total_runs
    median_lead = None
    if comparable_rows:
        median_lead = float(np.median([float(row["lead_steps"]) for row in comparable_rows]))

    informative = bool(symmetry_detected_rows) and bool(comparable_rows)
    if not informative:
        verdict = "INCONCLUSIVE"
    elif supportive_fraction >= 0.8 and median_lead is not None and median_lead <= 0:
        verdict = "SUPPORTED"
    elif supportive_fraction <= 0.2 or (median_lead is not None and median_lead > 0):
        verdict = "FALSIFIED"
    else:
        verdict = "INCONCLUSIVE"

    stats = {
        "total_runs": total_runs,
        "symmetry_detected_runs": len(symmetry_detected_rows),
        "comparable_runs": comparable_count,
        "supportive_runs": supportive_count,
        "falsifying_runs": sum(row["verdict_label"] == "falsifying" for row in run_rows),
        "symmetry_miss_runs": sum(row["verdict_label"] == "symmetry_miss" for row in run_rows),
        "drift_miss_runs": sum(row["verdict_label"] == "drift_miss" for row in run_rows),
        "comparable_fraction": comparable_fraction,
        "supportive_fraction_total": supportive_fraction,
        "median_lead_steps": median_lead,
        "informative": informative,
    }
    return verdict, stats


def run_benchmark2(
    output_root: str | Path | None = None,
    smoke: bool = False,
    quiet: bool = False,
) -> dict[str, object]:
    suite = _benchmark_suite(smoke=smoke)
    base_output = Path(output_root) if output_root is not None else default_benchmark_output_root()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = base_output / f"{timestamp}_{BENCHMARK_NAME}"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    step_rows: list[dict[str, object]] = []
    probe_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []

    combinations = [
        (seed, learning_rate, input_scale)
        for seed in suite.sweep.seeds
        for learning_rate in suite.sweep.learning_rates
        for input_scale in suite.sweep.input_scales
    ]

    for index, (seed, learning_rate, input_scale) in enumerate(combinations, start=1):
        if not quiet:
            print(
                f"[{index}/{len(combinations)}] {BENCHMARK_NAME}: "
                f"seed={seed} lr={learning_rate:.4f} input_scale={input_scale:.2f}",
                flush=True,
            )
        record, run_step_rows, run_probe_rows = execute_run(suite, seed, learning_rate, input_scale)
        step_rows.extend(run_step_rows)
        probe_rows.extend(run_probe_rows)
        run_rows.append(_serialize_record(record))

    benchmark2_verdict, verdict_stats = _benchmark_verdict(run_rows)
    representative_run_id = _choose_representative_run(run_rows)
    representative_summary = next(row for row in run_rows if row["run_id"] == representative_run_id)
    representative_steps = [row for row in step_rows if row["run_id"] == representative_run_id]
    representative_probes = [row for row in probe_rows if row["run_id"] == representative_run_id]

    runs_csv = output_dir / "runs.csv"
    summary_json = output_dir / "summary.json"
    timeseries_png = figures_dir / "representative_timeseries.png"
    ordering_png = figures_dir / "onset_ordering.png"

    _write_csv(runs_csv, run_rows)
    plot_representative_timeseries(representative_steps, representative_probes, representative_summary, timeseries_png)
    plot_onset_ordering(run_rows, ordering_png)

    summary = {
        "benchmark_name": BENCHMARK_NAME,
        "benchmark2_verdict": benchmark2_verdict,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "smoke": smoke,
        "direct_symmetry_detector": DIRECT_SYMMETRY_DETECTOR,
        "benchmark_horizon_steps": BENCHMARK_HORIZON,
        "libraries": library_versions(),
        "parameters": suite_to_dict(suite),
        "verdict_stats": verdict_stats,
        "representative_run_id": representative_run_id,
        "artifact_files": {
            "summary_json": str(summary_json),
            "runs_csv": str(runs_csv),
            "representative_timeseries_png": str(timeseries_png),
            "onset_ordering_png": str(ordering_png),
        },
        "run_summaries": run_rows,
    }
    summary_json.write_text(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Benchmark 2 for the instant-break ordering claim.")
    parser.add_argument("--output-dir", default=None, help="Override the default artifact root.")
    parser.add_argument("--smoke", action="store_true", help="Use the reduced smoke-test configuration.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-run progress logs.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_benchmark2(output_root=args.output_dir, smoke=args.smoke, quiet=args.quiet)

    print(f"Artifacts: {summary['output_dir']}")
    print(f"Benchmark 2 verdict: {summary['benchmark2_verdict']}")
    stats = summary["verdict_stats"]
    print(
        "Runs: "
        f"total={stats['total_runs']} "
        f"comparable={stats['comparable_runs']} "
        f"supportive={stats['supportive_runs']} "
        f"median_lead={stats['median_lead_steps']}"
    )


if __name__ == "__main__":
    main()
