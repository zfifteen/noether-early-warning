from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from .experiments import execute_run, library_versions
from .path_utils import repo_relative_path, repo_root
from .plotting import plot_onset_ordering, plot_representative_timeseries
from .suites import get_suite, suite_to_dict


BENCHMARK_NAME = "benchmark3"
DIRECT_SYMMETRY_DETECTOR = "covariance_mismatch"
OBSERVATION_BUDGET = 300
DETECTION_RATE_MARGIN = 0.2


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
    return repo_root() / "artifacts" / BENCHMARK_NAME


def _benchmark_suite(smoke: bool):
    return get_suite("main_paired_mlp", smoke=smoke)


def _effective_budget(smoke: bool) -> int:
    return 180 if smoke else OBSERVATION_BUDGET


def _serialize_record(record: object, budget: int) -> dict[str, object]:
    drift_onset = record.drift_onset_step
    symmetry_onset = record.detector_onsets[DIRECT_SYMMETRY_DETECTOR]
    drift_detected = drift_onset is not None and int(drift_onset) <= budget
    symmetry_detected = symmetry_onset is not None and int(symmetry_onset) <= budget

    if drift_detected and not symmetry_detected:
        verdict_label = "drift_only"
    elif drift_detected and symmetry_detected:
        verdict_label = "both_detected"
    elif symmetry_detected:
        verdict_label = "symmetry_only"
    else:
        verdict_label = "no_detection"

    return {
        "run_id": record.run_id,
        "seed": record.seed,
        "learning_rate": record.learning_rate,
        "input_scale": record.input_scale,
        "observation_budget_steps": budget,
        "drift_onset_step": drift_onset,
        "symmetry_onset_step": symmetry_onset,
        "drift_detected_within_budget": drift_detected,
        "symmetry_detected_within_budget": symmetry_detected,
        "verdict_label": verdict_label,
        "final_loss": record.final_loss,
        "early_curvature": record.early_curvature,
        "max_update_norm": record.max_update_norm,
        "max_symmetry_score": record.detector_max_scores[DIRECT_SYMMETRY_DETECTOR],
    }


def _choose_representative_run(run_rows: list[dict[str, object]]) -> str:
    drift_only = [row for row in run_rows if row["verdict_label"] == "drift_only"]
    if drift_only:
        return str(drift_only[0]["run_id"])
    both = [row for row in run_rows if row["verdict_label"] == "both_detected"]
    if both:
        return str(both[0]["run_id"])
    return str(run_rows[0]["run_id"])


def _benchmark_verdict(run_rows: list[dict[str, object]]) -> tuple[str, dict[str, object]]:
    total_runs = len(run_rows)
    drift_detected_runs = sum(bool(row["drift_detected_within_budget"]) for row in run_rows)
    symmetry_detected_runs = sum(bool(row["symmetry_detected_within_budget"]) for row in run_rows)
    drift_detection_rate = 0.0 if total_runs == 0 else drift_detected_runs / total_runs
    symmetry_detection_rate = 0.0 if total_runs == 0 else symmetry_detected_runs / total_runs
    rate_gap = drift_detection_rate - symmetry_detection_rate
    informative = drift_detected_runs > 0 or symmetry_detected_runs > 0

    if not informative:
        verdict = "INCONCLUSIVE"
    elif rate_gap >= DETECTION_RATE_MARGIN and drift_detection_rate > symmetry_detection_rate:
        verdict = "SUPPORTED"
    elif rate_gap <= 0.0:
        verdict = "FALSIFIED"
    else:
        verdict = "INCONCLUSIVE"

    stats = {
        "total_runs": total_runs,
        "drift_detected_runs": drift_detected_runs,
        "symmetry_detected_runs": symmetry_detected_runs,
        "drift_detection_rate": drift_detection_rate,
        "symmetry_detection_rate": symmetry_detection_rate,
        "detection_rate_gap": rate_gap,
        "margin_threshold": DETECTION_RATE_MARGIN,
        "informative": informative,
    }
    return verdict, stats


def run_benchmark3(
    output_root: str | Path | None = None,
    smoke: bool = False,
    quiet: bool = False,
) -> dict[str, object]:
    suite = _benchmark_suite(smoke=smoke)
    budget = _effective_budget(smoke)
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
        run_rows.append(_serialize_record(record, budget))

    benchmark3_verdict, verdict_stats = _benchmark_verdict(run_rows)
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
    plot_onset_ordering(
        [
            {
                "run_id": row["run_id"],
                "drift_onset_step": row["drift_onset_step"] if row["drift_detected_within_budget"] else None,
                "symmetry_onset_step": row["symmetry_onset_step"] if row["symmetry_detected_within_budget"] else None,
                "lead_steps": None
                if not (row["drift_detected_within_budget"] and row["symmetry_detected_within_budget"])
                else int(row["symmetry_onset_step"]) - int(row["drift_onset_step"]),
            }
            for row in run_rows
        ],
        ordering_png,
    )

    summary = {
        "benchmark_name": BENCHMARK_NAME,
        "benchmark3_verdict": benchmark3_verdict,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": repo_relative_path(output_dir),
        "smoke": smoke,
        "direct_symmetry_detector": DIRECT_SYMMETRY_DETECTOR,
        "observation_budget_steps": budget,
        "libraries": library_versions(),
        "parameters": suite_to_dict(suite),
        "verdict_stats": verdict_stats,
        "representative_run_id": representative_run_id,
        "artifact_files": {
            "summary_json": repo_relative_path(summary_json),
            "runs_csv": repo_relative_path(runs_csv),
            "representative_timeseries_png": repo_relative_path(timeseries_png),
            "onset_ordering_png": repo_relative_path(ordering_png),
        },
        "run_summaries": run_rows,
    }
    summary_json.write_text(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Benchmark 3 for the fixed-budget sensitivity claim.")
    parser.add_argument("--output-dir", default=None, help="Override the default artifact root.")
    parser.add_argument("--smoke", action="store_true", help="Use the reduced smoke-test configuration.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-run progress logs.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_benchmark3(output_root=args.output_dir, smoke=args.smoke, quiet=args.quiet)

    print(f"Artifacts: {summary['output_dir']}")
    print(f"Benchmark 3 verdict: {summary['benchmark3_verdict']}")
    stats = summary["verdict_stats"]
    print(
        "Detection rates: "
        f"drift={stats['drift_detection_rate']:.3f} "
        f"symmetry={stats['symmetry_detection_rate']:.3f} "
        f"gap={stats['detection_rate_gap']:.3f}"
    )


if __name__ == "__main__":
    main()
