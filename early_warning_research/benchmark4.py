from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import torch

from .detectors import detect_symmetry_onset
from .experiments import (
    _probe_seed,
    _run_id,
    _sample_train_batch,
    build_paired_dataset,
    compute_paired_probe_scores,
    execute_run,
    library_versions,
    set_seed,
)
from .path_utils import repo_relative_path, repo_root
from .models import PairedMLP
from .plotting import plot_onset_ordering, plot_representative_timeseries
from .suites import PairedMLPConfig, get_suite, suite_to_dict


BENCHMARK_NAME = "benchmark4"
BENCHMARK_HORIZON = 9600
DIRECT_SYMMETRY_DETECTOR = "covariance_mismatch"


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
    suite = get_suite("main_paired_mlp", smoke=smoke)
    return replace(suite, training=replace(suite.training, total_steps=BENCHMARK_HORIZON))


def _capture_alarm_state_and_score(
    suite,
    seed: int,
    learning_rate: float,
    input_scale: float,
    drift_onset_step: int | None,
) -> tuple[float | None, int | None]:
    if drift_onset_step is None:
        return None, None

    model_cfg = suite.model
    assert isinstance(model_cfg, PairedMLPConfig)

    set_seed(seed)
    data = build_paired_dataset(seed, input_scale, suite.training.train_size, suite.training.probe_size, model_cfg)
    model_seed = seed * 1000 + int(round(learning_rate * 10_000)) + int(round(input_scale * 100))
    model = PairedMLP(
        model_cfg.input_dim,
        model_cfg.pair_count,
        model_seed,
        model_cfg.init_noise,
        hidden_layers=model_cfg.hidden_layers,
        use_layer_norm=model_cfg.use_layer_norm,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    batch_gen = torch.Generator().manual_seed(model_seed + 17)

    for step in range(1, int(drift_onset_step) + 1):
        xb, yb = _sample_train_batch(data["x_train"], data["y_train"], suite.training.batch_size, batch_gen)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()

    scores = compute_paired_probe_scores(
        model,
        data["x_probe"],
        data["y_probe"],
        suite.probe.microbatches,
        suite.probe.microbatch_size,
        probe_seed=_probe_seed(seed, learning_rate, input_scale, int(drift_onset_step)),
    )
    return float(scores[DIRECT_SYMMETRY_DETECTOR]), int(drift_onset_step)


def _serialize_record(
    suite,
    record: object,
    run_probe_rows: list[dict[str, object]],
) -> dict[str, object]:
    drift_onset = record.drift_onset_step
    symmetry_onset = record.detector_onsets[DIRECT_SYMMETRY_DETECTOR]

    scheduled_probe_steps = [int(row["step"]) for row in run_probe_rows]
    scheduled_scores = [float(row[f"score_{DIRECT_SYMMETRY_DETECTOR}"]) for row in run_probe_rows]
    symmetry_result = detect_symmetry_onset(scheduled_scores, scheduled_probe_steps, suite.detector)
    alarm_score, alarm_state_step = _capture_alarm_state_and_score(
        suite,
        record.seed,
        record.learning_rate,
        record.input_scale,
        drift_onset,
    )

    if alarm_score is None:
        verdict_label = "no_alarm_state"
        alarm_threshold = None
        alarm_gap_to_threshold = None
    else:
        alarm_threshold = float(symmetry_result.threshold)
        alarm_gap_to_threshold = alarm_threshold - alarm_score
        verdict_label = "supportive" if alarm_score < alarm_threshold else "falsifying"

    return {
        "run_id": record.run_id,
        "seed": record.seed,
        "learning_rate": record.learning_rate,
        "input_scale": record.input_scale,
        "total_steps": record.total_steps,
        "drift_onset_step": drift_onset,
        "symmetry_onset_step": symmetry_onset,
        "alarm_state_step": alarm_state_step,
        "alarm_symmetry_score": alarm_score,
        "alarm_symmetry_threshold": alarm_threshold,
        "alarm_gap_to_threshold": alarm_gap_to_threshold,
        "alarm_state_exists": alarm_score is not None,
        "drift_detected": drift_onset is not None,
        "verdict_label": verdict_label,
        "final_loss": record.final_loss,
        "early_curvature": record.early_curvature,
        "max_update_norm": record.max_update_norm,
        "max_symmetry_score": record.detector_max_scores[DIRECT_SYMMETRY_DETECTOR],
    }


def _choose_representative_run(run_rows: list[dict[str, object]]) -> str:
    supportive = [row for row in run_rows if row["verdict_label"] == "supportive"]
    if supportive:
        chosen = min(supportive, key=lambda row: float(row["alarm_gap_to_threshold"]))
        return str(chosen["run_id"])
    alarm_rows = [row for row in run_rows if row["alarm_state_exists"]]
    if alarm_rows:
        return str(alarm_rows[0]["run_id"])
    return str(run_rows[0]["run_id"])


def _benchmark_verdict(run_rows: list[dict[str, object]]) -> tuple[str, dict[str, object]]:
    total_runs = len(run_rows)
    alarm_rows = [row for row in run_rows if row["alarm_state_exists"]]
    supportive_rows = [row for row in alarm_rows if row["verdict_label"] == "supportive"]
    drift_detected_rows = [row for row in run_rows if row["drift_detected"]]

    alarm_count = len(alarm_rows)
    supportive_count = len(supportive_rows)
    supportive_fraction = 0.0 if alarm_count == 0 else supportive_count / alarm_count
    informative = bool(drift_detected_rows) and bool(alarm_rows)

    if not informative:
        verdict = "INCONCLUSIVE"
    elif supportive_fraction >= 0.8:
        verdict = "SUPPORTED"
    elif supportive_fraction <= 0.2:
        verdict = "FALSIFIED"
    else:
        verdict = "INCONCLUSIVE"

    stats = {
        "total_runs": total_runs,
        "drift_detected_runs": len(drift_detected_rows),
        "alarm_state_runs": alarm_count,
        "supportive_runs": supportive_count,
        "falsifying_runs": sum(row["verdict_label"] == "falsifying" for row in alarm_rows),
        "no_alarm_state_runs": sum(row["verdict_label"] == "no_alarm_state" for row in run_rows),
        "supportive_fraction_alarm_state": supportive_fraction,
        "informative": informative,
    }
    return verdict, stats


def run_benchmark4(
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
        run_rows.append(_serialize_record(suite, record, run_probe_rows))

    benchmark4_verdict, verdict_stats = _benchmark_verdict(run_rows)
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
                "drift_onset_step": row["drift_onset_step"],
                "symmetry_onset_step": row["symmetry_onset_step"],
                "lead_steps": None
                if row["drift_onset_step"] is None or row["symmetry_onset_step"] is None
                else int(row["symmetry_onset_step"]) - int(row["drift_onset_step"]),
            }
            for row in run_rows
        ],
        ordering_png,
    )

    summary = {
        "benchmark_name": BENCHMARK_NAME,
        "benchmark4_verdict": benchmark4_verdict,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": repo_relative_path(output_dir),
        "smoke": smoke,
        "direct_symmetry_detector": DIRECT_SYMMETRY_DETECTOR,
        "benchmark_horizon_steps": BENCHMARK_HORIZON,
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
    parser = argparse.ArgumentParser(description="Run Benchmark 4 for the exact alarm-state separation claim.")
    parser.add_argument("--output-dir", default=None, help="Override the default artifact root.")
    parser.add_argument("--smoke", action="store_true", help="Use the reduced smoke-test configuration.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-run progress logs.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_benchmark4(output_root=args.output_dir, smoke=args.smoke, quiet=args.quiet)

    print(f"Artifacts: {summary['output_dir']}")
    print(f"Benchmark 4 verdict: {summary['benchmark4_verdict']}")
    stats = summary["verdict_stats"]
    print(
        "Alarm-state runs: "
        f"matched={stats['alarm_state_runs']} "
        f"supportive={stats['supportive_runs']} "
        f"supportive_fraction={stats['supportive_fraction_alarm_state']:.3f}"
    )


if __name__ == "__main__":
    main()
