#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from early_warning_research.detectors import detect_symmetry_onset
from early_warning_research.experiments import execute_run, library_versions
from early_warning_research.path_utils import repo_relative_path, repo_root
from early_warning_research.suites import DetectorConfig, SuiteConfig, get_suite, suite_to_dict


DIRECT_SYMMETRY_DETECTOR = "covariance_mismatch"
BENCHMARK_HORIZON = 9600
DENSE_PROBE_CADENCE = 5
CADENCE_VALUES = (5, 10, 15, 20, 30)
BASELINE_VALUES = (2, 3, 4)
CONSECUTIVE_VALUES = (1, 2, 3)
SWEEP_SUITE_NAMES = ("main_paired_mlp", "instant_break_control")


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


def _output_root() -> Path:
    return repo_root() / "artifacts" / "exploratory"


def _dense_suite(name: str) -> SuiteConfig:
    suite = get_suite(name, smoke=False)
    return replace(
        suite,
        training=replace(suite.training, total_steps=BENCHMARK_HORIZON),
        probe=replace(suite.probe, every_steps=DENSE_PROBE_CADENCE),
    )


def _structural_floor(cadence: int, baseline_probes: int, drift_window: int) -> int:
    return cadence * (baseline_probes + 1) - (drift_window + 1)


def _dense_run_rows(suite: SuiteConfig, quiet: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    combinations = [
        (seed, learning_rate, input_scale)
        for seed in suite.sweep.seeds
        for learning_rate in suite.sweep.learning_rates
        for input_scale in suite.sweep.input_scales
    ]

    for index, (seed, learning_rate, input_scale) in enumerate(combinations, start=1):
        if not quiet:
            print(
                f"[dense {suite.name} {index}/{len(combinations)}] "
                f"seed={seed} lr={learning_rate:.4f} input_scale={input_scale:.2f}",
                flush=True,
            )
        record, _step_rows, probe_rows = execute_run(suite, seed, learning_rate, input_scale)
        probe_steps = [int(row["step"]) for row in probe_rows]
        probe_scores = [float(row[f"score_{DIRECT_SYMMETRY_DETECTOR}"]) for row in probe_rows]
        rows.append(
            {
                "suite_name": suite.name,
                "run_id": record.run_id,
                "seed": record.seed,
                "learning_rate": record.learning_rate,
                "input_scale": record.input_scale,
                "drift_onset_step": record.drift_onset_step,
                "probe_steps": probe_steps,
                "probe_scores": probe_scores,
            }
        )
    return rows


def _setting_config(base: DetectorConfig, baseline_probes: int, consecutive: int) -> DetectorConfig:
    return replace(
        base,
        symmetry_baseline_probes=baseline_probes,
        symmetry_consecutive=consecutive,
    )


def _setting_run_rows(
    suite: SuiteConfig,
    dense_rows: list[dict[str, object]],
    cadence: int,
    baseline_probes: int,
    consecutive: int,
) -> list[dict[str, object]]:
    config = _setting_config(suite.detector, baseline_probes, consecutive)
    structural_floor = _structural_floor(cadence, baseline_probes, suite.detector.drift_window)
    rows: list[dict[str, object]] = []

    for row in dense_rows:
        filtered = [
            (step, score)
            for step, score in zip(row["probe_steps"], row["probe_scores"], strict=True)
            if step % cadence == 0
        ]
        probe_steps = [step for step, _score in filtered]
        scores = [score for _step, score in filtered]
        symmetry_result = detect_symmetry_onset(scores, probe_steps, config)
        drift_onset = row["drift_onset_step"]
        symmetry_onset = symmetry_result.onset_step
        lead = None if drift_onset is None or symmetry_onset is None else int(symmetry_onset) - int(drift_onset)
        excess_lead = None if lead is None else int(lead) - structural_floor
        comparable = lead is not None
        if suite.name == "main_paired_mlp":
            raw_supportive = comparable and int(lead) > 0
            excess_supportive = comparable and float(excess_lead) > 0
            excess_nonnegative = comparable and float(excess_lead) >= 0
        else:
            raw_supportive = comparable and int(lead) <= 0
            excess_supportive = comparable and float(excess_lead) <= 0
            excess_nonnegative = comparable and float(excess_lead) <= 0

        rows.append(
            {
                "suite_name": suite.name,
                "run_id": row["run_id"],
                "seed": row["seed"],
                "learning_rate": row["learning_rate"],
                "input_scale": row["input_scale"],
                "probe_cadence_steps": cadence,
                "symmetry_baseline_probes": baseline_probes,
                "symmetry_consecutive_hits": consecutive,
                "structural_floor_steps": structural_floor,
                "drift_onset_step": drift_onset,
                "symmetry_onset_step": symmetry_onset,
                "lead_steps": lead,
                "excess_lead_steps": excess_lead,
                "comparable": comparable,
                "raw_supportive": raw_supportive,
                "excess_supportive": excess_supportive,
                "excess_nonnegative_or_expected": excess_nonnegative,
            }
        )
    return rows


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))


def _summary_rows(setting_rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    by_suite_setting: dict[tuple[str, int, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in setting_rows:
        key = (
            str(row["suite_name"]),
            int(row["probe_cadence_steps"]),
            int(row["symmetry_baseline_probes"]),
            int(row["symmetry_consecutive_hits"]),
        )
        by_suite_setting[key].append(row)

    suite_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []

    for key in sorted(by_suite_setting):
        suite_name, cadence, baseline_probes, consecutive = key
        rows = by_suite_setting[key]
        comparable = [row for row in rows if row["comparable"]]
        raw_leads = [float(row["lead_steps"]) for row in comparable]
        excess_leads = [float(row["excess_lead_steps"]) for row in comparable]
        suite_rows.append(
            {
                "suite_name": suite_name,
                "probe_cadence_steps": cadence,
                "symmetry_baseline_probes": baseline_probes,
                "symmetry_consecutive_hits": consecutive,
                "structural_floor_steps": rows[0]["structural_floor_steps"],
                "total_runs": len(rows),
                "comparable_runs": len(comparable),
                "raw_median_lead_steps": _median(raw_leads),
                "excess_median_lead_steps": _median(excess_leads),
                "positive_excess_runs": sum(float(row["excess_lead_steps"]) > 0 for row in comparable),
                "nonnegative_excess_runs": sum(float(row["excess_lead_steps"]) >= 0 for row in comparable),
                "nonpositive_excess_runs": sum(float(row["excess_lead_steps"]) <= 0 for row in comparable),
                "negative_excess_runs": sum(float(row["excess_lead_steps"]) < 0 for row in comparable),
                "expected_sign_runs": sum(bool(row["excess_supportive"]) for row in comparable),
                "expected_sign_fraction": 0.0 if not comparable else sum(bool(row["excess_supportive"]) for row in comparable) / len(comparable),
                "low_signal_excess_median_steps": _median(
                    [float(row["excess_lead_steps"]) for row in comparable if float(row["input_scale"]) == 0.75]
                ),
            }
        )

    summary_map = {
        (str(row["suite_name"]), int(row["probe_cadence_steps"]), int(row["symmetry_baseline_probes"]), int(row["symmetry_consecutive_hits"])): row
        for row in suite_rows
    }

    for cadence in CADENCE_VALUES:
        for baseline_probes in BASELINE_VALUES:
            for consecutive in CONSECUTIVE_VALUES:
                b1 = summary_map[("main_paired_mlp", cadence, baseline_probes, consecutive)]
                b2 = summary_map[("instant_break_control", cadence, baseline_probes, consecutive)]
                pair_rows.append(
                    {
                        "probe_cadence_steps": cadence,
                        "symmetry_baseline_probes": baseline_probes,
                        "symmetry_consecutive_hits": consecutive,
                        "structural_floor_steps": b1["structural_floor_steps"],
                        "b1_raw_median_lead_steps": b1["raw_median_lead_steps"],
                        "b1_excess_median_lead_steps": b1["excess_median_lead_steps"],
                        "b1_positive_excess_runs": b1["positive_excess_runs"],
                        "b1_nonnegative_excess_runs": b1["nonnegative_excess_runs"],
                        "b1_low_signal_excess_median_steps": b1["low_signal_excess_median_steps"],
                        "b2_raw_median_lead_steps": b2["raw_median_lead_steps"],
                        "b2_excess_median_lead_steps": b2["excess_median_lead_steps"],
                        "b2_nonpositive_excess_runs": b2["nonpositive_excess_runs"],
                        "b2_negative_excess_runs": b2["negative_excess_runs"],
                        "clean_median_sign_split": (
                            b1["excess_median_lead_steps"] is not None
                            and b2["excess_median_lead_steps"] is not None
                            and float(b1["excess_median_lead_steps"]) > 0
                            and float(b2["excess_median_lead_steps"]) <= 0
                        ),
                        "all_b1_nonnegative_and_all_b2_nonpositive": (
                            int(b1["nonnegative_excess_runs"]) == int(b1["comparable_runs"])
                            and int(b2["nonpositive_excess_runs"]) == int(b2["comparable_runs"])
                        ),
                    }
                )

    cadence_baseline = {
        cadence: next(
            row
            for row in pair_rows
            if int(row["probe_cadence_steps"]) == cadence
            and int(row["symmetry_baseline_probes"]) == 3
            and int(row["symmetry_consecutive_hits"]) == 2
        )
        for cadence in CADENCE_VALUES
    }

    overall = {
        "total_detector_settings": len(pair_rows),
        "settings_with_clean_median_sign_split": sum(bool(row["clean_median_sign_split"]) for row in pair_rows),
        "settings_with_full_sign_consistency": sum(bool(row["all_b1_nonnegative_and_all_b2_nonpositive"]) for row in pair_rows),
        "baseline_setting_by_cadence": {
            str(cadence): {
                "structural_floor_steps": cadence_baseline[cadence]["structural_floor_steps"],
                "b1_raw_median_lead_steps": cadence_baseline[cadence]["b1_raw_median_lead_steps"],
                "b1_excess_median_lead_steps": cadence_baseline[cadence]["b1_excess_median_lead_steps"],
                "b2_raw_median_lead_steps": cadence_baseline[cadence]["b2_raw_median_lead_steps"],
                "b2_excess_median_lead_steps": cadence_baseline[cadence]["b2_excess_median_lead_steps"],
            }
            for cadence in CADENCE_VALUES
        },
    }
    return suite_rows, pair_rows, overall


def run_latency_sweep(output_root: str | Path | None = None, quiet: bool = False) -> dict[str, object]:
    base_output = Path(output_root) if output_root is not None else _output_root()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = base_output / f"{timestamp}_latency_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    suites = {name: _dense_suite(name) for name in SWEEP_SUITE_NAMES}
    dense_rows_by_suite: dict[str, list[dict[str, object]]] = {}
    dense_manifest_rows: list[dict[str, object]] = []
    for suite_name, suite in suites.items():
        dense_rows = _dense_run_rows(suite, quiet=quiet)
        dense_rows_by_suite[suite_name] = dense_rows
        for row in dense_rows:
            dense_manifest_rows.append(
                {
                    "suite_name": suite_name,
                    "run_id": row["run_id"],
                    "seed": row["seed"],
                    "learning_rate": row["learning_rate"],
                    "input_scale": row["input_scale"],
                    "drift_onset_step": row["drift_onset_step"],
                    "dense_probe_count": len(row["probe_steps"]),
                    "dense_probe_cadence_steps": DENSE_PROBE_CADENCE,
                }
            )

    setting_rows: list[dict[str, object]] = []
    for suite_name, suite in suites.items():
        dense_rows = dense_rows_by_suite[suite_name]
        for cadence in CADENCE_VALUES:
            for baseline_probes in BASELINE_VALUES:
                for consecutive in CONSECUTIVE_VALUES:
                    setting_rows.extend(
                        _setting_run_rows(
                            suite,
                            dense_rows,
                            cadence=cadence,
                            baseline_probes=baseline_probes,
                            consecutive=consecutive,
                        )
                    )

    suite_summary_rows, pair_summary_rows, overall = _summary_rows(setting_rows)

    dense_manifest_csv = output_dir / "dense_runs.csv"
    setting_runs_csv = output_dir / "setting_runs.csv"
    suite_summary_csv = output_dir / "suite_setting_summary.csv"
    pair_summary_csv = output_dir / "pair_setting_summary.csv"
    summary_json = output_dir / "summary.json"

    _write_csv(dense_manifest_csv, dense_manifest_rows)
    _write_csv(setting_runs_csv, setting_rows)
    _write_csv(suite_summary_csv, suite_summary_rows)
    _write_csv(pair_summary_csv, pair_summary_rows)

    summary = {
        "experiment_name": "latency_sweep",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": repo_relative_path(output_dir),
        "libraries": library_versions(),
        "dense_probe_cadence_steps": DENSE_PROBE_CADENCE,
        "sweep_parameters": {
            "probe_cadence_steps": list(CADENCE_VALUES),
            "symmetry_baseline_probes": list(BASELINE_VALUES),
            "symmetry_consecutive_hits": list(CONSECUTIVE_VALUES),
            "suites": list(SWEEP_SUITE_NAMES),
            "benchmark_horizon_steps": BENCHMARK_HORIZON,
        },
        "suite_parameters": {name: suite_to_dict(suite) for name, suite in suites.items()},
        "overall_summary": overall,
        "artifact_files": {
            "summary_json": repo_relative_path(summary_json),
            "dense_runs_csv": repo_relative_path(dense_manifest_csv),
            "setting_runs_csv": repo_relative_path(setting_runs_csv),
            "suite_setting_summary_csv": repo_relative_path(suite_summary_csv),
            "pair_setting_summary_csv": repo_relative_path(pair_summary_csv),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the latency-normalized cadence/confirmation sweep.")
    parser.add_argument("--output-dir", default=None, help="Override the default artifact root.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-run progress logs.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_latency_sweep(output_root=args.output_dir, quiet=args.quiet)
    overall = summary["overall_summary"]
    print(f"Artifacts: {summary['output_dir']}")
    print(
        "Latency sweep: "
        f"settings={overall['total_detector_settings']} "
        f"clean_sign_split={overall['settings_with_clean_median_sign_split']} "
        f"full_sign_consistency={overall['settings_with_full_sign_consistency']}"
    )


if __name__ == "__main__":
    main()
