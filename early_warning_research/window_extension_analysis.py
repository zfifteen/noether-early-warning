from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from .experiments import default_output_root, execute_run
from .suites import SuiteConfig, get_suite, suite_to_dict


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


def _run_grid(suite: SuiteConfig, quiet: bool) -> list[dict[str, object]]:
    combinations = [
        (seed, learning_rate, input_scale)
        for seed in suite.sweep.seeds
        for learning_rate in suite.sweep.learning_rates
        for input_scale in suite.sweep.input_scales
    ]
    rows: list[dict[str, object]] = []
    total = len(combinations)
    for index, (seed, learning_rate, input_scale) in enumerate(combinations, start=1):
        if not quiet:
            print(
                f"[base {index}/{total}] {suite.name}: seed={seed} lr={learning_rate:.4f} input_scale={input_scale:.2f}",
                flush=True,
            )
        record, _, _ = execute_run(suite, seed, learning_rate, input_scale)
        rows.append(
            {
                "run_id": record.run_id,
                "seed": seed,
                "learning_rate": learning_rate,
                "input_scale": input_scale,
                "total_steps": suite.training.total_steps,
                "drift_onset_step": record.drift_onset_step,
                "symmetry_onset_step": record.symmetry_onset_step,
                "lead_steps": record.lead_steps,
                "verdict_label": record.verdict_label,
                "consensus_verdict": record.consensus_verdict,
                "covariance_onset": record.detector_onsets.get("covariance_mismatch"),
                "mean_gradient_onset": record.detector_onsets.get("mean_gradient_mismatch"),
                "activation_onset": record.detector_onsets.get("activation_stat_mismatch"),
                "covariance_max_score": record.detector_max_scores.get("covariance_mismatch"),
            }
        )
    return rows


def _extension_rows(
    suite: SuiteConfig,
    base_rows: list[dict[str, object]],
    extension_steps: tuple[int, ...],
    quiet: bool,
) -> list[dict[str, object]]:
    censored_base_rows = [row for row in base_rows if row["verdict_label"] == "censored"]
    rows: list[dict[str, object]] = []
    total = len(censored_base_rows) * len(extension_steps)
    index = 0

    for total_steps in extension_steps:
        extended_suite = replace(suite, training=replace(suite.training, total_steps=total_steps))
        for base_row in censored_base_rows:
            index += 1
            seed = int(base_row["seed"])
            learning_rate = float(base_row["learning_rate"])
            input_scale = float(base_row["input_scale"])
            if not quiet:
                print(
                    f"[extend {index}/{total}] steps={total_steps} seed={seed} lr={learning_rate:.4f} input_scale={input_scale:.2f}",
                    flush=True,
                )
            record, _, _ = execute_run(extended_suite, seed, learning_rate, input_scale)
            rows.append(
                {
                    "run_id": record.run_id,
                    "seed": seed,
                    "learning_rate": learning_rate,
                    "input_scale": input_scale,
                    "base_total_steps": suite.training.total_steps,
                    "extended_total_steps": total_steps,
                    "base_drift_onset_step": base_row["drift_onset_step"],
                    "base_symmetry_onset_step": base_row["symmetry_onset_step"],
                    "base_verdict_label": base_row["verdict_label"],
                    "drift_onset_step": record.drift_onset_step,
                    "symmetry_onset_step": record.symmetry_onset_step,
                    "lead_steps": record.lead_steps,
                    "verdict_label": record.verdict_label,
                    "consensus_verdict": record.consensus_verdict,
                    "covariance_onset": record.detector_onsets.get("covariance_mismatch"),
                    "mean_gradient_onset": record.detector_onsets.get("mean_gradient_mismatch"),
                    "activation_onset": record.detector_onsets.get("activation_stat_mismatch"),
                    "covariance_max_score": record.detector_max_scores.get("covariance_mismatch"),
                }
            )
    return rows


def _window_summary(rows: list[dict[str, object]], total_steps: int) -> dict[str, object]:
    window_rows = [row for row in rows if int(row["extended_total_steps"]) == total_steps]
    supportive = [row for row in window_rows if row["verdict_label"] == "supportive"]
    censored = [row for row in window_rows if row["verdict_label"] == "censored"]
    covariance_onsets = [int(row["covariance_onset"]) for row in supportive if row["covariance_onset"] is not None]
    return {
        "extended_total_steps": total_steps,
        "resolved_supportive_runs": len(supportive),
        "remaining_censored_runs": len(censored),
        "supportive_fraction_of_original_censored": (len(supportive) / len(window_rows)) if window_rows else None,
        "median_covariance_onset_among_resolved": None if not covariance_onsets else int(sorted(covariance_onsets)[len(covariance_onsets) // 2]),
        "resolved_run_ids": [row["run_id"] for row in supportive],
        "still_censored_run_ids": [row["run_id"] for row in censored],
    }


def run_window_extension_analysis(
    suite_name: str,
    extension_steps: tuple[int, ...],
    output_root: Path | None = None,
    quiet: bool = False,
) -> dict[str, object]:
    suite = get_suite(suite_name, smoke=False)
    base_output = (
        output_root
        if output_root is not None
        else default_output_root().parent / "early_warning_research_window_extension"
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = base_output / f"{timestamp}_{suite_name}_window_extension"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_rows = _run_grid(suite, quiet=quiet)
    extension_rows = _extension_rows(suite, base_rows, extension_steps, quiet=quiet)

    base_censored_rows = [row for row in base_rows if row["verdict_label"] == "censored"]
    summaries = [_window_summary(extension_rows, total_steps) for total_steps in extension_steps]

    summary = {
        "suite_name": suite_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "base_suite": suite_to_dict(suite),
        "base_total_runs": len(base_rows),
        "base_censored_runs": len(base_censored_rows),
        "base_censored_run_ids": [row["run_id"] for row in base_censored_rows],
        "extension_steps": list(extension_steps),
        "window_summaries": summaries,
        "artifact_files": {
            "base_runs_csv": str(output_dir / "base_runs.csv"),
            "extended_runs_csv": str(output_dir / "extended_runs.csv"),
            "summary_json": str(output_dir / "summary.json"),
        },
    }

    _write_csv(output_dir / "base_runs.csv", base_rows)
    _write_csv(output_dir / "extended_runs.csv", extension_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Extend previously censored runs to test whether they resolve under longer windows.")
    parser.add_argument("--suite", default="main_paired_mlp", help="Suite name to analyze.")
    parser.add_argument(
        "--extension-steps",
        nargs="+",
        type=int,
        default=(1200, 2400, 4800, 9600),
        help="Longer training windows to test for base-window censored runs.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output root override.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logs.")
    args = parser.parse_args()

    summary = run_window_extension_analysis(
        suite_name=args.suite,
        extension_steps=tuple(args.extension_steps),
        output_root=args.output_dir,
        quiet=args.quiet,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
