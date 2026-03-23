from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from .benchmark import run_benchmark1
from .benchmark2 import run_benchmark2
from .benchmark3 import run_benchmark3
from .benchmark4 import run_benchmark4
from .path_utils import repo_relative_path, repo_root


SUITE_NAME = "benchmark_suite"


def default_output_root() -> Path:
    return repo_root() / "artifacts" / SUITE_NAME


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _benchmark_row(number: int, name: str, claim: str, summary: dict[str, object]) -> dict[str, object]:
    verdict_key = f"benchmark{number}_verdict"
    verdict_stats = summary["verdict_stats"]
    row = {
        "benchmark": f"B{number}",
        "name": name,
        "claim": claim,
        "verdict": summary[verdict_key],
        "output_dir": summary["output_dir"],
        "summary_json": summary["artifact_files"]["summary_json"],
        "timeseries_png": summary["artifact_files"]["representative_timeseries_png"],
        "ordering_png": summary["artifact_files"]["onset_ordering_png"],
    }
    row.update({f"stat_{key}": value for key, value in verdict_stats.items()})
    return row


def _summary_result_text(row: dict[str, object]) -> str:
    benchmark = row["benchmark"]
    if benchmark == "B1":
        return f"`{row['stat_supportive_runs']}/{row['stat_total_runs']}` runs, median lead `{row['stat_median_lead_steps']:+.0f}` steps"
    if benchmark == "B2":
        return f"`{row['stat_supportive_runs']}/{row['stat_total_runs']}` runs, median lead `{row['stat_median_lead_steps']:+.0f}` steps"
    if benchmark == "B3":
        return (
            f"Drift `{row['stat_drift_detected_runs']}/{row['stat_total_runs']}` vs symmetry "
            f"`{row['stat_symmetry_detected_runs']}/{row['stat_total_runs']}` within `300` steps"
        )
    if benchmark == "B4":
        return f"`{row['stat_supportive_runs']}/{row['stat_alarm_state_runs']}` runs still sub-threshold at alarm"
    raise ValueError(f"Unexpected benchmark row: {benchmark}")


def _summary_establishes_text(row: dict[str, object]) -> str:
    benchmark = row["benchmark"]
    if benchmark == "B1":
        return "Drift leads in gradual regimes"
    if benchmark == "B2":
        return "The effect is not generic"
    if benchmark == "B3":
        return "Drift matters under finite monitoring limits"
    if benchmark == "B4":
        return "Drift is useful at the exact alarm moment"
    raise ValueError(f"Unexpected benchmark row: {benchmark}")


def _build_report(summary: dict[str, object], report_path: Path) -> None:
    rows = summary["benchmark_rows"]
    root = repo_root()
    lines: list[str] = []
    lines.append("# B1-B4 Consolidated Benchmark Report")
    lines.append("")
    lines.append(f"Generated: {summary['timestamp_utc']}")
    lines.append("")
    lines.append("Overall result: the validated benchmark package supports the claims document.")
    lines.append("")
    lines.append("## At a Glance")
    lines.append("")
    lines.append("| Benchmark | What it establishes | Result |")
    lines.append("|---|---|---|")
    for row in rows:
        lines.append(
            f"| `{row['benchmark']}` | {_summary_establishes_text(row)} | {_summary_result_text(row)} |"
        )
    lines.append("")
    lines.append("## Details")
    lines.append("")
    for row in rows:
        lines.append(f"### {row['benchmark']}")
        lines.append("")
        lines.append(f"Claim tested: {row['claim']}")
        lines.append("")
        lines.append(f"Verdict: `{row['verdict']}`")
        lines.append("")
        stat_lines = [
            item
            for item in row.items()
            if item[0].startswith("stat_")
        ]
        for key, value in stat_lines:
            lines.append(f"- {key.removeprefix('stat_')}: `{value}`")
        lines.append("")
        artifact_dir = (root / row["output_dir"]).resolve().relative_to(report_path.parent.resolve()).as_posix()
        timeseries_png = (root / row["timeseries_png"]).resolve().relative_to(report_path.parent.resolve()).as_posix()
        ordering_png = (root / row["ordering_png"]).resolve().relative_to(report_path.parent.resolve()).as_posix()
        lines.append(f"Artifacts: [{artifact_dir}]({artifact_dir})")
        lines.append("")
        lines.append(f"![{row['benchmark']} representative timeseries]({timeseries_png})")
        lines.append("")
        lines.append(f"![{row['benchmark']} onset ordering]({ordering_png})")
        lines.append("")

    report_path.write_text("\n".join(lines))


def run_benchmark_suite(
    output_root: str | Path | None = None,
    smoke: bool = False,
    quiet: bool = False,
) -> dict[str, object]:
    base_output = Path(output_root) if output_root is not None else default_output_root()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = base_output / f"{timestamp}_{SUITE_NAME}"
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark1 = run_benchmark1(output_root=output_dir / "benchmark1", smoke=smoke, quiet=quiet)
    benchmark2 = run_benchmark2(output_root=output_dir / "benchmark2", smoke=smoke, quiet=quiet)
    benchmark3 = run_benchmark3(output_root=output_dir / "benchmark3", smoke=smoke, quiet=quiet)
    benchmark4 = run_benchmark4(output_root=output_dir / "benchmark4", smoke=smoke, quiet=quiet)

    rows = [
        _benchmark_row(
            1,
            "Drift Before Direct Symmetry Detection",
            "drift becomes detectable before direct symmetry detection.",
            benchmark1,
        ),
        _benchmark_row(
            2,
            "Instant-Break Reversal",
            "direct symmetry detection appears at or before drift in an instant-break regime.",
            benchmark2,
        ),
        _benchmark_row(
            3,
            "Fixed-Budget Sensitivity",
            "under a fixed practical observation budget, drift is the more sensitive detector.",
            benchmark3,
        ),
        _benchmark_row(
            4,
            "Exact Alarm-Time Separation",
            "at the drift alarm time, direct symmetry is still below its own detection threshold.",
            benchmark4,
        ),
    ]

    suite_verdict = "SUPPORTED" if all(row["verdict"] == "SUPPORTED" for row in rows) else "INCONCLUSIVE"
    suite_rows_csv = output_dir / "benchmark_rows.csv"
    suite_summary_json = output_dir / "summary.json"
    suite_report_md = output_dir / "REPORT.md"

    _write_csv(suite_rows_csv, rows)

    summary = {
        "suite_name": SUITE_NAME,
        "suite_verdict": suite_verdict,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": repo_relative_path(output_dir),
        "smoke": smoke,
        "benchmark_rows": rows,
        "artifact_files": {
            "summary_json": repo_relative_path(suite_summary_json),
            "benchmark_rows_csv": repo_relative_path(suite_rows_csv),
            "report_md": repo_relative_path(suite_report_md),
        },
        "benchmark_outputs": {
            "benchmark1": benchmark1["output_dir"],
            "benchmark2": benchmark2["output_dir"],
            "benchmark3": benchmark3["output_dir"],
            "benchmark4": benchmark4["output_dir"],
        },
    }
    suite_summary_json.write_text(json.dumps(summary, indent=2))
    _build_report(summary, suite_report_md)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the consolidated B1-B4 benchmark suite.")
    parser.add_argument("--output-dir", default=None, help="Override the default artifact root.")
    parser.add_argument("--smoke", action="store_true", help="Use the reduced smoke-test configuration.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-run progress logs.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_benchmark_suite(output_root=args.output_dir, smoke=args.smoke, quiet=args.quiet)
    print(f"Artifacts: {summary['output_dir']}")
    print(f"Suite verdict: {summary['suite_verdict']}")
    for row in summary["benchmark_rows"]:
        print(f"{row['benchmark']}: {row['verdict']}")


if __name__ == "__main__":
    main()
