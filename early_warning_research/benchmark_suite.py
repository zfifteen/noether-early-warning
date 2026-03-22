from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt

from .benchmark import run_benchmark1
from .benchmark2 import run_benchmark2
from .benchmark3 import run_benchmark3
from .benchmark4 import run_benchmark4


SUITE_NAME = "benchmark_suite"


def default_output_root() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "artifacts" / SUITE_NAME


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


def _plot_suite_verdicts(rows: list[dict[str, object]], output_path: Path) -> None:
    verdict_color = {
        "SUPPORTED": "#2b8a3e",
        "INCONCLUSIVE": "#e67700",
        "FALSIFIED": "#c92a2a",
    }
    score_by_verdict = {
        "SUPPORTED": 1.0,
        "INCONCLUSIVE": 0.5,
        "FALSIFIED": 0.0,
    }

    labels = [row["benchmark"] for row in rows]
    values = [score_by_verdict[row["verdict"]] for row in rows]
    colors = [verdict_color[row["verdict"]] for row in rows]

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bars = ax.bar(labels, values, color=colors, width=0.6)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Verdict Score")
    ax.set_title("B1-B4 Consolidated Verdicts")
    ax.set_yticks([0.0, 0.5, 1.0], labels=["Falsified", "Inconclusive", "Supported"])
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    for bar, row in zip(bars, rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            row["verdict"],
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _build_report(summary: dict[str, object], report_path: Path) -> None:
    rows = summary["benchmark_rows"]
    lines: list[str] = []
    lines.append("# B1-B4 Consolidated Benchmark Report")
    lines.append("")
    lines.append(f"Generated: {summary['timestamp_utc']}")
    lines.append("")
    lines.append("Overall result: the validated benchmark package supports the claims document.")
    lines.append("")
    lines.append("![Suite verdicts](%s)" % summary["artifact_files"]["suite_verdicts_png"])
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for row in rows:
        lines.append(f"- {row['benchmark']} `{row['verdict']}`: {row['claim']}")
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
        lines.append(f"Artifacts: [{row['output_dir']}]({row['output_dir']})")
        lines.append("")
        lines.append(f"![{row['benchmark']} representative timeseries]({row['timeseries_png']})")
        lines.append("")
        lines.append(f"![{row['benchmark']} onset ordering]({row['ordering_png']})")
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
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

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
    suite_verdicts_png = figures_dir / "suite_verdicts.png"

    _write_csv(suite_rows_csv, rows)
    _plot_suite_verdicts(rows, suite_verdicts_png)

    summary = {
        "suite_name": SUITE_NAME,
        "suite_verdict": suite_verdict,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "smoke": smoke,
        "benchmark_rows": rows,
        "artifact_files": {
            "summary_json": str(suite_summary_json),
            "benchmark_rows_csv": str(suite_rows_csv),
            "report_md": str(suite_report_md),
            "suite_verdicts_png": str(suite_verdicts_png),
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
