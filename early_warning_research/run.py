from __future__ import annotations

import argparse

from .benchmark import run_benchmark1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Benchmark 1 for the core early-warning claim.")
    parser.add_argument("--benchmark", choices=["benchmark1"], required=True)
    parser.add_argument("--output-dir", default=None, help="Override the default artifact root.")
    parser.add_argument("--smoke", action="store_true", help="Use the reduced smoke-test configuration.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-run progress logs.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_benchmark1(output_root=args.output_dir, smoke=args.smoke, quiet=args.quiet)

    print(f"Artifacts: {summary['output_dir']}")
    print(f"Benchmark 1 verdict: {summary['benchmark1_verdict']}")
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
