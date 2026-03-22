from __future__ import annotations

import argparse

from .benchmark import run_canonical_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the canonical early-warning confirmation benchmark.")
    parser.add_argument("--benchmark", choices=["canonical"], required=True)
    parser.add_argument("--output-dir", default=None, help="Override the default artifact root.")
    parser.add_argument("--smoke", action="store_true", help="Use the reduced smoke-test configuration.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-run progress logs.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_canonical_benchmark(output_root=args.output_dir, smoke=args.smoke, quiet=args.quiet)

    print(f"Artifacts: {summary['output_dir']}")
    print(f"Benchmark verdict: {summary['benchmark_verdict']}")
    print(f"Recommended stochastic detector: {summary['recommended_stochastic_direct_detector']}")
    for row in summary["suite_scorecards"]:
        print(
            f"{row['suite_name']}: "
            f"{'PASS' if row['suite_passed'] else 'FAIL'} "
            f"({row['summary']})"
        )


if __name__ == "__main__":
    main()
