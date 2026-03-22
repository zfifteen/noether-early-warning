from __future__ import annotations

import argparse

from .experiments import run_named_suite
from .suites import list_suite_names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the clean early-warning experiment suites.")
    parser.add_argument("--suite", choices=[*list_suite_names(), "all"], required=True)
    parser.add_argument("--output-dir", default=None, help="Override the default artifact root.")
    parser.add_argument("--smoke", action="store_true", help="Use the reduced smoke-test configuration.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-run progress logs.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_named_suite(args.suite, output_root=args.output_dir, smoke=args.smoke, quiet=args.quiet)

    if args.suite == "all":
        print(f"Artifacts: {summary['output_dir']}")
        print(f"Control guard passed: {summary['control_guard_passed']}")
        print(f"Overall claim verdict: {summary['overall_claim_verdict']}")
        for suite_name, suite_summary in summary["suites"].items():
            stats = suite_summary["verdict_stats"]
            print(
                f"{suite_name}: {suite_summary['verdict']} "
                f"(comparable={stats['comparable_runs']}, censored={stats['censored_runs']})"
            )
        return

    stats = summary["verdict_stats"]
    print(f"Artifacts: {summary['output_dir']}")
    print(f"Suite verdict: {summary['verdict']}")
    print(
        f"Comparable runs: {stats['comparable_runs']} | "
        f"Supportive: {stats['supportive_runs']} | "
        f"Falsifying: {stats['falsifying_runs']} | "
        f"Censored: {stats['censored_runs']}"
    )


if __name__ == "__main__":
    main()
