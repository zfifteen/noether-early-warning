#!/usr/bin/env python3
"""
NIM Full Pipeline Runner

Executes all experiments and generates figures + report data.

ACTUAL CODE EXECUTION -- not a simulation.
"""

import os
import sys
import time

# Ensure we're in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from experiments import run_all
from plot_results import generate_all_figures

def main():
    print("=" * 70)
    print("  NOETHER INCREMENT MONITOR (NIM)")
    print("  Research-Grade Experiment Pipeline")
    print("  Testing: L2 increment drift as early KSB detector")
    print("=" * 70)
    print()

    t0 = time.time()

    # Run all experiments
    print("PHASE 1: Running experiments...")
    all_results = run_all()

    # Generate figures and tables
    print("\nPHASE 2: Generating figures and statistical tables...")
    summary = generate_all_figures(results_base_dir='results', output_dir='figures')

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"  Results: results/")
    print(f"  Figures: figures/")
    print(f"{'=' * 70}")

if __name__ == '__main__':
    main()
