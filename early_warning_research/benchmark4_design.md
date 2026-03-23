# Benchmark 4 Design

Benchmark 4 tests one statement only:

at the drift alarm time, direct symmetry is still below its own detection threshold.

Benchmark 4 is an atomic benchmark. It does not stand in for the whole claims document.

Benchmark 4 uses:

- one suite: `main_paired_mlp`
- one direct symmetry detector: `covariance_mismatch`
- one fixed training horizon: `9600` steps for every run

There is no control in Benchmark 4, no second detector, no stochastic extension, no rerun logic, and no detector-package qualification. Benchmark 4 is a same-timepoint separation benchmark only.

For each run, Benchmark 4 records:

- drift onset step
- the exact saved model state immediately after the update at the drift onset step
- the direct symmetry score measured on that exact saved state
- the direct symmetry threshold used by the covariance detector for that run
- whether the exact-timepoint symmetry score is still below threshold

Benchmark 4 defines success as:

- supportive run = exact-timepoint symmetry score is strictly below the detector threshold
- falsifying run = exact-timepoint symmetry score is greater than or equal to the detector threshold

Benchmark 4 returns:

- `SUPPORTED` if the benchmark is informative and supportive fraction among exact alarm-state measurements is at least `0.8`
- `FALSIFIED` if the benchmark is informative and supportive fraction among exact alarm-state measurements is at most `0.2`
- `INCONCLUSIVE` otherwise

The benchmark is informative only if at least one run detects drift and has an exact alarm-state measurement.

Benchmark 4 does test:

- exact same-timepoint separation between the drift alarm and the direct symmetry threshold crossing

Benchmark 4 does not test:

- the ordering claim from Benchmark 1
- the instant-break reversal claim from Benchmark 2
- the fixed-budget sensitivity claim from Benchmark 3
- detector bakeoffs
- robustness or generalization across architectures or regimes

Benchmark 4 follows the atomic benchmark strategy described in `docs/benchmark_test_plan.md`.
