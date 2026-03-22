# Benchmark 3 Design

Benchmark 3 tests one statement only:

under a fixed practical observation budget, drift is the more sensitive detector.

Benchmark 3 is an atomic benchmark. It does not stand in for the whole claims document.

Benchmark 3 uses:

- one suite: `main_paired_mlp`
- one direct symmetry detector: `covariance_mismatch`
- one fixed practical observation budget: `300` training steps for every canonical run

There is no control in Benchmark 3, no second detector, no stochastic extension, no rerun logic, and no detector-package qualification. Benchmark 3 is a fixed-budget detection-rate comparison only.

For each run, Benchmark 3 records:

- whether drift is detected within the budget
- whether direct symmetry is detected within the same budget
- drift onset step if detected
- direct symmetry onset step if detected

Benchmark 3 compares:

- drift detection rate within the budget
- direct symmetry detection rate within the same budget
- detection-rate gap = `drift_detection_rate - symmetry_detection_rate`

Benchmark 3 returns:

- `SUPPORTED` if the benchmark is informative, drift detection rate exceeds direct symmetry detection rate by at least `0.2`, and drift detection rate is strictly greater than direct symmetry detection rate
- `FALSIFIED` if the benchmark is informative and the detection-rate gap is less than or equal to `0`
- `INCONCLUSIVE` otherwise

The benchmark is informative only if at least one run detects drift or direct symmetry within the budget.

Benchmark 3 does test:

- practical sensitivity under a fixed observation budget in the gradual claim-bearing suite

Benchmark 3 does not test:

- the ordering claim from Benchmark 1
- the instant-break reversal claim from Benchmark 2
- accumulation-versus-snapshot mechanism claims
- detector bakeoffs
- robustness or generalization across architectures or regimes

Benchmark 3 follows the atomic benchmark strategy described in `TEST_PLAN.md`.
