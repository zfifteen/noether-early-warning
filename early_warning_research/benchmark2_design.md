# Benchmark 2 Design

Benchmark 2 tests one statement only:

direct symmetry detection appears at or before drift in an instant-break regime.

Benchmark 2 is an atomic benchmark. It does not stand in for the whole claims document.

Benchmark 2 uses:

- one suite: `instant_break_control`
- one direct symmetry detector: `covariance_mismatch`
- one fixed training horizon: `9600` steps for every run

There is no second suite in Benchmark 2, no extra detector, no stochastic extension, no rerun logic, and no detector-package qualification. Every run is executed to the full horizon from the start.

For each run, Benchmark 2 records:

- drift onset step
- direct symmetry onset step
- lead = `symmetry_onset_step - drift_onset_step`
- whether the run is comparable, meaning both onsets were detected

Benchmark 2 defines success as reversed ordering relative to Benchmark 1:

- supportive run = `symmetry_onset_step <= drift_onset_step`
- falsifying run = `symmetry_onset_step > drift_onset_step`

Benchmark 2 returns:

- `SUPPORTED` if the benchmark is informative, supportive fraction among total runs is at least `0.8`, and median lead among comparable runs is less than or equal to `0`
- `FALSIFIED` if supportive fraction among total runs is at most `0.2`, or median lead among comparable runs is greater than `0`
- `INCONCLUSIVE` otherwise

The benchmark is informative only if at least one run detects direct symmetry onset and at least one run is comparable.

Benchmark 2 does test:

- regime-specific reversal of the ordering in the instant-break control

Benchmark 2 does not test:

- the positive gradual-breaking claim from Benchmark 1
- broad “more sensitive probe” language
- accumulation-versus-snapshot mechanism claims
- detector bakeoffs
- robustness or generalization across architectures or regimes

Benchmark 2 follows the atomic benchmark strategy described in `TEST_PLAN.md`.
