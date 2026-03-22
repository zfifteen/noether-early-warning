# Benchmark 1 Design

Benchmark 1 tests one statement only:

drift becomes detectable before direct symmetry detection.

Benchmark 1 uses:

- one suite: `main_paired_mlp`
- one direct symmetry detector: `covariance_mismatch`
- one fixed training horizon: `9600` steps for every run

There is no control in Benchmark 1, no stochastic suite, no detector bakeoff, no rerun policy, and no truncation-based censoring. Every run is executed to the full horizon from the start.

For each run, Benchmark 1 records:

- drift onset step
- direct symmetry onset step
- lead = `symmetry_onset_step - drift_onset_step`
- whether the run is comparable, meaning both onsets were detected

Benchmark 1 returns:

- `SUPPORTED` if the benchmark is informative, supportive fraction among total runs is at least `0.8`, and median lead among comparable runs is greater than `0`
- `FALSIFIED` if supportive fraction among total runs is at most `0.2`, or median lead among comparable runs is not greater than `0`
- `INCONCLUSIVE` otherwise

The benchmark is informative only if at least one run detects drift and at least one run is comparable.
