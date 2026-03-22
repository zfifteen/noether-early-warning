# Benchmark 1 Results

Benchmark 1 is the repo's canonical test of the core early-warning claim:

drift becomes detectable before direct symmetry detection.

Run it with:

`python -m early_warning_research.run --benchmark benchmark1`

Benchmark 1 uses only the claim-bearing `main_paired_mlp` suite, only the `covariance_mismatch` direct symmetry detector, and runs every trial to `9600` steps with no truncation and no reruns.

The benchmark writes:

- `summary.json`
- `runs.csv`
- `figures/representative_timeseries.png`
- `figures/onset_ordering.png`

The benchmark headline result is reported only as:

`benchmark1_verdict`

Nothing outside this one suite and this one detector is allowed to change that headline result.
