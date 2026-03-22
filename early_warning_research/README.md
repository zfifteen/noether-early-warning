This package exposes Benchmark 1 for the core early-warning claim only.

Benchmark 1 asks one question:

can drift become detectable before direct symmetry detection?

The public interface is:

`python -m early_warning_research.run --benchmark benchmark1`

Benchmark 1 runs only the claim-bearing paired-MLP suite, uses only `covariance_mismatch` as the direct symmetry detector, and executes all runs to a fixed uncensored horizon of `9600` steps.

Artifacts are written to `artifacts/benchmark1/` at the repository root by default. Human-facing narrative documents live under `docs/`.
