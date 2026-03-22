This package is the canonical runnable confirmation benchmark for the core early-warning claim only.

It is designed to answer one question cleanly: under a fixed benchmark matrix, fixed detector set, and fixed long-window resolution policy, does update-norm drift become detectable before direct symmetry detection, and do the controls and true negatives behave as they should.

The public interface is intentionally small:

`python -m early_warning_research.run --benchmark canonical`

Module layout:

- `run.py`: CLI entrypoint
- `benchmark.py`: canonical benchmark matrix, scoring, and artifact writing
- `suites.py`: reusable suite definitions used by the benchmark
- `models.py`: paired MLP and toy models
- `detectors.py`: drift and symmetry onset detectors
- `experiments.py`: low-level run execution and metric collection
- `plotting.py`: benchmark figures
- `tests/`: detector and benchmark coverage

Artifacts are written to `artifacts/canonical_benchmark/` at the repository root by default. Human-facing narrative documents live under `docs/`, and exploratory follow-ups live under `archive/notes/` plus `artifacts/exploratory/`.
