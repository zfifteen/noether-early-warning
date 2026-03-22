This package is the canonical runnable experiment bundle for the core early-warning claim only.

It is designed to answer one question well: can update-norm drift become detectable before direct symmetry detection, and does that ordering fail in an intentionally broken control.

The package keeps the public interface small:

`python -m early_warning_research.run --suite main_paired_mlp`

`python -m early_warning_research.run --suite instant_break_control`

`python -m early_warning_research.run --suite toy_sanity`

`python -m early_warning_research.run --suite all`

Module layout:

- `run.py`: CLI entrypoint
- `suites.py`: versioned suite definitions and default sweeps
- `models.py`: paired MLP and toy models
- `detectors.py`: drift and symmetry onset detectors
- `experiments.py`: execution, artifact writing, and verdict logic
- `plotting.py`: suite figures
- `tests/`: detector and pipeline coverage

Artifacts are written to `artifacts/early_warning_research/` at the repository root by default. Human-facing narrative documents live under `docs/`, not inside this package.
