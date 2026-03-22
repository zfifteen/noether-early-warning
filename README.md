# Noether Early Warning

This repository is centered on Benchmark 1 for the core early-warning claim:

update-norm drift becomes detectable before direct symmetry detection.

Benchmark 1 is intentionally narrow. It runs only the claim-bearing paired-MLP suite, uses only `covariance_mismatch` as the direct symmetry detector, and executes every run to a fixed uncensored horizon.

Run it from the repository root:

```bash
python -m early_warning_research.run --benchmark benchmark1
pytest -q early_warning_research/tests
```

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Benchmark 1 artifacts are written under `artifacts/benchmark1/` by default.

Historical materials remain under [`archive/`](archive) and [`docs/archive_map.md`](docs/archive_map.md), but they are not part of Benchmark 1.
