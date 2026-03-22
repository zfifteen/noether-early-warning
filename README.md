# Noether Early Warning

This repository is the canonical confirmation benchmark for the early-warning claim: update-norm drift may become detectable before direct symmetry detection in paired neural-network training.

The current frozen benchmark is the only canonical evidence path. It does not yet confirm the claim at the benchmark level, because one detector in the full-batch instant-break control fails the benchmark's control gate. The benchmark still shows strong positive evidence in multiple suites and localizes the current weakness to one direct detector.

The canonical documents are:

- [`docs/core_claim.md`](docs/core_claim.md)
- [`docs/benchmark_design.md`](docs/benchmark_design.md)
- [`docs/benchmark_results.md`](docs/benchmark_results.md)

Run the benchmark from the repository root:

```bash
python -m early_warning_research.run --benchmark canonical
pytest -q early_warning_research/tests
```

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Canonical benchmark outputs are written under `artifacts/canonical_benchmark/` by default.

Exploratory runs, follow-up analyses, and historical bundles are preserved as discovery material. See [`docs/archive_map.md`](docs/archive_map.md) and [`archive/`](archive) for that material.
