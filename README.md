# Noether Early Warning

This repository is the clean research artifact for the narrow early-warning claim: in the current experiment package, update-norm drift becomes detectable before direct symmetry detection in the main neural-network suite, and that ordering fails in the matched symmetry-broken control.

The canonical experiment package is [`early_warning_research/`](early_warning_research). It contains the runnable code, tests, and suite definitions for the current claim-bearing experiments.

The canonical human-readable entrypoints are:

- [`docs/core_claim.md`](docs/core_claim.md)
- [`docs/results.md`](docs/results.md)
- [`docs/real_world_examples.md`](docs/real_world_examples.md)
- [`docs/archive_map.md`](docs/archive_map.md)

The canonical commands are:

```bash
python -m early_warning_research.run --suite main_paired_mlp
python -m early_warning_research.run --suite instant_break_control
python -m early_warning_research.run --suite toy_sanity
python -m early_warning_research.run --suite all
pytest -q early_warning_research/tests
```

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Generated experiment outputs are written under `artifacts/early_warning_research/` by default.

Legacy experiments, prototypes, broad conjecture notes, and bundle exports are preserved under [`archive/`](archive). They remain available as historical context, but they are not the canonical path through the repo.
