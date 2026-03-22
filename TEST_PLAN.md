# Atomic Benchmark Test Plan

This repository is moving to an atomic benchmark strategy.

The claims in `docs/core_claim.md` will be broken into the smallest independently testable pieces. Each piece will get its own benchmark. Each benchmark will test one claim fragment only and return one headline verdict only.

This is the rule for the benchmark program:

- one benchmark, one claim sentence
- one benchmark, one headline verdict
- one benchmark may not bundle multiple subclaims into one result
- one benchmark may not absorb extra detector-package or robustness requirements unless those are part of the claim being tested
- controls, extensions, and broader robustness studies must be separate benchmarks

The workflow is:

- start from `docs/core_claim.md`
- split the claims into the smallest testable subclaims
- assign one benchmark to each subclaim
- keep each benchmark uncoupled from unrelated controls, detectors, or generalization questions

This means benchmark failure is narrow by design. If one benchmark fails, that falsifies only that one subclaim. It does not automatically falsify the whole claims document.

The current claims document breaks into four atomic empirical benchmark targets.

Benchmark 1 tests this statement only:

drift becomes detectable before direct symmetry detection.

Its current implementation path is:

`python -m early_warning_research.run --benchmark benchmark1`

Benchmark 1 is intentionally narrow. It does not stand in for the whole claims document. It covers only one atomic part of the claim set.

Benchmark 2 tests this statement only:

direct symmetry detection appears at or before drift in an instant-break regime.

Its current implementation path is:

`python -m early_warning_research.benchmark2`

Benchmark 2 is the regime-specific mirror of Benchmark 1. It exists to show that the Benchmark 1 ordering is not a generic detector artifact.

Benchmark 3 will test this statement only:

under a fixed practical observation budget, drift is the more sensitive detector.

This benchmark is the atomic version of the “more sensitive probe” language in `docs/core_claim.md`. It should stay in the same gradual claim-bearing suite as Benchmark 1, but it should use a fixed shorter observation budget and compare detection rates directly:

- drift detected within the budget
- direct symmetry detected within the same budget

Benchmark 3 should succeed only if drift detection rate exceeds direct symmetry detection rate by a pre-registered margin under that fixed budget. It should not say anything about instant-break controls, detector rankings, or broad generalization.

The planned implementation pair is:

- `early_warning_research/benchmark3.py`
- `early_warning_research/benchmark3_design.md`

Benchmark 4 will test this statement only:

at the drift alarm time, direct symmetry is still below its own detection threshold.

This benchmark is the atomic version of the accumulation-versus-snapshot part of the claims document. It should stay in the gradual claim-bearing suite and evaluate the direct symmetry score at the first probe at or after drift onset. It is not an ordering benchmark. It is a same-timepoint separation benchmark.

Benchmark 4 should succeed only if, in a high fraction of runs, the drift detector has already fired while the direct symmetry detector has still not crossed its threshold at that matched timepoint.

The planned implementation pair is:

- `early_warning_research/benchmark4.py`
- `early_warning_research/benchmark4_design.md`

The benchmark program therefore maps to the current claims document like this:

- Benchmark 1: earliest reliable evidence appears first in the conserved quantity
- Benchmark 2: the ordering is specific to gradual breaking and should reverse in an instant-break regime
- Benchmark 3: conservation drift is the more sensitive practical probe under finite observation budgets
- Benchmark 4: the drift signal is visible while the direct symmetry signal is still subthreshold at the same timepoint

Anything broader than those four pieces is not yet a separate benchmark target. Broader interpretation, mechanism language, and downstream implications should be added only after they are reduced to an equally narrow test sentence.
