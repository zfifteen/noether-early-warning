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

Benchmark 1 is the first example of this strategy.

Benchmark 1 tests this statement only:

drift becomes detectable before direct symmetry detection.

Its current implementation path is:

`python -m early_warning_research.run --benchmark benchmark1`

Benchmark 1 is intentionally narrow. It does not stand in for the whole claims document. It covers only one atomic part of the claim set.

Future benchmarks will be added one at a time. Each new benchmark will be defined only after the next smallest testable claim fragment has been identified from `docs/core_claim.md`.
