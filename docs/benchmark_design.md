# Canonical Benchmark Design

The canonical confirmation benchmark asks one question only: under fixed rules, does update-norm drift become detectable before direct symmetry detection, while matched controls and true negatives behave as they should.

The benchmark is run with:

`python -m early_warning_research.run --benchmark canonical`

The benchmark matrix is frozen to six suites.

- `full_batch_positive`: simple paired MLP, full-batch SGD
- `full_batch_instant_break`: same regime, but initialized in an already broken state
- `full_batch_fixed_point_negative`: exact-symmetric teacher/student fixed point with near-zero loss and near-zero gradients
- `stochastic_positive`: deeper paired MLP with LayerNorm and minibatch SGD
- `stochastic_instant_break`: stochastic matched broken-initialization control
- `stochastic_fixed_point_negative`: stochastic exact-symmetric fixed-point negative

The positive suites test the core claim. The instant-break suites test whether direct symmetry detectors reverse the ordering when symmetry is already broken. The fixed-point negatives test whether the benchmark can stay quiet when nothing meaningful should happen.

The short-window policy is also frozen.

- Full-batch suites run for 300 short-window steps.
- Stochastic suites run for 360 short-window steps.

The long-window resolution cap is fixed at 9600 steps. Every run executes the short window first. Any censored positive run is automatically rerun to the long cap with the same seed, architecture, data generation, and detector settings. Fixed-point negatives always run to the long cap, because their job is to prove non-firing under a meaningful negative regime. Instant-break controls are also extended if they censor, and any control run that does not eventually falsify counts against the benchmark.

The drift detector is fixed to the existing update-norm onset rule. The direct symmetry detector set is also frozen:

- `covariance_mismatch`
- `mean_gradient_mismatch`
- `activation_stat_mismatch`

No threshold sweeps are part of the canonical verdict. Threshold exploration remains discovery work and lives under `artifacts/exploratory/`.

For each detector and suite, the benchmark records:

- short-window comparable runs
- short-window censored runs
- resolved comparable runs after long-window follow-up
- supportive fraction among resolved positive runs
- falsifying fraction in instant-break controls
- drift false-positive rate in fixed-point negatives
- symmetry false-positive rate in fixed-point negatives
- median resolved lead

For positive suites, resolved lead is the resolved symmetry onset minus the short-window drift onset. If drift never fires in a positive run, that run counts against the benchmark.

The suite gates are frozen as follows.

- `full_batch_positive` passes if at least one direct detector has supportive fraction at least 0.8 and median resolved lead greater than 0.
- `stochastic_positive` passes if at least one direct detector has supportive fraction at least 0.8 and median resolved lead greater than 0.
- `full_batch_instant_break` and `stochastic_instant_break` pass only if all three direct detectors have falsifying fraction at least 0.8 and median lead below 0.
- `full_batch_fixed_point_negative` and `stochastic_fixed_point_negative` pass only if drift false-positive rate is at most 0.1 and direct symmetry false-positive rate is 0 for all three detectors.

The benchmark-level verdict is strict. It is `SUPPORTED` only if both positive suites pass, both instant-break controls pass, both fixed-point negatives pass, and the benchmark can recommend a stochastic direct detector. It is `FALSIFIED` if any instant-break control or fixed-point negative fails. Otherwise it is `INCONCLUSIVE`.

The stochastic detector recommendation is also rule-based. The recommended stochastic direct detector is the detector with the highest supportive fraction in `stochastic_positive`, excluding any detector that has a nonzero symmetry false-positive rate in `stochastic_fixed_point_negative` or a falsifying fraction below 0.8 in `stochastic_instant_break`. Ties break by larger median resolved lead, then lower short-window censoring.

Each canonical benchmark run writes:

- `benchmark_summary.json`
- `suite_scorecard.csv`
- `detector_scorecard.csv`
- `short_window_runs.csv`
- `resolved_runs.csv`
- `resolution_curve.csv`
- `fixed_point_negative_checks.csv`
- `figures/lead_distributions.png`
- `figures/censor_resolution.png`

The default output root is `artifacts/canonical_benchmark/`.
