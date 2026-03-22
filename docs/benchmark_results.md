# Canonical Benchmark Results

The current frozen canonical benchmark does not confirm the core claim at the benchmark level.

The overall benchmark verdict is `FALSIFIED` in the current run:

`artifacts/canonical_benchmark/20260322T142754Z_canonical/`

That benchmark verdict does not mean the early-warning effect disappeared. It means the frozen confirmation benchmark found a control failure in one member of the direct-detector set, so the repo cannot honestly claim benchmark-level confirmation yet.

The control failure is specific. `activation_stat_mismatch` fails the `full_batch_instant_break` gate. In that suite it falsifies only 15 of 27 runs, for a falsifying fraction of `0.5556`, below the required `0.8`. Because the benchmark requires all three direct detectors to pass both instant-break controls, that one detector failure is enough to make the overall benchmark verdict `FALSIFIED`.

What still held up is important.

- `full_batch_positive` passed. All three direct detectors met the positive-suite gate there.
- `stochastic_positive` passed. `mean_gradient_mismatch` cleared the stochastic positive gate with supportive fraction `0.8889` and median resolved lead `957`.
- `stochastic_instant_break` passed for all three detectors.
- `full_batch_fixed_point_negative` and `stochastic_fixed_point_negative` both stayed clean, with zero drift false positives and zero symmetry false positives across all detectors.
- The benchmark's recommended stochastic direct detector is `mean_gradient_mismatch`.

The detector scorecard from the current canonical run is:

- `full_batch_positive`
  - `covariance_mismatch`: supportive fraction `1.0000`, median resolved lead `84`, pass
  - `mean_gradient_mismatch`: supportive fraction `0.8519`, median resolved lead `38`, pass
  - `activation_stat_mismatch`: supportive fraction `0.9259`, median resolved lead `968`, pass
- `full_batch_instant_break`
  - `covariance_mismatch`: falsifying fraction `1.0000`, median resolved lead `-37`, pass
  - `mean_gradient_mismatch`: falsifying fraction `1.0000`, median resolved lead `-37`, pass
  - `activation_stat_mismatch`: falsifying fraction `0.5556`, median resolved lead `-36`, fail
- `stochastic_positive`
  - `covariance_mismatch`: supportive fraction `0.6667`, median resolved lead `2829`, fail
  - `mean_gradient_mismatch`: supportive fraction `0.8889`, median resolved lead `957`, pass
  - `activation_stat_mismatch`: supportive fraction `0.1852`, median resolved lead `4809`, fail
- `stochastic_instant_break`
  - `covariance_mismatch`: falsifying fraction `1.0000`, median resolved lead `-34`, pass
  - `mean_gradient_mismatch`: falsifying fraction `1.0000`, median resolved lead `-34`, pass
  - `activation_stat_mismatch`: falsifying fraction `0.8519`, median resolved lead `-34`, pass

The long-window resolution rule mattered. In `full_batch_positive`, the benchmark resolved all 27 runs after follow-up. In `stochastic_positive`, the long-window reruns also recovered many short-window censored runs, but detector quality diverged sharply: `mean_gradient_mismatch` resolved 25 runs with 24 supportive, while `covariance_mismatch` remained too insensitive and `activation_stat_mismatch` remained too sparse.

The practical conclusion is narrow but useful.

The benchmark currently says the repo does not yet have a clean detector set that supports the core claim under the full frozen confirmation standard. At the same time, it localizes the problem rather than muddying it. The early-warning effect still shows up strongly in the full-batch positive suite and in the stochastic suite under `mean_gradient_mismatch`. What fails is the attempt to promote `activation_stat_mismatch` into the canonical detector set without a clean full-batch instant-break control.

So the current benchmark-level answer is:

- the core claim is not benchmark-confirmed yet
- the benchmark is doing its job by identifying exactly where the confirmation story breaks
- `mean_gradient_mismatch` is the current best stochastic direct detector
- the next confirmation step is to revise or replace `activation_stat_mismatch`, then rerun the frozen benchmark rather than adding more ad hoc evidence
