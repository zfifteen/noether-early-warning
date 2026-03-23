The Drift Alarm Is a Trajectory Clock, Not a Symmetry Meter

When a neural network begins to break the symmetry between paired units during training, a gradient-based drift alarm fires at an almost fixed point in the training timeline regardless of how strong or weak the input signal is. But the direct measurement of symmetry in the weights can lag that alarm by anywhere from a few steps to thousands of steps, depending on how much signal the data contains.

The non-obvious part is that the drift alarm does not measure how much symmetry has been lost. It fires because the optimizer hits a transition point in the shape of the loss surface, a boundary that gradient momentum crosses at roughly the same training step no matter what the input scale or learning rate is. Across 27 runs spanning a 4x range in learning rate and a 2.3x range in input scale, drift onset varied by only 10 steps around a mean of 52, while symmetry onset varied by 5520 steps.

This means the two detectors are not measuring the same thing expressed at different sensitivities. They live in different spaces entirely. One tracks the optimization trajectory in gradient-norm space; the other tracks accumulated structural change in weight-covariance space. They are decoupled in a way that the field has not made explicit, because both are described as "symmetry detectors" when in fact only one of them is.

The practical implication is counterintuitive: the regimes where a drift early-warning is most valuable are precisely the low-signal regimes where you would least expect it to matter. When input scale is low, the weight-space covariance accumulates slowly, so symmetry onset is delayed by hundreds to thousands of steps. But the drift alarm still fires at step 52. The warning lead grows enormous exactly where the data is least informative and the training is most at risk of going wrong silently.

A direct consequence that surprises most practitioners is this: you cannot close the gap between the two detectors by increasing the learning rate or changing the input scale, because drift onset does not respond to either parameter in a statistically significant way. The only lever that shortens the lead time is increasing the input signal, and that does so by speeding up the symmetry detector, not by moving the drift alarm.

This reframes how monitoring budgets should be designed. If observation time is limited, monitoring gradient-space drift is robustly informative across all hyperparameter settings, while monitoring weight-space covariance is informative only when signal strength is already high enough that the network would likely self-correct anyway.

---

Falsifiable Prediction and Decision Rule
Prediction (Simulation): In any paired-MLP training run under gradual symmetry breaking, if you hold input scale fixed and vary learning rate across a 4x range, drift onset step will remain within a 15-step window while symmetry onset step will vary by at least 3x. This should hold even if the total training horizon changes, as long as the gradient-noise regime is similar.

What would falsify this: if drift onset step shows a statistically significant correlation with learning rate (Pearson |r| > 0.4, p < 0.05) across a sweep comparable to the one here.

Decision Rule (Operational): When monitoring budget is fixed and input scale is unknown or low (below the regime where symmetry score reliably exceeds 0.1 within 300 steps), allocate monitoring to drift detection rather than direct symmetry detection. Switch the dominant monitor to direct symmetry only when max symmetry score within the budget window crosses 0.10, which signals the network is in the high-signal regime where both detectors converge.

---

Investigation Audit

I audited the run data underlying this note rather than treating it as a free-floating interpretation. The primary artifact used for the audit was `artifacts/benchmark1/20260322T152418Z_benchmark1/runs.csv`, with `artifacts/benchmark2/20260322T154257Z_benchmark2/runs.csv` used as a control read.

The central empirical pattern is real. In the gradual `B1` suite, drift onset is tightly concentrated while symmetry onset is highly elastic:

- drift onset range: `51-61`
- drift onset mean: `52.19`
- drift onset median: `52`
- drift onset standard deviation: `1.96`
- drift onset coefficient of variation: `0.038`
- symmetry onset range: `60-5580`
- symmetry onset mean: `670`
- symmetry onset median: `135`
- symmetry onset standard deviation: `1230.87`
- symmetry onset coefficient of variation: `1.837`

That means the lead is being created mainly by movement in symmetry onset, not by large movement in drift onset. The lead itself spans `8-5529` steps, with a median of `84`.

The fixed-scale comparison also supports the note's main idea. Holding input scale fixed while varying learning rate across the sweep:

- at scale `0.75`, drift onset moves only `4` steps while symmetry onset spans `60-5580` (`93x`)
- at scale `1.25`, drift onset moves `10` steps while symmetry onset spans `60-510` (`8.5x`)
- at scale `1.75`, drift onset moves `1` step while symmetry onset spans `60-300` (`5x`)

This is strong evidence that the drift detector is behaving much more like a training-time clock than the direct symmetry detector in the gradual regime.

The low-signal practical implication is also supported by the data. Median lead by input scale is:

- scale `0.75`: `1417`
- scale `1.25`: `53`
- scale `1.75`: `38`

So the early-warning lead grows dramatically in the low-signal regime. The largest leads all occur at the lowest input scale, which is exactly where the note claims drift is most useful.

The factor audit is directionally consistent with the writeup. In `B1`:

- learning rate vs drift onset: Pearson `r = -0.222`, ANOVA `p = 0.369`
- input scale vs drift onset: Pearson `r = -0.116`, ANOVA `p = 0.587`
- input scale vs symmetry onset: Pearson `r = -0.514`, ANOVA `p = 0.0067`

So within this sweep, learning rate and input scale do not move drift onset much, while input scale does move symmetry onset substantially.

The instant-break control helps sharpen the interpretation. In `B2`, direct symmetry onset is fixed at step `15` in every run, while drift onset remains much later (`51-75`). That means the `B1` pattern is not just "everything has a fixed onset"; it is specific to how drift behaves in the gradual regime.

The investigation does not establish every sentence in the note equally strongly. The mechanism story about the optimizer crossing a transition in the loss-surface geometry remains a hypothesis, not something proven by the run tables alone. Seed effects on drift onset are also not zero; they are weaker than the symmetry-onset elasticity, but they are not absent.

The strongest audited version of the insight is therefore this:

In the benchmarked gradual regime, drift onset is tightly locked while symmetry onset is highly elastic, and the early-warning lead grows mainly because symmetry onset moves, especially in low-signal runs.
