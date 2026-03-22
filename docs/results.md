# Early Warning Summary

## Conclusion

The core claim is supported by the current clean experiment package.

In the main neural-network experiment, update-norm drift appeared before direct symmetry detection whenever both signals were actually measurable in the same run. The matched instant-break control behaved the opposite way: symmetry was detected first, then drift. That matters because it shows the package is not just biased toward declaring drift early in every case.

This is support, not universal proof. It means the claim survives the experiment we designed to test it, and it passes the control that should have broken it.

## What this means in plain language

The signal looks real in this setup.

The model starts showing a directional change in its update behavior before the direct symmetry detector can clearly say that symmetry has broken. That is the early-warning effect the package was built to isolate.

The strongest reason to take that seriously is the control. When the model starts from a symmetry-broken state, the result flips the other way. In other words, the package can say both “yes” and “no.” It is not just rewarding one outcome.

The toy sidecar does not add much either way right now. It came back inconclusive, so the evidence is really coming from the main neural-network suite plus the control.

## Technical evidence

The run summarized here is:

`python -m early_warning_research.run --suite all --quiet`

Artifacts for that run were written to:

`artifacts/early_warning_research/20260322T123337Z_all/`

Reference figures copied into this documentation set:

- `docs/figures/main_paired_mlp_representative_timeseries.png`
- `docs/figures/main_paired_mlp_onset_ordering.png`

### Main claim-bearing suite

Suite: `main_paired_mlp`

- Total runs: 27
- Comparable runs: 18
- Supportive runs: 18
- Falsifying runs: 0
- Censored runs: 9
- Median lead: 46 steps
- Verdict: `SUPPORTED`

What that means:

Comparable runs are the runs where both onset times were actually detected. In all 18 comparable runs, the drift detector fired first. The 9 censored runs were not counted as wins. Those are runs where drift was detected but symmetry never crossed the detector threshold within the fixed run window.

Representative examples:

- `seed0_lr0.0200_scale0.75`: drift at step 52, symmetry at step 210, lead = 158
- `seed0_lr0.0400_scale1.75`: drift at step 52, symmetry at step 75, lead = 23
- `seed2_lr0.0800_scale1.75`: drift at step 53, symmetry at step 90, lead = 37

### Instant-break control

Suite: `instant_break_control`

- Total runs: 27
- Comparable runs: 27
- Supportive runs: 0
- Falsifying runs: 27
- Censored runs: 0
- Median lead: -37 steps
- Verdict: `FALSIFIED`

What that means:

This control intentionally starts from a symmetry-broken state. In that setting, symmetry should be detectable before drift if the package is behaving properly. That is exactly what happened in every comparable run.

Representative examples:

- `seed0_lr0.0200_scale0.75`: drift at step 52, symmetry at step 15, lead = -37
- `seed0_lr0.0400_scale1.75`: drift at step 89, symmetry at step 15, lead = -74
- `seed1_lr0.0200_scale1.25`: drift at step 55, symmetry at step 15, lead = -40

### Toy sidecar

Suite: `toy_sanity`

- Total runs: 18
- Comparable runs: 0
- Supportive runs: 0
- Falsifying runs: 0
- Censored runs: 18
- Verdict: `INCONCLUSIVE`

What that means:

The toy suite did not produce a useful ordering result under the current settings. It does not support the claim, but it also does not count as a falsification. In this package it is only a sidecar sanity check, not the main evidence base.

## Overall verdict

- Control guard passed: `True`
- Overall claim verdict: `SUPPORTED`

The clean summary is:

Within this package, the main neural-network experiment supports the early-warning claim, the intentional symmetry-break control falsifies in the opposite direction as it should, and the toy sidecar is inconclusive.
