# Commitment Analysis

## Conclusion

The current data support a refinement of the early-warning claim, not a collapse of it.

Drift onset by itself does not distinguish which `main_paired_mlp` runs will later show direct symmetry detection within the observation window. Supportive and censored runs fire the drift alarm at essentially the same step.

What does separate the two groups is the short-horizon growth of the direct symmetry score after drift onset. In this experiment, runs that later become directly detectable show much larger post-onset symmetry-score growth over the next 60 steps than censored runs.

So the concrete answer is:

- the narrow claim still holds: drift becomes detectable before direct symmetry detection
- the stronger reading was incomplete: drift onset alone is not enough to tell whether the run is committing to observable symmetry escape within the current window
- the useful refinement is a two-stage interpretation:
  - drift onset marks precursor entry
  - post-onset symmetry-score growth distinguishes commitment from non-commitment within the window

## What Was Tested

The analysis reran the canonical `main_paired_mlp` suite and classified runs using the original covariance-mismatch direct symmetry detector:

- `supportive`: drift detected, symmetry later detected, positive lead
- `censored`: drift detected, symmetry not detected within the 300-step run window

For each run, the analysis compared supportive vs censored populations on:

- drift onset timing
- post-onset path-efficiency of the asymmetry vector
- post-onset growth of the direct symmetry score over 30 and 60 steps
- terminal symmetry score

Artifacts for this exploratory analysis are in:

`artifacts/exploratory/early_warning_research_commitment/20260322T134604Z_main_paired_mlp_commitment/`

## Concrete Results

Run counts:

- supportive runs: 18
- censored runs: 9

Drift onset timing does not separate the groups:

- supportive median drift onset: 52
- censored median drift onset: 52
- Mann-Whitney p-value: 0.122
- AUC for supportive vs censored: 0.327

The first geometric commitment metric that was tested did not separate the groups in a useful way:

- path-efficiency over 60 steps: AUC 0.370
- norm-growth ratio over 60 steps: AUC 0.500

The direct symmetry-score growth after onset does separate the groups:

- symmetry gain over 30 steps:
  - supportive median: 0.0135
  - censored median: 0.000659
  - p-value: 0.0193
  - AUC: 0.784
- symmetry gain over 60 steps:
  - supportive median: 0.0294
  - censored median: 0.000839
  - p-value: 0.000351
  - AUC: 0.932
- max symmetry gain over 60 steps:
  - supportive median: 0.0235
  - censored median: 0.00184
  - p-value: 0.00221
  - AUC: 0.870

As expected, the terminal symmetry score separates the groups almost perfectly, but that is too late to be the useful triage signal:

- supportive median terminal symmetry score: 0.211
- censored median terminal symmetry score: 0.0118
- p-value: 5.40e-05
- AUC: 0.988

## What This Means

The current evidence does not support the strongest version of the new idea, namely that supportive and censored runs are cleanly distinguished by a simple directionality metric on the asymmetry-vector path itself. That specific path-efficiency test did not work.

But the evidence does support a weaker and more concrete refinement:

- drift onset tells you the run has entered a precursor regime
- short-horizon growth in the direct symmetry score after onset tells you whether that precursor is actually committing within the current window

That makes the original hypothesis incomplete at the predictive level, not wrong at the detectability level.

The best revised statement is:

"Drift onset is an early warning of precursor entry; fast post-onset growth in direct symmetry signals indicates commitment to observable symmetry breaking within the current window."

## Remaining Caution

These censored runs are still censored runs, not proven permanent non-breaks. The current result is about what happens within the present observation window and with the present direct symmetry detector.

So the most careful interpretation is:

- drift onset alone is not enough to predict windowed escape
- post-onset symmetry-score growth is a strong candidate triage signal for windowed escape
- the stronger saddle-probing story needs a better signed trajectory metric before it is established

A follow-up window-extension analysis is archived in `archive/notes/2026-03-window-extension-analysis.md`. In the exploratory canonical suite used at the time, that follow-up showed that all nine originally censored runs eventually became supportive when the observation window was extended to 9600 steps.
