# Window Extension Analysis

## Conclusion

Yes, we should do uncensored runs, and once we do, the picture changes in an important way.

In the canonical `main_paired_mlp` suite, the nine runs that looked censored at the original 300-step window were not stable non-committing holdouts in this setup. They were late breakers. When the same runs were extended under the same detector rules, all nine eventually crossed the direct symmetry detector and became supportive runs.

That means the strongest version of the split-precursor claim is not supported by the current canonical suite. The data do not currently show a durable class of runs with early drift onset but no later direct symmetry escape. Instead, they show a wide distribution of escape times.

The early-warning claim therefore looks stronger than the 300-step view suggested:

- drift onset still appears early, around steps 51 to 55
- direct symmetry detection can occur much later, from step 300 to step 5580 in this follow-up
- the original censored class was largely an artifact of too short an observation window

The careful interpretation is not that the split-precursor idea is impossible. It is that the current canonical evidence does not require it.

## What Was Done

The exact `main_paired_mlp` sweep was rerun with its original 300-step window to identify the runs classified as `censored` under the primary covariance-mismatch detector.

Those same nine runs were then rerun with the same model, same data generation, same detector thresholds, and longer observation windows:

- 1200 steps
- 2400 steps
- 4800 steps
- 9600 steps

Artifacts for this exploratory analysis are in:

`artifacts/exploratory/early_warning_research_window_extension/20260322T135323Z_main_paired_mlp_window_extension/`

## Concrete Results

At the original 300-step window:

- total runs: 27
- censored runs: 9

As the window was extended, the original censored runs resolved as follows:

- at 1200 steps: 4 of 9 resolved, 5 remained censored
- at 2400 steps: 6 of 9 resolved, 3 remained censored
- at 4800 steps: 8 of 9 resolved, 1 remained censored
- at 9600 steps: 9 of 9 resolved, 0 remained censored

The latest covariance-detector onset among those original censored runs was 5580 steps.

Importantly, the drift onset did not move with the longer window. It stayed near the original early onset:

- typical drift onset remained around steps 51 to 55

So the concrete pattern is:

- early drift onset
- potentially very long delay
- eventual direct symmetry detection in every originally censored run tested here

## What This Means

This follow-up weakens the case for saying that drift onset splits cleanly into two fundamentally different fates in the canonical suite.

What the current evidence now supports is simpler:

- drift onset marks entry into a precursor regime very early
- the time to direct symmetry detectability can vary enormously
- some runs only looked like non-committing precursors because the observation window was too short

So the best current statement is:

"In the canonical paired-MLP suite, early drift onset is a robust precursor, while direct symmetry detection may lag by hundreds to thousands of steps."

## Remaining Caution

This does not prove that true non-committing precursors never exist.

It only says that in the current canonical suite, under the current detector and after extending the window to 9600 steps, the original censored runs all eventually became supportive. A genuine split-precursor result would need a different experiment that still preserves unresolved non-breaks even after a much longer observation window.
