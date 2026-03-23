Your repo is not just showing that drift fires before direct symmetry detection in the tested gradual regime. It also contains the pieces for a stronger claim:

**the observed lead can be decomposed into**

1. **structural detector latency** — delay caused purely by how the two detectors are implemented and sampled, and
2. **regime latency** — extra delay caused by the actual training dynamics.

That matters because it lets you say something sharper than “drift comes first.” It lets you ask:

**How much of the lead is genuinely in the system, and how much is just in the instrumentation?**

That is a better scientific object.

The repo already gives the ingredients. The drift detector runs on every training step, with a running mean, a 50-step window, and a 2-window confirmation rule. The symmetry detector is only checked on probe steps every 15 updates, uses 3 baseline probes, and then needs 2 consecutive triggered probes.

That means the two detectors do **not** have equal opportunity to fire. Even before any regime effect exists, the symmetry detector carries a built-in sampling and confirmation lag relative to drift. In your current config, the drift detector can in principle lock in around the 50-step neighborhood, while the symmetry detector’s first normal post-baseline onset cannot happen until the 4th probe, which at 15-step probing is around step 60.

So there is a built-in latency floor of about **10 steps** favoring drift.

That does **not** kill the result. Quite the opposite. Your reported median lead in B1 is **+84 steps**, and the instant-break control flips to **-37 steps**. The fact that the reversal survives the same detector machinery is strong evidence that the effect is not just an artifact of detector cadence. ([GitHub][1]) The exact alarm-state check in B4 also helps, because it partially breaks the probe-schedule objection by rescoring the symmetry observable at the drift alarm step itself rather than only at scheduled probes. ([GitHub][1])

So the new insight is this:

**Your benchmark is really measuring a latency competition between two observation channels.**
One channel is continuous and local in time (drift).
The other is sparse and thresholded through scheduled probes (symmetry).
The important quantity is not raw lead alone, but:

**excess lead = observed lead − detector-latency floor**

If that excess lead stays positive across the gradual regime and goes negative in the instant-break control, then you have isolated a more invariant object than “drift wins”: you have isolated a **regime-dependent excess warning horizon**.

That would be a cleaner and more portable concept than the current headline claim.

Here is the falsifiable prediction that drops out of this:

If you sweep only the probe cadence and confirmation settings while leaving the training dynamics fixed, then:

* the **raw** B1 lead should shrink as symmetry probing gets denser and easier,
* but the **excess lead** should remain positive in the gradual regime,
* and the B2 reversal should remain non-positive or negative after the same correction.

If instead the B1 advantage collapses to near zero once you normalize for detector latency, then the current result is mostly an instrumentation effect. If it survives, you have a much stronger result than the README currently states. The codebase is already set up for this kind of detector-threshold sensitivity work through its detector config and stability machinery.

I’d phrase the insight like this:

> The repo is not only showing earlier detection. It is implicitly showing that gradual symmetry breaking creates a positive excess warning horizon for drift after accounting for detector latency, while instant-break controls collapse or reverse that excess horizon.

That is new, tighter, and more transferable.

It also gives you a natural next benchmark family without changing the project’s core idea:
measure a **latency-normalized lead curve** across probe cadence, baseline count, consecutive-hit rules, and drift window size. If the sign is stable after normalization, your claim gets much harder to dismiss as “detector choice.” The repo already documents the current four-part evidence package and its main results, so this would read as a natural next layer rather than a rewrite. ([GitHub][1])

My honest read: this is the best new insight hiding in the repo right now. It upgrades the project from “drift seems earlier here” to “the regime creates a real warning-horizon surplus beyond detector mechanics.” That is a stronger idea.

---

Investigation Audit

I audited this lead against the detector implementation and the benchmark artifacts rather than treating it as a free-floating interpretation. The code audit used `early_warning_research/detectors.py` and `early_warning_research/suites.py`. The artifact audit used `artifacts/benchmark1/20260322T152418Z_benchmark1/runs.csv`, `artifacts/benchmark2/20260322T154257Z_benchmark2/runs.csv`, and `artifacts/benchmark4/20260322T161101Z_benchmark4/runs.csv`.

The detector mechanics are confirmed exactly as described, and they sharpen one number in the note. Drift detection uses a 10-step running mean, a 50-step window, and a 2-window confirmation rule. Under the current configuration, the earliest possible drift onset in a gradual run is step `51`: the first 50-step window cannot trigger because it defines the baseline, so the first possible triggered window ends at step `51`, and with two consecutive triggered windows the onset is recorded at the first trigger step. Symmetry detection runs only on probe steps every `15` updates, uses `3` baseline probes, and then requires `2` consecutive triggered probes. In the normal post-baseline path, the earliest possible symmetry onset is step `60`. That makes the structural detector-latency floor in the gradual regime `9` steps favoring drift, not merely “about 10.”

This floor is real, but it is much smaller than the observed `B1` lead. In `artifacts/benchmark1/20260322T152418Z_benchmark1/runs.csv`, raw lead ranges from `8` to `5529` steps with median `84`. After subtracting the `9`-step detector-latency floor, excess lead ranges from `-1` to `5520` with median `75`. Of the `27` gradual runs, `22` remain strictly positive, `26` are nonnegative, and only `1` goes slightly negative. So the `B1` result does not collapse under latency normalization.

The instant-break control sharpens the interpretation further. In `artifacts/benchmark2/20260322T154257Z_benchmark2/runs.csv`, symmetry onset is locked at step `15` in every run because the first two probe scores exceed the symmetry floor and the detector takes the special early-onset path already implemented in the code. Drift onset then occurs at `51-75`, producing raw leads from `-60` to `-36` with median `-37`. After the same `9`-step correction, excess lead remains negative in all `27` runs, with median `-46`. That means the reversal survives normalization; the detector machinery alone does not explain the sign flip.

The subgroup structure also supports the core idea. Drift onset in `B1` stays tight (`51-61`, mean `52.19`, coefficient of variation `0.038`), while symmetry onset remains highly elastic (`60-5580`, mean `670`, coefficient of variation `1.872`). Input scale strongly affects symmetry timing (one-way ANOVA `p = 0.0067`) but does not move drift timing much (`p = 0.587`). The largest excess leads all occur in the low-signal `scale = 0.75` regime, where median excess lead is `1408` steps. So the warning-horizon surplus grows exactly where the direct symmetry channel becomes slowest.

`B4` also supports the anti-artifact reading. It does not merely wait for the next scheduled symmetry probe. It rescored the symmetry observable at the exact saved drift-alarm state in `artifacts/benchmark4/20260322T161101Z_benchmark4/runs.csv`. In `24` of `27` runs, the alarm-state symmetry score remained below threshold at that exact step. That weakens the objection that the observed advantage is only a sparse-probe artifact.

The only part of this note that remains untested rather than already audited is the proposed cadence/confirmation sweep. The repo has enough structure to run that next experiment, but the current artifact set does not yet answer it. So the present audit supports the decomposition itself and the existence of a positive excess warning horizon in the benchmarked gradual regime; it does not yet establish invariance of that excess horizon across detector-setting sweeps.

---

Cadence and Confirmation Sweep: Experiment Design

The next experiment should test whether the excess warning horizon survives deliberate changes to detector cadence and confirmation rules while holding the training dynamics fixed. The object of interest is no longer raw lead by itself. It is the sign and magnitude of latency-normalized lead after the detector machinery has been changed on purpose.

The cleanest primary sweep is a symmetry-channel sweep with the drift detector held fixed at the current `B1/B2` settings. That isolates the question raised by this note: if the symmetry detector is made denser or easier to confirm, does the gradual-regime advantage remain after normalization? The primary sweep parameters should be:

- symmetry probe cadence `every_steps`: `{5, 10, 15, 20, 30}`
- symmetry baseline probes: `{2, 3, 4}`
- symmetry consecutive hits: `{1, 2, 3}`

Under the current implementation, the structural floor in the normal gradual path is:

`structural_floor = every_steps * (baseline_probes + 1) - (drift_window + 1)`

with the current drift detector held fixed at `drift_window = 50`, so the current benchmark configuration gives `15 * (3 + 1) - 51 = 9`. This formula should be computed directly for every sweep point and recorded alongside the observed lead. One subtle but important feature of the implementation is that confirmation count affects robustness more than the earliest timestamp itself, because onset is backdated to the first triggered step once the streak is confirmed. So the sweep should not assume that increasing confirmation necessarily increases the formal latency floor. It may instead reduce detection frequency or increase instability.

The benchmark matrix should be run on both existing regimes:

- `main_paired_mlp` for the gradual regime
- `instant_break_control` for the reversal control

The training sweep itself should stay unchanged from the validated suite:

- seeds: `{0, 1, 2}`
- learning rates: `{0.02, 0.04, 0.08}`
- input scales: `{0.75, 1.25, 1.75}`
- total runs per detector-setting point: `27` in `B1` and `27` in `B2`

For each detector-setting point, the analysis should record:

- raw lead distribution in `B1`
- excess lead distribution in `B1`
- raw lead distribution in `B2`
- excess lead distribution in `B2`
- comparable fraction in both regimes
- supportive fraction in `B1`
- reversal fraction in `B2`
- subgroup medians by input scale, since low-signal runs are where the current surplus is largest

The primary falsifiable predictions are:

1. As symmetry probing becomes denser or easier to confirm, raw `B1` lead should shrink.
2. Despite that shrinkage, median excess lead in `B1` should remain positive across most of the sweep.
3. The `B2` control should remain non-positive after the same correction.
4. The largest positive excess lead should continue to concentrate in the low-signal `scale = 0.75` regime.

The most useful summary outputs for this sweep are not pass/fail verdicts alone. They should be:

- a latency-normalized lead table by detector setting
- a sign-stability table showing positive / nonnegative / negative excess runs
- a heatmap of median excess lead across `every_steps x baseline_probes`
- a paired `B1` versus `B2` comparison plot for median excess lead

There is also a natural secondary sweep once the symmetry-only sweep is complete. That secondary pass should vary the confirmation rules on both channels:

- drift consecutive hits: `{1, 2, 3}`
- symmetry consecutive hits: `{1, 2, 3}`

The reason to keep this secondary is that, under the current onset-labeling semantics, confirmation count mainly changes whether an onset is declared reliably, not the theoretical earliest onset stamp. So it is more a robustness sweep than a pure latency sweep.

The experiment should be considered supportive of this insight if the following pattern survives:

- `B1` median excess lead remains positive across the great majority of settings
- `B2` median excess lead remains non-positive across the same settings
- the sign split between gradual and instant-break regimes remains clean even when raw lead moves substantially

It should be considered damaging to the insight if latency normalization causes the gradual-regime advantage to collapse broadly toward zero while also erasing the regime split. If that happens, the current result is mostly detector mechanics. If it does not happen, then this note's stronger claim is validated: the repo is measuring a real regime-dependent warning-horizon surplus, not just unequal detector opportunity.

---

Executed Sweep Results

I ran the primary symmetry-channel sweep described above and wrote the full outputs to `artifacts/exploratory/20260323T082341Z_latency_sweep`. The runner lives at `scripts/run_latency_sweep.py`. To keep the experiment exact while avoiding brute-force retraining at every cadence, each run was executed once with dense symmetry probing every `5` steps, and the coarser cadences `{10, 15, 20, 30}` were derived by subsampling those same probe traces. Because every cadence in the sweep is a multiple of `5`, this preserves the actual symmetry scores at the tested probe times.

The strongest result is that the regime split survives the entire sweep. Across all `45` detector settings, the median excess lead stayed positive in the gradual `B1` regime and non-positive in the instant-break `B2` control. The sweep therefore produced a clean median sign split in `45/45` settings. This is stronger than the original prediction, which only required the split to survive in most settings.

The median excess ranges are stable and separated:

- `B1` excess-median range across settings: `45` to `110`
- `B2` excess-median range across settings: `-121` to `-11`

So the latency-normalized object is not wobbling around zero. It stays positive in the gradual regime and negative in the instant-break regime throughout the tested grid.

The raw-lead prediction also behaves as expected. Under the current validated baseline/count setting (`baseline_probes = 3`, `symmetry_consecutive = 2`), making symmetry probing denser shrinks the raw `B1` lead while leaving excess lead positive:

- cadence `30`: structural floor `69`, `B1` raw median `128`, `B1` excess median `59`
- cadence `20`: structural floor `29`, `B1` raw median `108`, `B1` excess median `79`
- cadence `15`: structural floor `9`, `B1` raw median `84`, `B1` excess median `75`
- cadence `10`: structural floor `-11`, `B1` raw median `79`, `B1` excess median `90`
- cadence `5`: structural floor `-31`, `B1` raw median `74`, `B1` excess median `105`

So the raw lead does compress as the symmetry channel is given more opportunity, but the normalized lead does not collapse. The same cadence line in `B2` moves from raw median `-22` at cadence `30` to `-47` at cadence `5`, while excess lead stays negative throughout (`-91` to `-16`).

The easiest symmetry setting tested was cadence `5`, baseline probes `2`, and consecutive hits `1`. Even there, the gradual regime still retained a positive excess horizon:

- `B1`: raw median `53`, excess median `89`, positive excess in `27/27` runs
- `B2`: raw median `-47`, excess median `-11`, negative excess in `27/27` runs

So the effect does not disappear when the symmetry detector is made as permissive as this sweep allows.

The strictest tested setting was cadence `30`, baseline probes `4`, and consecutive hits `3`. There the raw lead becomes much larger, but the normalized result still shows the same sign split:

- `B1`: raw median `188`, excess median `89`
- `B2`: raw median `-22`, excess median `-121`

That makes the interpretation cleaner: changing cadence and confirmation rules moves the raw lead a lot, but it does not erase the regime-dependent excess horizon.

The run-level story is slightly weaker than the median story, and that distinction matters. Full sign consistency, meaning every gradual run stays nonnegative while every instant-break run stays nonpositive, held in `18/45` settings. So the strongest honest statement is not that every run is invariant under the sweep. It is that the regime-level excess-horizon split is stable across the whole tested grid.

The low-signal practical implication also survives. Under the original benchmark setting (`15/3/2`), the low-signal `scale = 0.75` subgroup still has median excess lead `1408`. Across the full sweep, the largest low-signal excess medians reach `1873`. So the warning-horizon surplus continues to be largest in the regime where the direct symmetry channel is slowest.

Taken together, this experiment supports the stronger version of the note. The repo is not only showing raw detector ordering. It is showing a regime-dependent excess warning horizon that survives deliberate changes to probe cadence, baseline count, and confirmation rules. Detector mechanics affect the magnitude of raw lead, but they do not explain away the sign split between gradual and instant-break regimes in this tested grid.

[1]: https://github.com/zfifteen/noether-early-warning "GitHub - zfifteen/noether-early-warning: Atomic benchmark suite showing drift can act as an early warning before direct symmetry detection in gradual-breaking regimes, with reversal controls, finite-budget sensitivity tests, and exact alarm-time validation. · GitHub"
