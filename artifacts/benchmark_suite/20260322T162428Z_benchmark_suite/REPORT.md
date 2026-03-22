# B1-B4 Consolidated Benchmark Report

Generated: 2026-03-22T16:29:08.131012+00:00

Overall result: the validated benchmark package supports the claims document.

![Suite verdicts](/Users/velocityworks/IdeaProjects/noether-early-warning/artifacts/benchmark_suite/20260322T162428Z_benchmark_suite/figures/suite_verdicts.png)

## Summary

- B1 `SUPPORTED`: drift becomes detectable before direct symmetry detection.
- B2 `SUPPORTED`: direct symmetry detection appears at or before drift in an instant-break regime.
- B3 `SUPPORTED`: under a fixed practical observation budget, drift is the more sensitive detector.
- B4 `SUPPORTED`: at the drift alarm time, direct symmetry is still below its own detection threshold.

## Details

### B1

Claim tested: drift becomes detectable before direct symmetry detection.

Verdict: `SUPPORTED`

- total_runs: `27`
- drift_detected_runs: `27`
- comparable_runs: `27`
- supportive_runs: `27`
- falsifying_runs: `0`
- symmetry_miss_runs: `0`
- drift_miss_runs: `0`
- comparable_fraction: `1.0`
- supportive_fraction_total: `1.0`
- median_lead_steps: `84.0`
- informative: `True`

Artifacts: [/Users/velocityworks/IdeaProjects/noether-early-warning/artifacts/benchmark_suite/20260322T162428Z_benchmark_suite/benchmark1/20260322T162428Z_benchmark1](/Users/velocityworks/IdeaProjects/noether-early-warning/artifacts/benchmark_suite/20260322T162428Z_benchmark_suite/benchmark1/20260322T162428Z_benchmark1)

![B1 representative timeseries](/Users/velocityworks/IdeaProjects/noether-early-warning/artifacts/benchmark_suite/20260322T162428Z_benchmark_suite/benchmark1/20260322T162428Z_benchmark1/figures/representative_timeseries.png)

![B1 onset ordering](/Users/velocityworks/IdeaProjects/noether-early-warning/artifacts/benchmark_suite/20260322T162428Z_benchmark_suite/benchmark1/20260322T162428Z_benchmark1/figures/onset_ordering.png)

### B2

Claim tested: direct symmetry detection appears at or before drift in an instant-break regime.

Verdict: `SUPPORTED`

- total_runs: `27`
- symmetry_detected_runs: `27`
- comparable_runs: `27`
- supportive_runs: `27`
- falsifying_runs: `0`
- symmetry_miss_runs: `0`
- drift_miss_runs: `0`
- comparable_fraction: `1.0`
- supportive_fraction_total: `1.0`
- median_lead_steps: `-37.0`
- informative: `True`

Artifacts: [/Users/velocityworks/IdeaProjects/noether-early-warning/artifacts/benchmark_suite/20260322T162428Z_benchmark_suite/benchmark2/20260322T162601Z_benchmark2](/Users/velocityworks/IdeaProjects/noether-early-warning/artifacts/benchmark_suite/20260322T162428Z_benchmark_suite/benchmark2/20260322T162601Z_benchmark2)

![B2 representative timeseries](/Users/velocityworks/IdeaProjects/noether-early-warning/artifacts/benchmark_suite/20260322T162428Z_benchmark_suite/benchmark2/20260322T162601Z_benchmark2/figures/representative_timeseries.png)

![B2 onset ordering](/Users/velocityworks/IdeaProjects/noether-early-warning/artifacts/benchmark_suite/20260322T162428Z_benchmark_suite/benchmark2/20260322T162601Z_benchmark2/figures/onset_ordering.png)

### B3

Claim tested: under a fixed practical observation budget, drift is the more sensitive detector.

Verdict: `SUPPORTED`

- total_runs: `27`
- drift_detected_runs: `27`
- symmetry_detected_runs: `18`
- drift_detection_rate: `1.0`
- symmetry_detection_rate: `0.6666666666666666`
- detection_rate_gap: `0.33333333333333337`
- margin_threshold: `0.2`
- informative: `True`

Artifacts: [/Users/velocityworks/IdeaProjects/noether-early-warning/artifacts/benchmark_suite/20260322T162428Z_benchmark_suite/benchmark3/20260322T162732Z_benchmark3](/Users/velocityworks/IdeaProjects/noether-early-warning/artifacts/benchmark_suite/20260322T162428Z_benchmark_suite/benchmark3/20260322T162732Z_benchmark3)

![B3 representative timeseries](/Users/velocityworks/IdeaProjects/noether-early-warning/artifacts/benchmark_suite/20260322T162428Z_benchmark_suite/benchmark3/20260322T162732Z_benchmark3/figures/representative_timeseries.png)

![B3 onset ordering](/Users/velocityworks/IdeaProjects/noether-early-warning/artifacts/benchmark_suite/20260322T162428Z_benchmark_suite/benchmark3/20260322T162732Z_benchmark3/figures/onset_ordering.png)

### B4

Claim tested: at the drift alarm time, direct symmetry is still below its own detection threshold.

Verdict: `SUPPORTED`

- total_runs: `27`
- drift_detected_runs: `27`
- alarm_state_runs: `27`
- supportive_runs: `24`
- falsifying_runs: `3`
- no_alarm_state_runs: `0`
- supportive_fraction_alarm_state: `0.8888888888888888`
- informative: `True`

Artifacts: [/Users/velocityworks/IdeaProjects/noether-early-warning/artifacts/benchmark_suite/20260322T162428Z_benchmark_suite/benchmark4/20260322T162735Z_benchmark4](/Users/velocityworks/IdeaProjects/noether-early-warning/artifacts/benchmark_suite/20260322T162428Z_benchmark_suite/benchmark4/20260322T162735Z_benchmark4)

![B4 representative timeseries](/Users/velocityworks/IdeaProjects/noether-early-warning/artifacts/benchmark_suite/20260322T162428Z_benchmark_suite/benchmark4/20260322T162735Z_benchmark4/figures/representative_timeseries.png)

![B4 onset ordering](/Users/velocityworks/IdeaProjects/noether-early-warning/artifacts/benchmark_suite/20260322T162428Z_benchmark_suite/benchmark4/20260322T162735Z_benchmark4/figures/onset_ordering.png)
