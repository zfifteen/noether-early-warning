from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import linregress

from .suites import DetectorConfig


EPS = 1e-12


@dataclass(frozen=True)
class DriftWindowStat:
    window_end_step: int
    slope: float
    p_value: float
    effect: float
    triggered: bool


@dataclass(frozen=True)
class DriftDetectionResult:
    onset_step: int | None
    baseline_mean: float
    smoothed_series: list[float]
    window_stats: list[DriftWindowStat]


@dataclass(frozen=True)
class SymmetryProbeStat:
    step: int
    score: float
    threshold: float
    z_score: float | None
    triggered: bool


@dataclass(frozen=True)
class SymmetryDetectionResult:
    onset_step: int | None
    baseline_mean: float
    baseline_std: float
    threshold: float
    probe_stats: list[SymmetryProbeStat]


def running_mean(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    out: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        out.append(float(np.mean(values[start : idx + 1])))
    return out


def detect_drift_onset(
    update_norms: list[float],
    steps: list[int],
    config: DetectorConfig,
) -> DriftDetectionResult:
    smoothed = running_mean(update_norms, config.drift_running_mean_window)
    if len(smoothed) < config.drift_window:
        return DriftDetectionResult(onset_step=None, baseline_mean=0.0, smoothed_series=smoothed, window_stats=[])

    baseline = float(np.mean(smoothed[: config.drift_window]))
    window_stats: list[DriftWindowStat] = []
    onset: int | None = None
    streak = 0
    first_trigger_step: int | None = None

    for end in range(config.drift_window, len(smoothed) + 1):
        segment = np.asarray(smoothed[end - config.drift_window : end], dtype=float)
        x = np.arange(segment.size, dtype=float)
        slope, _, _, p_value, _ = linregress(x, segment)
        effect = abs(float(np.mean(segment)) - baseline) / (abs(baseline) + EPS)
        triggered = effect >= config.drift_effect_floor and p_value < config.drift_p_threshold
        stat = DriftWindowStat(
            window_end_step=steps[end - 1],
            slope=float(slope),
            p_value=float(p_value),
            effect=float(effect),
            triggered=triggered,
        )
        window_stats.append(stat)

        if triggered:
            streak += 1
            if first_trigger_step is None:
                first_trigger_step = stat.window_end_step
            if streak >= config.drift_consecutive and onset is None:
                onset = first_trigger_step
        else:
            streak = 0
            first_trigger_step = None

    return DriftDetectionResult(
        onset_step=onset,
        baseline_mean=baseline,
        smoothed_series=smoothed,
        window_stats=window_stats,
    )


def detect_symmetry_onset(
    scores: list[float],
    probe_steps: list[int],
    config: DetectorConfig,
) -> SymmetryDetectionResult:
    if not scores:
        return SymmetryDetectionResult(
            onset_step=None,
            baseline_mean=0.0,
            baseline_std=0.0,
            threshold=config.symmetry_floor,
            probe_stats=[],
        )

    probe_stats: list[SymmetryProbeStat] = []
    onset: int | None = None

    if len(scores) >= config.symmetry_consecutive and all(
        score >= config.symmetry_floor for score in scores[: config.symmetry_consecutive]
    ):
        for step, score in zip(probe_steps, scores):
            probe_stats.append(
                SymmetryProbeStat(
                    step=step,
                    score=float(score),
                    threshold=float(config.symmetry_floor),
                    z_score=None,
                    triggered=score >= config.symmetry_floor,
                )
            )
        return SymmetryDetectionResult(
            onset_step=probe_steps[0],
            baseline_mean=float(np.mean(scores[: min(len(scores), config.symmetry_baseline_probes)])),
            baseline_std=float(np.std(scores[: min(len(scores), config.symmetry_baseline_probes)], ddof=0)),
            threshold=float(config.symmetry_floor),
            probe_stats=probe_stats,
        )

    baseline_count = min(len(scores), config.symmetry_baseline_probes)
    baseline = np.asarray(scores[:baseline_count], dtype=float)
    baseline_mean = float(np.mean(baseline))
    baseline_std = float(np.std(baseline, ddof=0))
    threshold = max(baseline_mean + config.symmetry_z_threshold * baseline_std, config.symmetry_floor)

    streak = 0
    first_trigger_step: int | None = None
    for step, score in zip(probe_steps, scores):
        z_score = None if baseline_std <= EPS else (score - baseline_mean) / baseline_std
        triggered = score >= threshold
        probe_stats.append(
            SymmetryProbeStat(
                step=step,
                score=float(score),
                threshold=float(threshold),
                z_score=None if z_score is None else float(z_score),
                triggered=triggered,
            )
        )

    for stat in probe_stats[baseline_count:]:
        if stat.triggered:
            streak += 1
            if first_trigger_step is None:
                first_trigger_step = stat.step
            if streak >= config.symmetry_consecutive and onset is None:
                onset = first_trigger_step
        else:
            streak = 0
            first_trigger_step = None

    return SymmetryDetectionResult(
        onset_step=onset,
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        threshold=float(threshold),
        probe_stats=probe_stats,
    )
