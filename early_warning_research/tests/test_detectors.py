from __future__ import annotations

from early_warning_research.detectors import detect_drift_onset, detect_symmetry_onset
from early_warning_research.suites import DetectorConfig


def test_drift_detector_finds_onset_on_trending_series() -> None:
    config = DetectorConfig(drift_window=20, drift_running_mean_window=4, drift_effect_floor=0.02)
    steps = list(range(1, 121))
    update_norms = [1.0] * 30 + [1.0 - 0.01 * (idx - 30) for idx in range(30, 120)]
    result = detect_drift_onset(update_norms, steps, config)
    assert result.onset_step is not None
    assert result.onset_step < 80


def test_drift_detector_returns_none_for_flat_series() -> None:
    config = DetectorConfig(drift_window=20, drift_running_mean_window=4, drift_effect_floor=0.02)
    steps = list(range(1, 121))
    update_norms = [1.0] * 120
    result = detect_drift_onset(update_norms, steps, config)
    assert result.onset_step is None


def test_symmetry_detector_finds_delayed_onset() -> None:
    config = DetectorConfig(symmetry_baseline_probes=3, symmetry_z_threshold=2.0, symmetry_floor=0.01)
    probe_steps = [15, 30, 45, 60, 75, 90, 105]
    scores = [0.002, 0.003, 0.0025, 0.004, 0.019, 0.024, 0.03]
    result = detect_symmetry_onset(scores, probe_steps, config)
    assert result.onset_step == 75


def test_symmetry_detector_handles_immediate_break() -> None:
    config = DetectorConfig(symmetry_baseline_probes=3, symmetry_floor=0.02, symmetry_consecutive=2)
    probe_steps = [15, 30, 45]
    scores = [0.05, 0.06, 0.07]
    result = detect_symmetry_onset(scores, probe_steps, config)
    assert result.onset_step == 15
