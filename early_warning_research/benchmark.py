from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from .detectors import detect_drift_onset, detect_symmetry_onset
from .experiments import execute_run, library_versions
from .models import PairedMLP
from .plotting import plot_benchmark_lead_distributions, plot_benchmark_resolution_curves
from .suites import PairedMLPConfig, SuiteConfig, get_suite, suite_to_dict


BENCHMARK_NAME = "canonical"
DETECTOR_NAMES = ("covariance_mismatch", "mean_gradient_mismatch", "activation_stat_mismatch")
LONG_WINDOW_CAP = 9600


@dataclass(frozen=True)
class BenchmarkSuite:
    name: str
    role: Literal["positive", "instant_break", "fixed_point_negative"]
    suite: SuiteConfig
    short_window_steps: int
    long_window_cap: int = LONG_WINDOW_CAP


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def default_benchmark_output_root() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "artifacts" / "canonical_benchmark"


def _run_id(seed: int, learning_rate: float, input_scale: float) -> str:
    return f"seed{seed}_lr{learning_rate:.4f}_scale{input_scale:.2f}"


def _model_seed(seed: int, learning_rate: float, input_scale: float) -> int:
    return seed * 1000 + int(round(learning_rate * 10_000)) + int(round(input_scale * 100))


def _make_fixed_point_negative_suite(base_suite: SuiteConfig, name: str, description: str) -> SuiteConfig:
    model_cfg = base_suite.model
    assert isinstance(model_cfg, PairedMLPConfig)
    return replace(
        base_suite,
        name=name,
        description=description,
        model=replace(model_cfg, init_noise=0.0),
    )


def build_canonical_benchmark(smoke: bool = False) -> dict[str, BenchmarkSuite]:
    full_positive = get_suite("main_paired_mlp", smoke=smoke)
    full_control = get_suite("instant_break_control", smoke=smoke)
    stochastic_positive = get_suite("main_paired_mlp_stochastic", smoke=smoke)
    stochastic_control = get_suite("instant_break_control_stochastic", smoke=smoke)

    full_negative = _make_fixed_point_negative_suite(
        full_positive,
        name="full_batch_fixed_point_negative",
        description="Exact-symmetric fixed-point negative built from the full-batch paired MLP.",
    )
    stochastic_negative = _make_fixed_point_negative_suite(
        stochastic_positive,
        name="stochastic_fixed_point_negative",
        description="Exact-symmetric fixed-point negative built from the stochastic paired MLP.",
    )

    return {
        "full_batch_positive": BenchmarkSuite(
            name="full_batch_positive",
            role="positive",
            suite=replace(
                full_positive,
                name="full_batch_positive",
                description="Canonical full-batch paired MLP confirmation suite.",
            ),
            short_window_steps=full_positive.training.total_steps,
        ),
        "full_batch_instant_break": BenchmarkSuite(
            name="full_batch_instant_break",
            role="instant_break",
            suite=replace(
                full_control,
                name="full_batch_instant_break",
                description="Canonical full-batch instant-break control.",
            ),
            short_window_steps=full_control.training.total_steps,
        ),
        "full_batch_fixed_point_negative": BenchmarkSuite(
            name="full_batch_fixed_point_negative",
            role="fixed_point_negative",
            suite=full_negative,
            short_window_steps=full_positive.training.total_steps,
        ),
        "stochastic_positive": BenchmarkSuite(
            name="stochastic_positive",
            role="positive",
            suite=replace(
                stochastic_positive,
                name="stochastic_positive",
                description="Canonical stochastic paired MLP confirmation suite.",
            ),
            short_window_steps=stochastic_positive.training.total_steps,
        ),
        "stochastic_instant_break": BenchmarkSuite(
            name="stochastic_instant_break",
            role="instant_break",
            suite=replace(
                stochastic_control,
                name="stochastic_instant_break",
                description="Canonical stochastic instant-break control.",
            ),
            short_window_steps=stochastic_control.training.total_steps,
        ),
        "stochastic_fixed_point_negative": BenchmarkSuite(
            name="stochastic_fixed_point_negative",
            role="fixed_point_negative",
            suite=stochastic_negative,
            short_window_steps=stochastic_positive.training.total_steps,
        ),
    }


def build_fixed_point_negative_pair(
    suite: SuiteConfig,
    seed: int,
    learning_rate: float,
    input_scale: float,
) -> tuple[PairedMLP, dict[str, torch.Tensor], float]:
    model_cfg = suite.model
    assert isinstance(model_cfg, PairedMLPConfig)
    if model_cfg.init_noise != 0.0:
        raise ValueError("fixed-point negative helper expects init_noise=0")

    rng = np.random.RandomState(seed)
    total = suite.training.train_size + suite.training.probe_size
    x = rng.randn(total, model_cfg.input_dim) * input_scale
    x_tensor = torch.tensor(x, dtype=torch.float64)
    run_seed = _model_seed(seed, learning_rate, input_scale)

    teacher = PairedMLP(
        model_cfg.input_dim,
        model_cfg.pair_count,
        run_seed,
        model_cfg.init_noise,
        hidden_layers=model_cfg.hidden_layers,
        use_layer_norm=model_cfg.use_layer_norm,
    )
    student = PairedMLP(
        model_cfg.input_dim,
        model_cfg.pair_count,
        run_seed,
        model_cfg.init_noise,
        hidden_layers=model_cfg.hidden_layers,
        use_layer_norm=model_cfg.use_layer_norm,
    )

    with torch.no_grad():
        y_tensor = teacher(x_tensor)
    train_size = suite.training.train_size
    data = {
        "x_train": x_tensor[:train_size],
        "y_train": y_tensor[:train_size],
        "x_probe": x_tensor[train_size:],
        "y_probe": y_tensor[train_size:],
    }
    initial_loss = float(torch.mean((student(data["x_train"]) - data["y_train"]) ** 2).item())
    return student, data, initial_loss


def _negative_probe_steps(total_steps: int, every_steps: int) -> list[int]:
    steps = list(range(every_steps, total_steps + 1, every_steps))
    if not steps or steps[-1] != total_steps:
        steps.append(total_steps)
    return steps


def _classify_short_status(role: str, drift_onset: int | None, symmetry_onset: int | None) -> str:
    if role == "positive":
        if drift_onset is None:
            return "drift_miss"
        if symmetry_onset is None:
            return "censored"
        return "supportive" if symmetry_onset > drift_onset else "falsifying"

    if role == "instant_break":
        if symmetry_onset is None:
            return "censored" if drift_onset is not None else "no_signal"
        if drift_onset is None or symmetry_onset <= drift_onset:
            return "falsifying"
        return "supportive"

    return "negative_clean"


def _classify_resolved_status(role: str, short_drift_onset: int | None, resolved_symmetry_onset: int | None) -> str:
    if role == "positive":
        if short_drift_onset is None:
            return "drift_miss"
        if resolved_symmetry_onset is None:
            return "unresolved"
        return "supportive" if resolved_symmetry_onset > short_drift_onset else "falsifying"

    if role == "instant_break":
        if resolved_symmetry_onset is None:
            return "unresolved" if short_drift_onset is not None else "no_signal"
        if short_drift_onset is None or resolved_symmetry_onset <= short_drift_onset:
            return "falsifying"
        return "supportive"

    return "negative_clean"


def _build_short_row(
    benchmark_suite: BenchmarkSuite,
    record: object,
    detector_name: str,
) -> dict[str, object]:
    short_drift = record.drift_onset_step
    short_symmetry = record.detector_onsets[detector_name]
    short_lead = None if short_drift is None or short_symmetry is None else short_symmetry - short_drift
    short_status = _classify_short_status(benchmark_suite.role, short_drift, short_symmetry)
    return {
        "suite_name": benchmark_suite.name,
        "suite_role": benchmark_suite.role,
        "detector_name": detector_name,
        "run_id": record.run_id,
        "seed": record.seed,
        "learning_rate": record.learning_rate,
        "input_scale": record.input_scale,
        "short_window_steps": benchmark_suite.short_window_steps,
        "long_window_cap": benchmark_suite.long_window_cap,
        "short_drift_onset_step": short_drift,
        "short_symmetry_onset_step": short_symmetry,
        "short_lead_steps": short_lead,
        "short_status": short_status,
        "short_comparable": short_lead is not None,
        "short_censored": short_status == "censored",
        "short_drift_detected": short_drift is not None,
        "short_symmetry_detected": short_symmetry is not None,
        "short_max_symmetry_score": record.detector_max_scores[detector_name],
    }


def _build_resolved_row(
    short_row: dict[str, object],
    benchmark_suite: BenchmarkSuite,
    long_record: object | None,
) -> dict[str, object]:
    detector_name = str(short_row["detector_name"])
    short_symmetry = short_row["short_symmetry_onset_step"]
    long_symmetry = None if long_record is None else long_record.detector_onsets[detector_name]
    resolved_symmetry = short_symmetry if short_symmetry is not None else long_symmetry
    short_drift = short_row["short_drift_onset_step"]
    resolved_lead = None if short_drift is None or resolved_symmetry is None else int(resolved_symmetry) - int(short_drift)
    resolved_status = _classify_resolved_status(
        benchmark_suite.role,
        None if short_drift is None else int(short_drift),
        None if resolved_symmetry is None else int(resolved_symmetry),
    )
    return {
        **short_row,
        "reran_to_long_cap": long_record is not None,
        "long_total_steps": None if long_record is None else long_record.total_steps,
        "long_drift_onset_step": None if long_record is None else long_record.drift_onset_step,
        "long_symmetry_onset_step": long_symmetry,
        "resolved_symmetry_onset_step": resolved_symmetry,
        "resolved_lead_steps": resolved_lead,
        "resolved_status": resolved_status,
    }


def _build_negative_rows(
    benchmark_suite: BenchmarkSuite,
    seed: int,
    learning_rate: float,
    input_scale: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    student, data, initial_loss = build_fixed_point_negative_pair(
        benchmark_suite.suite,
        seed,
        learning_rate,
        input_scale,
    )
    loss = torch.mean((student(data["x_train"]) - data["y_train"]) ** 2)
    student.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = float(
        torch.linalg.vector_norm(
            torch.cat([param.grad.detach().flatten() for param in student.parameters() if param.grad is not None])
        ).item()
    )

    run_id = _run_id(seed, learning_rate, input_scale)
    long_steps = list(range(1, benchmark_suite.long_window_cap + 1))
    update_norms = [0.0] * benchmark_suite.long_window_cap
    long_drift = detect_drift_onset(update_norms, long_steps, benchmark_suite.suite.detector)
    long_probe_steps = _negative_probe_steps(benchmark_suite.long_window_cap, benchmark_suite.suite.probe.every_steps)
    zero_scores = [0.0] * len(long_probe_steps)
    long_symmetry_results = {
        detector_name: detect_symmetry_onset(zero_scores, long_probe_steps, benchmark_suite.suite.detector)
        for detector_name in DETECTOR_NAMES
    }

    short_rows: list[dict[str, object]] = []
    resolved_rows: list[dict[str, object]] = []
    for detector_name in DETECTOR_NAMES:
        short_row = {
            "suite_name": benchmark_suite.name,
            "suite_role": benchmark_suite.role,
            "detector_name": detector_name,
            "run_id": run_id,
            "seed": seed,
            "learning_rate": learning_rate,
            "input_scale": input_scale,
            "short_window_steps": benchmark_suite.short_window_steps,
            "long_window_cap": benchmark_suite.long_window_cap,
            "short_drift_onset_step": None,
            "short_symmetry_onset_step": None,
            "short_lead_steps": None,
            "short_status": "negative_clean",
            "short_comparable": False,
            "short_censored": False,
            "short_drift_detected": False,
            "short_symmetry_detected": False,
            "short_max_symmetry_score": 0.0,
        }
        short_rows.append(short_row)
        resolved_rows.append(
            {
                **short_row,
                "reran_to_long_cap": False,
                "long_total_steps": benchmark_suite.long_window_cap,
                "long_drift_onset_step": long_drift.onset_step,
                "long_symmetry_onset_step": long_symmetry_results[detector_name].onset_step,
                "resolved_symmetry_onset_step": long_symmetry_results[detector_name].onset_step,
                "resolved_lead_steps": None,
                "resolved_status": "negative_clean",
            }
        )

    metadata = {
        "suite_name": benchmark_suite.name,
        "run_id": run_id,
        "seed": seed,
        "learning_rate": learning_rate,
        "input_scale": input_scale,
        "initial_loss": initial_loss,
        "first_step_grad_norm": grad_norm,
    }
    return short_rows, resolved_rows, metadata


def _needs_long_followup(benchmark_suite: BenchmarkSuite, short_rows: list[dict[str, object]]) -> bool:
    if benchmark_suite.role == "positive":
        short_drift_detected = any(bool(row["short_drift_detected"]) for row in short_rows)
        return short_drift_detected and any(str(row["short_status"]) == "censored" for row in short_rows)
    if benchmark_suite.role == "instant_break":
        return any(not bool(row["short_symmetry_detected"]) for row in short_rows)
    return False


def _scorecard_metrics(rows: list[dict[str, object]], role: str) -> dict[str, object]:
    total_runs = len(rows)
    short_comparable = sum(bool(row["short_comparable"]) for row in rows)
    short_censored = sum(str(row["short_status"]) == "censored" for row in rows)
    resolved_leads = [int(row["resolved_lead_steps"]) for row in rows if row["resolved_lead_steps"] is not None]
    resolved_comparable = len(resolved_leads)
    median_resolved_lead = None if not resolved_leads else float(np.median(resolved_leads))

    supportive_runs = sum(str(row["resolved_status"]) == "supportive" for row in rows)
    falsifying_runs = sum(str(row["resolved_status"]) == "falsifying" for row in rows)
    unresolved_runs = sum(str(row["resolved_status"]) == "unresolved" for row in rows)
    drift_miss_runs = sum(str(row["resolved_status"]) == "drift_miss" for row in rows)
    no_signal_runs = sum(str(row["resolved_status"]) == "no_signal" for row in rows)

    drift_false_positive_runs = sum(row["long_drift_onset_step"] is not None for row in rows)
    symmetry_false_positive_runs = sum(
        str(row["resolved_status"]) in {"symmetry_false_positive", "both_false_positive"}
        or row["resolved_symmetry_onset_step"] is not None
        for row in rows
    )

    supportive_fraction = 0.0 if total_runs == 0 else supportive_runs / total_runs
    falsifying_fraction = 0.0 if total_runs == 0 else falsifying_runs / total_runs
    drift_false_positive_rate = 0.0 if total_runs == 0 else drift_false_positive_runs / total_runs
    symmetry_false_positive_rate = 0.0 if total_runs == 0 else symmetry_false_positive_runs / total_runs

    passes_suite_gate = False
    if role == "positive":
        passes_suite_gate = supportive_fraction >= 0.8 and median_resolved_lead is not None and median_resolved_lead > 0
    elif role == "instant_break":
        passes_suite_gate = falsifying_fraction >= 0.8 and median_resolved_lead is not None and median_resolved_lead < 0
    elif role == "fixed_point_negative":
        passes_suite_gate = drift_false_positive_rate <= 0.1 and symmetry_false_positive_rate == 0.0

    return {
        "total_runs": total_runs,
        "short_window_comparable_runs": short_comparable,
        "short_window_censored_runs": short_censored,
        "resolved_comparable_runs": resolved_comparable,
        "supportive_runs": supportive_runs,
        "supportive_fraction": supportive_fraction,
        "falsifying_runs": falsifying_runs,
        "falsifying_fraction": falsifying_fraction,
        "unresolved_runs": unresolved_runs,
        "drift_miss_runs": drift_miss_runs,
        "no_signal_runs": no_signal_runs,
        "drift_false_positive_runs": drift_false_positive_runs,
        "drift_false_positive_rate": drift_false_positive_rate,
        "symmetry_false_positive_runs": symmetry_false_positive_runs,
        "symmetry_false_positive_rate": symmetry_false_positive_rate,
        "median_resolved_lead_steps": median_resolved_lead,
        "passes_suite_gate": passes_suite_gate,
    }


def _build_detector_scorecards(
    benchmark_suites: dict[str, BenchmarkSuite],
    resolved_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    scorecards: list[dict[str, object]] = []
    for suite_name, benchmark_suite in benchmark_suites.items():
        for detector_name in DETECTOR_NAMES:
            rows = [
                row
                for row in resolved_rows
                if row["suite_name"] == suite_name and row["detector_name"] == detector_name
            ]
            metrics = _scorecard_metrics(rows, benchmark_suite.role)
            scorecards.append(
                {
                    "suite_name": suite_name,
                    "suite_role": benchmark_suite.role,
                    "detector_name": detector_name,
                    **metrics,
                }
            )
    return scorecards


def _build_suite_scorecards(
    benchmark_suites: dict[str, BenchmarkSuite],
    detector_scorecards: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for suite_name, benchmark_suite in benchmark_suites.items():
        suite_cards = [row for row in detector_scorecards if row["suite_name"] == suite_name]
        if benchmark_suite.role == "positive":
            qualifying = [row["detector_name"] for row in suite_cards if row["passes_suite_gate"]]
            suite_passed = bool(qualifying)
            summary = ",".join(qualifying)
        elif benchmark_suite.role == "instant_break":
            suite_passed = all(bool(row["passes_suite_gate"]) for row in suite_cards)
            summary = "all_detectors" if suite_passed else "control_failure"
        else:
            suite_passed = (
                suite_cards[0]["drift_false_positive_rate"] <= 0.1
                and all(row["symmetry_false_positive_rate"] == 0.0 for row in suite_cards)
            )
            summary = "negative_clean" if suite_passed else "negative_false_positive"

        rows.append(
            {
                "suite_name": suite_name,
                "suite_role": benchmark_suite.role,
                "short_window_steps": benchmark_suite.short_window_steps,
                "long_window_cap": benchmark_suite.long_window_cap,
                "suite_passed": suite_passed,
                "summary": summary,
            }
        )
    return rows


def _recommended_stochastic_detector(detector_scorecards: list[dict[str, object]]) -> str | None:
    positive_cards = {
        row["detector_name"]: row
        for row in detector_scorecards
        if row["suite_name"] == "stochastic_positive"
    }
    control_cards = {
        row["detector_name"]: row
        for row in detector_scorecards
        if row["suite_name"] == "stochastic_instant_break"
    }
    negative_cards = {
        row["detector_name"]: row
        for row in detector_scorecards
        if row["suite_name"] == "stochastic_fixed_point_negative"
    }

    candidates: list[dict[str, object]] = []
    for detector_name in DETECTOR_NAMES:
        positive = positive_cards[detector_name]
        control = control_cards[detector_name]
        negative = negative_cards[detector_name]
        if control["falsifying_fraction"] < 0.8:
            continue
        if negative["symmetry_false_positive_rate"] > 0.0:
            continue
        candidates.append(positive)

    if not candidates:
        return None

    candidates.sort(
        key=lambda row: (
            float(row["supportive_fraction"]),
            float("-inf") if row["median_resolved_lead_steps"] is None else float(row["median_resolved_lead_steps"]),
            -float(row["short_window_censored_runs"]) / max(1, int(row["total_runs"])),
        ),
        reverse=True,
    )
    return str(candidates[0]["detector_name"])


def _resolution_curve_rows(resolved_rows: list[dict[str, object]], benchmark_suites: dict[str, BenchmarkSuite]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for suite_name, benchmark_suite in benchmark_suites.items():
        if benchmark_suite.role != "positive":
            continue
        for detector_name in DETECTOR_NAMES:
            detector_rows = [
                row
                for row in resolved_rows
                if row["suite_name"] == suite_name and row["detector_name"] == detector_name
            ]
            censored_rows = [row for row in detector_rows if str(row["short_status"]) == "censored"]
            total = len(censored_rows)
            if total == 0:
                rows.append(
                    {
                        "suite_name": suite_name,
                        "detector_name": detector_name,
                        "step": benchmark_suite.short_window_steps,
                        "resolved_count": 0,
                        "remaining_count": 0,
                        "short_censored_total": 0,
                        "resolved_fraction": 1.0,
                    }
                )
                continue
            resolution_steps = sorted(
                int(row["resolved_symmetry_onset_step"])
                for row in censored_rows
                if row["resolved_symmetry_onset_step"] is not None
            )
            checkpoints = sorted(set([benchmark_suite.short_window_steps, *resolution_steps, benchmark_suite.long_window_cap]))
            for step in checkpoints:
                resolved_count = sum(resolution_step <= step for resolution_step in resolution_steps)
                rows.append(
                    {
                        "suite_name": suite_name,
                        "detector_name": detector_name,
                        "step": step,
                        "resolved_count": resolved_count,
                        "remaining_count": total - resolved_count,
                        "short_censored_total": total,
                        "resolved_fraction": resolved_count / total,
                    }
                )
    return rows


def _benchmark_verdict(suite_scorecards: list[dict[str, object]], recommended_stochastic_detector: str | None) -> str:
    passes = {row["suite_name"]: bool(row["suite_passed"]) for row in suite_scorecards}
    positive_pass = passes["full_batch_positive"] and passes["stochastic_positive"]
    controls_pass = passes["full_batch_instant_break"] and passes["stochastic_instant_break"]
    negatives_pass = passes["full_batch_fixed_point_negative"] and passes["stochastic_fixed_point_negative"]

    if controls_pass and negatives_pass and positive_pass and recommended_stochastic_detector is not None:
        return "SUPPORTED"
    if not controls_pass or not negatives_pass:
        return "FALSIFIED"
    return "INCONCLUSIVE"


def run_canonical_benchmark(
    output_root: str | Path | None = None,
    smoke: bool = False,
    quiet: bool = False,
) -> dict[str, object]:
    benchmark_suites = build_canonical_benchmark(smoke=smoke)
    base_output = Path(output_root) if output_root is not None else default_benchmark_output_root()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = base_output / f"{timestamp}_{BENCHMARK_NAME}"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    short_rows: list[dict[str, object]] = []
    resolved_rows: list[dict[str, object]] = []
    negative_metadata_rows: list[dict[str, object]] = []
    suite_parameters: dict[str, dict[str, object]] = {}

    for benchmark_suite in benchmark_suites.values():
        suite_parameters[benchmark_suite.name] = {
            "role": benchmark_suite.role,
            "short_window_steps": benchmark_suite.short_window_steps,
            "long_window_cap": benchmark_suite.long_window_cap,
            "suite": suite_to_dict(benchmark_suite.suite),
        }
        combinations = [
            (seed, learning_rate, input_scale)
            for seed in benchmark_suite.suite.sweep.seeds
            for learning_rate in benchmark_suite.suite.sweep.learning_rates
            for input_scale in benchmark_suite.suite.sweep.input_scales
        ]

        for index, (seed, learning_rate, input_scale) in enumerate(combinations, start=1):
            if not quiet:
                print(
                    f"[{benchmark_suite.name} {index}/{len(combinations)}] "
                    f"seed={seed} lr={learning_rate:.4f} input_scale={input_scale:.2f}",
                    flush=True,
                )

            if benchmark_suite.role == "fixed_point_negative":
                run_short_rows, run_resolved_rows, metadata = _build_negative_rows(
                    benchmark_suite,
                    seed,
                    learning_rate,
                    input_scale,
                )
                short_rows.extend(run_short_rows)
                resolved_rows.extend(run_resolved_rows)
                negative_metadata_rows.append(metadata)
                continue

            short_record, _, _ = execute_run(benchmark_suite.suite, seed, learning_rate, input_scale)
            run_short_rows = [
                _build_short_row(benchmark_suite, short_record, detector_name)
                for detector_name in DETECTOR_NAMES
            ]
            short_rows.extend(run_short_rows)

            long_record = None
            if _needs_long_followup(benchmark_suite, run_short_rows):
                long_suite = replace(
                    benchmark_suite.suite,
                    training=replace(benchmark_suite.suite.training, total_steps=benchmark_suite.long_window_cap),
                )
                long_record, _, _ = execute_run(long_suite, seed, learning_rate, input_scale)

            run_resolved_rows = [
                _build_resolved_row(short_row, benchmark_suite, long_record)
                for short_row in run_short_rows
            ]
            resolved_rows.extend(run_resolved_rows)

    detector_scorecards = _build_detector_scorecards(benchmark_suites, resolved_rows)
    suite_scorecards = _build_suite_scorecards(benchmark_suites, detector_scorecards)
    resolution_curve_rows = _resolution_curve_rows(resolved_rows, benchmark_suites)
    recommended_detector = _recommended_stochastic_detector(detector_scorecards)
    benchmark_verdict = _benchmark_verdict(suite_scorecards, recommended_detector)

    short_runs_csv = output_dir / "short_window_runs.csv"
    resolved_runs_csv = output_dir / "resolved_runs.csv"
    detector_scorecard_csv = output_dir / "detector_scorecard.csv"
    suite_scorecard_csv = output_dir / "suite_scorecard.csv"
    resolution_curve_csv = output_dir / "resolution_curve.csv"
    negative_metadata_csv = output_dir / "fixed_point_negative_checks.csv"
    benchmark_summary_json = output_dir / "benchmark_summary.json"
    lead_distributions_png = figures_dir / "lead_distributions.png"
    censor_resolution_png = figures_dir / "censor_resolution.png"

    _write_csv(short_runs_csv, short_rows)
    _write_csv(resolved_runs_csv, resolved_rows)
    _write_csv(detector_scorecard_csv, detector_scorecards)
    _write_csv(suite_scorecard_csv, suite_scorecards)
    _write_csv(resolution_curve_csv, resolution_curve_rows)
    _write_csv(negative_metadata_csv, negative_metadata_rows)
    plot_benchmark_lead_distributions(resolved_rows, lead_distributions_png)
    plot_benchmark_resolution_curves(resolution_curve_rows, censor_resolution_png)

    summary = {
        "benchmark_name": BENCHMARK_NAME,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "smoke": smoke,
        "libraries": library_versions(),
        "long_window_cap": LONG_WINDOW_CAP,
        "benchmark_verdict": benchmark_verdict,
        "recommended_stochastic_direct_detector": recommended_detector,
        "suite_parameters": suite_parameters,
        "suite_scorecards": suite_scorecards,
        "detector_scorecards": detector_scorecards,
        "artifact_files": {
            "benchmark_summary_json": str(benchmark_summary_json),
            "suite_scorecard_csv": str(suite_scorecard_csv),
            "detector_scorecard_csv": str(detector_scorecard_csv),
            "short_window_runs_csv": str(short_runs_csv),
            "resolved_runs_csv": str(resolved_runs_csv),
            "resolution_curve_csv": str(resolution_curve_csv),
            "fixed_point_negative_checks_csv": str(negative_metadata_csv),
            "lead_distributions_png": str(lead_distributions_png),
            "censor_resolution_png": str(censor_resolution_png),
        },
    }
    benchmark_summary_json.write_text(json.dumps(summary, indent=2))
    return summary
