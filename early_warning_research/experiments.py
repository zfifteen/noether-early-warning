from __future__ import annotations

import csv
import json
import math
import platform
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import scipy
import torch
import torch.nn.functional as F

from .detectors import detect_drift_onset, detect_symmetry_onset
from .models import PairedMLP, ToyScaleProduct
from .plotting import plot_detector_suite_comparison, plot_onset_ordering, plot_representative_timeseries
from .suites import (
    DetectorConfig,
    PairedMLPConfig,
    SuiteConfig,
    ToyScaleProductConfig,
    get_suite,
    list_suite_names,
    suite_to_dict,
)


EPS = 1e-12
PAIRED_DETECTOR_NAMES = ("covariance_mismatch", "mean_gradient_mismatch", "activation_stat_mismatch")
TOY_DETECTOR_NAMES = ("scale_charge_mismatch",)
PRIMARY_DETECTOR_BY_KIND = {
    "paired_mlp": "covariance_mismatch",
    "toy_scale_product": "scale_charge_mismatch",
}


@dataclass
class RunRecord:
    suite_name: str
    run_id: str
    model_kind: str
    seed: int
    learning_rate: float
    input_scale: float
    total_steps: int
    drift_onset_step: int | None
    symmetry_onset_step: int | None
    lead_steps: int | None
    comparable: bool
    censored_symmetry: bool
    drift_detected: bool
    symmetry_detected: bool
    verdict_label: str
    final_loss: float
    early_curvature: float
    max_update_norm: float
    max_symmetry_score: float
    detector_onsets: dict[str, int | None]
    detector_leads: dict[str, int | None]
    detector_comparable: dict[str, bool]
    detector_censored: dict[str, bool]
    detector_symmetry_detected: dict[str, bool]
    detector_verdicts: dict[str, str]
    detector_max_scores: dict[str, float]
    consensus_symmetry_onset_step: int | None
    consensus_supportive_detectors: int
    consensus_falsifying_detectors: int
    consensus_censored_detectors: int
    consensus_verdict: str


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


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


def _to_float64_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.tensor(array, dtype=torch.float64)


def detector_names_for_suite(suite: SuiteConfig) -> tuple[str, ...]:
    return PAIRED_DETECTOR_NAMES if suite.kind == "paired_mlp" else TOY_DETECTOR_NAMES


def primary_detector_name(suite: SuiteConfig) -> str:
    return PRIMARY_DETECTOR_BY_KIND[suite.kind]


def build_paired_dataset(
    seed: int,
    input_scale: float,
    train_size: int,
    probe_size: int,
    model_cfg: PairedMLPConfig,
) -> dict[str, torch.Tensor]:
    rng = np.random.RandomState(seed)
    total = train_size + probe_size
    x = rng.randn(total, model_cfg.input_dim) * input_scale

    teacher_w = rng.randn(model_cfg.teacher_hidden, model_cfg.input_dim) / math.sqrt(model_cfg.input_dim)
    teacher_out = rng.randn(model_cfg.teacher_hidden) / math.sqrt(model_cfg.teacher_hidden)
    hidden = np.tanh(x @ teacher_w.T)
    y = hidden @ teacher_out + 0.05 * np.sin(0.7 * x[:, 0])

    return {
        "x_train": _to_float64_tensor(x[:train_size]),
        "y_train": _to_float64_tensor(y[:train_size]),
        "x_probe": _to_float64_tensor(x[train_size:]),
        "y_probe": _to_float64_tensor(y[train_size:]),
    }


def build_toy_dataset(
    seed: int,
    input_scale: float,
    train_size: int,
    probe_size: int,
    model_cfg: ToyScaleProductConfig,
) -> dict[str, torch.Tensor]:
    rng = np.random.RandomState(seed)
    total = train_size + probe_size
    x = rng.randn(total, 1) * input_scale
    y = model_cfg.teacher_slope * x[:, 0] + 0.05 * np.sin(0.9 * x[:, 0]) + 0.02 * rng.randn(total)
    return {
        "x_train": _to_float64_tensor(x[:train_size]),
        "y_train": _to_float64_tensor(y[:train_size]),
        "x_probe": _to_float64_tensor(x[train_size:]),
        "y_probe": _to_float64_tensor(y[train_size:]),
    }


def empirical_covariance(samples: torch.Tensor) -> torch.Tensor:
    if samples.shape[0] < 2:
        return torch.zeros((samples.shape[1], samples.shape[1]), dtype=samples.dtype)
    centered = samples - samples.mean(dim=0, keepdim=True)
    return centered.t() @ centered / (samples.shape[0] - 1)


def paired_prediction_from_flat(flat_params: torch.Tensor, x: torch.Tensor, model: PairedMLP) -> torch.Tensor:
    hidden_dim = model.hidden_dim
    input_split = model.input_dim * hidden_dim
    input_weight = flat_params[:input_split].view(hidden_dim, model.input_dim)
    cursor = input_split

    hidden = torch.tanh(x @ input_weight.t())
    if model.use_layer_norm:
        hidden = F.layer_norm(hidden, (hidden_dim,))

    if model.hidden_layers == 2:
        hidden_split = hidden_dim * hidden_dim
        hidden_weight = flat_params[cursor : cursor + hidden_split].view(hidden_dim, hidden_dim)
        cursor += hidden_split
        hidden = torch.tanh(hidden @ hidden_weight.t())

    output_weight = flat_params[cursor:].view(1, hidden_dim)
    return (hidden @ output_weight.t()).squeeze(-1)


def toy_prediction_from_flat(flat_params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    u, v = flat_params[0], flat_params[1]
    return (u * v) * x.squeeze(-1)


def estimate_curvature(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    flat = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone().requires_grad_(True)

    if isinstance(model, PairedMLP):
        loss_fn = lambda params: torch.mean((paired_prediction_from_flat(params, x, model) - y) ** 2)
    else:
        loss_fn = lambda params: torch.mean((toy_prediction_from_flat(params, x) - y) ** 2)

    hessian = torch.autograd.functional.hessian(loss_fn, flat, vectorize=True)
    eigvals = torch.linalg.eigvalsh(hessian.detach())
    return float(torch.max(torch.abs(eigvals)).item())


def _relative_vector_mismatch(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(left - right)
    denominator = torch.linalg.vector_norm(left) + torch.linalg.vector_norm(right) + EPS
    return float((numerator / denominator).item())


def _activation_stat_mismatch(left: torch.Tensor, right: torch.Tensor) -> float:
    left_stats = torch.stack(
        [
            left.mean(),
            left.std(unbiased=False),
            left.abs().mean(),
            torch.sqrt(torch.mean(left.square()) + EPS),
        ]
    )
    right_stats = torch.stack(
        [
            right.mean(),
            right.std(unbiased=False),
            right.abs().mean(),
            torch.sqrt(torch.mean(right.square()) + EPS),
        ]
    )
    return _relative_vector_mismatch(left_stats, right_stats)


def _detector_verdict_label(drift_onset: int | None, symmetry_onset: int | None) -> tuple[str, int | None, bool, bool]:
    lead_steps = None if drift_onset is None or symmetry_onset is None else symmetry_onset - drift_onset
    comparable = lead_steps is not None
    censored = drift_onset is not None and symmetry_onset is None
    if comparable and lead_steps > 0:
        return "supportive", lead_steps, True, False
    if comparable:
        return "falsifying", lead_steps, True, False
    if censored:
        return "censored", None, False, True
    return "no_ordering", None, False, False


def _consensus_onset(detector_onsets: dict[str, int | None]) -> int | None:
    detected = sorted(onset for onset in detector_onsets.values() if onset is not None)
    if len(detected) < 2:
        return None
    return detected[1]


def _probe_seed(seed: int, learning_rate: float, input_scale: float, step: int) -> int:
    return seed * 100_000 + int(round(learning_rate * 10_000)) * 100 + int(round(input_scale * 100)) + step


def _run_id(seed: int, learning_rate: float, input_scale: float) -> str:
    return f"seed{seed}_lr{learning_rate:.4f}_scale{input_scale:.2f}"


def _sample_train_batch(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    batch_size: int | None,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size is None or batch_size >= x_train.shape[0]:
        return x_train, y_train
    idx = torch.randint(0, x_train.shape[0], (batch_size,), generator=gen)
    return x_train[idx], y_train[idx]


def compute_paired_probe_scores(
    model: PairedMLP,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    microbatches: int,
    microbatch_size: int,
    probe_seed: int,
) -> dict[str, float]:
    gen = torch.Generator().manual_seed(probe_seed)
    loss_fn = torch.nn.MSELoss()
    pair_samples: dict[tuple[int, int], tuple[list[torch.Tensor], list[torch.Tensor]]] = {
        pair: ([], []) for pair in model.pairs
    }

    for _ in range(microbatches):
        idx = torch.randint(0, x_probe.shape[0], (microbatch_size,), generator=gen)
        xb = x_probe[idx]
        yb = y_probe[idx]

        model.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb)
        loss.backward()

        weight_grad = model.monitored_weight_grad().detach()
        readout_grad = model.output_layer.weight.grad.detach().squeeze(0)
        for left, right in model.pairs:
            pair_samples[(left, right)][0].append(torch.cat([weight_grad[left], readout_grad[left : left + 1]]))
            pair_samples[(left, right)][1].append(torch.cat([weight_grad[right], readout_grad[right : right + 1]]))

    with torch.no_grad():
        hidden = model.hidden_representation(x_probe).detach()

    covariance_scores: list[float] = []
    mean_gradient_scores: list[float] = []
    activation_scores: list[float] = []

    for (left, right), (left_samples, right_samples) in pair_samples.items():
        left_stack = torch.stack(left_samples)
        right_stack = torch.stack(right_samples)
        covariance_scores.append(
            float(
                (
                    torch.linalg.matrix_norm(empirical_covariance(left_stack) - empirical_covariance(right_stack))
                    / (
                        torch.linalg.matrix_norm(empirical_covariance(left_stack))
                        + torch.linalg.matrix_norm(empirical_covariance(right_stack))
                        + EPS
                    )
                ).item()
            )
        )
        mean_gradient_scores.append(_relative_vector_mismatch(left_stack.mean(dim=0), right_stack.mean(dim=0)))
        activation_scores.append(_activation_stat_mismatch(hidden[:, left], hidden[:, right]))

    return {
        "covariance_mismatch": float(np.mean(covariance_scores)),
        "mean_gradient_mismatch": float(np.mean(mean_gradient_scores)),
        "activation_stat_mismatch": float(np.mean(activation_scores)),
    }


def compute_toy_probe_scores(model: ToyScaleProduct, x_probe: torch.Tensor, y_probe: torch.Tensor) -> dict[str, float]:
    loss_fn = torch.nn.MSELoss()
    model.zero_grad(set_to_none=True)
    loss = loss_fn(model(x_probe), y_probe)
    loss.backward()
    term_u = model.u.detach() * model.u.grad.detach()
    term_v = model.v.detach() * model.v.grad.detach()
    numerator = abs((term_u - term_v).item())
    denominator = abs(term_u.item()) + abs(term_v.item()) + EPS
    return {"scale_charge_mismatch": float(numerator / denominator)}


def _attach_drift_stats(
    step_rows: list[dict[str, object]],
    drift_result: object,
) -> None:
    drift_stats_by_step = {item.window_end_step: item for item in drift_result.window_stats}
    for row, running_mean_value in zip(step_rows, drift_result.smoothed_series):
        row["running_mean_update_norm"] = running_mean_value
        stat = drift_stats_by_step.get(int(row["step"]))
        row["drift_window_slope"] = None if stat is None else stat.slope
        row["drift_window_p_value"] = None if stat is None else stat.p_value
        row["drift_window_effect"] = None if stat is None else stat.effect
        row["drift_window_triggered"] = None if stat is None else stat.triggered


def _attach_detector_stats(
    probe_rows: list[dict[str, object]],
    detector_results: dict[str, object],
    primary_detector: str,
) -> None:
    detector_stats_by_step = {
        name: {item.step: item for item in result.probe_stats}
        for name, result in detector_results.items()
    }
    for row in probe_rows:
        step = int(row["step"])
        for detector_name, result in detector_results.items():
            stat = detector_stats_by_step[detector_name].get(step)
            row[f"threshold_{detector_name}"] = None if stat is None else stat.threshold
            row[f"z_score_{detector_name}"] = None if stat is None else stat.z_score
            row[f"triggered_{detector_name}"] = None if stat is None else stat.triggered
        row["symmetry_score"] = row[f"score_{primary_detector}"]
        row["symmetry_threshold"] = row[f"threshold_{primary_detector}"]
        row["symmetry_z_score"] = row[f"z_score_{primary_detector}"]
        row["symmetry_triggered"] = row[f"triggered_{primary_detector}"]


def _build_run_record(
    suite: SuiteConfig,
    run_id: str,
    seed: int,
    learning_rate: float,
    input_scale: float,
    drift_result: object,
    detector_results: dict[str, object],
    final_loss: float,
    early_curvature: float,
    update_norms: list[float],
    detector_score_history: dict[str, list[float]],
) -> RunRecord:
    primary_detector = primary_detector_name(suite)
    detector_onsets = {name: result.onset_step for name, result in detector_results.items()}
    detector_leads: dict[str, int | None] = {}
    detector_comparable: dict[str, bool] = {}
    detector_censored: dict[str, bool] = {}
    detector_symmetry_detected: dict[str, bool] = {}
    detector_verdicts: dict[str, str] = {}
    detector_max_scores = {
        name: max(scores) if scores else 0.0 for name, scores in detector_score_history.items()
    }

    for detector_name, onset_step in detector_onsets.items():
        verdict_label, lead_steps, comparable, censored = _detector_verdict_label(drift_result.onset_step, onset_step)
        detector_leads[detector_name] = lead_steps
        detector_comparable[detector_name] = comparable
        detector_censored[detector_name] = censored
        detector_symmetry_detected[detector_name] = onset_step is not None
        detector_verdicts[detector_name] = verdict_label

    primary_onset = detector_onsets[primary_detector]
    primary_label, primary_lead, primary_comparable, primary_censored = _detector_verdict_label(
        drift_result.onset_step,
        primary_onset,
    )

    consensus_symmetry_onset_step = (
        _consensus_onset(detector_onsets) if suite.kind == "paired_mlp" else primary_onset
    )
    consensus_supportive = sum(label == "supportive" for label in detector_verdicts.values())
    consensus_falsifying = sum(label == "falsifying" for label in detector_verdicts.values())
    consensus_censored = sum(label == "censored" for label in detector_verdicts.values())
    if suite.kind == "paired_mlp" and consensus_supportive >= 2:
        consensus_verdict = "supportive"
    elif suite.kind == "paired_mlp" and consensus_falsifying >= 2:
        consensus_verdict = "falsifying"
    elif suite.kind == "paired_mlp" and consensus_censored >= 2:
        consensus_verdict = "censored"
    else:
        consensus_verdict = primary_label

    return RunRecord(
        suite_name=suite.name,
        run_id=run_id,
        model_kind=suite.kind,
        seed=seed,
        learning_rate=learning_rate,
        input_scale=input_scale,
        total_steps=suite.training.total_steps,
        drift_onset_step=drift_result.onset_step,
        symmetry_onset_step=primary_onset,
        lead_steps=primary_lead,
        comparable=primary_comparable,
        censored_symmetry=primary_censored,
        drift_detected=drift_result.onset_step is not None,
        symmetry_detected=primary_onset is not None,
        verdict_label=primary_label,
        final_loss=final_loss,
        early_curvature=early_curvature,
        max_update_norm=max(update_norms) if update_norms else 0.0,
        max_symmetry_score=detector_max_scores[primary_detector],
        detector_onsets=detector_onsets,
        detector_leads=detector_leads,
        detector_comparable=detector_comparable,
        detector_censored=detector_censored,
        detector_symmetry_detected=detector_symmetry_detected,
        detector_verdicts=detector_verdicts,
        detector_max_scores=detector_max_scores,
        consensus_symmetry_onset_step=consensus_symmetry_onset_step,
        consensus_supportive_detectors=consensus_supportive,
        consensus_falsifying_detectors=consensus_falsifying,
        consensus_censored_detectors=consensus_censored,
        consensus_verdict=consensus_verdict,
    )


def execute_paired_run(
    suite: SuiteConfig,
    seed: int,
    learning_rate: float,
    input_scale: float,
) -> tuple[RunRecord, list[dict[str, object]], list[dict[str, object]]]:
    model_cfg = suite.model
    assert isinstance(model_cfg, PairedMLPConfig)

    set_seed(seed)
    data = build_paired_dataset(seed, input_scale, suite.training.train_size, suite.training.probe_size, model_cfg)
    model_seed = seed * 1000 + int(round(learning_rate * 10_000)) + int(round(input_scale * 100))
    model = PairedMLP(
        model_cfg.input_dim,
        model_cfg.pair_count,
        model_seed,
        model_cfg.init_noise,
        hidden_layers=model_cfg.hidden_layers,
        use_layer_norm=model_cfg.use_layer_norm,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    batch_gen = torch.Generator().manual_seed(model_seed + 17)

    curvature_slice = slice(0, min(64, suite.training.probe_size))
    early_curvature = estimate_curvature(model, data["x_probe"][curvature_slice], data["y_probe"][curvature_slice])

    run_id = _run_id(seed, learning_rate, input_scale)
    step_rows: list[dict[str, object]] = []
    probe_rows: list[dict[str, object]] = []
    update_norms: list[float] = []
    probe_steps: list[int] = []
    detector_score_history = {name: [] for name in detector_names_for_suite(suite)}
    final_loss = 0.0

    for step in range(1, suite.training.total_steps + 1):
        xb, yb = _sample_train_batch(data["x_train"], data["y_train"], suite.training.batch_size, batch_gen)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb)
        loss.backward()
        grad_vec = torch.cat([param.grad.detach().flatten() for param in model.parameters() if param.grad is not None])
        update_norm = float((learning_rate * torch.linalg.vector_norm(grad_vec)).item())
        update_norms.append(update_norm)
        final_loss = float(loss.item())
        step_rows.append(
            {
                "suite_name": suite.name,
                "run_id": run_id,
                "step": step,
                "update_norm": update_norm,
                "loss": final_loss,
            }
        )
        optimizer.step()

        if step % suite.probe.every_steps == 0 or step == suite.training.total_steps:
            scores = compute_paired_probe_scores(
                model,
                data["x_probe"],
                data["y_probe"],
                suite.probe.microbatches,
                suite.probe.microbatch_size,
                probe_seed=_probe_seed(seed, learning_rate, input_scale, step),
            )
            probe_steps.append(step)
            row = {
                "suite_name": suite.name,
                "run_id": run_id,
                "probe_index": len(probe_steps) - 1,
                "step": step,
                "curvature": early_curvature,
            }
            for detector_name, score in scores.items():
                detector_score_history[detector_name].append(score)
                row[f"score_{detector_name}"] = score
            probe_rows.append(row)

    steps = list(range(1, suite.training.total_steps + 1))
    drift_result = detect_drift_onset(update_norms, steps, suite.detector)
    detector_results = {
        name: detect_symmetry_onset(scores, probe_steps, suite.detector)
        for name, scores in detector_score_history.items()
    }

    _attach_drift_stats(step_rows, drift_result)
    _attach_detector_stats(probe_rows, detector_results, primary_detector_name(suite))

    record = _build_run_record(
        suite,
        run_id,
        seed,
        learning_rate,
        input_scale,
        drift_result,
        detector_results,
        final_loss,
        early_curvature,
        update_norms,
        detector_score_history,
    )
    return record, step_rows, probe_rows


def execute_toy_run(
    suite: SuiteConfig,
    seed: int,
    learning_rate: float,
    input_scale: float,
) -> tuple[RunRecord, list[dict[str, object]], list[dict[str, object]]]:
    model_cfg = suite.model
    assert isinstance(model_cfg, ToyScaleProductConfig)

    set_seed(seed)
    data = build_toy_dataset(seed, input_scale, suite.training.train_size, suite.training.probe_size, model_cfg)
    rng = np.random.RandomState(seed)
    model = ToyScaleProduct(
        u_init=model_cfg.init_u + model_cfg.init_noise * rng.randn(),
        v_init=model_cfg.init_v - model_cfg.init_noise * rng.randn(),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    curvature_slice = slice(0, min(64, suite.training.probe_size))
    early_curvature = estimate_curvature(model, data["x_probe"][curvature_slice], data["y_probe"][curvature_slice])

    run_id = _run_id(seed, learning_rate, input_scale)
    step_rows: list[dict[str, object]] = []
    probe_rows: list[dict[str, object]] = []
    update_norms: list[float] = []
    probe_steps: list[int] = []
    detector_score_history = {name: [] for name in detector_names_for_suite(suite)}
    final_loss = 0.0

    for step in range(1, suite.training.total_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(data["x_train"]), data["y_train"])
        loss.backward()
        grad_vec = torch.cat([param.grad.detach().flatten() for param in model.parameters() if param.grad is not None])
        update_norm = float((learning_rate * torch.linalg.vector_norm(grad_vec)).item())
        update_norms.append(update_norm)
        final_loss = float(loss.item())
        step_rows.append(
            {
                "suite_name": suite.name,
                "run_id": run_id,
                "step": step,
                "update_norm": update_norm,
                "loss": final_loss,
            }
        )
        optimizer.step()

        if step % suite.probe.every_steps == 0 or step == suite.training.total_steps:
            scores = compute_toy_probe_scores(model, data["x_probe"], data["y_probe"])
            probe_steps.append(step)
            row = {
                "suite_name": suite.name,
                "run_id": run_id,
                "probe_index": len(probe_steps) - 1,
                "step": step,
                "curvature": early_curvature,
            }
            for detector_name, score in scores.items():
                detector_score_history[detector_name].append(score)
                row[f"score_{detector_name}"] = score
            probe_rows.append(row)

    steps = list(range(1, suite.training.total_steps + 1))
    drift_result = detect_drift_onset(update_norms, steps, suite.detector)
    detector_results = {
        name: detect_symmetry_onset(scores, probe_steps, suite.detector)
        for name, scores in detector_score_history.items()
    }

    _attach_drift_stats(step_rows, drift_result)
    _attach_detector_stats(probe_rows, detector_results, primary_detector_name(suite))

    record = _build_run_record(
        suite,
        run_id,
        seed,
        learning_rate,
        input_scale,
        drift_result,
        detector_results,
        final_loss,
        early_curvature,
        update_norms,
        detector_score_history,
    )
    return record, step_rows, probe_rows


def execute_run(
    suite: SuiteConfig,
    seed: int,
    learning_rate: float,
    input_scale: float,
) -> tuple[RunRecord, list[dict[str, object]], list[dict[str, object]]]:
    if suite.kind == "paired_mlp":
        return execute_paired_run(suite, seed, learning_rate, input_scale)
    return execute_toy_run(suite, seed, learning_rate, input_scale)


def compute_suite_verdict(run_rows: list[dict[str, object]]) -> tuple[str, dict[str, object]]:
    comparable = [row for row in run_rows if row["comparable"]]
    positive = [row for row in comparable if row["lead_steps"] > 0]
    nonpositive = [row for row in comparable if row["lead_steps"] <= 0]
    censored = [row for row in run_rows if row["censored_symmetry"]]

    verdict = "INCONCLUSIVE"
    if comparable:
        positive_fraction = len(positive) / len(comparable)
        nonpositive_fraction = len(nonpositive) / len(comparable)
        median_lead = float(np.median([row["lead_steps"] for row in comparable]))
        if positive_fraction >= 0.7 and median_lead > 0:
            verdict = "SUPPORTED"
        elif nonpositive_fraction >= 0.5:
            verdict = "FALSIFIED"
    else:
        positive_fraction = 0.0
        nonpositive_fraction = 0.0
        median_lead = None

    stats = {
        "total_runs": len(run_rows),
        "comparable_runs": len(comparable),
        "supportive_runs": len(positive),
        "falsifying_runs": len(nonpositive),
        "censored_runs": len(censored),
        "positive_fraction": positive_fraction,
        "nonpositive_fraction": nonpositive_fraction,
        "median_lead_steps": median_lead,
    }
    return verdict, stats


def _verdict_row_from_record(record: RunRecord, detector_name: str) -> dict[str, object]:
    return {
        "run_id": record.run_id,
        "comparable": record.detector_comparable[detector_name],
        "lead_steps": record.detector_leads[detector_name],
        "censored_symmetry": record.detector_censored[detector_name],
    }


def compute_detector_suite_summaries(
    run_records: list[RunRecord],
    detector_names: tuple[str, ...],
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    summaries: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    for detector_name in detector_names:
        verdict_rows = [_verdict_row_from_record(record, detector_name) for record in run_records]
        verdict, verdict_stats = compute_suite_verdict(verdict_rows)
        lead_distribution = [
            record.detector_leads[detector_name]
            for record in run_records
            if record.detector_leads[detector_name] is not None
        ]
        summary = {
            "detector_name": detector_name,
            "verdict": verdict,
            "verdict_stats": verdict_stats,
            "lead_distribution": lead_distribution,
        }
        summaries[detector_name] = summary
        rows.append(
            {
                "detector_name": detector_name,
                "verdict": verdict,
                **verdict_stats,
            }
        )
    return summaries, rows


def compute_robust_suite_verdict(detector_summaries: dict[str, dict[str, object]]) -> tuple[str, dict[str, object]]:
    supported = sorted(
        detector_name
        for detector_name, summary in detector_summaries.items()
        if summary["verdict"] == "SUPPORTED"
    )
    falsified = sorted(
        detector_name
        for detector_name, summary in detector_summaries.items()
        if summary["verdict"] == "FALSIFIED"
    )
    inconclusive = sorted(
        detector_name
        for detector_name, summary in detector_summaries.items()
        if summary["verdict"] == "INCONCLUSIVE"
    )
    if len(supported) >= 2:
        verdict = "SUPPORTED"
    elif len(falsified) >= 2:
        verdict = "FALSIFIED"
    else:
        verdict = "INCONCLUSIVE"
    return verdict, {
        "supported_detectors": supported,
        "falsified_detectors": falsified,
        "inconclusive_detectors": inconclusive,
        "supported_detector_count": len(supported),
        "falsified_detector_count": len(falsified),
        "inconclusive_detector_count": len(inconclusive),
    }


def choose_representative_run(run_summaries: list[dict[str, object]]) -> str | None:
    if not run_summaries:
        return None
    comparable = [row for row in run_summaries if row["comparable"] and row["lead_steps"] > 0]
    if comparable:
        comparable.sort(key=lambda row: row["lead_steps"], reverse=True)
        return str(comparable[0]["run_id"])
    detected = [row for row in run_summaries if row["drift_detected"]]
    if detected:
        return str(detected[0]["run_id"])
    return str(run_summaries[0]["run_id"])


def library_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "matplotlib": matplotlib.__version__,
    }


def _serialize_run_records(run_records: list[RunRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in run_records:
        row = {
            "suite_name": record.suite_name,
            "run_id": record.run_id,
            "model_kind": record.model_kind,
            "seed": record.seed,
            "learning_rate": record.learning_rate,
            "input_scale": record.input_scale,
            "total_steps": record.total_steps,
            "drift_onset_step": record.drift_onset_step,
            "symmetry_onset_step": record.symmetry_onset_step,
            "lead_steps": record.lead_steps,
            "comparable": record.comparable,
            "censored_symmetry": record.censored_symmetry,
            "drift_detected": record.drift_detected,
            "symmetry_detected": record.symmetry_detected,
            "verdict_label": record.verdict_label,
            "final_loss": record.final_loss,
            "early_curvature": record.early_curvature,
            "max_update_norm": record.max_update_norm,
            "max_symmetry_score": record.max_symmetry_score,
            "consensus_symmetry_onset_step": record.consensus_symmetry_onset_step,
            "consensus_supportive_detectors": record.consensus_supportive_detectors,
            "consensus_falsifying_detectors": record.consensus_falsifying_detectors,
            "consensus_censored_detectors": record.consensus_censored_detectors,
            "consensus_verdict": record.consensus_verdict,
        }
        for detector_name in sorted(record.detector_onsets):
            row[f"symmetry_onset_step_{detector_name}"] = record.detector_onsets[detector_name]
            row[f"lead_steps_{detector_name}"] = record.detector_leads[detector_name]
            row[f"comparable_{detector_name}"] = record.detector_comparable[detector_name]
            row[f"censored_symmetry_{detector_name}"] = record.detector_censored[detector_name]
            row[f"symmetry_detected_{detector_name}"] = record.detector_symmetry_detected[detector_name]
            row[f"verdict_label_{detector_name}"] = record.detector_verdicts[detector_name]
            row[f"max_symmetry_score_{detector_name}"] = record.detector_max_scores[detector_name]
        rows.append(row)
    return rows


def build_detector_config_variants(base: DetectorConfig) -> dict[str, DetectorConfig]:
    return {
        "baseline": base,
        "local_low": DetectorConfig(
            drift_window=max(12, int(round(base.drift_window * 0.8))),
            drift_running_mean_window=max(4, int(round(base.drift_running_mean_window * 0.8))),
            drift_effect_floor=base.drift_effect_floor * 0.8,
            drift_p_threshold=base.drift_p_threshold,
            drift_consecutive=base.drift_consecutive,
            symmetry_baseline_probes=base.symmetry_baseline_probes,
            symmetry_z_threshold=base.symmetry_z_threshold * 0.8,
            symmetry_floor=base.symmetry_floor * 0.8,
            symmetry_consecutive=base.symmetry_consecutive,
        ),
        "local_high": DetectorConfig(
            drift_window=int(round(base.drift_window * 1.2)),
            drift_running_mean_window=max(4, int(round(base.drift_running_mean_window * 1.2))),
            drift_effect_floor=base.drift_effect_floor * 1.2,
            drift_p_threshold=base.drift_p_threshold,
            drift_consecutive=base.drift_consecutive,
            symmetry_baseline_probes=base.symmetry_baseline_probes,
            symmetry_z_threshold=base.symmetry_z_threshold * 1.2,
            symmetry_floor=base.symmetry_floor * 1.2,
            symmetry_consecutive=base.symmetry_consecutive,
        ),
        "looser": DetectorConfig(
            drift_window=max(12, int(round(base.drift_window * 0.75))),
            drift_running_mean_window=max(4, int(round(base.drift_running_mean_window * 0.8))),
            drift_effect_floor=base.drift_effect_floor * 0.75,
            drift_p_threshold=base.drift_p_threshold,
            drift_consecutive=base.drift_consecutive,
            symmetry_baseline_probes=base.symmetry_baseline_probes,
            symmetry_z_threshold=base.symmetry_z_threshold * 0.75,
            symmetry_floor=base.symmetry_floor * 0.8,
            symmetry_consecutive=base.symmetry_consecutive,
        ),
        "stricter": DetectorConfig(
            drift_window=int(round(base.drift_window * 1.25)),
            drift_running_mean_window=max(4, int(round(base.drift_running_mean_window * 1.2))),
            drift_effect_floor=base.drift_effect_floor * 1.25,
            drift_p_threshold=base.drift_p_threshold,
            drift_consecutive=base.drift_consecutive,
            symmetry_baseline_probes=base.symmetry_baseline_probes,
            symmetry_z_threshold=base.symmetry_z_threshold * 1.25,
            symmetry_floor=base.symmetry_floor * 1.2,
            symmetry_consecutive=base.symmetry_consecutive,
        ),
    }


def summarize_threshold_stability(
    suite: SuiteConfig,
    run_records: list[RunRecord],
    step_rows: list[dict[str, object]],
    probe_rows: list[dict[str, object]],
    detector_summaries: dict[str, dict[str, object]],
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    if suite.kind != "paired_mlp":
        return None, []

    step_map: dict[str, list[dict[str, object]]] = defaultdict(list)
    probe_map: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in step_rows:
        step_map[str(row["run_id"])].append(row)
    for row in probe_rows:
        probe_map[str(row["run_id"])].append(row)
    for rows in step_map.values():
        rows.sort(key=lambda row: int(row["step"]))
    for rows in probe_map.values():
        rows.sort(key=lambda row: int(row["step"]))

    variants = build_detector_config_variants(suite.detector)
    preset_summaries: dict[str, dict[str, object]] = {}
    table_rows: list[dict[str, object]] = []

    for preset_name, config in variants.items():
        preset_detector_summaries: dict[str, dict[str, object]] = {}
        for detector_name in detector_names_for_suite(suite):
            verdict_rows: list[dict[str, object]] = []
            for record in run_records:
                run_step_rows = step_map[record.run_id]
                run_probe_rows = probe_map[record.run_id]
                steps = [int(row["step"]) for row in run_step_rows]
                update_norms = [float(row["update_norm"]) for row in run_step_rows]
                drift_result = detect_drift_onset(update_norms, steps, config)
                probe_steps = [int(row["step"]) for row in run_probe_rows]
                scores = [float(row[f"score_{detector_name}"]) for row in run_probe_rows]
                symmetry_result = detect_symmetry_onset(scores, probe_steps, config)
                _, lead_steps, comparable, censored = _detector_verdict_label(
                    drift_result.onset_step,
                    symmetry_result.onset_step,
                )
                verdict_rows.append(
                    {
                        "run_id": record.run_id,
                        "comparable": comparable,
                        "lead_steps": lead_steps,
                        "censored_symmetry": censored,
                    }
                )

            verdict, verdict_stats = compute_suite_verdict(verdict_rows)
            preset_detector_summaries[detector_name] = {
                "verdict": verdict,
                "verdict_stats": verdict_stats,
            }
            table_rows.append(
                {
                    "suite_name": suite.name,
                    "preset_name": preset_name,
                    "detector_name": detector_name,
                    "verdict": verdict,
                    **verdict_stats,
                }
            )

        robust_verdict, robust_details = compute_robust_suite_verdict(preset_detector_summaries)
        preset_summaries[preset_name] = {
            "robust_verdict": robust_verdict,
            "robust_details": robust_details,
            "detector_summaries": preset_detector_summaries,
        }

    baseline_robust_verdict = preset_summaries["baseline"]["robust_verdict"]
    detector_flip_counts = {
        detector_name: sum(
            preset_summaries[preset_name]["detector_summaries"][detector_name]["verdict"]
            != detector_summaries[detector_name]["verdict"]
            for preset_name in preset_summaries
            if preset_name != "baseline"
        )
        for detector_name in detector_names_for_suite(suite)
    }
    robust_flip_count = sum(
        preset_summaries[preset_name]["robust_verdict"] != baseline_robust_verdict
        for preset_name in preset_summaries
        if preset_name != "baseline"
    )

    summary = {
        "suite_name": suite.name,
        "baseline_robust_verdict": baseline_robust_verdict,
        "robust_flip_count": robust_flip_count,
        "detector_flip_counts": detector_flip_counts,
        "preset_summaries": preset_summaries,
    }
    return summary, table_rows


def run_suite(suite: SuiteConfig, output_dir: Path, quiet: bool = False) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    run_records: list[RunRecord] = []
    step_rows: list[dict[str, object]] = []
    probe_rows: list[dict[str, object]] = []

    combinations = [
        (seed, learning_rate, input_scale)
        for seed in suite.sweep.seeds
        for learning_rate in suite.sweep.learning_rates
        for input_scale in suite.sweep.input_scales
    ]
    for index, (seed, learning_rate, input_scale) in enumerate(combinations, start=1):
        if not quiet:
            print(
                f"[{index}/{len(combinations)}] {suite.name}: seed={seed} lr={learning_rate:.4f} input_scale={input_scale:.2f}",
                flush=True,
            )
        record, run_step_rows, run_probe_rows = execute_run(suite, seed, learning_rate, input_scale)
        run_records.append(record)
        step_rows.extend(run_step_rows)
        probe_rows.extend(run_probe_rows)

    run_summaries = _serialize_run_records(run_records)
    primary_verdict, primary_verdict_stats = compute_suite_verdict(
        [
            {
                "run_id": record.run_id,
                "comparable": record.comparable,
                "lead_steps": record.lead_steps,
                "censored_symmetry": record.censored_symmetry,
            }
            for record in run_records
        ]
    )
    detector_names = detector_names_for_suite(suite)
    detector_summaries, detector_summary_rows = compute_detector_suite_summaries(run_records, detector_names)

    if suite.kind == "paired_mlp":
        verdict, robustness_details = compute_robust_suite_verdict(detector_summaries)
    else:
        verdict = primary_verdict
        robustness_details = {
            "supported_detectors": [primary_detector_name(suite)] if verdict == "SUPPORTED" else [],
            "falsified_detectors": [primary_detector_name(suite)] if verdict == "FALSIFIED" else [],
            "inconclusive_detectors": [primary_detector_name(suite)] if verdict == "INCONCLUSIVE" else [],
            "supported_detector_count": 1 if verdict == "SUPPORTED" else 0,
            "falsified_detector_count": 1 if verdict == "FALSIFIED" else 0,
            "inconclusive_detector_count": 1 if verdict == "INCONCLUSIVE" else 0,
        }

    stability_summary, stability_rows = summarize_threshold_stability(
        suite,
        run_records,
        step_rows,
        probe_rows,
        detector_summaries,
    )
    representative_run_id = choose_representative_run(run_summaries)

    representative_steps = [row for row in step_rows if row["run_id"] == representative_run_id]
    representative_probes = [row for row in probe_rows if row["run_id"] == representative_run_id]
    representative_summary = next((row for row in run_summaries if row["run_id"] == representative_run_id), None)

    runs_csv = output_dir / "runs.csv"
    step_csv = output_dir / "step_metrics.csv"
    probe_csv = output_dir / "probe_metrics.csv"
    summary_json = output_dir / "summary.json"
    detector_summary_csv = output_dir / "detector_summary.csv"
    detector_summary_json = output_dir / "detector_summary.json"
    stability_summary_csv = output_dir / "stability_table.csv"
    stability_summary_json = output_dir / "stability_summary.json"
    timeseries_png = figures_dir / "representative_timeseries.png"
    ordering_png = figures_dir / "onset_ordering.png"

    _write_csv(runs_csv, run_summaries)
    _write_csv(step_csv, step_rows)
    _write_csv(probe_csv, probe_rows)
    _write_csv(detector_summary_csv, detector_summary_rows)
    _write_csv(stability_summary_csv, stability_rows)

    if representative_summary is not None:
        plot_representative_timeseries(representative_steps, representative_probes, representative_summary, timeseries_png)
    plot_onset_ordering(run_summaries, ordering_png)

    detector_summary_payload = {
        "suite_name": suite.name,
        "primary_detector": primary_detector_name(suite),
        "primary_detector_verdict": primary_verdict,
        "primary_detector_verdict_stats": primary_verdict_stats,
        "suite_verdict": verdict,
        "robustness_details": robustness_details,
        "detector_summaries": detector_summaries,
    }
    detector_summary_json.write_text(json.dumps(detector_summary_payload, indent=2))
    if stability_summary is not None:
        stability_summary_json.write_text(json.dumps(stability_summary, indent=2))

    summary = {
        "suite_name": suite.name,
        "description": suite.description,
        "verdict": verdict,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": suite_to_dict(suite),
        "seed_list": list(suite.sweep.seeds),
        "detector_thresholds": asdict(suite.detector),
        "libraries": library_versions(),
        "output_dir": str(output_dir),
        "representative_run_id": representative_run_id,
        "primary_detector": primary_detector_name(suite),
        "primary_detector_verdict": primary_verdict,
        "verdict_stats": primary_verdict_stats,
        "primary_detector_verdict_stats": primary_verdict_stats,
        "detector_summaries": detector_summaries,
        "robustness_details": robustness_details,
        "stability_summary": stability_summary,
        "artifact_files": {
            "summary_json": str(summary_json),
            "runs_csv": str(runs_csv),
            "step_metrics_csv": str(step_csv),
            "probe_metrics_csv": str(probe_csv),
            "detector_summary_csv": str(detector_summary_csv),
            "detector_summary_json": str(detector_summary_json),
            "stability_table_csv": str(stability_summary_csv),
            "stability_summary_json": str(stability_summary_json) if stability_summary is not None else None,
            "representative_timeseries_png": str(timeseries_png),
            "onset_ordering_png": str(ordering_png),
        },
        "run_summaries": run_summaries,
    }

    summary_json.write_text(json.dumps(summary, indent=2))
    return summary


def _bundle_rows_from_suite_summaries(summaries: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for suite_name, suite_summary in summaries.items():
        if "detector_summaries" not in suite_summary:
            continue
        for detector_name, detector_summary in suite_summary["detector_summaries"].items():
            rows.append(
                {
                    "suite_name": suite_name,
                    "detector_name": detector_name,
                    "verdict": detector_summary["verdict"],
                    **detector_summary["verdict_stats"],
                }
            )
    return rows


def default_output_root() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "artifacts" / "exploratory" / "early_warning_research"


def run_named_suite(
    name: str,
    output_root: str | Path | None = None,
    smoke: bool = False,
    quiet: bool = False,
) -> dict[str, object]:
    base_output = Path(output_root) if output_root is not None else default_output_root()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    invocation_dir = base_output / f"{timestamp}_{name}"

    if name == "all":
        summaries: dict[str, dict[str, object]] = {}
        for suite_name in list_suite_names():
            summaries[suite_name] = run_suite(get_suite(suite_name, smoke=smoke), invocation_dir / suite_name, quiet=quiet)

        figures_dir = invocation_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        detector_comparison_png = figures_dir / "detector_suite_comparison.png"
        plot_detector_suite_comparison(summaries, detector_comparison_png)

        bundle_rows = _bundle_rows_from_suite_summaries(summaries)
        detector_bundle_csv = invocation_dir / "detector_bundle_summary.csv"
        detector_bundle_json = invocation_dir / "detector_bundle_summary.json"
        _write_csv(detector_bundle_csv, bundle_rows)

        control_guard_passed = (
            summaries["instant_break_control"]["verdict"] == "FALSIFIED"
            and summaries["instant_break_control_stochastic"]["verdict"] == "FALSIFIED"
        )
        main_supported = (
            summaries["main_paired_mlp"]["verdict"] == "SUPPORTED"
            and summaries["main_paired_mlp_stochastic"]["verdict"] == "SUPPORTED"
        )
        if control_guard_passed and main_supported:
            overall_claim_verdict = "SUPPORTED"
        elif not control_guard_passed:
            overall_claim_verdict = "FALSIFIED"
        else:
            overall_claim_verdict = "INCONCLUSIVE"

        detector_bundle_payload = {
            "control_guard_passed": control_guard_passed,
            "main_suite_support_passed": main_supported,
            "overall_claim_verdict": overall_claim_verdict,
            "suite_detector_rows": bundle_rows,
        }
        detector_bundle_json.write_text(json.dumps(detector_bundle_payload, indent=2))

        combined = {
            "suite_name": "all",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(invocation_dir),
            "smoke": smoke,
            "libraries": library_versions(),
            "control_guard_passed": control_guard_passed,
            "overall_claim_verdict": overall_claim_verdict,
            "artifact_files": {
                "summary_json": str(invocation_dir / "summary.json"),
                "detector_bundle_summary_csv": str(detector_bundle_csv),
                "detector_bundle_summary_json": str(detector_bundle_json),
                "detector_suite_comparison_png": str(detector_comparison_png),
            },
            "suites": summaries,
        }
        (invocation_dir / "summary.json").write_text(json.dumps(combined, indent=2))
        return combined

    suite = get_suite(name, smoke=smoke)
    return run_suite(suite, invocation_dir, quiet=quiet)
