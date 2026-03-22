from __future__ import annotations

import csv
import json
import math
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import scipy
import torch

from .detectors import detect_drift_onset, detect_symmetry_onset
from .models import PairedMLP, ToyScaleProduct
from .plotting import plot_onset_ordering, plot_representative_timeseries
from .suites import (
    PairedMLPConfig,
    SuiteConfig,
    ToyScaleProductConfig,
    get_suite,
    list_suite_names,
    suite_to_dict,
)


EPS = 1e-12


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


def build_paired_dataset(seed: int, input_scale: float, train_size: int, probe_size: int, model_cfg: PairedMLPConfig) -> dict[str, torch.Tensor]:
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


def build_toy_dataset(seed: int, input_scale: float, train_size: int, probe_size: int, model_cfg: ToyScaleProductConfig) -> dict[str, torch.Tensor]:
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


def paired_prediction_from_flat(flat_params: torch.Tensor, x: torch.Tensor, input_dim: int, hidden_dim: int) -> torch.Tensor:
    split = input_dim * hidden_dim
    fc1_weight = flat_params[:split].view(hidden_dim, input_dim)
    fc2_weight = flat_params[split:].view(1, hidden_dim)
    hidden = torch.tanh(x @ fc1_weight.t())
    return (hidden @ fc2_weight.t()).squeeze(-1)


def toy_prediction_from_flat(flat_params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    u, v = flat_params[0], flat_params[1]
    return (u * v) * x.squeeze(-1)


def estimate_curvature(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    flat = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone().requires_grad_(True)

    if isinstance(model, PairedMLP):
        hidden_dim = model.hidden_dim
        input_dim = model.input_dim
        loss_fn = lambda params: torch.mean((paired_prediction_from_flat(params, x, input_dim, hidden_dim) - y) ** 2)
    else:
        loss_fn = lambda params: torch.mean((toy_prediction_from_flat(params, x) - y) ** 2)

    hessian = torch.autograd.functional.hessian(loss_fn, flat, vectorize=True)
    eigvals = torch.linalg.eigvalsh(hessian.detach())
    return float(torch.max(torch.abs(eigvals)).item())


def compute_paired_symmetry_score(
    model: PairedMLP,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    microbatches: int,
    microbatch_size: int,
    probe_seed: int,
) -> float:
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

        fc1_grad = model.fc1.weight.grad.detach()
        fc2_grad = model.fc2.weight.grad.detach().squeeze(0)
        for left, right in model.pairs:
            pair_samples[(left, right)][0].append(torch.cat([fc1_grad[left], fc2_grad[left : left + 1]]))
            pair_samples[(left, right)][1].append(torch.cat([fc1_grad[right], fc2_grad[right : right + 1]]))

    pair_scores: list[float] = []
    for left_samples, right_samples in pair_samples.values():
        cov_left = empirical_covariance(torch.stack(left_samples))
        cov_right = empirical_covariance(torch.stack(right_samples))
        denom = torch.linalg.matrix_norm(cov_left) + torch.linalg.matrix_norm(cov_right) + EPS
        mismatch = torch.linalg.matrix_norm(cov_left - cov_right) / denom
        pair_scores.append(float(mismatch.item()))
    return float(np.mean(pair_scores))


def compute_toy_symmetry_score(model: ToyScaleProduct, x_probe: torch.Tensor, y_probe: torch.Tensor) -> float:
    loss_fn = torch.nn.MSELoss()
    model.zero_grad(set_to_none=True)
    loss = loss_fn(model(x_probe), y_probe)
    loss.backward()
    term_u = model.u.detach() * model.u.grad.detach()
    term_v = model.v.detach() * model.v.grad.detach()
    numerator = abs((term_u - term_v).item())
    denominator = abs(term_u.item()) + abs(term_v.item()) + EPS
    return float(numerator / denominator)


def _run_id(seed: int, learning_rate: float, input_scale: float) -> str:
    return f"seed{seed}_lr{learning_rate:.4f}_scale{input_scale:.2f}"


def _probe_seed(seed: int, learning_rate: float, input_scale: float, step: int) -> int:
    return seed * 100_000 + int(round(learning_rate * 10_000)) * 100 + int(round(input_scale * 100)) + step


def execute_paired_run(suite: SuiteConfig, seed: int, learning_rate: float, input_scale: float) -> tuple[RunRecord, list[dict[str, object]], list[dict[str, object]]]:
    model_cfg = suite.model
    assert isinstance(model_cfg, PairedMLPConfig)

    set_seed(seed)
    data = build_paired_dataset(seed, input_scale, suite.training.train_size, suite.training.probe_size, model_cfg)
    model_seed = seed * 1000 + int(round(learning_rate * 10_000)) + int(round(input_scale * 100))
    model = PairedMLP(model_cfg.input_dim, model_cfg.pair_count, model_seed, model_cfg.init_noise)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    curvature_slice = slice(0, min(64, suite.training.probe_size))
    early_curvature = estimate_curvature(model, data["x_probe"][curvature_slice], data["y_probe"][curvature_slice])

    run_id = _run_id(seed, learning_rate, input_scale)
    step_rows: list[dict[str, object]] = []
    probe_rows: list[dict[str, object]] = []
    update_norms: list[float] = []
    symmetry_scores: list[float] = []
    probe_steps: list[int] = []
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
            symmetry_score = compute_paired_symmetry_score(
                model,
                data["x_probe"],
                data["y_probe"],
                suite.probe.microbatches,
                suite.probe.microbatch_size,
                probe_seed=_probe_seed(seed, learning_rate, input_scale, step),
            )
            symmetry_scores.append(symmetry_score)
            probe_steps.append(step)
            probe_rows.append(
                {
                    "suite_name": suite.name,
                    "run_id": run_id,
                    "probe_index": len(probe_steps) - 1,
                    "step": step,
                    "symmetry_score": symmetry_score,
                    "curvature": early_curvature,
                }
            )

    steps = list(range(1, suite.training.total_steps + 1))
    drift_result = detect_drift_onset(update_norms, steps, suite.detector)
    symmetry_result = detect_symmetry_onset(symmetry_scores, probe_steps, suite.detector)

    drift_stats_by_step = {item.window_end_step: item for item in drift_result.window_stats}
    for row, running_mean_value in zip(step_rows, drift_result.smoothed_series):
        row["running_mean_update_norm"] = running_mean_value
        stat = drift_stats_by_step.get(int(row["step"]))
        row["drift_window_slope"] = None if stat is None else stat.slope
        row["drift_window_p_value"] = None if stat is None else stat.p_value
        row["drift_window_effect"] = None if stat is None else stat.effect
        row["drift_window_triggered"] = None if stat is None else stat.triggered

    symmetry_stats_by_step = {item.step: item for item in symmetry_result.probe_stats}
    for row in probe_rows:
        stat = symmetry_stats_by_step.get(int(row["step"]))
        row["symmetry_threshold"] = None if stat is None else stat.threshold
        row["symmetry_z_score"] = None if stat is None else stat.z_score
        row["symmetry_triggered"] = None if stat is None else stat.triggered

    lead_steps = None
    if drift_result.onset_step is not None and symmetry_result.onset_step is not None:
        lead_steps = symmetry_result.onset_step - drift_result.onset_step

    comparable = lead_steps is not None
    censored_symmetry = drift_result.onset_step is not None and symmetry_result.onset_step is None
    verdict_label = (
        "supportive" if comparable and lead_steps > 0 else "falsifying" if comparable and lead_steps <= 0 else "censored"
        if censored_symmetry
        else "no_ordering"
    )
    record = RunRecord(
        suite_name=suite.name,
        run_id=run_id,
        model_kind=suite.kind,
        seed=seed,
        learning_rate=learning_rate,
        input_scale=input_scale,
        total_steps=suite.training.total_steps,
        drift_onset_step=drift_result.onset_step,
        symmetry_onset_step=symmetry_result.onset_step,
        lead_steps=lead_steps,
        comparable=comparable,
        censored_symmetry=censored_symmetry,
        drift_detected=drift_result.onset_step is not None,
        symmetry_detected=symmetry_result.onset_step is not None,
        verdict_label=verdict_label,
        final_loss=final_loss,
        early_curvature=early_curvature,
        max_update_norm=max(update_norms) if update_norms else 0.0,
        max_symmetry_score=max(symmetry_scores) if symmetry_scores else 0.0,
    )
    return record, step_rows, probe_rows


def execute_toy_run(suite: SuiteConfig, seed: int, learning_rate: float, input_scale: float) -> tuple[RunRecord, list[dict[str, object]], list[dict[str, object]]]:
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
    symmetry_scores: list[float] = []
    probe_steps: list[int] = []
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
            symmetry_score = compute_toy_symmetry_score(model, data["x_probe"], data["y_probe"])
            symmetry_scores.append(symmetry_score)
            probe_steps.append(step)
            probe_rows.append(
                {
                    "suite_name": suite.name,
                    "run_id": run_id,
                    "probe_index": len(probe_steps) - 1,
                    "step": step,
                    "symmetry_score": symmetry_score,
                    "curvature": early_curvature,
                }
            )

    steps = list(range(1, suite.training.total_steps + 1))
    drift_result = detect_drift_onset(update_norms, steps, suite.detector)
    symmetry_result = detect_symmetry_onset(symmetry_scores, probe_steps, suite.detector)

    drift_stats_by_step = {item.window_end_step: item for item in drift_result.window_stats}
    for row, running_mean_value in zip(step_rows, drift_result.smoothed_series):
        row["running_mean_update_norm"] = running_mean_value
        stat = drift_stats_by_step.get(int(row["step"]))
        row["drift_window_slope"] = None if stat is None else stat.slope
        row["drift_window_p_value"] = None if stat is None else stat.p_value
        row["drift_window_effect"] = None if stat is None else stat.effect
        row["drift_window_triggered"] = None if stat is None else stat.triggered

    symmetry_stats_by_step = {item.step: item for item in symmetry_result.probe_stats}
    for row in probe_rows:
        stat = symmetry_stats_by_step.get(int(row["step"]))
        row["symmetry_threshold"] = None if stat is None else stat.threshold
        row["symmetry_z_score"] = None if stat is None else stat.z_score
        row["symmetry_triggered"] = None if stat is None else stat.triggered

    lead_steps = None
    if drift_result.onset_step is not None and symmetry_result.onset_step is not None:
        lead_steps = symmetry_result.onset_step - drift_result.onset_step

    comparable = lead_steps is not None
    censored_symmetry = drift_result.onset_step is not None and symmetry_result.onset_step is None
    verdict_label = (
        "supportive" if comparable and lead_steps > 0 else "falsifying" if comparable and lead_steps <= 0 else "censored"
        if censored_symmetry
        else "no_ordering"
    )
    record = RunRecord(
        suite_name=suite.name,
        run_id=run_id,
        model_kind=suite.kind,
        seed=seed,
        learning_rate=learning_rate,
        input_scale=input_scale,
        total_steps=suite.training.total_steps,
        drift_onset_step=drift_result.onset_step,
        symmetry_onset_step=symmetry_result.onset_step,
        lead_steps=lead_steps,
        comparable=comparable,
        censored_symmetry=censored_symmetry,
        drift_detected=drift_result.onset_step is not None,
        symmetry_detected=symmetry_result.onset_step is not None,
        verdict_label=verdict_label,
        final_loss=final_loss,
        early_curvature=early_curvature,
        max_update_norm=max(update_norms) if update_norms else 0.0,
        max_symmetry_score=max(symmetry_scores) if symmetry_scores else 0.0,
    )
    return record, step_rows, probe_rows


def execute_run(suite: SuiteConfig, seed: int, learning_rate: float, input_scale: float) -> tuple[RunRecord, list[dict[str, object]], list[dict[str, object]]]:
    if suite.kind == "paired_mlp":
        return execute_paired_run(suite, seed, learning_rate, input_scale)
    return execute_toy_run(suite, seed, learning_rate, input_scale)


def compute_suite_verdict(run_summaries: list[dict[str, object]]) -> tuple[str, dict[str, object]]:
    comparable = [row for row in run_summaries if row["comparable"]]
    positive = [row for row in comparable if row["lead_steps"] > 0]
    nonpositive = [row for row in comparable if row["lead_steps"] <= 0]
    censored = [row for row in run_summaries if row["censored_symmetry"]]

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
        "total_runs": len(run_summaries),
        "comparable_runs": len(comparable),
        "supportive_runs": len(positive),
        "falsifying_runs": len(nonpositive),
        "censored_runs": len(censored),
        "positive_fraction": positive_fraction,
        "nonpositive_fraction": nonpositive_fraction,
        "median_lead_steps": median_lead,
    }
    return verdict, stats


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
    return [asdict(record) for record in run_records]


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
    verdict, verdict_stats = compute_suite_verdict(run_summaries)
    representative_run_id = choose_representative_run(run_summaries)

    representative_steps = [row for row in step_rows if row["run_id"] == representative_run_id]
    representative_probes = [row for row in probe_rows if row["run_id"] == representative_run_id]
    representative_summary = next((row for row in run_summaries if row["run_id"] == representative_run_id), None)

    runs_csv = output_dir / "runs.csv"
    step_csv = output_dir / "step_metrics.csv"
    probe_csv = output_dir / "probe_metrics.csv"
    summary_json = output_dir / "summary.json"
    timeseries_png = figures_dir / "representative_timeseries.png"
    ordering_png = figures_dir / "onset_ordering.png"

    _write_csv(runs_csv, run_summaries)
    _write_csv(step_csv, step_rows)
    _write_csv(probe_csv, probe_rows)

    if representative_summary is not None:
        plot_representative_timeseries(representative_steps, representative_probes, representative_summary, timeseries_png)
    plot_onset_ordering(run_summaries, ordering_png)

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
        "verdict_stats": verdict_stats,
        "artifact_files": {
            "summary_json": str(summary_json),
            "runs_csv": str(runs_csv),
            "step_metrics_csv": str(step_csv),
            "probe_metrics_csv": str(probe_csv),
            "representative_timeseries_png": str(timeseries_png),
            "onset_ordering_png": str(ordering_png),
        },
        "run_summaries": run_summaries,
    }

    summary_json.write_text(json.dumps(summary, indent=2))
    return summary


def default_output_root() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "artifacts" / "early_warning_research"


def run_named_suite(name: str, output_root: str | Path | None = None, smoke: bool = False, quiet: bool = False) -> dict[str, object]:
    base_output = Path(output_root) if output_root is not None else default_output_root()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    invocation_dir = base_output / f"{timestamp}_{name}"

    if name == "all":
        summaries: dict[str, dict[str, object]] = {}
        for suite_name in list_suite_names():
            summaries[suite_name] = run_suite(get_suite(suite_name, smoke=smoke), invocation_dir / suite_name, quiet=quiet)
        control_verdict = summaries["instant_break_control"]["verdict"]
        overall_claim_verdict = summaries["main_paired_mlp"]["verdict"] if control_verdict == "FALSIFIED" else "FALSIFIED"
        combined = {
            "suite_name": "all",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(invocation_dir),
            "smoke": smoke,
            "libraries": library_versions(),
            "control_guard_passed": control_verdict == "FALSIFIED",
            "overall_claim_verdict": overall_claim_verdict,
            "suites": summaries,
        }
        (invocation_dir / "summary.json").write_text(json.dumps(combined, indent=2))
        return combined

    suite = get_suite(name, smoke=smoke)
    return run_suite(suite, invocation_dir, quiet=quiet)
