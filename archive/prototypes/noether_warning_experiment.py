#!/usr/bin/env python3
"""
Empirical falsification harness for the README conjectures.

This script operationalizes two claims:
1. Update-norm drift becomes detectable before symmetry loss in the gradient
   covariance structure.
2. Drift amplitude is better explained by lr / curvature than by lr alone.

The experiment uses a tiny permutation-symmetric MLP trained on synthetic data.
It prints a verdict of VINDICATED, FALSIFIED, or INCONCLUSIVE based on the
observed runs rather than assuming the claim is true.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import torch


torch.set_default_dtype(torch.float64)


EPS = 1e-12


@dataclass
class ExperimentConfig:
    scenario: str
    seeds: list[int]
    learning_rates: list[float]
    input_scales: list[float]
    tau_max: float
    batch_size: int
    probe_every: int
    probe_microbatches: int
    probe_batch_size: int
    train_size: int
    eval_size: int
    input_dim: int
    hidden_dim: int
    init_noise: float
    symmetry_floor: float
    drift_floor: float
    detection_z: float
    detection_consecutive: int


@dataclass
class RunResult:
    scenario: str
    seed: int
    lr: float
    input_scale: float
    steps: int
    drift_onset_step: int | None
    symmetry_onset_step: int | None
    drift_without_symmetry: bool
    lead_steps: int | None
    drift_amplitude: float
    update_norm_slope: float
    early_curvature: float
    first_probe_step: int
    last_probe_step: int


class SymmetricMLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, seed: int, init_noise: float) -> None:
        super().__init__()
        if hidden_dim != 2:
            raise ValueError("This harness currently assumes exactly two symmetric hidden units.")
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim, bias=False)
        self.out = torch.nn.Linear(hidden_dim, 1, bias=False)

        gen = torch.Generator().manual_seed(seed)
        base_in = torch.randn(input_dim, generator=gen) / math.sqrt(input_dim)
        base_out = torch.randn(1, generator=gen).item() / math.sqrt(hidden_dim)
        asym = torch.randn(input_dim, generator=gen)
        asym = asym / (torch.linalg.vector_norm(asym) + EPS)
        out_asym = torch.randn(1, generator=gen).item()

        with torch.no_grad():
            self.fc1.weight[0].copy_(base_in + init_noise * asym)
            self.fc1.weight[1].copy_(base_in - init_noise * asym)
            self.out.weight[0, 0] = base_out + init_noise * out_asym
            self.out.weight[0, 1] = base_out - init_noise * out_asym

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.fc1(x))
        return self.out(hidden).squeeze(-1)


def parse_csv_floats(raw: str) -> list[float]:
    return [float(chunk) for chunk in raw.split(",") if chunk.strip()]


def parse_csv_ints(raw: str) -> list[int]:
    return [int(chunk) for chunk in raw.split(",") if chunk.strip()]


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    scenario_noise = {
        "hypothesis": args.init_noise,
        "instant-break": max(args.init_noise, 0.08),
    }[args.scenario]

    return ExperimentConfig(
        scenario=args.scenario,
        seeds=parse_csv_ints(args.seeds),
        learning_rates=parse_csv_floats(args.learning_rates),
        input_scales=parse_csv_floats(args.input_scales),
        tau_max=args.tau_max,
        batch_size=args.batch_size,
        probe_every=args.probe_every,
        probe_microbatches=args.probe_microbatches,
        probe_batch_size=args.probe_batch_size,
        train_size=args.train_size,
        eval_size=args.eval_size,
        input_dim=args.input_dim,
        hidden_dim=2,
        init_noise=scenario_noise,
        symmetry_floor=args.symmetry_floor,
        drift_floor=args.drift_floor,
        detection_z=args.detection_z,
        detection_consecutive=args.detection_consecutive,
    )


def build_dataset(seed: int, input_scale: float, train_size: int, eval_size: int, input_dim: int) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    total = train_size + eval_size
    x = torch.randn(total, input_dim, generator=gen) * input_scale

    teacher_w1 = torch.randn(input_dim, generator=gen) / math.sqrt(input_dim)
    teacher_w2 = torch.randn(input_dim, generator=gen) / math.sqrt(input_dim)
    teacher_out = torch.tensor([1.1, -0.9], dtype=torch.get_default_dtype())

    hidden = torch.stack([torch.tanh(x @ teacher_w1), torch.tanh(x @ teacher_w2)], dim=1)
    y = hidden @ teacher_out + 0.05 * torch.sin(0.7 * x[:, 0])

    return {
        "x_train": x[:train_size],
        "y_train": y[:train_size],
        "x_eval": x[train_size:],
        "y_eval": y[train_size:],
    }


def functional_forward(flat_params: torch.Tensor, x: torch.Tensor, input_dim: int, hidden_dim: int) -> torch.Tensor:
    split = input_dim * hidden_dim
    fc1_weight = flat_params[:split].view(hidden_dim, input_dim)
    out_weight = flat_params[split:].view(1, hidden_dim)
    hidden = torch.tanh(x @ fc1_weight.t())
    return (hidden @ out_weight.t()).squeeze(-1)


def mse_from_flat(flat_params: torch.Tensor, x: torch.Tensor, y: torch.Tensor, input_dim: int, hidden_dim: int) -> torch.Tensor:
    pred = functional_forward(flat_params, x, input_dim, hidden_dim)
    return torch.mean((pred - y) ** 2)


def compute_curvature(model: SymmetricMLP, x: torch.Tensor, y: torch.Tensor, input_dim: int, hidden_dim: int) -> float:
    flat = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone().requires_grad_(True)
    hessian = torch.autograd.functional.hessian(
        lambda params: mse_from_flat(params, x, y, input_dim, hidden_dim),
        flat,
        vectorize=True,
    )
    eigvals = torch.linalg.eigvalsh(hessian.detach())
    return float(torch.max(torch.abs(eigvals)).item())


def empirical_covariance(samples: torch.Tensor) -> torch.Tensor:
    if samples.shape[0] < 2:
        return torch.zeros((samples.shape[1], samples.shape[1]), dtype=samples.dtype)
    centered = samples - samples.mean(dim=0, keepdim=True)
    return centered.t() @ centered / (samples.shape[0] - 1)


def symmetry_score(
    model: SymmetricMLP,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    probe_microbatches: int,
    probe_batch_size: int,
    probe_seed: int,
) -> float:
    gen = torch.Generator().manual_seed(probe_seed)
    grads_0 = []
    grads_1 = []

    for _ in range(probe_microbatches):
        idx = torch.randint(0, x_eval.shape[0], (probe_batch_size,), generator=gen)
        xb = x_eval[idx]
        yb = y_eval[idx]

        model.zero_grad(set_to_none=True)
        loss = torch.mean((model(xb) - yb) ** 2)
        loss.backward()

        fc_grad = model.fc1.weight.grad.detach()
        out_grad = model.out.weight.grad.detach().squeeze(0)

        grads_0.append(torch.cat([fc_grad[0], out_grad[0:1]]))
        grads_1.append(torch.cat([fc_grad[1], out_grad[1:2]]))

    cov_0 = empirical_covariance(torch.stack(grads_0))
    cov_1 = empirical_covariance(torch.stack(grads_1))
    denom = torch.linalg.matrix_norm(cov_0) + torch.linalg.matrix_norm(cov_1) + EPS
    return float((torch.linalg.matrix_norm(cov_0 - cov_1) / denom).item())


def detect_onset(
    values: Iterable[float],
    steps: Iterable[int],
    *,
    z_score: float,
    floor: float,
    consecutive: int,
) -> int | None:
    values = list(values)
    steps = list(steps)
    if len(values) < 4:
        return None

    if len(values) >= consecutive and all(value >= floor for value in values[:consecutive]):
        return steps[0]

    baseline_count = max(3, min(5, len(values) // 3))
    baseline = np.asarray(values[:baseline_count], dtype=float)
    mean = float(baseline.mean())
    std = float(baseline.std(ddof=0))
    threshold = max(mean + z_score * std, floor)

    streak = 0
    candidate: int | None = None
    for step, value in zip(steps[baseline_count:], values[baseline_count:]):
        if value >= threshold:
            streak += 1
            if candidate is None:
                candidate = step
            if streak >= consecutive:
                return candidate
        else:
            streak = 0
            candidate = None

    return None


def relative_drift_series(values: Iterable[float]) -> list[float]:
    values = np.asarray(list(values), dtype=float)
    if values.size == 0:
        return []
    baseline_count = max(2, min(4, values.size // 4))
    baseline = float(np.mean(values[:baseline_count]))
    return (np.abs(values - baseline) / (abs(baseline) + EPS)).tolist()


def fit_slope(x: Iterable[float], y: Iterable[float]) -> float:
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    if x.size < 2 or np.allclose(x, x[0]):
        return 0.0
    design = np.column_stack([np.ones_like(x), x])
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coeffs[1])


def run_training(
    cfg: ExperimentConfig,
    *,
    seed: int,
    lr: float,
    input_scale: float,
    steps: int,
) -> dict[str, list[float] | list[int]]:
    dataset = build_dataset(seed, input_scale, cfg.train_size, cfg.eval_size, cfg.input_dim)
    model = SymmetricMLP(cfg.input_dim, cfg.hidden_dim, seed, cfg.init_noise)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    batch_gen = torch.Generator().manual_seed(seed + 100)

    step_norms: list[float] = []
    probe_steps: list[int] = []
    probe_step_norms: list[float] = []
    symmetry_scores: list[float] = []
    curvatures: list[float] = []

    for step in range(1, steps + 1):
        idx = torch.randint(0, cfg.train_size, (cfg.batch_size,), generator=batch_gen)
        xb = dataset["x_train"][idx]
        yb = dataset["y_train"][idx]

        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(xb) - yb) ** 2)
        loss.backward()

        prev = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
        optimizer.step()
        curr = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
        step_norm = float(torch.linalg.vector_norm(curr - prev).item())
        step_norms.append(step_norm)

        if step % cfg.probe_every == 0 or step == steps:
            probe_steps.append(step)
            probe_step_norms.append(step_norm)
            symmetry_scores.append(
                symmetry_score(
                    model,
                    dataset["x_eval"],
                    dataset["y_eval"],
                    cfg.probe_microbatches,
                    cfg.probe_batch_size,
                    probe_seed=seed * 1_000 + step,
                )
            )
            curvature_batch = slice(0, min(cfg.probe_batch_size * 2, cfg.eval_size))
            curvatures.append(
                compute_curvature(
                    model,
                    dataset["x_eval"][curvature_batch],
                    dataset["y_eval"][curvature_batch],
                    cfg.input_dim,
                    cfg.hidden_dim,
                )
            )

    return {
        "step_norms": step_norms,
        "probe_steps": probe_steps,
        "probe_step_norms": probe_step_norms,
        "symmetry_scores": symmetry_scores,
        "curvatures": curvatures,
    }


def compute_r2(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    design = np.column_stack([np.ones_like(x), x])
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coeffs
    residual = float(np.sum((y - fitted) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    if total <= EPS:
        return 0.0
    return max(0.0, 1.0 - residual / total)


def summarize_run(cfg: ExperimentConfig, seed: int, lr: float, input_scale: float) -> RunResult:
    steps = max(20, int(round(cfg.tau_max / lr)))
    main = run_training(cfg, seed=seed, lr=lr, input_scale=input_scale, steps=steps)

    probe_steps = list(main["probe_steps"])
    main_probe_norms = list(main["probe_step_norms"])
    drift_series = relative_drift_series(main_probe_norms)
    symmetry_series = list(main["symmetry_scores"])
    curvatures = np.asarray(list(main["curvatures"]), dtype=float)

    drift_onset = detect_onset(
        drift_series,
        probe_steps,
        z_score=cfg.detection_z,
        floor=cfg.drift_floor,
        consecutive=cfg.detection_consecutive,
    )
    symmetry_onset = detect_onset(
        symmetry_series,
        probe_steps,
        z_score=cfg.detection_z,
        floor=cfg.symmetry_floor,
        consecutive=cfg.detection_consecutive,
    )

    if symmetry_onset is None:
        drift_window = drift_series
        drift_without_symmetry = drift_onset is not None
    else:
        onset_index = next(i for i, step in enumerate(probe_steps) if step >= symmetry_onset)
        drift_window = drift_series[: onset_index + 1]
        drift_without_symmetry = False

    first_curvatures = curvatures[: max(2, min(4, len(curvatures)))]
    early_curvature = float(np.median(first_curvatures)) if len(first_curvatures) else float("nan")
    drift_amplitude = float(np.max(drift_window)) if drift_window else 0.0
    update_norm_slope = abs(fit_slope(np.asarray(probe_steps, dtype=float) * lr, drift_series))

    lead_steps = None
    if drift_onset is not None and symmetry_onset is not None:
        lead_steps = symmetry_onset - drift_onset

    return RunResult(
        scenario=cfg.scenario,
        seed=seed,
        lr=lr,
        input_scale=input_scale,
        steps=steps,
        drift_onset_step=drift_onset,
        symmetry_onset_step=symmetry_onset,
        drift_without_symmetry=drift_without_symmetry,
        lead_steps=lead_steps,
        drift_amplitude=drift_amplitude,
        update_norm_slope=update_norm_slope,
        early_curvature=early_curvature,
        first_probe_step=probe_steps[0],
        last_probe_step=probe_steps[-1],
    )


def verdict_early_warning(results: list[RunResult]) -> tuple[str, dict[str, float | int]]:
    comparable = [result for result in results if result.lead_steps is not None]
    support = [result for result in comparable if result.lead_steps > 0]
    falsify = [result for result in comparable if result.lead_steps <= 0]
    symmetry_without_drift = [
        result
        for result in results
        if result.symmetry_onset_step is not None and result.drift_onset_step is None
    ]
    drift_without_symmetry = [result for result in results if result.drift_without_symmetry]

    evaluable = len(comparable) + len(symmetry_without_drift)
    if evaluable < 4:
        verdict = "INCONCLUSIVE"
    else:
        support_fraction = (len(support) + 0.5 * len(drift_without_symmetry)) / evaluable
        falsify_fraction = (len(falsify) + len(symmetry_without_drift)) / evaluable
        median_lead = float(np.median([item.lead_steps for item in comparable])) if comparable else 0.0

        if support_fraction >= 0.67 and median_lead > 0:
            verdict = "SUPPORTED"
        elif falsify_fraction >= 0.67:
            verdict = "FALSIFIED"
        else:
            verdict = "INCONCLUSIVE"

    stats: dict[str, float | int] = {
        "comparable_runs": len(comparable),
        "supporting_runs": len(support),
        "falsifying_runs": len(falsify),
        "symmetry_without_drift": len(symmetry_without_drift),
        "drift_without_symmetry": len(drift_without_symmetry),
    }
    if comparable:
        stats["median_lead_steps"] = float(np.median([item.lead_steps for item in comparable]))
    return verdict, stats


def verdict_scaling(results: list[RunResult]) -> tuple[str, dict[str, float]]:
    usable = [
        result
        for result in results
        if result.update_norm_slope > 0.0 and math.isfinite(result.early_curvature) and result.early_curvature > 0.0
    ]
    if len(usable) < 5:
        return "INCONCLUSIVE", {"usable_runs": float(len(usable))}

    y = np.log(np.asarray([result.update_norm_slope for result in usable]) + 1e-6)
    lr_values = np.asarray([result.lr for result in usable], dtype=float)
    curvature = np.asarray([result.early_curvature for result in usable], dtype=float)

    score_lr = compute_r2(np.log(lr_values + 1e-6), y)
    score_ratio = compute_r2(np.log((lr_values / curvature) + 1e-6), y)
    score_product = compute_r2(np.log((lr_values * curvature) + 1e-6), y)

    if score_ratio >= score_lr + 0.05 and score_ratio >= score_product + 0.03:
        verdict = "SUPPORTED"
    elif score_lr >= score_ratio + 0.05 or score_product >= score_ratio + 0.05:
        verdict = "FALSIFIED"
    else:
        verdict = "INCONCLUSIVE"

    return verdict, {
        "usable_runs": float(len(usable)),
        "r2_lr": round(score_lr, 4),
        "r2_lr_over_curvature": round(score_ratio, 4),
        "r2_lr_times_curvature": round(score_product, 4),
    }


def overall_verdict(early_warning: str, scaling: str) -> str:
    if early_warning == "FALSIFIED" or scaling == "FALSIFIED":
        return "FALSIFIED"
    if early_warning == "SUPPORTED" and scaling == "SUPPORTED":
        return "VINDICATED"
    return "INCONCLUSIVE"


def format_run_table(results: list[RunResult]) -> str:
    lines = [
        "seed  lr     scale  drift_onset  symmetry_onset  lead  drift_amp  drift_slope  curvature",
        "----  -----  -----  -----------  --------------  ----  ---------  -----------  ---------",
    ]
    for result in results:
        lead = "na" if result.lead_steps is None else str(result.lead_steps)
        drift = "na" if result.drift_onset_step is None else str(result.drift_onset_step)
        sym = "na" if result.symmetry_onset_step is None else str(result.symmetry_onset_step)
        lines.append(
            f"{result.seed:<4}  {result.lr:<5.3f}  {result.input_scale:<5.2f}  "
            f"{drift:<11}  {sym:<14}  {lead:<4}  {result.drift_amplitude:<9.4f}  {result.update_norm_slope:<11.4f}  "
            f"{result.early_curvature:<9.4f}"
        )
    return "\n".join(lines)


def run_suite(cfg: ExperimentConfig) -> dict[str, object]:
    results = []
    total = len(cfg.seeds) * len(cfg.learning_rates) * len(cfg.input_scales)
    run_index = 0
    for seed in cfg.seeds:
        for lr in cfg.learning_rates:
            for input_scale in cfg.input_scales:
                run_index += 1
                print(
                    f"[{run_index}/{total}] seed={seed} lr={lr:.4f} input_scale={input_scale:.2f}",
                    flush=True,
                )
                results.append(summarize_run(cfg, seed=seed, lr=lr, input_scale=input_scale))

    early_warning, early_stats = verdict_early_warning(results)
    scaling, scaling_stats = verdict_scaling(results)
    overall = overall_verdict(early_warning, scaling)

    return {
        "config": asdict(cfg),
        "results": [asdict(result) for result in results],
        "verdicts": {
            "early_warning": early_warning,
            "scaling": scaling,
            "overall": overall,
        },
        "stats": {
            "early_warning": early_stats,
            "scaling": scaling_stats,
        },
        "table": format_run_table(results),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["hypothesis", "instant-break"], default="hypothesis")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--learning-rates", default="0.02,0.04,0.08")
    parser.add_argument("--input-scales", default="0.75,1.25,1.75")
    parser.add_argument("--tau-max", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--probe-every", type=int, default=15)
    parser.add_argument("--probe-microbatches", type=int, default=8)
    parser.add_argument("--probe-batch-size", type=int, default=32)
    parser.add_argument("--train-size", type=int, default=1024)
    parser.add_argument("--eval-size", type=int, default=256)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--init-noise", type=float, default=1e-4)
    parser.add_argument("--symmetry-floor", type=float, default=0.02)
    parser.add_argument("--drift-floor", type=float, default=0.25)
    parser.add_argument("--detection-z", type=float, default=2.5)
    parser.add_argument("--detection-consecutive", type=int, default=2)
    parser.add_argument("--json", action="store_true", help="Emit the full report as JSON.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = build_config(args)
    report = run_suite(cfg)

    print()
    print(f"Scenario: {cfg.scenario}")
    print(report["table"])
    print()
    print(f"Early-warning verdict: {report['verdicts']['early_warning']}")
    print(f"Scaling verdict: {report['verdicts']['scaling']}")
    print(f"Overall verdict: {report['verdicts']['overall']}")
    print(f"Early-warning stats: {json.dumps(report['stats']['early_warning'], sort_keys=True)}")
    print(f"Scaling stats: {json.dumps(report['stats']['scaling'], sort_keys=True)}")

    if args.json:
        print()
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
