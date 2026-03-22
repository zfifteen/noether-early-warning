from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch

from .detectors import detect_drift_onset, detect_symmetry_onset
from .experiments import build_paired_dataset, default_output_root
from .models import PairedMLP
from .suites import PairedMLPConfig, SuiteConfig, get_suite


EPS = 1e-12


def empirical_covariance(samples: torch.Tensor) -> torch.Tensor:
    if samples.shape[0] < 2:
        return torch.zeros((samples.shape[1], samples.shape[1]), dtype=samples.dtype)
    centered = samples - samples.mean(dim=0, keepdim=True)
    return centered.t() @ centered / (samples.shape[0] - 1)


def original_covariance_score(
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

        weight_grad = model.monitored_weight_grad().detach()
        readout_grad = model.output_layer.weight.grad.detach().squeeze(0)
        for left, right in model.pairs:
            pair_samples[(left, right)][0].append(torch.cat([weight_grad[left], readout_grad[left : left + 1]]))
            pair_samples[(left, right)][1].append(torch.cat([weight_grad[right], readout_grad[right : right + 1]]))

    pair_scores: list[float] = []
    for left_samples, right_samples in pair_samples.values():
        cov_left = empirical_covariance(torch.stack(left_samples))
        cov_right = empirical_covariance(torch.stack(right_samples))
        denom = torch.linalg.matrix_norm(cov_left) + torch.linalg.matrix_norm(cov_right) + EPS
        mismatch = torch.linalg.matrix_norm(cov_left - cov_right) / denom
        pair_scores.append(float(mismatch.item()))
    return float(np.mean(pair_scores))


def asymmetry_vector(model: PairedMLP) -> torch.Tensor:
    layer = model.hidden_layer if model.hidden_layer is not None else model.input_layer
    weight_values = layer.weight.detach()
    readout = model.output_layer.weight.detach().squeeze(0)
    pieces: list[torch.Tensor] = []
    for left, right in model.pairs:
        pieces.append(weight_values[left] - weight_values[right])
        pieces.append(readout[left : left + 1] - readout[right : right + 1])
    return torch.cat(pieces)


def probe_seed(seed: int, learning_rate: float, input_scale: float, step: int) -> int:
    return seed * 100_000 + int(round(learning_rate * 10_000)) * 100 + int(round(input_scale * 100)) + step


def run_id(seed: int, learning_rate: float, input_scale: float) -> str:
    return f"seed{seed}_lr{learning_rate:.4f}_scale{input_scale:.2f}"


def sample_train_batch(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    batch_size: int | None,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size is None or batch_size >= x_train.shape[0]:
        return x_train, y_train
    idx = torch.randint(0, x_train.shape[0], (batch_size,), generator=gen)
    return x_train[idx], y_train[idx]


def path_efficiency(vectors: list[torch.Tensor], onset_step: int, horizon: int) -> float | None:
    start_idx = max(0, onset_step - 1)
    end_idx = min(len(vectors) - 1, start_idx + horizon)
    if end_idx <= start_idx:
        return None
    segment = vectors[start_idx : end_idx + 1]
    displacement = torch.linalg.vector_norm(segment[-1] - segment[0]).item()
    path = sum(torch.linalg.vector_norm(segment[idx + 1] - segment[idx]).item() for idx in range(len(segment) - 1))
    if path <= EPS:
        return 0.0
    return float(displacement / path)


def norm_growth_ratio(vectors: list[torch.Tensor], onset_step: int, horizon: int) -> float | None:
    start_idx = max(0, onset_step - 1)
    end_idx = min(len(vectors) - 1, start_idx + horizon)
    if end_idx <= start_idx:
        return None
    norms = [torch.linalg.vector_norm(vector).item() for vector in vectors[start_idx : end_idx + 1]]
    numerator = norms[-1] - norms[0]
    denominator = sum(abs(norms[idx + 1] - norms[idx]) for idx in range(len(norms) - 1))
    if denominator <= EPS:
        return 0.0
    return float(numerator / denominator)


def auc_score(values: list[float], labels: list[int]) -> float | None:
    positives = [value for value, label in zip(values, labels) if label == 1]
    negatives = [value for value, label in zip(values, labels) if label == 0]
    if not positives or not negatives:
        return None
    ranks = scipy.stats.rankdata(values)
    positive_rank_sum = float(sum(rank for rank, label in zip(ranks, labels) if label == 1))
    n_pos = len(positives)
    n_neg = len(negatives)
    return float((positive_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def symmetry_growth_metrics(
    probe_steps: list[int],
    symmetry_scores: list[float],
    drift_onset_step: int,
    horizon: int,
) -> tuple[float, float]:
    base_index = next((idx for idx, step in enumerate(probe_steps) if step >= drift_onset_step), len(probe_steps) - 1)
    target_index = next((idx for idx, step in enumerate(probe_steps) if step >= drift_onset_step + horizon), len(probe_steps) - 1)
    window_scores = [
        symmetry_scores[idx]
        for idx, step in enumerate(probe_steps)
        if drift_onset_step <= step <= drift_onset_step + horizon
    ]
    if not window_scores:
        window_scores = [symmetry_scores[base_index]]
    base_score = symmetry_scores[base_index]
    target_score = symmetry_scores[target_index]
    return float(target_score - base_score), float(max(window_scores) - base_score)


def group_stats(rows: list[dict[str, object]], key: str) -> dict[str, object]:
    supportive = [float(row[key]) for row in rows if row["verdict_label"] == "supportive" and row[key] is not None]
    censored = [float(row[key]) for row in rows if row["verdict_label"] == "censored" and row[key] is not None]
    if supportive and censored:
        stat = scipy.stats.mannwhitneyu(supportive, censored, alternative="two-sided")
        labels = [1] * len(supportive) + [0] * len(censored)
        auc = auc_score(supportive + censored, labels)
        p_value = float(stat.pvalue)
    else:
        auc = None
        p_value = None
    return {
        "supportive_count": len(supportive),
        "censored_count": len(censored),
        "supportive_median": None if not supportive else float(np.median(supportive)),
        "censored_median": None if not censored else float(np.median(censored)),
        "supportive_mean": None if not supportive else float(np.mean(supportive)),
        "censored_mean": None if not censored else float(np.mean(censored)),
        "mannwhitney_p_value": p_value,
        "auc_supportive_vs_censored": auc,
    }


def plot_commitment_separation(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    supportive = [float(row["path_efficiency_60"]) for row in rows if row["verdict_label"] == "supportive" and row["path_efficiency_60"] is not None]
    censored = [float(row["path_efficiency_60"]) for row in rows if row["verdict_label"] == "censored" and row["path_efficiency_60"] is not None]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    data = [supportive, censored]
    labels = ["supportive", "censored"]
    box = ax.boxplot(data, patch_artist=True, labels=labels)
    colors = ["#0b7285", "#c92a2a"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for median in box["medians"]:
        median.set_color("#212529")
    ax.set_ylabel("Path efficiency over 60 steps")
    ax.set_title("Post-onset commitment separates supportive vs censored runs")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_commitment_analysis(
    suite_name: str,
    horizon_short: int,
    horizon_long: int,
    output_root: Path | None = None,
    quiet: bool = False,
) -> dict[str, object]:
    suite = get_suite(suite_name, smoke=False)
    if suite.kind != "paired_mlp":
        raise ValueError("commitment analysis currently supports paired_mlp suites only")

    model_cfg = suite.model
    assert isinstance(model_cfg, PairedMLPConfig)

    base_output = (
        output_root
        if output_root is not None
        else default_output_root().parent / "early_warning_research_commitment"
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = base_output / f"{timestamp}_{suite_name}_commitment"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []

    combinations = [
        (seed, learning_rate, input_scale)
        for seed in suite.sweep.seeds
        for learning_rate in suite.sweep.learning_rates
        for input_scale in suite.sweep.input_scales
    ]
    for index, (seed, learning_rate, input_scale) in enumerate(combinations, start=1):
        if not quiet:
            print(
                f"[{index}/{len(combinations)}] {suite_name}: seed={seed} lr={learning_rate:.4f} input_scale={input_scale:.2f}",
                flush=True,
            )

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

        update_norms: list[float] = []
        covariance_scores: list[float] = []
        probe_steps: list[int] = []
        asymmetry_vectors: list[torch.Tensor] = []
        final_loss = 0.0

        for step in range(1, suite.training.total_steps + 1):
            xb, yb = sample_train_batch(data["x_train"], data["y_train"], suite.training.batch_size, batch_gen)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            grad_vec = torch.cat([param.grad.detach().flatten() for param in model.parameters() if param.grad is not None])
            update_norms.append(float((learning_rate * torch.linalg.vector_norm(grad_vec)).item()))
            final_loss = float(loss.item())
            optimizer.step()
            asymmetry_vectors.append(asymmetry_vector(model))

            if step % suite.probe.every_steps == 0 or step == suite.training.total_steps:
                score = original_covariance_score(
                    model,
                    data["x_probe"],
                    data["y_probe"],
                    suite.probe.microbatches,
                    suite.probe.microbatch_size,
                    probe_seed(seed, learning_rate, input_scale, step),
                )
                covariance_scores.append(score)
                probe_steps.append(step)

        steps = list(range(1, suite.training.total_steps + 1))
        drift_result = detect_drift_onset(update_norms, steps, suite.detector)
        symmetry_result = detect_symmetry_onset(covariance_scores, probe_steps, suite.detector)

        lead_steps = None
        if drift_result.onset_step is not None and symmetry_result.onset_step is not None:
            lead_steps = symmetry_result.onset_step - drift_result.onset_step
        if lead_steps is not None and lead_steps > 0:
            verdict_label = "supportive"
        elif lead_steps is not None:
            verdict_label = "falsifying"
        elif drift_result.onset_step is not None and symmetry_result.onset_step is None:
            verdict_label = "censored"
        else:
            verdict_label = "no_ordering"

        row = {
            "run_id": run_id(seed, learning_rate, input_scale),
            "seed": seed,
            "learning_rate": learning_rate,
            "input_scale": input_scale,
            "drift_onset_step": drift_result.onset_step,
            "symmetry_onset_step": symmetry_result.onset_step,
            "lead_steps": lead_steps,
            "verdict_label": verdict_label,
            "final_loss": final_loss,
            "terminal_symmetry_score": covariance_scores[-1] if covariance_scores else None,
            "path_efficiency_30": None if drift_result.onset_step is None else path_efficiency(asymmetry_vectors, drift_result.onset_step, horizon_short),
            "path_efficiency_60": None if drift_result.onset_step is None else path_efficiency(asymmetry_vectors, drift_result.onset_step, horizon_long),
            "norm_growth_ratio_30": None if drift_result.onset_step is None else norm_growth_ratio(asymmetry_vectors, drift_result.onset_step, horizon_short),
            "norm_growth_ratio_60": None if drift_result.onset_step is None else norm_growth_ratio(asymmetry_vectors, drift_result.onset_step, horizon_long),
            "symmetry_gain_30": None,
            "symmetry_gain_60": None,
            "symmetry_max_gain_30": None,
            "symmetry_max_gain_60": None,
        }
        if drift_result.onset_step is not None and covariance_scores:
            gain_short, max_gain_short = symmetry_growth_metrics(probe_steps, covariance_scores, drift_result.onset_step, horizon_short)
            gain_long, max_gain_long = symmetry_growth_metrics(probe_steps, covariance_scores, drift_result.onset_step, horizon_long)
            row["symmetry_gain_30"] = gain_short
            row["symmetry_gain_60"] = gain_long
            row["symmetry_max_gain_30"] = max_gain_short
            row["symmetry_max_gain_60"] = max_gain_long
        rows.append(row)

    summary = {
        "suite_name": suite_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "parameters": {
            "horizon_short": horizon_short,
            "horizon_long": horizon_long,
            "suite": suite_name,
            "classifier": "supportive vs censored under covariance-mismatch direct symmetry detection",
        },
        "counts": {
            "supportive_runs": sum(row["verdict_label"] == "supportive" for row in rows),
            "censored_runs": sum(row["verdict_label"] == "censored" for row in rows),
            "falsifying_runs": sum(row["verdict_label"] == "falsifying" for row in rows),
            "no_ordering_runs": sum(row["verdict_label"] == "no_ordering" for row in rows),
        },
        "onset_timing_stats": group_stats(rows, "drift_onset_step"),
        "path_efficiency_30_stats": group_stats(rows, "path_efficiency_30"),
        "path_efficiency_60_stats": group_stats(rows, "path_efficiency_60"),
        "norm_growth_ratio_30_stats": group_stats(rows, "norm_growth_ratio_30"),
        "norm_growth_ratio_60_stats": group_stats(rows, "norm_growth_ratio_60"),
        "symmetry_gain_30_stats": group_stats(rows, "symmetry_gain_30"),
        "symmetry_gain_60_stats": group_stats(rows, "symmetry_gain_60"),
        "symmetry_max_gain_30_stats": group_stats(rows, "symmetry_max_gain_30"),
        "symmetry_max_gain_60_stats": group_stats(rows, "symmetry_max_gain_60"),
        "terminal_symmetry_score_stats": group_stats(rows, "terminal_symmetry_score"),
    }

    runs_csv = output_dir / "runs.csv"
    summary_json = output_dir / "summary.json"
    figure_png = output_dir / "commitment_separation.png"
    with runs_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary_json.write_text(json.dumps(summary, indent=2))
    plot_commitment_separation(rows, figure_png)
    summary["artifact_files"] = {
        "runs_csv": str(runs_csv),
        "summary_json": str(summary_json),
        "commitment_separation_png": str(figure_png),
    }
    summary_json.write_text(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze post-onset commitment in paired MLP runs.")
    parser.add_argument("--suite", default="main_paired_mlp")
    parser.add_argument("--horizon-short", type=int, default=30)
    parser.add_argument("--horizon-long", type=int, default=60)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_commitment_analysis(
        suite_name=args.suite,
        horizon_short=args.horizon_short,
        horizon_long=args.horizon_long,
        output_root=None if args.output_dir is None else Path(args.output_dir),
        quiet=args.quiet,
    )
    print(f"Artifacts: {summary['output_dir']}")
    print(f"Supportive runs: {summary['counts']['supportive_runs']} | Censored runs: {summary['counts']['censored_runs']}")
    print(f"Drift-onset AUC: {summary['onset_timing_stats']['auc_supportive_vs_censored']}")
    print(f"Symmetry-gain-{args.horizon_long} AUC: {summary['symmetry_gain_60_stats']['auc_supportive_vs_censored']}")
    print(f"Terminal-symmetry-score AUC: {summary['terminal_symmetry_score_stats']['auc_supportive_vs_censored']}")


if __name__ == "__main__":
    main()
