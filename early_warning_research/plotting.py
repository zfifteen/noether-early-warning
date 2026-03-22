from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_representative_timeseries(
    step_rows: list[dict[str, object]],
    probe_rows: list[dict[str, object]],
    run_summary: dict[str, object],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax_left = plt.subplots(figsize=(9, 4.8))

    steps = [int(row["step"]) for row in step_rows]
    running_mean = [float(row["running_mean_update_norm"]) for row in step_rows]
    ax_left.plot(steps, running_mean, color="#0b7285", linewidth=2.0, label="Running mean of ||Δw_t||")
    ax_left.set_xlabel("Training step")
    ax_left.set_ylabel("Running mean of ||Δw_t||", color="#0b7285")
    ax_left.tick_params(axis="y", labelcolor="#0b7285")

    drift_onset = run_summary.get("drift_onset_step")
    if drift_onset is not None:
        ax_left.axvline(int(drift_onset), color="#0b7285", linestyle="--", linewidth=1.4)

    ax_right = ax_left.twinx()
    if probe_rows:
        probe_steps = [int(row["step"]) for row in probe_rows]
        symmetry_scores = [float(row["symmetry_score"]) for row in probe_rows]
        ax_right.plot(
            probe_steps,
            symmetry_scores,
            color="#e8590c",
            marker="o",
            linewidth=1.6,
            markersize=3.8,
            label="Symmetry score",
        )
        threshold = probe_rows[0].get("symmetry_threshold")
        if threshold is not None:
            ax_right.axhline(float(threshold), color="#e8590c", linestyle=":", linewidth=1.2)

    symmetry_onset = run_summary.get("symmetry_onset_step")
    if symmetry_onset is not None:
        ax_right.axvline(int(symmetry_onset), color="#e8590c", linestyle="--", linewidth=1.2)
    ax_right.set_ylabel("Symmetry score", color="#e8590c")
    ax_right.tick_params(axis="y", labelcolor="#e8590c")

    fig.suptitle(f"Representative run: {run_summary['run_id']}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_onset_ordering(run_summaries: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, max(4.2, 0.35 * len(run_summaries) + 1.6)))

    ordered = sorted(
        run_summaries,
        key=lambda row: (
            row["lead_steps"] is None,
            10**9 if row["lead_steps"] is None else row["lead_steps"],
            row["run_id"],
        ),
    )

    for index, row in enumerate(ordered):
        drift = row["drift_onset_step"]
        symmetry = row["symmetry_onset_step"]
        y = index
        if drift is not None and symmetry is not None:
            ax.plot([int(drift), int(symmetry)], [y, y], color="#ced4da", linewidth=2.0, zorder=1)
        if drift is not None:
            ax.scatter(int(drift), y, color="#0b7285", s=34, zorder=2)
        if symmetry is not None:
            ax.scatter(int(symmetry), y, color="#e8590c", marker="s", s=34, zorder=3)

    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels([row["run_id"] for row in ordered], fontsize=8)
    ax.set_xlabel("Onset step")
    ax.set_title("Drift onset vs symmetry onset")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_detector_suite_comparison(suite_summaries: dict[str, dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels: list[str] = []
    data: list[list[float]] = []
    colors: list[str] = []
    color_map = {
        "covariance_mismatch": "#e8590c",
        "mean_gradient_mismatch": "#0b7285",
        "activation_stat_mismatch": "#2b8a3e",
    }

    for suite_name, suite_summary in suite_summaries.items():
        detector_summaries = suite_summary.get("detector_summaries", {})
        for detector_name in ("covariance_mismatch", "mean_gradient_mismatch", "activation_stat_mismatch"):
            detector_summary = detector_summaries.get(detector_name)
            if detector_summary is None:
                continue
            lead_distribution = detector_summary.get("lead_distribution", [])
            if not lead_distribution:
                continue
            labels.append(f"{suite_name}\n{detector_name}")
            data.append([float(value) for value in lead_distribution])
            colors.append(color_map[detector_name])

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.55 * max(1, len(labels)) + 1.8)))
    if data:
        box = ax.boxplot(data, orientation="horizontal", patch_artist=True, tick_labels=labels)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        for median in box["medians"]:
            median.set_color("#212529")
        ax.axvline(0, color="#868e96", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Lead steps (symmetry onset - drift onset)")
        ax.set_title("Lead distributions by suite and detector")
    else:
        ax.text(0.5, 0.5, "No comparable detector leads available", ha="center", va="center")
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_benchmark_lead_distributions(resolved_rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels: list[str] = []
    data: list[list[float]] = []
    colors: list[str] = []
    color_map = {
        "covariance_mismatch": "#e8590c",
        "mean_gradient_mismatch": "#0b7285",
        "activation_stat_mismatch": "#2b8a3e",
    }

    suite_order = (
        "full_batch_positive",
        "stochastic_positive",
        "full_batch_instant_break",
        "stochastic_instant_break",
    )
    for suite_name in suite_order:
        for detector_name in ("covariance_mismatch", "mean_gradient_mismatch", "activation_stat_mismatch"):
            lead_distribution = [
                float(row["resolved_lead_steps"])
                for row in resolved_rows
                if row["suite_name"] == suite_name
                and row["detector_name"] == detector_name
                and row["resolved_lead_steps"] is not None
            ]
            if not lead_distribution:
                continue
            labels.append(f"{suite_name}\n{detector_name}")
            data.append(lead_distribution)
            colors.append(color_map[detector_name])

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.55 * max(1, len(labels)) + 1.8)))
    if data:
        box = ax.boxplot(data, orientation="horizontal", patch_artist=True, tick_labels=labels)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        for median in box["medians"]:
            median.set_color("#212529")
        ax.axvline(0, color="#868e96", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Resolved lead steps (symmetry onset - short-window drift onset)")
        ax.set_title("Resolved lead distributions by suite and detector")
    else:
        ax.text(0.5, 0.5, "No resolved leads available", ha="center", va="center")
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_benchmark_resolution_curves(resolution_rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    suites = ("full_batch_positive", "stochastic_positive")
    color_map = {
        "covariance_mismatch": "#e8590c",
        "mean_gradient_mismatch": "#0b7285",
        "activation_stat_mismatch": "#2b8a3e",
    }

    for ax, suite_name in zip(axes, suites):
        suite_rows = [row for row in resolution_rows if row["suite_name"] == suite_name]
        if not suite_rows:
            ax.text(0.5, 0.5, "No censored runs", ha="center", va="center")
            ax.set_axis_off()
            continue
        for detector_name in ("covariance_mismatch", "mean_gradient_mismatch", "activation_stat_mismatch"):
            detector_rows = sorted(
                (row for row in suite_rows if row["detector_name"] == detector_name),
                key=lambda row: int(row["step"]),
            )
            if not detector_rows:
                continue
            steps = [int(row["step"]) for row in detector_rows]
            resolved_fraction = [float(row["resolved_fraction"]) for row in detector_rows]
            ax.step(
                steps,
                resolved_fraction,
                where="post",
                linewidth=2.0,
                color=color_map[detector_name],
                label=detector_name,
            )
        ax.set_title(suite_name)
        ax.set_xlabel("Training step")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Resolved fraction of short-window censored runs")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
