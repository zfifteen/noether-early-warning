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
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.text(0.5, 0.5, "Detector comparison is not part of Benchmark 1.", ha="center", va="center")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
