from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "docs" / "findings_assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "repo_banner.png"

    plt.rcParams["font.family"] = "DejaVu Sans"

    fig = plt.figure(figsize=(16, 5), dpi=200, facecolor="#07111b")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Deep, restrained background glow
    x = np.linspace(0, 1, 900)
    y = np.linspace(0, 1, 300)
    xx, yy = np.meshgrid(x, y)
    glow_left = np.exp(-((xx - 0.18) ** 2 / 0.03 + (yy - 0.68) ** 2 / 0.20))
    glow_right = np.exp(-((xx - 0.82) ** 2 / 0.04 + (yy - 0.28) ** 2 / 0.12))
    base = np.zeros((y.size, x.size, 4))
    base[..., 0] = 0.03 + 0.05 * glow_right
    base[..., 1] = 0.06 + 0.10 * glow_left + 0.03 * glow_right
    base[..., 2] = 0.10 + 0.12 * glow_left + 0.10 * glow_right
    base[..., 3] = 1.0
    ax.imshow(base, extent=(0, 1, 0, 1), origin="lower", aspect="auto", zorder=0)

    # Quiet threshold line
    threshold_y = 0.50
    ax.plot([0.49, 0.95], [threshold_y, threshold_y], color=(1, 1, 1, 0.12), lw=1.2, ls=(0, (2, 4)), zorder=2)

    # Main signal curves
    t = np.linspace(0, 1, 500)
    x0 = 0.49 + 0.42 * t
    drift = 0.29 + 0.40 / (1 + np.exp(-14 * (t - 0.44)))
    symmetry = 0.27 + 0.44 / (1 + np.exp(-16 * (t - 0.67)))

    drift_color = "#5fd4ff"
    sym_color = "#ff8cb2"

    ax.plot(x0, drift, color=drift_color, lw=5.0, solid_capstyle="round", zorder=4)
    ax.plot(x0, symmetry, color=sym_color, lw=5.0, solid_capstyle="round", zorder=4)

    # Subtle lead cue
    alarm_x = x0[np.argmax(drift > threshold_y)]
    ax.plot([alarm_x, alarm_x], [0.21, 0.74], color=(0.37, 0.83, 1.0, 0.18), lw=1.3, zorder=3)

    # Minimal text block
    ax.text(
        0.075,
        0.63,
        "Noether Early Warning",
        color="#ecf3fb",
        fontsize=34,
        fontweight="bold",
        ha="left",
        va="center",
        zorder=5,
    )
    ax.text(
        0.077,
        0.47,
        "Drift before direct symmetry detection",
        color="#adc0d4",
        fontsize=16,
        ha="left",
        va="center",
        zorder=5,
    )

    # Small legend, not a lecture
    ax.text(0.078, 0.31, "drift", color=drift_color, fontsize=13, ha="left", va="center", zorder=5)
    ax.text(0.122, 0.31, "arrives first in gradual regimes", color="#8fa3b8", fontsize=13, ha="left", va="center", zorder=5)

    ax.text(0.72, 0.72, "drift", color=drift_color, fontsize=12, ha="left", va="center", zorder=5)
    ax.text(0.79, 0.78, "symmetry", color=sym_color, fontsize=12, ha="left", va="center", zorder=5)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor(), pad_inches=0)
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
