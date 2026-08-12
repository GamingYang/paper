"""Generate an adaptive MMD weight curve with axes and grid for mechanism figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

LAMBDA_SCALE = 5.5


def main() -> None:
    soh = np.linspace(1.00, 0.80, 400)
    y = np.exp(LAMBDA_SCALE * (1.0 - soh))

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.unicode_minus": False,
            "path.simplify": True,
        }
    )

    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.grid(True, which="major", color="#D7D7D7", linestyle="--", linewidth=1.05, alpha=0.78, zorder=0)

    ax.plot(
        soh,
        y,
        color="#D9C8FF",
        lw=13.0,
        alpha=0.82,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=2,
    )
    ax.plot(
        soh,
        y,
        color="#5B0DAD",
        lw=7.2,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=3,
    )

    key_soh = np.array([1.00, 0.95, 0.90, 0.85, 0.80])
    key_y = np.exp(LAMBDA_SCALE * (1.0 - key_soh))
    ax.scatter(key_soh, key_y, s=120, color="#2E2E2E", edgecolors="white", linewidths=1.3, zorder=4)

    ax.set_xlim(1.005, 0.795)
    ax.set_ylim(0.86, 3.18)
    ax.set_xticks([1.00, 0.95, 0.90, 0.85, 0.80])
    ax.set_yticks([1.0, 1.5, 2.0, 2.5, 3.0])
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.tick_params(axis="both", labelsize=14, direction="out", length=5.5, width=1.5, pad=6)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_color("#2A2A2A")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.8)
        spine.set_color("#333333")

    fig.tight_layout(pad=0.55)
    png_path = FIG_DIR / "mmd_weight_adaptive_curve_mechanism_axes_grid.png"
    svg_path = FIG_DIR / "mmd_weight_adaptive_curve_mechanism_axes_grid.svg"
    fig.savefig(png_path, dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(svg_path, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print(f"saved: {png_path}")
    print(f"saved: {svg_path}")


if __name__ == "__main__":
    main()
