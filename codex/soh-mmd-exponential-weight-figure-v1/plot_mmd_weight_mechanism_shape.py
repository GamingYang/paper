"""Generate a label-free adaptive MMD weight curve for mechanism diagrams."""

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
    x = (1.00 - soh) / 0.20
    y = np.exp(LAMBDA_SCALE * (1.0 - soh))

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.unicode_minus": False,
            "path.simplify": True,
        }
    )

    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    # A soft under-stroke keeps the curve readable when placed over mechanism art.
    ax.plot(
        x,
        y,
        color="#D9C8FF",
        lw=14.0,
        alpha=0.80,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=1,
    )
    ax.plot(
        x,
        y,
        color="#5B0DAD",
        lw=8.0,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=2,
    )

    key_soh = np.array([1.00, 0.95, 0.90, 0.85, 0.80])
    key_x = (1.00 - key_soh) / 0.20
    key_y = np.exp(LAMBDA_SCALE * (1.0 - key_soh))
    ax.scatter(key_x, key_y, s=145, color="#2E2E2E", edgecolors="white", linewidths=1.4, zorder=3)

    ax.set_xlim(-0.035, 1.035)
    ax.set_ylim(0.82, 3.16)
    ax.axis("off")

    png_path = FIG_DIR / "mmd_weight_adaptive_curve_mechanism_clean.png"
    svg_path = FIG_DIR / "mmd_weight_adaptive_curve_mechanism_clean.svg"
    fig.savefig(png_path, dpi=600, transparent=True, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(svg_path, transparent=True, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    print(f"saved: {png_path}")
    print(f"saved: {svg_path}")


if __name__ == "__main__":
    main()
