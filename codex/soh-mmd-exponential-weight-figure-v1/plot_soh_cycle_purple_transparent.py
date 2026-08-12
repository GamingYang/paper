"""Generate a transparent purple SOH decline curve with segmentation dots."""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "codex" / "soh-mmd-exponential-weight-figure-v1"
FIG_DIR = RUN_DIR / "figures"
RAW_PATH = ROOT / "our_data" / "10-1.pkl"
RATED_CAPACITY_MAH = 1100.0


def load_nominal_soh() -> tuple[str, np.ndarray, np.ndarray]:
    with RAW_PATH.open("rb") as f:
        obj = pickle.load(f)
    cell_id = next(iter(obj))
    dq = obj[cell_id]["dq"]
    cycles = np.array(sorted(dq.keys()), dtype=float)
    capacity_mah = np.array([float(dq[int(c)]) for c in cycles], dtype=float)
    soh = capacity_mah / RATED_CAPACITY_MAH
    return cell_id, cycles, soh


def smooth_curve(y: np.ndarray, window: int = 25) -> np.ndarray:
    return pd.Series(y).rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cell_id, cycles, soh = load_nominal_soh()
    soh_plot = smooth_curve(soh)
    x = (cycles - cycles.min()) / (cycles.max() - cycles.min())

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.unicode_minus": False,
            "path.simplify": True,
            "path.simplify_threshold": 0.12,
        }
    )

    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    ax.plot(
        x,
        soh_plot,
        color="#D9C8FF",
        lw=14.0,
        alpha=0.80,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=1,
    )
    ax.plot(
        x,
        soh_plot,
        color="#5B0DAD",
        lw=8.0,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=2,
    )

    key_x = np.linspace(0.0, 1.0, 5)
    key_y = np.interp(key_x, x, soh_plot)
    ax.scatter(key_x, key_y, s=145, color="#2E2E2E", edgecolors="white", linewidths=1.4, zorder=3)

    ax.set_xlim(-0.035, 1.035)
    ax.set_ylim(float(soh_plot.min()) - 0.025, float(soh_plot.max()) + 0.025)
    ax.axis("off")

    png_path = FIG_DIR / f"soh_vs_cycles_{cell_id}_purple_transparent_dots.png"
    svg_path = FIG_DIR / f"soh_vs_cycles_{cell_id}_purple_transparent_dots.svg"
    fig.savefig(png_path, dpi=600, transparent=True, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(svg_path, transparent=True, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    print(f"saved: {png_path}")
    print(f"saved: {svg_path}")


if __name__ == "__main__":
    main()
