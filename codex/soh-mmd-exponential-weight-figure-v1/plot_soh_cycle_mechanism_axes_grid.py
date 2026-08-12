"""Generate a single-cell SOH-vs-cycle curve using the project battery data."""

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
    cycles = np.array(sorted(dq.keys()), dtype=int)
    capacity_mah = np.array([float(dq[int(c)]) for c in cycles], dtype=float)
    soh = capacity_mah / RATED_CAPACITY_MAH
    return cell_id, cycles, soh


def smooth_curve(y: np.ndarray, window: int = 23) -> np.ndarray:
    series = pd.Series(y)
    return series.rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cell_id, cycles, soh = load_nominal_soh()
    soh_plot = smooth_curve(soh)

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
        cycles,
        soh_plot,
        color="#DDF5E8",
        lw=13.0,
        alpha=0.90,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=2,
    )
    ax.plot(
        cycles,
        soh_plot,
        color="#8FD19E",
        lw=7.2,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=3,
    )

    ax.set_xlim(0, cycles.max() + 30)
    ax.set_ylim(0.78, 1.09)
    ax.set_xticks([0, 400, 800, 1200, 1600])
    ax.set_yticks([0.8, 0.9, 1.0, 1.1])
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
    png_path = FIG_DIR / f"soh_vs_cycles_{cell_id}_mechanism_axes_grid_green.png"
    svg_path = FIG_DIR / f"soh_vs_cycles_{cell_id}_mechanism_axes_grid_green.svg"
    fig.savefig(png_path, dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(svg_path, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print(f"saved: {png_path}")
    print(f"saved: {svg_path}")


if __name__ == "__main__":
    main()
