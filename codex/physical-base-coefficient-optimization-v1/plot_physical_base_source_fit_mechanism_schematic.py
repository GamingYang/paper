"""Generate a text-light mechanism schematic for the physical-base source fit."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "codex" / "physical-base-coefficient-optimization-v1"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
GLOBAL_MAX_CYCLE = 3000.0


def find_processed_clean(root: Path) -> Path:
    for child in root.iterdir():
        candidate = child / "processed_clean"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate processed_clean under project root.")


def design_matrix(p_norm: np.ndarray) -> np.ndarray:
    return np.column_stack([p_norm**3, p_norm**2, p_norm, np.ones_like(p_norm)])


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = find_processed_clean(ROOT)
    p_norm = np.load(data_dir / "clean_source_p.npy").astype(np.float64).reshape(-1) / GLOBAL_MAX_CYCLE
    y = np.load(data_dir / "clean_source_y_nominal.npy").astype(np.float64).reshape(-1)

    bin_df = pd.read_csv(TABLE_DIR / "source_soh_binned_for_physical_base.csv")
    coef = pd.read_csv(TABLE_DIR / "physical_base_coefficients.csv").iloc[0]
    beta = coef[["a", "b", "c", "d"]].to_numpy(dtype=np.float64)

    curve_p = np.linspace(float(p_norm.min()), float(p_norm.max()), 420)
    curve_y = design_matrix(curve_p) @ beta

    rng = np.random.default_rng(20260801)
    sample_idx = rng.choice(len(p_norm), size=min(7800, len(p_norm)), replace=False)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.unicode_minus": False,
            "path.simplify": True,
        }
    )

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True, which="major", color="#D7D7D7", linestyle="--", linewidth=1.05, alpha=0.76, zorder=0)

    ax.scatter(
        p_norm[sample_idx],
        y[sample_idx],
        s=13,
        color="#AAB4C1",
        alpha=0.115,
        linewidths=0,
        zorder=1,
    )
    ax.fill_between(
        bin_df["p_mid"].to_numpy(),
        bin_df["q10"].to_numpy(),
        bin_df["q90"].to_numpy(),
        color="#DCEAF8",
        alpha=0.88,
        linewidth=0,
        zorder=2,
    )
    ax.plot(
        bin_df["p_mid"],
        bin_df["mean"],
        color="#D9ECFF",
        lw=10.5,
        alpha=0.90,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=3,
    )
    ax.plot(
        bin_df["p_mid"],
        bin_df["mean"],
        color="#1F77B4",
        lw=5.4,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=4,
    )
    ax.plot(
        curve_p,
        curve_y,
        color="#FFD2B8",
        lw=11.0,
        alpha=0.82,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=5,
    )
    ax.plot(
        curve_p,
        curve_y,
        color="#D04A06",
        lw=5.8,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=6,
    )

    ax.set_xlim(float(p_norm.min()) - 0.015, float(p_norm.max()) + 0.015)
    ymin = min(float(y.min()), float(curve_y.min()), float(bin_df["q10"].min())) - 0.012
    ymax = max(float(y.max()), float(curve_y.max()), float(bin_df["q90"].max())) + 0.012
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.set_yticks([0.80, 0.90, 1.00, 1.10])
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
    png_path = FIG_DIR / "fig_physical_base_source_fit_mechanism_schematic.png"
    svg_path = FIG_DIR / "fig_physical_base_source_fit_mechanism_schematic.svg"
    fig.savefig(png_path, dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(svg_path, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print(f"saved: {png_path}")
    print(f"saved: {svg_path}")


if __name__ == "__main__":
    main()
