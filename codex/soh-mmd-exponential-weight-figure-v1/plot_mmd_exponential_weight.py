"""
Plot the degradation-aware exponential MMD weight used in the SOH model.

Style source:
paper-plot-skills-main/plot-from-data, style: line_training_curve.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

LAMBDA_SCALE = 5.5
TARGET_MMD_WEIGHT = 5e-5


def main() -> None:
    soh = np.linspace(1.00, 0.80, 201)
    dynamic_multiplier = np.exp(LAMBDA_SCALE * (1.0 - soh))
    static_multiplier = np.ones_like(soh)
    effective_weight = TARGET_MMD_WEIGHT * dynamic_multiplier
    static_effective_weight = TARGET_MMD_WEIGHT * static_multiplier

    data = pd.DataFrame(
        {
            "soh": soh,
            "dynamic_multiplier": dynamic_multiplier,
            "static_multiplier": static_multiplier,
            "dynamic_effective_mmd_weight": effective_weight,
            "static_effective_mmd_weight": static_effective_weight,
        }
    )
    csv_path = ROOT / "mmd_exponential_weight_curve.csv"
    data.to_csv(csv_path, index=False)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "text.usetex": False,
            "axes.unicode_minus": False,
        }
    )

    c_dynamic = "#5B0DAD"
    c_static = "#5BBCCA"
    c_marker = "#333333"

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(
        soh,
        dynamic_multiplier,
        color=c_dynamic,
        lw=2.0,
        label="Exponential lifecycle-adaptive MMD",
        zorder=3,
    )
    ax.plot(
        soh,
        static_multiplier,
        color=c_static,
        lw=1.8,
        ls="--",
        label="Static MMD baseline",
        zorder=2,
    )

    key_soh = np.array([1.00, 0.95, 0.90, 0.85, 0.80])
    key_weight = np.exp(LAMBDA_SCALE * (1.0 - key_soh))
    ax.scatter(key_soh, key_weight, s=30, color=c_marker, zorder=4)
    for x, y in zip(key_soh, key_weight):
        ax.annotate(
            f"{y:.2f}x",
            xy=(x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=c_marker,
        )

    ax.set_xlim(1.005, 0.795)
    ax.set_ylim(0.85, 3.25)
    ax.set_xticks([1.00, 0.95, 0.90, 0.85, 0.80])
    ax.set_yticks([1.0, 1.5, 2.0, 2.5, 3.0])
    ax.set_xlabel("Estimated SOH from degradation prior", fontsize=12)
    ax.set_ylabel("MMD alignment multiplier", fontsize=12)
    ax.tick_params(labelsize=10, direction="out", length=4, width=0.8)

    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(1.0)

    ax.grid(False)
    ax.legend(
        loc="upper left",
        fontsize=9.5,
        frameon=True,
        facecolor="white",
        edgecolor="#AAAAAA",
        framealpha=1.0,
        borderpad=0.55,
        labelspacing=0.35,
        handlelength=2.5,
        handletextpad=0.6,
    )

    ax.text(
        0.885,
        1.15,
        r"multiplier = exp(5.5 * (1 - SOH))",
        fontsize=10,
        color="#444444",
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#BBBBBB", "linewidth": 0.8},
    )

    fig.tight_layout(pad=0.8)
    png_path = FIG_DIR / "soh_mmd_exponential_weight_curve.png"
    pdf_path = FIG_DIR / "soh_mmd_exponential_weight_curve.pdf"
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)

    report_path = ROOT / "mmd_exponential_weight_curve_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# MMD Exponential Weight Curve",
                "",
                "This figure visualizes the lifecycle-adaptive MMD multiplier used by the SOH model.",
                "",
                f"- Formula: `exp({LAMBDA_SCALE} * (1 - SOH))`.",
                f"- Static MMD baseline multiplier: `1.0`.",
                f"- Outer target MMD coefficient in training: `{TARGET_MMD_WEIGHT}`.",
                "- SOH range shown: 1.00 to 0.80, matching the nominal SOH range of the fixed split.",
                "",
                "Key points:",
                "",
                "| SOH | Dynamic multiplier | Effective MMD coefficient |",
                "|---:|---:|---:|",
            ]
            + [
                f"| {x:.2f} | {y:.4f} | {TARGET_MMD_WEIGHT * y:.8f} |"
                for x, y in zip(key_soh, key_weight)
            ]
            + [
                "",
                "Outputs:",
                "",
                f"- `{png_path}`",
                f"- `{pdf_path}`",
                f"- `{csv_path}`",
            ]
        ),
        encoding="utf-8",
    )

    print(f"saved: {png_path}")
    print(f"saved: {pdf_path}")
    print(f"saved: {csv_path}")
    print(f"saved: {report_path}")


if __name__ == "__main__":
    main()
