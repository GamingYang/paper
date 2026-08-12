from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from PIL import Image


ROOT = Path(r"C:\Users\ROG\Desktop\paper")
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


def load_source_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_dir = find_processed_clean(ROOT)
    p = np.load(data_dir / "clean_source_p.npy").astype(np.float64).reshape(-1)
    y = np.load(data_dir / "clean_source_y_nominal.npy").astype(np.float64).reshape(-1)
    bidx = np.load(data_dir / "clean_source_battery_index.npy").astype(np.int64).reshape(-1)
    batch = np.load(data_dir / "clean_source_batch.npy").astype(np.int64).reshape(-1)
    return p / GLOBAL_MAX_CYCLE, y, bidx, batch


def paper_style() -> dict:
    font_path = font_manager.findfont("Times New Roman", fallback_to_default=False)
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.2,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 8.0,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return {"times_new_roman_path": font_path}


def design_matrix(p_norm: np.ndarray) -> np.ndarray:
    return np.column_stack([p_norm**3, p_norm**2, p_norm, np.ones_like(p_norm)])


def objective_stats(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    gram = (x.T @ x) / len(y)
    xy = (x.T @ y) / len(y)
    yy = float((y @ y) / len(y))
    return gram, xy, yy


def mse_from_beta(beta: np.ndarray, gram: np.ndarray, xy: np.ndarray, yy: float) -> np.ndarray:
    beta = np.asarray(beta)
    return np.einsum("...i,ij,...j->...", beta, gram, beta) - 2.0 * np.einsum("...i,i->...", beta, xy) + yy


def coefficient_ranges(beta_star: np.ndarray, gram: np.ndarray, base_mse: float) -> list[np.ndarray]:
    # Radius is set by expected RMSE growth, so all four coefficients have visually
    # comparable objective variation despite different numeric scales.
    target_delta = base_mse * 1.8
    radii = np.sqrt(target_delta / np.maximum(np.diag(gram), 1e-12))
    radii = np.maximum(radii, np.maximum(0.08 * np.abs(beta_star), 0.01))
    return [np.linspace(v - r, v + r, 81) for v, r in zip(beta_star, radii)]


def format_coeff(value: float) -> str:
    return f"{value:+.4f}"


def plot_corner_grid(
    beta_star: np.ndarray,
    ranges: list[np.ndarray],
    gram: np.ndarray,
    xy: np.ndarray,
    yy: float,
    base_rmse: float,
    out_png: Path,
    out_pdf: Path,
) -> None:
    names = ["a: p^3", "b: p^2", "c: p", "d"]
    short = ["a", "b", "c", "d"]
    cmap = mpl.colormaps["YlGnBu_r"]

    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    pair_values = []
    cached = {}
    for x_idx, y_idx in pairs:
        gx, gy = np.meshgrid(ranges[x_idx], ranges[y_idx])
        betas = np.zeros(gx.shape + (4,), dtype=np.float64)
        betas[...] = beta_star
        betas[..., x_idx] = gx
        betas[..., y_idx] = gy
        rmse_inc = (np.sqrt(np.maximum(mse_from_beta(betas, gram, xy, yy), 0.0)) - base_rmse) * 100.0
        cached[(x_idx, y_idx)] = (gx, gy, rmse_inc)
        pair_values.append(rmse_inc)
    vmax = float(np.percentile(np.concatenate([v.ravel() for v in pair_values]), 97))
    vmax = max(vmax, 0.05)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)

    fig, axes = plt.subplots(2, 3, figsize=(10.6, 5.35), constrained_layout=False)
    fig.subplots_adjust(left=0.088, right=0.905, top=0.955, bottom=0.105, wspace=0.30, hspace=0.34)

    for ax, (x_idx, y_idx) in zip(axes.ravel(), pairs):
        gx, gy, rmse_inc = cached[(x_idx, y_idx)]
        ax.contourf(gx, gy, rmse_inc, levels=28, cmap=cmap, norm=norm)
        ax.contour(gx, gy, rmse_inc, levels=7, colors="white", linewidths=0.45, alpha=0.72)
        ax.scatter(
            [beta_star[x_idx]],
            [beta_star[y_idx]],
            s=25,
            color="#C2410C",
            edgecolor="white",
            lw=0.55,
            zorder=5,
        )
        ax.axvline(beta_star[x_idx], color="white", lw=0.62, ls="--", alpha=0.85)
        ax.axhline(beta_star[y_idx], color="white", lw=0.62, ls="--", alpha=0.85)
        ax.set_xlabel(names[x_idx])
        ax.set_ylabel(names[y_idx], rotation=0, ha="right", va="center", labelpad=7)
        ax.set_title(f"{short[x_idx]}-{short[y_idx]} slice", pad=4.5, fontweight="bold")
        ax.tick_params(axis="x", rotation=25, pad=1.5, length=2.2)
        ax.tick_params(axis="y", pad=1.5, length=2.2)
        for spine in ax.spines.values():
            spine.set_linewidth(0.75)
            spine.set_color("#374151")

    cax = fig.add_axes([0.925, 0.18, 0.018, 0.58])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("RMSE increase (SOH %)", labelpad=6)
    cbar.ax.tick_params(labelsize=7.2, length=2.2)

    coeff_line = "  ".join(
        [
            f"a={format_coeff(beta_star[0])}",
            f"b={format_coeff(beta_star[1])}",
            f"c={format_coeff(beta_star[2])}",
            f"d={format_coeff(beta_star[3])}",
        ]
    )

    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_base_fit(
    p_norm: np.ndarray,
    y: np.ndarray,
    beta_star: np.ndarray,
    out_png: Path,
    out_pdf: Path,
) -> None:
    order = np.argsort(p_norm)
    p_sorted = p_norm[order]
    y_sorted = y[order]
    curve_p = np.linspace(p_norm.min(), p_norm.max(), 400)
    curve_x = design_matrix(curve_p)
    curve_y = curve_x @ beta_star

    bins = np.linspace(p_norm.min(), p_norm.max(), 46)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_norm >= lo) & (p_norm < hi)
        if mask.sum() < 10:
            continue
        rows.append(
            {
                "p_mid": float((lo + hi) / 2),
                "mean": float(np.mean(y[mask])),
                "q10": float(np.quantile(y[mask], 0.10)),
                "q90": float(np.quantile(y[mask], 0.90)),
                "n": int(mask.sum()),
            }
        )
    bin_df = pd.DataFrame(rows)
    bin_df.to_csv(TABLE_DIR / "source_soh_binned_for_physical_base.csv", index=False, encoding="utf-8-sig")

    rng = np.random.default_rng(20260723)
    sample_idx = rng.choice(len(p_sorted), size=min(6500, len(p_sorted)), replace=False)

    fig, ax = plt.subplots(figsize=(7.1, 4.25))
    ax.scatter(
        p_norm[sample_idx],
        y[sample_idx],
        s=4.0,
        color="#9CA3AF",
        alpha=0.16,
        linewidths=0,
        label="Source samples",
    )
    ax.fill_between(
        bin_df["p_mid"],
        bin_df["q10"],
        bin_df["q90"],
        color="#D8E4F2",
        alpha=0.82,
        linewidth=0,
        label="10-90% source range",
    )
    ax.plot(bin_df["p_mid"], bin_df["mean"], color="#1F77B4", lw=1.55, label="Binned source mean")
    ax.plot(curve_p, curve_y, color="#C2410C", lw=2.0, label="Low-order prior")
    ax.set_xlabel("Normalized cycle position")
    ax.set_ylabel("SOH")
    ax.set_title("Source-domain fit of the low-order nonlinear degradation prior", loc="left", fontweight="bold")
    ax.grid(True, color="#E5E7EB", linestyle=":", linewidth=0.7)
    ax.legend(frameon=True, facecolor="white", edgecolor="#D1D5DB", framealpha=0.96, loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(float(p_norm.min()) - 0.015, float(p_norm.max()) + 0.015)
    ymin = min(float(y.min()), float(curve_y.min())) - 0.015
    ymax = max(float(y.max()), float(curve_y.max())) + 0.015
    ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def inspect_image(path: Path) -> dict:
    with Image.open(path) as im:
        return {"path": str(path), "width": im.width, "height": im.height, "mode": im.mode}


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    font_info = paper_style()

    p_norm, y, bidx, batch = load_source_data()
    x = design_matrix(p_norm)
    beta_star = np.polyfit(p_norm, y, 3).astype(np.float64)
    gram, xy, yy = objective_stats(x, y)
    base_mse = float(mse_from_beta(beta_star, gram, xy, yy))
    base_rmse = float(np.sqrt(max(base_mse, 0.0)))
    base_mae = float(np.mean(np.abs(x @ beta_star - y)))
    ranges = coefficient_ranges(beta_star, gram, base_mse)

    grid_png = FIG_DIR / "fig_physical_base_abcd_optimization_grid.png"
    grid_pdf = FIG_DIR / "fig_physical_base_abcd_optimization_grid.pdf"
    fit_png = FIG_DIR / "fig_physical_base_source_fit.png"
    fit_pdf = FIG_DIR / "fig_physical_base_source_fit.pdf"

    plot_corner_grid(beta_star, ranges, gram, xy, yy, base_rmse, grid_png, grid_pdf)
    plot_base_fit(p_norm, y, beta_star, fit_png, fit_pdf)

    summary = {
        "data_protocol": "Batch 1-8 source training samples only",
        "n_source_samples": int(len(y)),
        "n_source_cells": int(len(np.unique(bidx))),
        "source_batches": [int(v) for v in np.unique(batch)],
        "global_max_cycle": GLOBAL_MAX_CYCLE,
        "coefficients_order": "a,b,c,d for a*p^3 + b*p^2 + c*p + d",
        "coefficients": {
            "a": float(beta_star[0]),
            "b": float(beta_star[1]),
            "c": float(beta_star[2]),
            "d": float(beta_star[3]),
        },
        "source_rmse": base_rmse,
        "source_mae": base_mae,
        "times_new_roman_path": font_info["times_new_roman_path"],
        "figures": [inspect_image(grid_png), inspect_image(fit_png)],
    }
    (OUT_DIR / "physical_base_coefficient_optimization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "a": beta_star[0],
                "b": beta_star[1],
                "c": beta_star[2],
                "d": beta_star[3],
                "source_rmse": base_rmse,
                "source_mae": base_mae,
                "n_source_samples": len(y),
                "n_source_cells": len(np.unique(bidx)),
            }
        ]
    ).to_csv(TABLE_DIR / "physical_base_coefficients.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
