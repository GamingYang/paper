from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats


ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = ROOT / "codex" / "batch10-soh-prediction-figure-v1"
FIG_DIR = RUN_DIR / "figures"
TABLE_DIR = RUN_DIR / "tables"

PRED_CSV = ROOT / "codex" / "6月29日最新证据链" / "tables" / "batch10_final_predictions_by_cycle.csv"
METRIC_CSV = ROOT / "codex" / "6月29日最新证据链" / "tables" / "batch10_per_cell_metrics_final.csv"


COLORS = {
    "true": "#102A43",
    "pred": "#08A6A6",
    "pred_edge": "#065A60",
    "positive": "#D95F0E",
    "negative": "#2C7FB8",
    "shade": "#F2F5F7",
    "grid": "#E5E7EB",
    "text": "#1F2933",
    "best": "#C0392B",
    "density": "#B8B8B8",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def polish_axes(ax: plt.Axes, *, grid_axis: str | None = "both") -> None:
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.85)
        ax.spines[side].set_color("#333333")
    ax.tick_params(length=3, width=0.75, color="#333333")
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=COLORS["grid"], linestyle="--", linewidth=0.55, zorder=0)
        ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.11,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(PRED_CSV)
    if "y_pred" not in df:
        df["y_pred"] = df["final-retained-v1"]
    df["error"] = df["y_pred"] - df["y_true"]
    df["abs_error"] = df["error"].abs()
    metrics = (
        df.groupby(["battery_index", "cell_id"], as_index=False)
        .agg(
            n_points=("cycle", "count"),
            cycle_min=("cycle", "min"),
            cycle_max=("cycle", "max"),
            rmse=("error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            mae=("abs_error", "mean"),
            mape_percent=("abs_error", lambda x: float(np.mean(x / df.loc[x.index, "y_true"]) * 100.0)),
            signed_bias_pred_minus_true=("error", "mean"),
            max_abs_error=("abs_error", "max"),
        )
        .sort_values("rmse")
    )
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(TABLE_DIR / "batch10_per_cell_metrics_recomputed.csv", index=False, encoding="utf-8-sig")

    if METRIC_CSV.exists():
        source_metrics = pd.read_csv(METRIC_CSV)
        source_metrics.to_csv(TABLE_DIR / "batch10_per_cell_metrics_source.csv", index=False, encoding="utf-8-sig")
    return df, metrics


def residual_density_inset(ax: plt.Axes, residual: np.ndarray, *, loc: str = "upper right") -> plt.Axes:
    inset = inset_axes(ax, width="34%", height="29%", loc=loc, borderpad=0.75)
    res_scaled = residual * 1000.0
    if len(res_scaled) > 5 and np.std(res_scaled) > 1e-9:
        xs = np.linspace(np.percentile(res_scaled, 0.5), np.percentile(res_scaled, 99.5), 240)
        kde = stats.gaussian_kde(res_scaled)
        ys = kde(xs)
        inset.fill_between(xs, ys, color=COLORS["density"], alpha=0.78, linewidth=0)
        inset.plot(xs, ys, color="#777777", linewidth=0.75)
    else:
        inset.hist(res_scaled, bins=12, color=COLORS["density"], alpha=0.78)
    inset.axvline(0, color="#555555", linewidth=0.7, linestyle="--")
    inset.set_yticks([])
    inset.tick_params(axis="x", labelsize=6.8, length=2)
    inset.set_xlabel(r"Residual ($\times 10^{-3}$)", fontsize=6.8, labelpad=0.5)
    inset.set_title("density", fontsize=7.2, pad=1)
    for spine in inset.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color("#777777")
    return inset


def plot_prediction_panel(
    ax: plt.Axes,
    g: pd.DataFrame,
    *,
    title: str,
    metric_row: pd.Series,
    cmap_norm: mpl.colors.Normalize,
    show_ylabel: bool,
    panel: str | None = None,
    highlight_best: bool = False,
) -> mpl.collections.PathCollection:
    cycle = g["cycle"].to_numpy()
    true = g["y_true"].to_numpy()
    pred = g["y_pred"].to_numpy()
    err = pred - true
    abs_err = np.abs(err)

    ax.axhspan(1.00, 1.13, color="#EAF4FB", alpha=0.35, zorder=0)
    ax.axhspan(0.90, 1.00, color="#F7FBF2", alpha=0.35, zorder=0)
    ax.axhspan(0.78, 0.90, color="#FFF4E6", alpha=0.38, zorder=0)
    ax.axhline(1.0, color="#8A8A8A", linewidth=0.75, linestyle=(0, (3, 2)), zorder=1)
    ax.axhline(0.9, color="#B8B8B8", linewidth=0.65, linestyle=(0, (2, 2)), zorder=1)
    ax.axhline(0.8, color="#C0392B", linewidth=0.75, linestyle=(0, (3, 2)), zorder=1)

    ax.fill_between(
        cycle,
        true,
        pred,
        where=err >= 0,
        color=COLORS["positive"],
        alpha=0.16,
        interpolate=True,
        linewidth=0,
        zorder=2,
    )
    ax.fill_between(
        cycle,
        true,
        pred,
        where=err < 0,
        color=COLORS["negative"],
        alpha=0.15,
        interpolate=True,
        linewidth=0,
        zorder=2,
    )
    ax.plot(cycle, true, color=COLORS["true"], linewidth=1.95, zorder=4, label="Actual SOH")
    stride = max(1, len(g) // 520)
    sc = ax.scatter(
        cycle[::stride],
        pred[::stride],
        c=abs_err[::stride] * 1000.0,
        cmap="YlOrRd",
        norm=cmap_norm,
        s=9.5,
        edgecolors="none",
        alpha=0.92,
        zorder=5,
        label="Predicted SOH",
    )
    ax.plot(cycle, pred, color=COLORS["pred_edge"], linewidth=0.55, alpha=0.65, zorder=4)

    metric_text = (
        f"RMSE={metric_row.rmse * 1000:.2f}\n"
        f"MAE={metric_row.mae * 1000:.2f}\n"
        f"Bias={metric_row.signed_bias_pred_minus_true * 1000:+.2f}"
    )
    bbox_edge = COLORS["best"] if highlight_best else "#C9CDD2"
    ax.text(
        0.035,
        0.055,
        metric_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.8,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=bbox_edge, linewidth=0.85),
    )
    residual_density_inset(ax, err, loc="upper right")
    ax.set_title(title, pad=5, fontweight="bold" if highlight_best else "normal", color=COLORS["best"] if highlight_best else "black")
    ax.set_xlabel("Cycle number")
    if show_ylabel:
        ax.set_ylabel("SOH")
    ax.set_ylim(0.785, 1.13)
    ax.set_xlim(max(0, cycle.min() - 25), cycle.max() + 40)
    polish_axes(ax)
    if panel:
        panel_label(ax, panel)
    return sc


def make_eight_cell_figure(df: pd.DataFrame, metrics: pd.DataFrame) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    metric_by_cell = metrics.set_index("cell_id")
    ordered_cells = sorted(df["cell_id"].unique(), key=lambda s: int(s.split("-")[1]))
    best_cell = metrics.iloc[0]["cell_id"]
    max_abs = df["abs_error"].quantile(0.995) * 1000.0
    norm = mpl.colors.Normalize(vmin=0.0, vmax=max(8.0, max_abs))

    fig, axes = plt.subplots(2, 4, figsize=(15.2, 7.9), sharey=True)
    fig.subplots_adjust(left=0.055, right=0.92, bottom=0.165, top=0.955, wspace=0.22, hspace=0.34)

    letters = list("abcdefgh")
    last_sc = None
    for i, (ax, cell_id) in enumerate(zip(axes.ravel(), ordered_cells)):
        g = df[df["cell_id"] == cell_id].sort_values("cycle")
        metric = metric_by_cell.loc[cell_id]
        title = f"{cell_id} / Battery {int(metric.battery_index)}"
        last_sc = plot_prediction_panel(
            ax,
            g,
            title=title,
            metric_row=metric,
            cmap_norm=norm,
            show_ylabel=i % 4 == 0,
            panel=letters[i],
            highlight_best=cell_id == best_cell,
        )

    handles = [
        Line2D([0], [0], color=COLORS["true"], lw=2.1, label="Actual SOH"),
        Line2D([0], [0], marker="o", linestyle="", color="#D95F0E", markersize=5.0, label="Predicted SOH"),
        mpl.patches.Patch(facecolor=COLORS["positive"], alpha=0.16, label="Prediction > actual"),
        mpl.patches.Patch(facecolor=COLORS["negative"], alpha=0.15, label="Prediction < actual"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=True, edgecolor="#D5D5D5", bbox_to_anchor=(0.49, 0.025))
    cax = fig.add_axes([0.935, 0.245, 0.014, 0.62])
    cbar = fig.colorbar(last_sc, cax=cax)
    cbar.set_label(r"Absolute error ($\times 10^{-3}$ SOH)")

    out = FIG_DIR / "Figure_Batch10_per_cell_SOH_prediction_curves_enhanced_2x4.png"
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(out.with_suffix(f".{suffix}"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def make_best_cell_figure(df: pd.DataFrame, metrics: pd.DataFrame) -> Path:
    best = metrics.iloc[0]
    cell_id = best["cell_id"]
    g = df[df["cell_id"] == cell_id].sort_values("cycle").copy()
    cycle = g["cycle"].to_numpy()
    true = g["y_true"].to_numpy()
    pred = g["y_pred"].to_numpy()
    err = pred - true
    abs_err = np.abs(err)
    max_idx = int(np.argmax(abs_err))

    norm = mpl.colors.Normalize(vmin=0.0, vmax=max(8.0, np.quantile(df["abs_error"], 0.995) * 1000.0))

    fig = plt.figure(figsize=(9.4, 6.7))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.6, 1.1], left=0.09, right=0.9, bottom=0.11, top=0.955, hspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    ax_res = fig.add_subplot(gs[1, 0], sharex=ax)

    ax.axhspan(1.00, 1.13, color="#EAF4FB", alpha=0.35, zorder=0)
    ax.axhspan(0.90, 1.00, color="#F7FBF2", alpha=0.35, zorder=0)
    ax.axhspan(0.78, 0.90, color="#FFF4E6", alpha=0.38, zorder=0)
    ax.axhline(1.0, color="#8A8A8A", linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    ax.axhline(0.9, color="#B8B8B8", linewidth=0.7, linestyle=(0, (2, 2)), zorder=1)
    ax.axhline(0.8, color="#C0392B", linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    ax.fill_between(cycle, true, pred, where=err >= 0, color=COLORS["positive"], alpha=0.18, interpolate=True, linewidth=0)
    ax.fill_between(cycle, true, pred, where=err < 0, color=COLORS["negative"], alpha=0.17, interpolate=True, linewidth=0)
    ax.plot(cycle, true, color=COLORS["true"], linewidth=2.4, label="Actual SOH", zorder=4)
    sc = ax.scatter(
        cycle,
        pred,
        c=abs_err * 1000.0,
        cmap="YlOrRd",
        norm=norm,
        s=8,
        edgecolors="none",
        alpha=0.88,
        label="Predicted SOH",
        zorder=5,
    )
    ax.plot(cycle, pred, color=COLORS["pred_edge"], linewidth=0.7, alpha=0.65, zorder=4)
    ax.scatter(cycle[max_idx], pred[max_idx], s=72, facecolor="none", edgecolor=COLORS["best"], linewidth=1.35, zorder=7)
    ax.annotate(
        f"Max abs. error = {abs_err[max_idx] * 1000:.2f}",
        xy=(cycle[max_idx], pred[max_idx]),
        xytext=(0.62, 0.32),
        textcoords=ax.transAxes,
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color=COLORS["best"], lw=0.9),
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=COLORS["best"], linewidth=0.8),
    )
    metric_text = (
        f"RMSE = {best.rmse * 1000:.2f} $\\times 10^{{-3}}$ SOH\n"
        f"MAE = {best.mae * 1000:.2f} $\\times 10^{{-3}}$ SOH\n"
        f"MAPE = {best.mape_percent:.3f}%\n"
        f"Bias = {best.signed_bias_pred_minus_true * 1000:+.2f} $\\times 10^{{-3}}$"
    )
    ax.text(
        0.035,
        0.05,
        metric_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.4,
        bbox=dict(boxstyle="round,pad=0.38", facecolor="white", edgecolor="#C9CDD2", linewidth=0.85),
    )
    residual_density_inset(ax, err, loc="upper right")
    ax.set_ylabel("SOH")
    ax.set_ylim(0.785, 1.13)
    ax.set_xlim(max(0, cycle.min() - 25), cycle.max() + 35)
    polish_axes(ax)
    ax.legend(frameon=False, loc="upper left")
    ax.tick_params(labelbottom=False)

    ax_res.axhline(0, color="#444444", linewidth=0.8)
    ax_res.fill_between(cycle, 0, err * 1000.0, where=err >= 0, color=COLORS["positive"], alpha=0.32, interpolate=True)
    ax_res.fill_between(cycle, 0, err * 1000.0, where=err < 0, color=COLORS["negative"], alpha=0.32, interpolate=True)
    ax_res.plot(cycle, err * 1000.0, color="#4B5563", linewidth=0.7)
    ax_res.set_xlabel("Cycle number")
    ax_res.set_ylabel(r"Residual ($\times 10^{-3}$)")
    lim = max(8.0, np.quantile(np.abs(err * 1000.0), 0.995) * 1.25)
    ax_res.set_ylim(-lim, lim)
    polish_axes(ax_res, grid_axis="both")

    cax = fig.add_axes([0.925, 0.25, 0.018, 0.48])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label(r"Absolute error ($\times 10^{-3}$ SOH)")

    out = FIG_DIR / "Figure_Batch10_best_RMSE_cell_SOH_prediction_detail.png"
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(out.with_suffix(f".{suffix}"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def write_report(metrics: pd.DataFrame, eight_path: Path, best_path: Path) -> None:
    best = metrics.iloc[0]
    report = f"""# Batch10 SOH Prediction Figures v1

Generated: 2026-07-29

## Purpose

Regenerate the Batch10 per-cell SOH prediction visualization using a richer
paper-style layout inspired by the supplied reference image. The figures use
the accepted Batch10 final-model prediction table and do not retrain or select
models.

## Input Evidence

- `{PRED_CSV.relative_to(ROOT)}`
- `{METRIC_CSV.relative_to(ROOT)}`

## Outputs

- Eight-cell overview: `{eight_path.relative_to(ROOT)}`
- Best-cell detail: `{best_path.relative_to(ROOT)}`

Both figures are exported as PNG, PDF, and SVG.

## Best Cell

The smallest Batch10 per-cell RMSE is `{best.cell_id}` / Battery
`{int(best.battery_index)}`:

- RMSE: {best.rmse:.6f} SOH
- MAE: {best.mae:.6f} SOH
- MAPE: {best.mape_percent:.3f}%
- Signed bias: {best.signed_bias_pred_minus_true:+.6f} SOH

## Visual Encoding

- Solid dark curve: actual SOH.
- Coloured prediction points: predicted SOH, coloured by absolute SOH error.
- Orange/blue fill: signed residual direction.
- Inset: residual-density distribution in `x10^-3` SOH units.
"""
    (RUN_DIR / "batch10_soh_prediction_figures_v1_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df, metrics = load_data()
    eight = make_eight_cell_figure(df, metrics)
    best = make_best_cell_figure(df, metrics)
    write_report(metrics, eight, best)
    print(eight)
    print(best)


if __name__ == "__main__":
    main()
