from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(r"C:\Users\ROG\Desktop\paper")
PRED_CSV = ROOT / "codex" / "paper-update-20260420" / "target_test_predictions_all_models_20260420.csv"
SUMMARY_XLSX = ROOT / "codex" / "visualization-update-20260420" / "teal_ci_visualization_summary_20260420.xlsx"
OUT_DIR = ROOT / "codex" / "visual-test-clean-20260420"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "final-retained-v1"
OUT_PNG = OUT_DIR / "final-retained-v1_batch10_2x4_clean_abs_error_bars.png"
OUT_JSON = OUT_DIR / "final-retained-v1_batch10_2x4_clean_abs_error_bars_summary.json"

DISPLAY_SMOOTH = True
SAVGOL_WINDOW = 71
SAVGOL_POLY = 3
ERROR_STRIDE_MIN = 14
ERROR_MAX_BARS = 70

COLOR_TRUE = "#17413F"
COLOR_PRED = "#00A6A6"
COLOR_BAND = "#BFEDE7"
COLOR_ERR = "#0B7F7A"
COLOR_GRID = "#CFE5E1"
COLOR_TEXT = "#102D2B"

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 1.15
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.unicode_minus"] = False


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(np.abs(y_pred - y_true)))


def smooth_line(y, window=SAVGOL_WINDOW, poly=SAVGOL_POLY):
    y = np.asarray(y).reshape(-1)
    if not DISPLAY_SMOOTH or len(y) < 9:
        return y.copy()
    try:
        from scipy.signal import savgol_filter

        win = min(window, len(y) if len(y) % 2 == 1 else len(y) - 1)
        if win <= poly:
            win = poly + 3 if (poly + 3) % 2 == 1 else poly + 4
        win = min(win, len(y) if len(y) % 2 == 1 else len(y) - 1)
        if win < 7:
            return y.copy()
        return savgol_filter(y, window_length=win, polyorder=min(poly, win - 2), mode="interp")
    except Exception:
        # Fallback only affects display, not metrics.
        s = pd.Series(y)
        win = min(window, len(y))
        if win < 5:
            return y.copy()
        return s.rolling(window=win, center=True, min_periods=1).mean().to_numpy()


def load_ci_half_width():
    if not SUMMARY_XLSX.exists():
        return 0.014
    df = pd.read_excel(SUMMARY_XLSX, sheet_name="overall")
    row = df[df["model"] == MODEL]
    if row.empty:
        return 0.014
    return float(row.iloc[0]["ci_half_width_95"])


def main():
    if not PRED_CSV.exists():
        raise FileNotFoundError(f"Missing prediction CSV: {PRED_CSV}")

    df = pd.read_csv(PRED_CSV)
    need_cols = {"battery_index", "cycle", "y_true", MODEL}
    missing = need_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in prediction CSV: {sorted(missing)}")

    ci_half = load_ci_half_width()
    batteries = sorted(df["battery_index"].unique().tolist())

    fig, axes = plt.subplots(2, 4, figsize=(20, 9.2), constrained_layout=False)
    axes = axes.flatten()

    legend_handles = [
        Line2D([0], [0], color=COLOR_TRUE, lw=2.0, label="Measured SOH"),
        Line2D([0], [0], color=COLOR_PRED, lw=2.0, label="Predicted SOH (display-smoothed)"),
        Patch(facecolor=COLOR_BAND, edgecolor=COLOR_BAND, alpha=0.38, label="95% empirical band"),
        Patch(facecolor=COLOR_ERR, edgecolor=COLOR_ERR, alpha=0.16, label="Absolute error bars"),
    ]

    cell_rows = []
    all_abs_err = np.abs(df[MODEL].to_numpy() - df["y_true"].to_numpy())
    # Deliberately enlarge error axis so bars stay visually low and do not cover SOH curves.
    err_ylim = max(0.08, float(np.percentile(all_abs_err, 99)) * 3.0)

    for i, battery in enumerate(batteries):
        ax = axes[i]
        ax_err = ax.twinx()
        g = df[df["battery_index"] == battery].sort_values("cycle")

        x = g["cycle"].to_numpy(dtype=float)
        yt = g["y_true"].to_numpy(dtype=float)
        yp_raw = g[MODEL].to_numpy(dtype=float)
        yp_show = smooth_line(yp_raw)
        lo_show = smooth_line(yp_raw - ci_half)
        hi_show = smooth_line(yp_raw + ci_half)
        abs_err = np.abs(yp_raw - yt)

        stride = max(ERROR_STRIDE_MIN, len(x) // ERROR_MAX_BARS)
        xs = x[::stride]
        es = abs_err[::stride]
        width = max(5.0, float(np.median(np.diff(x))) * stride * 0.75) if len(x) > 1 else 5.0

        ax_err.bar(xs, es, width=width, color=COLOR_ERR, alpha=0.16, linewidth=0, zorder=0)
        ax_err.set_ylim(0, err_ylim)
        ax_err.set_ylabel("Abs. Error", fontsize=9, color=COLOR_ERR)
        ax_err.tick_params(axis="y", labelsize=8, colors=COLOR_ERR)
        ax_err.grid(False)
        ax_err.set_zorder(0)
        ax.set_zorder(1)
        ax.patch.set_alpha(0)

        ax.fill_between(x, lo_show, hi_show, color=COLOR_BAND, alpha=0.38, linewidth=0, zorder=1)
        ax.plot(x, yt, color=COLOR_TRUE, linewidth=2.0, zorder=3)
        ax.plot(x, yp_show, color=COLOR_PRED, linewidth=2.0, zorder=4)

        cell_rmse = rmse(yt, yp_raw)
        cell_mae = mae(yt, yp_raw)
        cell_rows.append(
            {
                "battery_index": int(battery),
                "rmse_raw": cell_rmse,
                "mae_raw": cell_mae,
                "max_abs_error": float(np.max(abs_err)),
                "n_points": int(len(g)),
            }
        )

        y_min = min(float(np.min(yt)), float(np.min(lo_show)), float(np.min(yp_show)))
        y_max = max(float(np.max(yt)), float(np.max(hi_show)), float(np.max(yp_show)))
        pad = max(0.025, 0.06 * (y_max - y_min))
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_title(f"Cell {int(battery)} | RMSE = {cell_rmse:.4f}", fontsize=13, fontweight="bold", color=COLOR_TEXT)
        ax.set_xlabel("Cycle", fontsize=11)
        ax.set_ylabel("SOH", fontsize=11)
        ax.grid(True, linestyle="--", color=COLOR_GRID, alpha=0.55)
        ax.tick_params(axis="both", labelsize=10)

    for j in range(len(batteries), 8):
        axes[j].axis("off")

    pooled = rmse(df["y_true"].to_numpy(), df[MODEL].to_numpy())
    macro = float(np.mean([r["rmse_raw"] for r in cell_rows]))
    mae_all = mae(df["y_true"].to_numpy(), df[MODEL].to_numpy())

    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=4, frameon=False, fontsize=12)
    fig.suptitle(
        "Batch 10 SOH Prediction - Clean 2 x 4 Visualization\nFinal retained model, smoothed display line, empirical band, absolute error bars",
        fontsize=20,
        y=1.035,
        fontweight="bold",
        color=COLOR_TEXT,
    )
    fig.text(
        0.5,
        0.018,
        f"Pooled RMSE = {pooled:.5f}   |   Macro RMSE = {macro:.5f}   |   MAE = {mae_all:.5f}   |   95% half-width = {ci_half:.5f}",
        ha="center",
        fontsize=13,
        color=COLOR_TEXT,
    )
    plt.tight_layout(rect=[0, 0.055, 1, 0.925])
    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "model": MODEL,
        "figure_path": str(OUT_PNG),
        "pooled_rmse_raw": pooled,
        "macro_rmse_raw": macro,
        "mae_raw": mae_all,
        "ci_half_width_95": ci_half,
        "display_smooth": DISPLAY_SMOOTH,
        "savgol_window": SAVGOL_WINDOW,
        "savgol_poly": SAVGOL_POLY,
        "error_type": "absolute_error_bars_nonnegative",
        "error_ylim": err_ylim,
        "note": "Metrics use raw predictions. Only the plotted prediction and band are display-smoothed.",
        "cell_metrics": cell_rows,
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
