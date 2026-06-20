from __future__ import annotations

import json
import math
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle


RUN_NAME = "final-figure-table-pack-v2"
SCRIPT_DIR = Path(__file__).resolve().parent
CODEX_DIR = SCRIPT_DIR.parent

STRICT_DIR = CODEX_DIR / "strict-eol-protocol-v1"
ROBUST_DIR = CODEX_DIR / "strict-eol-robustness-v1"
BUDGET_DIR = CODEX_DIR / "target-label-budget-sensitivity-v1"
RAW_V3_DIR = CODEX_DIR / "strict-raw-provenance-v3"
NEURAL_V1_DIR = CODEX_DIR / "neural-eol-model-v1"
RESIDUAL_DIR = CODEX_DIR / "eol-ablation-residual-v1"
CONFORMAL_DIR = CODEX_DIR / "strict-eol-conformal-uncertainty-v1"


PATHS = {
    "strict_candidate": STRICT_DIR / "strict-eol-protocol-v1_candidate_validation_summary.csv",
    "strict_selected": STRICT_DIR / "strict-eol-protocol-v1_selected_test_summary.csv",
    "strict_predictions": STRICT_DIR / "strict-eol-protocol-v1_selected_test_predictions.csv",
    "robust_summary": ROBUST_DIR / "strict-eol-robustness-v1_summary.json",
    "robust_per_cell": ROBUST_DIR / "strict-eol-robustness-v1_per_cell.csv",
    "budget_aggregate": BUDGET_DIR / "budget_aggregate_results.csv",
    "budget_subset": BUDGET_DIR / "budget_subset_results.csv",
    "budget_summary": BUDGET_DIR / "target-label-budget-sensitivity-v1_summary.json",
    "raw_summary": RAW_V3_DIR / "strict-raw-provenance-v3_summary.json",
    "raw_selected_features": RAW_V3_DIR / "selected_raw_features_v3.csv",
    "raw_feature_audit": RAW_V3_DIR / "raw_feature_audit_v3.csv",
    "raw_model_results": RAW_V3_DIR / "raw_model_results_v3.csv",
    "neural_v1_selected": NEURAL_V1_DIR / "neural-eol-model-v1_selected_test_summary.csv",
    "residual_selected": RESIDUAL_DIR / "selected_test_summary.csv",
    "conformal_summary": CONFORMAL_DIR / "strict-eol-conformal-uncertainty-v1_summary.json",
}


FIGURE_BASES = {
    "protocol": SCRIPT_DIR / "figure_protocol_flowchart_v2",
    "batch9_heatmap": SCRIPT_DIR / "figure_batch9_candidate_horizon_model_selection_v2",
    "pred_vs_true": SCRIPT_DIR / "figure_batch10_predicted_vs_true_eol_v2",
    "per_cell_error": SCRIPT_DIR / "figure_batch10_per_cell_abs_error_v2",
    "budget": SCRIPT_DIR / "figure_target_label_budget_sensitivity_v2",
    "raw_provenance": SCRIPT_DIR / "figure_raw_provenance_v3_summary",
    "evidence_hierarchy": SCRIPT_DIR / "figure_evidence_hierarchy_v2",
}


TABLES = {
    "main_csv": SCRIPT_DIR / "table_main_results_v2.csv",
    "main_md": SCRIPT_DIR / "table_main_results_v2.md",
    "budget_csv": SCRIPT_DIR / "table_target_label_budget.csv",
    "budget_md": SCRIPT_DIR / "table_target_label_budget.md",
    "raw_csv": SCRIPT_DIR / "table_raw_provenance_v3.csv",
    "raw_md": SCRIPT_DIR / "table_raw_provenance_v3.md",
}


REPORT_PATH = SCRIPT_DIR / f"{RUN_NAME}_report.md"


MODEL_LABELS = {
    "extra_trees": "ExtraTrees",
    "random_forest": "RandomForest",
    "gbr": "GBR",
    "svr_rbf": "SVR-RBF",
    "elastic_net": "ElasticNet",
    "ridge": "Ridge",
    "kernel_ridge_rbf": "KernelRidge-RBF",
    "source_mean_eol": "Source mean",
    "source_median_eol": "Source median",
    "neural_eol_qnet": "Neural QNet",
}


FAMILY_LABELS = {
    "capacity": "Capacity",
    "dq": "dQ",
    "time_status": "Time/status",
    "voltage_current": "Voltage/current",
    "fused": "Fused",
    "other": "Other",
}


COLORS = {
    "blue": "#2f6f9f",
    "teal": "#4c956c",
    "orange": "#d08c33",
    "gray": "#6f6f6f",
    "light_blue": "#e8f2f8",
    "light_teal": "#edf6ef",
    "light_orange": "#fff2e3",
    "red": "#b22222",
    "purple": "#8064a2",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def require_inputs() -> None:
    missing = [str(path) for path in PATHS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_inputs() -> dict[str, object]:
    require_inputs()
    return {
        "strict_candidate": pd.read_csv(PATHS["strict_candidate"]),
        "strict_selected": pd.read_csv(PATHS["strict_selected"]),
        "strict_predictions": pd.read_csv(PATHS["strict_predictions"]),
        "robust_summary": read_json(PATHS["robust_summary"]),
        "robust_per_cell": pd.read_csv(PATHS["robust_per_cell"]),
        "budget_aggregate": pd.read_csv(PATHS["budget_aggregate"]),
        "budget_subset": pd.read_csv(PATHS["budget_subset"]),
        "budget_summary": read_json(PATHS["budget_summary"]),
        "raw_summary": read_json(PATHS["raw_summary"]),
        "raw_selected_features": pd.read_csv(PATHS["raw_selected_features"]),
        "raw_feature_audit": pd.read_csv(PATHS["raw_feature_audit"]),
        "raw_model_results": pd.read_csv(PATHS["raw_model_results"]),
        "neural_v1_selected": pd.read_csv(PATHS["neural_v1_selected"]),
        "residual_selected": pd.read_csv(PATHS["residual_selected"]),
        "conformal_summary": read_json(PATHS["conformal_summary"]),
    }


def safe_float(value: object) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def fmt_float(value: object, digits: int = 2, signed: bool = False) -> str:
    x = safe_float(value)
    if x is None:
        return ""
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


def fmt_int(value: object) -> str:
    x = safe_float(value)
    if x is None:
        return ""
    return str(int(round(x)))


def protocol_label(protocol: object) -> str:
    text = str(protocol)
    if text == "source_only":
        return "Source only"
    if text == "source_plus_target_adapt_loocv":
        return "Source + target-adapt LOOCV"
    if text == "source_plus_target_adapt_nested_loocv_residual":
        return "Residual nested LOOCV"
    if text == "source_plus_k_batch9_labels":
        return "Source + k Batch9 labels"
    return text


def short_protocol_label(protocol: object) -> str:
    text = str(protocol)
    if text == "source_only":
        return "Source only"
    if text == "source_plus_target_adapt_loocv":
        return "S+T LOOCV"
    if text == "source_plus_target_adapt_nested_loocv_residual":
        return "Residual LOOCV"
    if text == "source_plus_k_batch9_labels":
        return "S+k Batch9"
    return text


def model_label(model: object) -> str:
    return MODEL_LABELS.get(str(model), str(model))


def family_label(family: object) -> str:
    return FAMILY_LABELS.get(str(family), str(family))


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    out = df.fillna("").copy()
    columns = [str(col) for col in out.columns]
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in out.iterrows():
        values = []
        for col in out.columns:
            text = str(row[col]).replace("\n", "<br>").replace("|", "\\|")
            values.append(text)
        rows.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def save_table(df: pd.DataFrame, csv_path: Path, md_path: Path) -> None:
    df.to_csv(csv_path, index=False, encoding="utf-8")
    write_markdown_table(df, md_path)


def save_figure(fig: plt.Figure, base: Path) -> list[Path]:
    outputs = []
    for ext in ["png", "svg", "pdf"]:
        path = base.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def get_strict_row(strict_selected: pd.DataFrame, scope: str) -> pd.Series:
    rows = strict_selected[strict_selected["selection_scope"].eq(scope)]
    if rows.empty:
        raise ValueError(f"Missing strict selected row: {scope}")
    return rows.iloc[0]


def get_neural_v1_row(neural_v1: pd.DataFrame) -> pd.Series:
    rows = neural_v1[neural_v1["selection_scope"].eq("selected_neural_by_batch9")]
    if rows.empty:
        raise ValueError("Missing neural v1 selected_neural_by_batch9 row")
    return rows.iloc[0]


def draw_protocol_flowchart() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.set_axis_off()

    boxes = [
        {
            "xy": (0.055, 0.47),
            "w": 0.25,
            "h": 0.32,
            "face": COLORS["light_blue"],
            "edge": COLORS["blue"],
            "title": "Batch 1-8",
            "body": "Source domain\nmodel fitting\n61 cells",
        },
        {
            "xy": (0.375, 0.47),
            "w": 0.25,
            "h": 0.32,
            "face": COLORS["light_teal"],
            "edge": COLORS["teal"],
            "title": "Batch 9",
            "body": "Labeled target-adapt\nBatch9-only selection\n8 cells",
        },
        {
            "xy": (0.695, 0.47),
            "w": 0.25,
            "h": 0.32,
            "face": COLORS["light_orange"],
            "edge": COLORS["orange"],
            "title": "Batch 10",
            "body": "Blind target test\nfinal reporting only\n8 cells",
        },
    ]
    for box in boxes:
        patch = FancyBboxPatch(
            box["xy"],
            box["w"],
            box["h"],
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=1.8,
            facecolor=box["face"],
            edgecolor=box["edge"],
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        x, y = box["xy"]
        ax.text(
            x + box["w"] / 2,
            y + box["h"] - 0.08,
            box["title"],
            ha="center",
            va="center",
            fontsize=15,
            weight="bold",
            color=box["edge"],
            transform=ax.transAxes,
        )
        ax.text(
            x + box["w"] / 2,
            y + 0.115,
            box["body"],
            ha="center",
            va="center",
            fontsize=10.5,
            linespacing=1.25,
            color="#222222",
            transform=ax.transAxes,
        )

    for start, end in [((0.305, 0.63), (0.375, 0.63)), ((0.625, 0.63), (0.695, 0.63))]:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#444444"),
        )

    ax.text(
        0.5,
        0.92,
        "Strict cross-batch EOL/RUL protocol",
        ha="center",
        va="center",
        fontsize=15.5,
        weight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.31,
        "EOL-first formulation: predicted RUL at horizon H = predicted EOL - H",
        ha="center",
        va="center",
        fontsize=10.2,
        color="#333333",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.13,
        "Guard: Batch 10 is not used for model, horizon, feature, hyperparameter, checkpoint, or seed selection.",
        ha="center",
        va="center",
        fontsize=9.6,
        color="#333333",
        transform=ax.transAxes,
    )
    save_figure(fig, FIGURE_BASES["protocol"])


def draw_batch9_candidate_heatmap(candidates: pd.DataFrame) -> None:
    df = candidates.copy()
    df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce").astype("Int64")
    df["selection_eol_rmse"] = pd.to_numeric(df["selection_eol_rmse"], errors="coerce")
    df["row_label"] = df.apply(
        lambda row: f"{protocol_label(row['protocol'])}\n{model_label(row['model'])}", axis=1
    )
    pivot = df.pivot_table(
        index="row_label",
        columns="horizon",
        values="selection_eol_rmse",
        aggfunc="min",
    ).sort_index(axis=1)
    pivot = pivot.loc[pivot.min(axis=1).sort_values().index]
    values = pivot.to_numpy(dtype=float)
    clipped = np.clip(values, 0, 300)

    fig_h = max(7.0, 0.39 * len(pivot) + 2.0)
    fig, ax = plt.subplots(figsize=(10.6, fig_h))
    im = ax.imshow(clipped, cmap="YlGnBu_r", aspect="auto", vmin=50, vmax=300)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(int(col)) for col in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8.2)
    ax.set_xlabel("Observed horizon (cycles)")
    ax.set_title("Batch 9 validation RMSE used for candidate selection")
    ax.set_ylabel("")
    ax.grid(False)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isnan(value):
                label = ""
            elif value > 999:
                label = ">999"
            elif value > 300:
                label = ">300"
            else:
                label = f"{value:.0f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=7.4, color="#1f1f1f")

    selected_label = "Source + target-adapt LOOCV\nExtraTrees"
    if selected_label in pivot.index and 1000 in pivot.columns:
        i = list(pivot.index).index(selected_label)
        j = list(pivot.columns).index(1000)
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=COLORS["red"], lw=2.3))
        ax.text(j, i - 0.44, "selected", ha="center", va="top", fontsize=7.3, color=COLORS["red"], weight="bold")

    ax.set_xticks(np.arange(-0.5, len(pivot.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.033, pad=0.025)
    cbar.set_label("Batch 9 RMSE (cycles; clipped at 300)")
    ax.text(
        0.0,
        -0.12,
        "All values are Batch 9 validation metrics. Batch 10 is not used in this selection heatmap.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        color="#444444",
    )
    fig.tight_layout()
    save_figure(fig, FIGURE_BASES["batch9_heatmap"])


def draw_batch10_pred_vs_true(per_cell: pd.DataFrame, robust_summary: dict[str, object]) -> None:
    df = per_cell.copy()
    for col in ["true_eol_cycle", "pred_eol_cycle", "eol_error_cycles"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    metrics = robust_summary["blind_test_metrics"]

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    errors = df["eol_error_cycles"]
    colors = np.where(errors >= 0, COLORS["blue"], COLORS["orange"])
    ax.scatter(
        df["true_eol_cycle"],
        df["pred_eol_cycle"],
        s=76,
        c=colors,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    low = min(df["true_eol_cycle"].min(), df["pred_eol_cycle"].min()) - 80
    high = max(df["true_eol_cycle"].max(), df["pred_eol_cycle"].max()) + 80
    ax.plot([low, high], [low, high], color="#555555", lw=1.2, linestyle="--", label="Ideal")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("True EOL (cycles)")
    ax.set_ylabel("Predicted EOL (cycles)")
    ax.set_title("Batch 10 blind-test EOL predictions")

    for _, row in df.iterrows():
        dx = -12 if row["cell_id"] in {"10-4"} else 8
        dy = 8 if row["cell_id"] != "10-7" else -12
        ha = "right" if dx < 0 else "left"
        ax.annotate(
            str(row["cell_id"]),
            (row["true_eol_cycle"], row["pred_eol_cycle"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            ha=ha,
            color="#333333",
        )

    note = (
        f"n = {int(metrics['n_cells'])} cells\n"
        f"RMSE = {float(metrics['rmse_cycles']):.2f} cycles\n"
        f"MAE = {float(metrics['mae_cycles']):.2f} cycles\n"
        f"Bias = {float(metrics['bias_mean_error_cycles']):+.2f} cycles\n"
        f"Pearson r = {float(metrics['pearson_true_pred']):.3f}"
    )
    ax.text(
        0.04,
        0.96,
        note,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.1,
        bbox=dict(boxstyle="round,pad=0.34", facecolor="white", edgecolor="#cccccc"),
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, FIGURE_BASES["pred_vs_true"])


def draw_batch10_abs_error(per_cell: pd.DataFrame, robust_summary: dict[str, object]) -> None:
    df = per_cell.copy()
    df["abs_eol_error_cycles"] = pd.to_numeric(df["abs_eol_error_cycles"], errors="coerce")
    df["eol_error_cycles"] = pd.to_numeric(df["eol_error_cycles"], errors="coerce")
    df = df.sort_values("cell_id")
    mae = float(robust_summary["blind_test_metrics"]["mae_cycles"])
    max_cell = robust_summary["per_cell"]["max_abs_error_cell_id"]

    fig, ax = plt.subplots(figsize=(7.5, 4.9))
    colors = np.where(df["eol_error_cycles"] >= 0, COLORS["blue"], COLORS["orange"])
    bars = ax.bar(df["cell_id"], df["abs_eol_error_cycles"], color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(mae, color="#333333", linestyle="--", linewidth=1.2, label=f"MAE = {mae:.2f}")
    ax.set_ylabel("Absolute EOL/RUL error (cycles)")
    ax.set_xlabel("Batch 10 cell")
    ax.set_title("Batch 10 per-cell absolute error")
    ax.set_ylim(0, max(df["abs_eol_error_cycles"].max() * 1.22, mae * 1.5))
    ax.legend(loc="upper right")
    for bar, cell, value in zip(bars, df["cell_id"], df["abs_eol_error_cycles"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2.0,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.4,
            weight="bold" if cell == max_cell else "normal",
        )
    ax.text(
        0.01,
        -0.18,
        "Blue bars overpredict EOL; orange bars underpredict EOL. At H = 1000, EOL and derived RUL errors are identical.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#444444",
    )
    fig.tight_layout()
    save_figure(fig, FIGURE_BASES["per_cell_error"])


def draw_budget_sensitivity(aggregate: pd.DataFrame, subsets: pd.DataFrame) -> None:
    agg = aggregate.copy()
    sub = subsets.copy()
    for col in [
        "budget_k",
        "rmse_mean",
        "rmse_median",
        "rmse_min",
        "rmse_q25",
        "rmse_q75",
        "rmse_max",
        "mae_mean",
    ]:
        if col in agg.columns:
            agg[col] = pd.to_numeric(agg[col], errors="coerce")
    for col in ["budget_k", "subset_index_within_budget", "target_test_eol_rmse"]:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    agg = agg.sort_values("budget_k")
    sub = sub.sort_values(["budget_k", "subset_index_within_budget"])

    fig, ax = plt.subplots(figsize=(7.7, 5.2))

    for budget, group in sub.groupby("budget_k"):
        n = max(len(group), 1)
        offsets = np.linspace(-0.16, 0.16, n) if n > 1 else np.array([0.0])
        ax.scatter(
            np.full(n, budget) + offsets,
            group["target_test_eol_rmse"],
            s=28,
            color="#9a9a9a",
            alpha=0.55,
            edgecolor="white",
            linewidth=0.35,
            zorder=2,
        )

    for _, row in agg.iterrows():
        x = row["budget_k"]
        ax.vlines(x, row["rmse_min"], row["rmse_max"], color=COLORS["gray"], lw=1.2, zorder=1)
        ax.add_patch(
            Rectangle(
                (x - 0.13, row["rmse_q25"]),
                0.26,
                row["rmse_q75"] - row["rmse_q25"],
                facecolor=COLORS["light_blue"],
                edgecolor=COLORS["blue"],
                linewidth=1.0,
                zorder=3,
            )
        )
        ax.hlines(row["rmse_median"], x - 0.13, x + 0.13, color=COLORS["blue"], lw=1.8, zorder=4)

    ax.plot(agg["budget_k"], agg["rmse_mean"], marker="o", color=COLORS["orange"], lw=2.1, label="Mean Batch10 RMSE")
    ax.plot(agg["budget_k"], agg["mae_mean"], marker="s", color=COLORS["teal"], lw=1.8, label="Mean Batch10 MAE")
    ax.set_xticks([0, 1, 2, 4, 8])
    ax.set_xlabel("Labeled Batch 9 cells")
    ax.set_ylabel("Batch 10 error (cycles)")
    ax.set_title("Target-label budget sensitivity under fixed ExtraTrees, H = 1000")
    ax.legend(loc="upper right")
    ax.text(
        0.01,
        -0.18,
        "Gray dots are predefined subsets. Box/range summaries are descriptive only; no subset is selected by Batch 10.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#444444",
    )
    fig.tight_layout()
    save_figure(fig, FIGURE_BASES["budget"])


def draw_raw_provenance_summary(
    selected_features: pd.DataFrame, feature_audit: pd.DataFrame, raw_models: pd.DataFrame
) -> None:
    audit = feature_audit.copy()
    selected = selected_features.copy()
    models = raw_models.copy()
    candidate_counts = (
        audit[truthy(audit["used_in_model_candidate_pool"])]
        .groupby("feature_family")
        .size()
        .rename("candidate_count")
    )
    selected_counts = selected.groupby("feature_family").size().rename("selected_count")
    family_df = (
        pd.concat([candidate_counts, selected_counts], axis=1)
        .fillna(0)
        .astype(int)
        .reindex(["capacity", "dq", "time_status", "voltage_current"])
        .fillna(0)
    )

    models["selection_eol_rmse"] = pd.to_numeric(models["selection_eol_rmse"], errors="coerce")
    top_models = models.sort_values("selection_eol_rmse").head(10).copy()
    is_selected = truthy(top_models["is_selected_by_batch9"])
    labels = [
        f"r{int(row.selection_rank)} {row.feature_strategy}\n{short_protocol_label(row.protocol)}, {model_label(row.model)}"
        for row in top_models.itertuples(index=False)
    ]

    fig = plt.figure(figsize=(13.0, 7.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.55], height_ratios=[1.0, 1.25], wspace=0.52, hspace=0.38)
    ax_counts = fig.add_subplot(gs[0, 0])
    ax_rank = fig.add_subplot(gs[0, 1])
    ax_features = fig.add_subplot(gs[1, :])

    x = np.arange(len(family_df))
    ax_counts.bar(x - 0.17, family_df["candidate_count"], width=0.34, color=COLORS["blue"], label="Candidate")
    ax_counts.bar(x + 0.17, family_df["selected_count"], width=0.34, color=COLORS["orange"], label="Selected")
    ax_counts.set_xticks(x)
    ax_counts.set_xticklabels([family_label(idx) for idx in family_df.index], rotation=20, ha="right")
    ax_counts.set_ylabel("Feature count")
    ax_counts.set_title("Raw v3 feature families")
    ax_counts.legend(loc="upper right")
    for xpos, val in zip(x - 0.17, family_df["candidate_count"]):
        ax_counts.text(xpos, val + 5, str(int(val)), ha="center", va="bottom", fontsize=8)
    for xpos, val in zip(x + 0.17, family_df["selected_count"]):
        ax_counts.text(xpos, val + 5, str(int(val)), ha="center", va="bottom", fontsize=8)

    y = np.arange(len(top_models))[::-1]
    bar_colors = np.where(is_selected.to_numpy()[::-1], COLORS["orange"], COLORS["blue"])
    ax_rank.barh(y, top_models["selection_eol_rmse"].to_numpy()[::-1], color=bar_colors)
    ax_rank.set_yticks(y)
    ax_rank.set_yticklabels(labels[::-1], fontsize=7.2)
    ax_rank.tick_params(axis="y", pad=3)
    ax_rank.set_xlabel("Batch 9 RMSE (cycles)")
    ax_rank.set_title("Top raw v3 candidates by Batch 9 validation")
    ax_rank.grid(axis="x", alpha=0.25)
    for yy, value in zip(y, top_models["selection_eol_rmse"].to_numpy()[::-1]):
        ax_rank.text(value + 1.0, yy, f"{value:.1f}", ha="left", va="center", fontsize=7.6)

    ax_features.set_axis_off()
    ax_features.set_title("Selected raw features: corr_capacity_top20 + SVR-RBF, H = 1000", loc="left", pad=10)
    selected = selected.sort_values("rank").head(20)
    cols = 2
    rows_per_col = 10
    for i, row in selected.iterrows():
        rank = int(row["rank"])
        col = 0 if rank <= rows_per_col else 1
        pos = rank - 1 if col == 0 else rank - rows_per_col - 1
        x0 = 0.02 + col * 0.49
        y0 = 0.88 - pos * 0.082
        ax_features.text(x0, y0, f"{rank:02d}", ha="left", va="center", fontsize=8.6, weight="bold")
        ax_features.text(
            x0 + 0.06,
            y0,
            textwrap.shorten(str(row["feature"]), width=54, placeholder="..."),
            ha="left",
            va="center",
            fontsize=8.1,
            family="monospace",
        )
    ax_features.text(
        0.02,
        0.02,
        "All selected features are capacity-family summaries from raw our_data/*.pkl observed cycles 1..1000. Ranks are selected-feature order, not coefficients.",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )
    fig.suptitle("Strict raw-provenance v3 selected-feature and family summary", fontsize=15.0, weight="bold", y=0.985)
    save_figure(fig, FIGURE_BASES["raw_provenance"])


def draw_evidence_hierarchy(main_table: pd.DataFrame) -> None:
    df = main_table.copy()
    fig, ax = plt.subplots(figsize=(13.8, 8.0))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.02, 0.96, "Evidence hierarchy for the current EOL/RUL package", fontsize=15, weight="bold", ha="left")
    ax.text(
        0.02,
        0.915,
        "Rows separate formal strict evidence, comparators, diagnostics, prototypes, and negative/risk-control results.",
        fontsize=9.2,
        color="#444444",
        ha="left",
    )

    headers = ["Evidence line", "Metric", "Role", "Safe interpretation"]
    x_positions = [0.03, 0.265, 0.425, 0.64]
    wrap_widths = [28, 18, 28, 56]
    y_header = 0.84
    y_start = 0.765
    row_h = 0.112
    for x, header in zip(x_positions, headers):
        ax.text(x, y_header, header, fontsize=9.2, weight="bold", ha="left", va="center")
    ax.plot([0.025, 0.975], [y_header - 0.03, y_header - 0.03], color="#cccccc", lw=1.0)

    role_colors = {
        "Formal main": COLORS["light_blue"],
        "Comparator": COLORS["light_teal"],
        "Diagnostic": "#f2f2f2",
        "Prototype": "#f8f0fb",
        "Negative": COLORS["light_orange"],
        "Uncertainty": "#f0f5ff",
    }
    for idx, row in df.iterrows():
        y = y_start - idx * row_h
        role = str(row["evidence_level"])
        face = role_colors.get(role, "white")
        ax.add_patch(Rectangle((0.02, y - row_h * 0.45), 0.96, row_h * 0.9, facecolor=face, edgecolor="white", lw=1.0))
        metric = row["batch10_rmse_cycles"]
        if str(row["result_line"]).startswith("Uncertainty"):
            metric = "80% half-width 129.42 cycles; coverage 8/8"
        elif metric:
            metric = f"RMSE {metric}, MAE {row['batch10_mae_cycles']}"
        else:
            metric = row["uncertainty_or_diagnostic_metric"]
        cells = [
            row["result_line"],
            metric,
            row["paper_role"],
            row["safe_interpretation"],
        ]
        for x, width, text in zip(x_positions, wrap_widths, cells):
            wrapped = textwrap.fill(str(text), width=width)
            ax.text(x, y, wrapped, fontsize=7.7, ha="left", va="center", color="#222222", linespacing=1.08)
    ax.text(
        0.02,
        0.075,
        "Batch 10 values are final reporting only. They are not used here to select models or promote subset-specific results.",
        fontsize=8.6,
        color="#444444",
        ha="left",
    )
    save_figure(fig, FIGURE_BASES["evidence_hierarchy"])


def build_main_results_table(data: dict[str, object]) -> pd.DataFrame:
    strict_selected = data["strict_selected"].copy()
    raw_summary = data["raw_summary"]
    neural = get_neural_v1_row(data["neural_v1_selected"])
    residual = data["residual_selected"].iloc[0]
    conformal = data["conformal_summary"]

    main = get_strict_row(strict_selected, "global_best_by_batch9")
    source_only = get_strict_row(strict_selected, "best_horizon_for_source_only_extra_trees")
    raw = raw_summary["selected_test_row"]
    interval80 = conformal["conformal_interval_summary"]["80"]

    rows = [
        {
            "evidence_level": "Formal main",
            "result_line": "Strict main baseline",
            "source_artifact": "strict-eol-protocol-v1; strict-eol-robustness-v1",
            "protocol": protocol_label(main["protocol"]),
            "model_or_strategy": "ExtraTrees, strict processed fused features",
            "horizon_cycles": fmt_int(main["horizon"]),
            "batch9_rmse_cycles": fmt_float(main["selection_eol_rmse"]),
            "batch9_mae_cycles": fmt_float(main["selection_eol_mae"]),
            "batch10_rmse_cycles": fmt_float(main["target_test_eol_rmse"]),
            "batch10_mae_cycles": fmt_float(main["target_test_eol_mae"]),
            "n_test_cells": fmt_int(main["target_test_n_cells"]),
            "uncertainty_or_diagnostic_metric": "",
            "paper_role": "Current main quantitative proof-of-concept",
            "safe_interpretation": "Batch9-selected strict baseline; Batch10 is final blind-test reporting only.",
            "do_not_overclaim": "Not final proposed architecture; supervised labeled Batch9 adaptation.",
        },
        {
            "evidence_level": "Comparator",
            "result_line": "Strict source-only comparator",
            "source_artifact": "strict-eol-protocol-v1; target-label-budget-sensitivity-v1",
            "protocol": protocol_label(source_only["protocol"]),
            "model_or_strategy": "ExtraTrees, no Batch9 adaptation",
            "horizon_cycles": fmt_int(source_only["horizon"]),
            "batch9_rmse_cycles": fmt_float(source_only["selection_eol_rmse"]),
            "batch9_mae_cycles": fmt_float(source_only["selection_eol_mae"]),
            "batch10_rmse_cycles": fmt_float(source_only["target_test_eol_rmse"]),
            "batch10_mae_cycles": fmt_float(source_only["target_test_eol_mae"]),
            "n_test_cells": fmt_int(source_only["target_test_n_cells"]),
            "uncertainty_or_diagnostic_metric": "",
            "paper_role": "Comparator for labeled target-adapt gain",
            "safe_interpretation": "Shows the fixed model without labeled Batch9 target cells.",
            "do_not_overclaim": "Not a globally selected main result.",
        },
        {
            "evidence_level": "Diagnostic",
            "result_line": "Raw-provenance v3 diagnostic",
            "source_artifact": "strict-raw-provenance-v3",
            "protocol": protocol_label(raw["protocol"]),
            "model_or_strategy": f"{raw['feature_strategy']} + {model_label(raw['model'])}",
            "horizon_cycles": fmt_int(raw["horizon"]),
            "batch9_rmse_cycles": fmt_float(raw["selection_eol_rmse"]),
            "batch9_mae_cycles": fmt_float(raw["selection_eol_mae"]),
            "batch10_rmse_cycles": fmt_float(raw["target_test_eol_rmse"]),
            "batch10_mae_cycles": fmt_float(raw["target_test_eol_mae"]),
            "n_test_cells": fmt_int(raw["target_test_n_cells"]),
            "uncertainty_or_diagnostic_metric": "20 selected capacity-family raw features",
            "paper_role": "Raw-feature provenance and online-availability diagnostic",
            "safe_interpretation": "Direct raw our_data/*.pkl features support traceability, not superiority over the strict main result.",
            "do_not_overclaim": "Does not beat strict ExtraTrees Batch10 RMSE; not raw-only superiority.",
        },
        {
            "evidence_level": "Prototype",
            "result_line": "Neural v1 prototype",
            "source_artifact": "neural-eol-model-v1",
            "protocol": protocol_label(neural["protocol"]),
            "model_or_strategy": model_label(neural["model"]),
            "horizon_cycles": fmt_int(neural["horizon"]),
            "batch9_rmse_cycles": fmt_float(neural["selection_eol_rmse"]),
            "batch9_mae_cycles": fmt_float(neural["selection_eol_mae"]),
            "batch10_rmse_cycles": fmt_float(neural["target_test_eol_rmse"]),
            "batch10_mae_cycles": fmt_float(neural["target_test_eol_mae"]),
            "n_test_cells": fmt_int(neural["target_test_n_cells"]),
            "uncertainty_or_diagnostic_metric": "",
            "paper_role": "Physics-consistent neural prototype",
            "safe_interpretation": "Useful prototype, but weaker than the strict ExtraTrees main result on Batch10.",
            "do_not_overclaim": "No neural superiority claim.",
        },
        {
            "evidence_level": "Negative",
            "result_line": "Residual calibration negative result",
            "source_artifact": "eol-ablation-residual-v1",
            "protocol": protocol_label(residual["protocol"]),
            "model_or_strategy": f"{residual['residual_method']} on {model_label(residual['base_model'])}",
            "horizon_cycles": fmt_int(residual["horizon"]),
            "batch9_rmse_cycles": fmt_float(residual["selection_eol_rmse"]),
            "batch9_mae_cycles": fmt_float(residual["selection_eol_mae"]),
            "batch10_rmse_cycles": fmt_float(residual["target_test_eol_rmse"]),
            "batch10_mae_cycles": fmt_float(residual["target_test_eol_mae"]),
            "n_test_cells": fmt_int(residual["target_test_n_cells"]),
            "uncertainty_or_diagnostic_metric": "",
            "paper_role": "Negative/risk-control evidence",
            "safe_interpretation": "Batch9 improvement does not transfer; Batch10 worsens, so the residual correction is not retained.",
            "do_not_overclaim": "No residual-improvement claim.",
        },
        {
            "evidence_level": "Uncertainty",
            "result_line": "Uncertainty diagnostic",
            "source_artifact": "strict-eol-conformal-uncertainty-v1",
            "protocol": protocol_label(conformal["fixed_setting"]["protocol"]),
            "model_or_strategy": "Fixed strict ExtraTrees + split conformal diagnostic",
            "horizon_cycles": fmt_int(conformal["fixed_setting"]["horizon_cycles"]),
            "batch9_rmse_cycles": fmt_float(conformal["batch9_calibration"]["rmse_cycles"]),
            "batch9_mae_cycles": fmt_float(conformal["batch9_calibration"]["mae_cycles"]),
            "batch10_rmse_cycles": fmt_float(conformal["batch10_metrics"]["rmse_cycles"]),
            "batch10_mae_cycles": fmt_float(conformal["batch10_metrics"]["mae_cycles"]),
            "n_test_cells": fmt_int(conformal["batch10_metrics"]["n_cells"]),
            "uncertainty_or_diagnostic_metric": (
                f"80% finite interval half-width {fmt_float(interval80['q_abs_error_cycles'])} cycles; "
                f"coverage {fmt_int(interval80['covered_cells'])}/{fmt_int(interval80['n_batch10_cells'])}"
            ),
            "paper_role": "Limited uncertainty diagnostic",
            "safe_interpretation": "Finite 80% interval only; 90% and 95% standard split-conformal intervals are unbounded with eight calibration cells.",
            "do_not_overclaim": "No strong calibrated-uncertainty claim.",
        },
    ]
    return pd.DataFrame(rows)


def build_budget_table(aggregate: pd.DataFrame) -> pd.DataFrame:
    agg = aggregate.copy()
    numeric_cols = [
        "budget_k",
        "n_subsets",
        "rmse_mean",
        "rmse_std",
        "rmse_min",
        "rmse_q25",
        "rmse_median",
        "rmse_q75",
        "rmse_max",
        "mae_mean",
        "mae_min",
        "mae_median",
        "mae_max",
    ]
    for col in numeric_cols:
        if col in agg.columns:
            agg[col] = pd.to_numeric(agg[col], errors="coerce")
    rows = []
    for _, row in agg.sort_values("budget_k").iterrows():
        rows.append(
            {
                "labeled_batch9_cells": fmt_int(row["budget_k"]),
                "n_predefined_subsets": fmt_int(row["n_subsets"]),
                "subset_generation_strategy": row["subset_generation_strategy"],
                "batch10_rmse_mean_cycles": fmt_float(row["rmse_mean"]),
                "batch10_rmse_median_cycles": fmt_float(row["rmse_median"]),
                "batch10_rmse_iqr_cycles": f"{fmt_float(row['rmse_q25'])}-{fmt_float(row['rmse_q75'])}",
                "batch10_rmse_range_cycles": f"{fmt_float(row['rmse_min'])}-{fmt_float(row['rmse_max'])}",
                "batch10_mae_mean_cycles": fmt_float(row["mae_mean"]),
                "batch10_mae_median_cycles": fmt_float(row["mae_median"]),
                "batch10_mae_range_cycles": f"{fmt_float(row['mae_min'])}-{fmt_float(row['mae_max'])}",
                "source_artifact": "target-label-budget-sensitivity-v1/budget_aggregate_results.csv",
                "safe_interpretation": "Sensitivity aggregate only; no subset is selected or promoted by Batch10.",
            }
        )
    return pd.DataFrame(rows)


def build_raw_provenance_table(
    raw_summary: dict[str, object],
    selected_features: pd.DataFrame,
    feature_audit: pd.DataFrame,
    raw_models: pd.DataFrame,
) -> pd.DataFrame:
    audit = feature_audit.copy()
    selected = selected_features.copy()
    models = raw_models.copy()
    candidate_counts = (
        audit[truthy(audit["used_in_model_candidate_pool"])]
        .groupby("feature_family")
        .size()
        .to_dict()
    )
    selected_counts = selected.groupby("feature_family").size().to_dict()
    forbidden_excluded = int(
        (
            truthy(audit["forbidden_input"])
            & ~truthy(audit["used_in_model_candidate_pool"])
        ).sum()
    )
    selected_row = raw_summary["selected_test_row"]
    rows: list[dict[str, object]] = [
        {
            "row_type": "selected_batch9_candidate",
            "feature_family": selected_row["feature_family_scope"],
            "candidate_features": fmt_int(raw_summary["candidate_feature_count"]),
            "selected_features": fmt_int(selected_row["n_features"]),
            "model_or_strategy": f"{selected_row['feature_strategy']} + {model_label(selected_row['model'])}",
            "protocol": protocol_label(selected_row["protocol"]),
            "horizon_cycles": fmt_int(selected_row["horizon"]),
            "batch9_rmse_cycles": fmt_float(selected_row["selection_eol_rmse"]),
            "batch9_mae_cycles": fmt_float(selected_row["selection_eol_mae"]),
            "batch10_rmse_cycles": fmt_float(selected_row["target_test_eol_rmse"]),
            "batch10_mae_cycles": fmt_float(selected_row["target_test_eol_mae"]),
            "source_fields_or_provenance": "direct raw our_data/*.pkl; observed cycles 1..1000",
            "safe_interpretation": "Strongest raw v3 diagnostic; supports traceable online features, not raw-only superiority.",
        }
    ]
    for family in ["capacity", "dq", "time_status", "voltage_current"]:
        source_note = raw_summary["family_source_notes"].get(family, "")
        rows.append(
            {
                "row_type": "feature_family_pool",
                "feature_family": family_label(family),
                "candidate_features": fmt_int(candidate_counts.get(family, 0)),
                "selected_features": fmt_int(selected_counts.get(family, 0)),
                "model_or_strategy": "",
                "protocol": "",
                "horizon_cycles": "1000",
                "batch9_rmse_cycles": "",
                "batch9_mae_cycles": "",
                "batch10_rmse_cycles": "",
                "batch10_mae_cycles": "",
                "source_fields_or_provenance": source_note,
                "safe_interpretation": "Candidate/selected count summary only.",
            }
        )
    rows.append(
        {
            "row_type": "forbidden_input_audit",
            "feature_family": "forbidden metadata/labels",
            "candidate_features": "0",
            "selected_features": "0",
            "model_or_strategy": "",
            "protocol": "",
            "horizon_cycles": "1000",
            "batch9_rmse_cycles": "",
            "batch9_mae_cycles": "",
            "batch10_rmse_cycles": "",
            "batch10_mae_cycles": "",
            "source_fields_or_provenance": f"{forbidden_excluded} forbidden audit rows excluded from model candidate pool",
            "safe_interpretation": "Batch/cell IDs, EOL/RUL labels, full-life and future-cycle metadata are excluded from model inputs.",
        }
    )

    models["selection_eol_rmse"] = pd.to_numeric(models["selection_eol_rmse"], errors="coerce")
    top3 = models.sort_values("selection_eol_rmse").head(3)
    for _, row in top3.iterrows():
        rows.append(
            {
                "row_type": "batch9_candidate_rank",
                "feature_family": row["feature_family_scope"],
                "candidate_features": fmt_int(row["n_features_median"]),
                "selected_features": fmt_int(row["n_features_median"]) if truthy(pd.Series([row["is_selected_by_batch9"]])).iloc[0] else "",
                "model_or_strategy": f"rank {fmt_int(row['selection_rank'])}: {row['feature_strategy']} + {model_label(row['model'])}",
                "protocol": protocol_label(row["protocol"]),
                "horizon_cycles": fmt_int(row["horizon"]),
                "batch9_rmse_cycles": fmt_float(row["selection_eol_rmse"]),
                "batch9_mae_cycles": fmt_float(row["selection_eol_mae"]),
                "batch10_rmse_cycles": fmt_float(row["target_test_eol_rmse"]),
                "batch10_mae_cycles": fmt_float(row["target_test_eol_mae"]),
                "source_fields_or_provenance": "raw_model_results_v3.csv",
                "safe_interpretation": "Ranked by Batch9 only; nonselected candidates are not Batch10-evaluated.",
            }
        )
    return pd.DataFrame(rows)


def write_report(
    data: dict[str, object],
    main_table: pd.DataFrame,
    budget_table: pd.DataFrame,
    raw_table: pd.DataFrame,
) -> None:
    generated = datetime.now().isoformat(timespec="seconds")
    figure_files = {key: [base.with_suffix(ext) for ext in [".png", ".svg", ".pdf"]] for key, base in FIGURE_BASES.items()}
    lines = [
        f"# {RUN_NAME}",
        "",
        f"Generated: {generated}",
        "",
        "## Protocol Guard",
        "",
        "- This package only regenerates paper-ready data-derived figures and tables from accepted CSV/JSON/MD evidence.",
        "- It does not train models, tune hyperparameters, search features, choose horizons, or select by Batch 10.",
        "- Batch 1-8 are source cells, Batch 9 is labeled target-development/adaptation and selection, and Batch 10 is final blind-test reporting only.",
        "- Batch 10 plots report the locked strict baseline selected from Batch 9; they are not used to choose models.",
        "- Target-label budget rows are sensitivity evidence only; low-error subsets are not promoted as selected models.",
        "- Raw-provenance v3 is diagnostic feature-traceability evidence, not raw-only superiority.",
        "",
        "## Figures",
        "",
        f"- `{figure_files['protocol'][0].name}` / SVG / PDF",
        "  - Source: authoritative mainline, strict protocol reports, and project split facts.",
        "  - Safe interpretation: fixed supervised small labeled target-domain adaptation protocol.",
        "  - Boundary: not unsupervised adaptation, not cross-dataset validation.",
        "",
        f"- `{figure_files['batch9_heatmap'][0].name}` / SVG / PDF",
        f"  - Source: `{PATHS['strict_candidate']}`.",
        "  - Safe interpretation: Batch9 validation selects source_plus_target_adapt_loocv + ExtraTrees + 1000 cycles.",
        "  - Boundary: color scale is clipped for readability and contains no Batch10 ranking.",
        "",
        f"- `{figure_files['pred_vs_true'][0].name}` / SVG / PDF",
        f"  - Source: `{PATHS['robust_per_cell']}` and `{PATHS['robust_summary']}`.",
        "  - Safe interpretation: final Batch10 predicted-vs-true EOL for the Batch9-selected strict baseline.",
        "  - Boundary: correlation and bias are over eight blind-test cells only.",
        "",
        f"- `{figure_files['per_cell_error'][0].name}` / SVG / PDF",
        f"  - Source: `{PATHS['robust_per_cell']}` and `{PATHS['robust_summary']}`.",
        "  - Safe interpretation: per-cell absolute EOL/RUL error at H = 1000.",
        "  - Boundary: small n means each cell strongly affects summary metrics.",
        "",
        f"- `{figure_files['budget'][0].name}` / SVG / PDF",
        f"  - Source: `{PATHS['budget_aggregate']}` and `{PATHS['budget_subset']}`.",
        "  - Safe interpretation: fixed ExtraTrees 1000-cycle sensitivity to predefined labeled Batch9 budgets.",
        "  - Boundary: no subset is selected by Batch10; k=8 reproduces the strict main result.",
        "",
        f"- `{figure_files['raw_provenance'][0].name}` / SVG / PDF",
        f"  - Source: `{PATHS['raw_selected_features']}`, `{PATHS['raw_feature_audit']}`, and `{PATHS['raw_model_results']}`.",
        "  - Safe interpretation: selected raw v3 model uses 20 capacity-family raw features computed from observed cycles only.",
        "  - Boundary: provenance/availability diagnostic, not a replacement for the strict main baseline.",
        "",
        f"- `{figure_files['evidence_hierarchy'][0].name}` / SVG / PDF",
        "  - Source: strict protocol/robustness, target-label budget, raw-provenance v3, neural v1, residual ablation, and conformal uncertainty summaries.",
        "  - Safe interpretation: separates formal main evidence, comparator, diagnostics, prototype, negative result, and limited uncertainty.",
        "  - Boundary: no SOTA, neural-superiority, residual-improvement, raw-only-superiority, or strong-uncertainty claim.",
        "",
        "## Tables",
        "",
        "- `table_main_results_v2.csv` / `table_main_results_v2.md`",
        "  - Sources: strict protocol and robustness, raw-provenance v3, neural v1, residual ablation, conformal uncertainty.",
        "  - Safe interpretation: compact evidence hierarchy for manuscript main table or supplementary summary.",
        "",
        "- `table_target_label_budget.csv` / `table_target_label_budget.md`",
        "  - Sources: budget aggregate CSV and subset CSV.",
        "  - Safe interpretation: mean/median/IQR/range summarize all predefined subsets at each labeled Batch9 budget.",
        "",
        "- `table_raw_provenance_v3.csv` / `table_raw_provenance_v3.md`",
        "  - Sources: raw v3 selected features, feature audit, model results, and summary JSON.",
        "  - Safe interpretation: raw feature family, forbidden-input audit, and Batch9-selected diagnostic summary.",
        "",
        "## Key Numerical Snapshot",
        "",
        "- Strict main result: Batch10 RMSE/MAE = 56.19/49.42 cycles, selected by Batch9 only.",
        "- Source-only comparator: Batch10 RMSE/MAE = 64.27/55.21 cycles.",
        "- Target-label budget mean RMSE: k=0/1/2/4/8 gives 64.27/61.17/60.55/59.64/56.19 cycles.",
        "- Raw-provenance v3: corr_capacity_top20 + SVR-RBF gives Batch10 RMSE/MAE = 74.12/63.16 cycles.",
        "- Neural v1 prototype: Batch10 RMSE/MAE = 68.68/60.30 cycles.",
        "- Residual calibration negative result: Batch10 RMSE/MAE = 100.26/92.90 cycles.",
        "- Conformal uncertainty diagnostic: finite 80% half-width = 129.42 cycles with 8/8 Batch10 coverage; 90% and 95% standard intervals are unbounded with n_cal=8.",
        "",
        "## Main Table Snapshot",
        "",
    ]
    lines.extend(main_table.to_string(index=False).splitlines())
    lines.extend(
        [
            "",
            "## Remaining Limits",
            "",
            "- Batch10 contains only eight cells.",
            "- The main strict result uses historical processed capacity/SOH-like features and should be worded as measured/estimated historical capacity-SOH information.",
            "- Raw v3 uses direct raw pkl features and helps provenance defense, but its Batch10 error is higher than the strict main baseline.",
            "- Uncertainty evidence is limited to diagnostic coverage/interval reporting, not strong calibrated uncertainty.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_outputs() -> None:
    outputs = []
    for base in FIGURE_BASES.values():
        outputs.extend([base.with_suffix(ext) for ext in [".png", ".svg", ".pdf"]])
    outputs.extend(TABLES.values())
    outputs.append(REPORT_PATH)
    missing = [str(path) for path in outputs if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError("Missing or empty outputs:\n" + "\n".join(missing))


def main() -> None:
    setup_style()
    data = read_inputs()

    draw_protocol_flowchart()
    draw_batch9_candidate_heatmap(data["strict_candidate"])
    draw_batch10_pred_vs_true(data["robust_per_cell"], data["robust_summary"])
    draw_batch10_abs_error(data["robust_per_cell"], data["robust_summary"])
    draw_budget_sensitivity(data["budget_aggregate"], data["budget_subset"])
    draw_raw_provenance_summary(data["raw_selected_features"], data["raw_feature_audit"], data["raw_model_results"])

    main_table = build_main_results_table(data)
    budget_table = build_budget_table(data["budget_aggregate"])
    raw_table = build_raw_provenance_table(
        data["raw_summary"],
        data["raw_selected_features"],
        data["raw_feature_audit"],
        data["raw_model_results"],
    )

    draw_evidence_hierarchy(main_table)

    save_table(main_table, TABLES["main_csv"], TABLES["main_md"])
    save_table(budget_table, TABLES["budget_csv"], TABLES["budget_md"])
    save_table(raw_table, TABLES["raw_csv"], TABLES["raw_md"])
    write_report(data, main_table, budget_table, raw_table)

    validate_outputs()
    figure_count = len(FIGURE_BASES) * 3
    table_count = len(TABLES)
    print(f"{RUN_NAME}: generated {figure_count} figure files, {table_count} table files, and report in {SCRIPT_DIR}")


if __name__ == "__main__":
    main()
