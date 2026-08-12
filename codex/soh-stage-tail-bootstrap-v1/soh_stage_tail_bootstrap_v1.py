from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WORKSPACE = Path(r"C:\Users\ROG\Desktop\paper")
DEFAULT_OUTPUT_DIR = WORKSPACE / "codex" / "soh-stage-tail-bootstrap-v1"
DEFAULT_PREDICTION_WORKBOOK = (
    WORKSPACE
    / "codex"
    / "visualization-update-20260420"
    / "teal_ci_visualization_summary_20260420.xlsx"
)
DEFAULT_PREDICTION_CSV = (
    WORKSPACE
    / "codex"
    / "paper-update-20260420"
    / "target_test_predictions_all_models_20260420.csv"
)

MODELS = [
    "final-retained-v1",
    "no-physical-base-v1",
    "no-mmd-v1",
    "no-attention-v1",
]
ABLATION_MODELS = [m for m in MODELS if m != "final-retained-v1"]
MODEL_COLORS = {
    "final-retained-v1": "#007C89",
    "no-physical-base-v1": "#CC6677",
    "no-mmd-v1": "#DDCC77",
    "no-attention-v1": "#88CCEE",
}
MODEL_LABELS = {
    "final-retained-v1": "Final retained",
    "no-physical-base-v1": "No physical base",
    "no-mmd-v1": "No MMD",
    "no-attention-v1": "No attention",
}

STAGE_LABELS = ["early", "mid", "late"]
STAGE_BOUNDS = {
    "early": "[0, 1/3]",
    "mid": "(1/3, 2/3]",
    "late": "(2/3, 1]",
}
SOH_BIN_EDGES = [-np.inf, 0.85, 0.90, 0.95, 1.00, np.inf]
SOH_BIN_LABELS = [
    "soh_le_0.85",
    "0.85_lt_soh_le_0.90",
    "0.90_lt_soh_le_0.95",
    "0.95_lt_soh_le_1.00",
    "soh_gt_1.00",
]
SOH_BIN_DESCRIPTIONS = {
    "soh_le_0.85": "y_true <= 0.85",
    "0.85_lt_soh_le_0.90": "0.85 < y_true <= 0.90",
    "0.90_lt_soh_le_0.95": "0.90 < y_true <= 0.95",
    "0.95_lt_soh_le_1.00": "0.95 < y_true <= 1.00",
    "soh_gt_1.00": "y_true > 1.00",
}

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260621
TAIL_POSITION_THRESHOLD = 0.80
ROLLING_WINDOW_CYCLES = 21
SLOPE_EPS = 1e-6


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_pred - y_true))))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def signed_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred - y_true))


def mape_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.where(np.abs(y_true) < 1e-12, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_pred - y_true) / denominator)) * 100.0)


def metric_row(
    frame: pd.DataFrame,
    model: str,
    extra: Dict[str, object] | None = None,
) -> Dict[str, object]:
    y_true = frame["y_true"].to_numpy(dtype=float)
    y_pred = frame[model].to_numpy(dtype=float)
    err = y_pred - y_true
    row: Dict[str, object] = {
        "model": model,
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "signed_bias_pred_minus_true": signed_bias(y_true, y_pred),
        "max_abs_error": float(np.max(np.abs(err))) if len(err) else np.nan,
        "mape_percent": mape_percent(y_true, y_pred),
        "sample_count": int(len(frame)),
        "cell_count": int(frame["battery_index"].nunique()),
    }
    if extra:
        row.update(extra)
    return row


def read_predictions(prediction_workbook: Path, prediction_csv: Path) -> pd.DataFrame:
    if prediction_workbook.exists():
        xl = pd.ExcelFile(prediction_workbook)
        if "target_test_predictions" not in xl.sheet_names:
            raise ValueError(f"target_test_predictions sheet missing in {prediction_workbook}")
        df = pd.read_excel(prediction_workbook, sheet_name="target_test_predictions")
        source_path = prediction_workbook
    elif prediction_csv.exists():
        df = pd.read_csv(prediction_csv)
        source_path = prediction_csv
    else:
        raise FileNotFoundError(
            f"Could not find prediction workbook {prediction_workbook} or CSV {prediction_csv}"
        )

    required = {"sample_index", "battery_index", "batch", "cycle", "y_true", *MODELS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required prediction columns: {missing}")

    df = df.copy()
    for col in ["sample_index", "battery_index", "batch", "cycle", "y_true", *MODELS]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    df = df.sort_values(["battery_index", "cycle", "sample_index"]).reset_index(drop=True)
    df.attrs["prediction_source_path"] = str(source_path)
    return df


def add_cycle_position_and_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cell_min = out.groupby("battery_index")["cycle"].transform("min")
    cell_max = out.groupby("battery_index")["cycle"].transform("max")
    denom = (cell_max - cell_min).replace(0, np.nan)
    out["cycle_position_norm"] = ((out["cycle"] - cell_min) / denom).fillna(1.0)
    out["cycle_position_norm"] = out["cycle_position_norm"].clip(lower=0.0, upper=1.0)
    out["cycle_stage"] = pd.cut(
        out["cycle_position_norm"],
        bins=[-1e-12, 1.0 / 3.0, 2.0 / 3.0, 1.0],
        labels=STAGE_LABELS,
        include_lowest=True,
        right=True,
    )
    out["soh_bin"] = pd.cut(
        out["y_true"],
        bins=SOH_BIN_EDGES,
        labels=SOH_BIN_LABELS,
        include_lowest=True,
        right=True,
    )
    out["tail_last_20pct"] = out["cycle_position_norm"] > TAIL_POSITION_THRESHOLD
    return out


def load_json_provenance(workspace: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for model in MODELS:
        path = workspace / "codex" / model / f"{model}_best_metrics.json"
        if not path.exists():
            rows.append({"model": model, "json_path": str(path), "json_found": False})
            continue
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if "summary" in data:
            summary = data["summary"]
            rows.append(
                {
                    "model": model,
                    "json_path": str(path),
                    "json_found": True,
                    "protocol": summary.get("protocol"),
                    "data_protocol": summary.get("data_protocol"),
                    "best_epoch_selected_by_target_adapt": summary.get(
                        "best_epoch_selected_by_target_adapt"
                    ),
                    "best_target_adapt_rmse": summary.get("best_target_adapt_rmse"),
                    "json_final_target_test_rmse": summary.get("final_target_test_rmse"),
                    "seed": summary.get("seed"),
                }
            )
        else:
            config = data.get("config", {})
            target_test_metrics = [
                m for m in data.get("metrics", []) if m.get("split") == "target_test"
            ]
            target_test = target_test_metrics[0] if target_test_metrics else {}
            rows.append(
                {
                    "model": model,
                    "json_path": str(path),
                    "json_found": True,
                    "protocol": config.get("protocol"),
                    "data_protocol": config.get("data_protocol"),
                    "best_epoch_selected_by_target_adapt": config.get(
                        "best_epoch_selected_by_target_adapt"
                    ),
                    "best_target_adapt_rmse": config.get("best_target_adapt_rmse"),
                    "json_final_target_test_rmse": target_test.get(
                        "pooled_rmse", config.get("final_target_test_rmse")
                    ),
                    "seed": config.get("seed"),
                }
            )
    return pd.DataFrame(rows)


def compute_per_cell_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for model in MODELS:
        for cell, group in df.groupby("battery_index", sort=True):
            row = metric_row(group, model)
            row.update(
                {
                    "battery_index": int(cell),
                    "cycle_min": float(group["cycle"].min()),
                    "cycle_max": float(group["cycle"].max()),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def compute_overall_metrics(
    df: pd.DataFrame, per_cell_metrics: pd.DataFrame, provenance: pd.DataFrame
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for model in MODELS:
        row = metric_row(
            df,
            model,
            extra={
                "batch_values": ",".join(str(int(x)) for x in sorted(df["batch"].unique())),
                "cycle_min": float(df["cycle"].min()),
                "cycle_max": float(df["cycle"].max()),
                "y_true_min": float(df["y_true"].min()),
                "y_true_max": float(df["y_true"].max()),
            },
        )
        cells = per_cell_metrics[per_cell_metrics["model"] == model]
        row["macro_rmse_mean_of_cell_rmse"] = float(cells["rmse"].mean())
        row["macro_mae_mean_of_cell_mae"] = float(cells["mae"].mean())
        rows.append(row)
    overall = pd.DataFrame(rows)
    if not provenance.empty:
        keep = [
            "model",
            "best_epoch_selected_by_target_adapt",
            "best_target_adapt_rmse",
            "json_final_target_test_rmse",
            "protocol",
            "data_protocol",
            "seed",
            "json_path",
        ]
        present = [c for c in keep if c in provenance.columns]
        overall = overall.merge(provenance[present], on="model", how="left")
        overall["recomputed_minus_json_target_test_rmse"] = (
            overall["rmse"] - overall["json_final_target_test_rmse"]
        )
    return overall


def compute_stage_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for model in MODELS:
        for stage in STAGE_LABELS:
            group = df[df["cycle_stage"] == stage]
            if group.empty:
                continue
            row = metric_row(
                group,
                model,
                extra={
                    "cycle_stage": stage,
                    "stage_bounds": STAGE_BOUNDS[stage],
                    "cycle_position_min": float(group["cycle_position_norm"].min()),
                    "cycle_position_max": float(group["cycle_position_norm"].max()),
                },
            )
            rows.append(row)
    return pd.DataFrame(rows)[
        [
            "model",
            "cycle_stage",
            "stage_bounds",
            "rmse",
            "mae",
            "signed_bias_pred_minus_true",
            "max_abs_error",
            "mape_percent",
            "sample_count",
            "cell_count",
            "cycle_position_min",
            "cycle_position_max",
        ]
    ]


def compute_soh_bin_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for model in MODELS:
        for bin_label in SOH_BIN_LABELS:
            group = df[df["soh_bin"] == bin_label]
            if group.empty:
                continue
            row = metric_row(
                group,
                model,
                extra={
                    "soh_bin": bin_label,
                    "soh_bin_definition": SOH_BIN_DESCRIPTIONS[bin_label],
                    "y_true_min_in_bin": float(group["y_true"].min()),
                    "y_true_max_in_bin": float(group["y_true"].max()),
                },
            )
            rows.append(row)
    return pd.DataFrame(rows)[
        [
            "model",
            "soh_bin",
            "soh_bin_definition",
            "rmse",
            "mae",
            "signed_bias_pred_minus_true",
            "max_abs_error",
            "mape_percent",
            "sample_count",
            "cell_count",
            "y_true_min_in_bin",
            "y_true_max_in_bin",
        ]
    ]


def compute_tail_metrics(df: pd.DataFrame) -> pd.DataFrame:
    tail = df[df["tail_last_20pct"]].copy()
    rows: List[Dict[str, object]] = []
    for model in MODELS:
        positive_count = 0
        slope_pairs = 0
        mean_positive_slope_values: List[float] = []
        for _, cell_group in df.groupby("battery_index", sort=True):
            cell_group = cell_group.sort_values("cycle").copy()
            cell_group["smoothed_pred"] = (
                cell_group[model]
                .rolling(window=ROLLING_WINDOW_CYCLES, min_periods=1)
                .mean()
            )
            cell_tail = cell_group[cell_group["tail_last_20pct"]].copy()
            diffs = cell_tail["smoothed_pred"].diff().dropna()
            positives = diffs[diffs > SLOPE_EPS]
            positive_count += int(len(positives))
            slope_pairs += int(len(diffs))
            if len(positives):
                mean_positive_slope_values.extend(positives.tolist())
        row = metric_row(
            tail,
            model,
            extra={
                "tail_definition": f"cycle_position_norm > {TAIL_POSITION_THRESHOLD:.2f}",
                "tail_cycle_position_threshold": TAIL_POSITION_THRESHOLD,
                "rolling_window_cycles": ROLLING_WINDOW_CYCLES,
                "rolling_definition": (
                    "per-cell trailing rolling mean of predictions, sorted by cycle, "
                    "window=21 cycles, min_periods=1"
                ),
                "positive_slope_epsilon": SLOPE_EPS,
                "slope_pair_count": slope_pairs,
                "monotonic_violation_count": positive_count,
                "positive_predicted_slope_fraction": (
                    float(positive_count / slope_pairs) if slope_pairs else np.nan
                ),
                "mean_positive_smoothed_slope": (
                    float(np.mean(mean_positive_slope_values))
                    if mean_positive_slope_values
                    else 0.0
                ),
                "cycle_position_min": float(tail["cycle_position_norm"].min()),
                "cycle_position_max": float(tail["cycle_position_norm"].max()),
            },
        )
        rows.append(row)
    return pd.DataFrame(rows)[
        [
            "model",
            "tail_definition",
            "rmse",
            "mae",
            "signed_bias_pred_minus_true",
            "max_abs_error",
            "mape_percent",
            "sample_count",
            "cell_count",
            "rolling_window_cycles",
            "rolling_definition",
            "positive_slope_epsilon",
            "slope_pair_count",
            "monotonic_violation_count",
            "positive_predicted_slope_fraction",
            "mean_positive_smoothed_slope",
            "cycle_position_min",
            "cycle_position_max",
        ]
    ]


def compute_bootstrap_ci(per_cell_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    cells = sorted(per_cell_metrics["battery_index"].unique())
    n_cells = len(cells)
    indices = rng.integers(0, n_cells, size=(BOOTSTRAP_RESAMPLES, n_cells))
    for model in MODELS:
        cells_frame = (
            per_cell_metrics[per_cell_metrics["model"] == model]
            .sort_values("battery_index")
            .reset_index(drop=True)
        )
        rmse_values = cells_frame["rmse"].to_numpy(dtype=float)
        mae_values = cells_frame["mae"].to_numpy(dtype=float)
        rmse_resampled = rmse_values[indices].mean(axis=1)
        mae_resampled = mae_values[indices].mean(axis=1)
        rows.append(
            {
                "model": model,
                "n_cells": n_cells,
                "n_resamples": BOOTSTRAP_RESAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "observed_macro_rmse": float(rmse_values.mean()),
                "bootstrap_mean_macro_rmse": float(rmse_resampled.mean()),
                "macro_rmse_p2_5": float(np.percentile(rmse_resampled, 2.5)),
                "macro_rmse_p97_5": float(np.percentile(rmse_resampled, 97.5)),
                "observed_macro_mae": float(mae_values.mean()),
                "bootstrap_mean_macro_mae": float(mae_resampled.mean()),
                "macro_mae_p2_5": float(np.percentile(mae_resampled, 2.5)),
                "macro_mae_p97_5": float(np.percentile(mae_resampled, 97.5)),
                "interval_note": (
                    "Empirical cell-level bootstrap over Batch 10 cells; "
                    "not calibrated predictive uncertainty."
                ),
            }
        )
    return pd.DataFrame(rows)


def rows_for_delta(
    source: pd.DataFrame,
    diagnostic_group: str,
    subset_columns: Iterable[str],
    metric_columns: Iterable[str],
) -> List[Dict[str, object]]:
    subset_columns = list(subset_columns)
    metric_columns = list(metric_columns)
    rows: List[Dict[str, object]] = []
    if subset_columns:
        grouped = source.groupby(subset_columns, dropna=False, sort=False)
        groups = grouped
    else:
        groups = [((), source)]

    for subset_key, group in groups:
        if not isinstance(subset_key, tuple):
            subset_key = (subset_key,)
        subset_dict = dict(zip(subset_columns, subset_key))
        subset_label = "overall"
        if subset_dict:
            subset_label = " | ".join(f"{k}={v}" for k, v in subset_dict.items())
        final_rows = group[group["model"] == "final-retained-v1"]
        if final_rows.empty:
            continue
        final_row = final_rows.iloc[0]
        for model in ABLATION_MODELS:
            model_rows = group[group["model"] == model]
            if model_rows.empty:
                continue
            model_row = model_rows.iloc[0]
            for metric in metric_columns:
                if metric not in group.columns:
                    continue
                value = float(model_row[metric])
                final_value = float(final_row[metric])
                delta = value - final_value
                rows.append(
                    {
                        "diagnostic_group": diagnostic_group,
                        "subset": subset_label,
                        **subset_dict,
                        "model": model,
                        "metric": metric,
                        "model_value": value,
                        "final_value": final_value,
                        "delta_model_minus_final": delta,
                        "percent_delta_vs_final": (
                            float(delta / final_value * 100.0)
                            if abs(final_value) > 1e-12
                            else np.nan
                        ),
                    }
                )
    return rows


def compute_final_vs_ablations_deltas(
    overall: pd.DataFrame,
    stage: pd.DataFrame,
    soh_bin: pd.DataFrame,
    tail: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    rows += rows_for_delta(
        overall,
        "overall_metrics_recomputed",
        [],
        [
            "rmse",
            "macro_rmse_mean_of_cell_rmse",
            "mae",
            "macro_mae_mean_of_cell_mae",
            "signed_bias_pred_minus_true",
            "max_abs_error",
            "mape_percent",
        ],
    )
    rows += rows_for_delta(
        stage,
        "cycle_stage_metrics",
        ["cycle_stage"],
        ["rmse", "mae", "signed_bias_pred_minus_true", "max_abs_error"],
    )
    rows += rows_for_delta(
        soh_bin,
        "soh_bin_metrics",
        ["soh_bin"],
        ["rmse", "mae", "signed_bias_pred_minus_true", "max_abs_error"],
    )
    rows += rows_for_delta(
        tail,
        "tail_metrics",
        [],
        [
            "rmse",
            "mae",
            "signed_bias_pred_minus_true",
            "max_abs_error",
            "positive_predicted_slope_fraction",
            "monotonic_violation_count",
        ],
    )
    rows += rows_for_delta(
        bootstrap,
        "cell_bootstrap_ci",
        [],
        [
            "observed_macro_rmse",
            "bootstrap_mean_macro_rmse",
            "observed_macro_mae",
            "bootstrap_mean_macro_mae",
        ],
    )
    return pd.DataFrame(rows)


def definition_sheet(prediction_source: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item": "prediction_source",
                "definition": prediction_source,
            },
            {
                "item": "protocol_scope",
                "definition": (
                    "No-retraining diagnostic over saved Batch 10 target_test SOH "
                    "predictions only; Batch 10 is not used for tuning or selection."
                ),
            },
            {
                "item": "source_domain",
                "definition": "Batch 1-8 are the source-domain training batches in processed_clean_fixed_split.",
            },
            {
                "item": "target_development",
                "definition": "Batch 9 is the target-domain development/adaptation batch used by the upstream model runs.",
            },
            {
                "item": "blind_test",
                "definition": "Batch 10 is the held-out blind-test batch; this diagnostic only reports saved Batch 10 predictions.",
            },
            {
                "item": "cycle_position_norm",
                "definition": (
                    "(cycle - per-cell min cycle) / (per-cell max cycle - per-cell min cycle)."
                ),
            },
            {
                "item": "cycle_stage_bins",
                "definition": "early=[0,1/3], mid=(1/3,2/3], late=(2/3,1].",
            },
            {
                "item": "soh_bins",
                "definition": "; ".join(SOH_BIN_DESCRIPTIONS.values()),
            },
            {
                "item": "signed_bias",
                "definition": "mean(prediction - y_true); positive means overprediction.",
            },
            {
                "item": "tail_last_20pct",
                "definition": f"cycle_position_norm > {TAIL_POSITION_THRESHOLD:.2f}.",
            },
            {
                "item": "tail_monotonicity",
                "definition": (
                    "Per-cell predictions are smoothed with a trailing rolling mean "
                    f"(window={ROLLING_WINDOW_CYCLES}, min_periods=1). Adjacent "
                    f"smoothed increases > {SLOPE_EPS:g} inside the tail count as "
                    "monotonic violations."
                ),
            },
            {
                "item": "bootstrap_ci",
                "definition": (
                    f"{BOOTSTRAP_RESAMPLES} cell-level bootstrap resamples over the "
                    "8 Batch 10 cells. Intervals are empirical resampling summaries, "
                    "not calibrated predictive uncertainty."
                ),
            },
            {
                "item": "mmd_interpretation",
                "definition": (
                    "MMD is treated as a lightweight cross-domain enhancement; this "
                    "diagnostic should not be used to overclaim it."
                ),
            },
            {
                "item": "cubic_base_interpretation",
                "definition": (
                    "The cubic base is treated as a low-order nonlinear degradation "
                    "prior, not as a strong electrochemical mechanism."
                ),
            },
        ]
    )


def _annotate_bars(ax: plt.Axes, fmt: str = "{:.4f}") -> None:
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    for patch in ax.patches:
        height = patch.get_height()
        if not np.isfinite(height):
            continue
        if height >= 0:
            y = height + 0.02 * span
            va = "bottom"
        else:
            y = height - 0.04 * span
            va = "top"
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            y,
            fmt.format(height),
            ha="center",
            va=va,
            fontsize=7,
            rotation=90,
        )


def make_figures(stage: pd.DataFrame, tail: pd.DataFrame, output_dir: Path) -> List[Path]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )

    saved: List[Path] = []

    pivot = (
        stage.pivot(index="cycle_stage", columns="model", values="rmse")
        .reindex(STAGE_LABELS)
        .loc[:, MODELS]
    )
    x = np.arange(len(STAGE_LABELS))
    width = 0.18
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, model in enumerate(MODELS):
        offsets = x + (i - (len(MODELS) - 1) / 2) * width
        ax.bar(
            offsets,
            pivot[model].to_numpy(dtype=float),
            width=width,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["Early", "Mid", "Late"])
    ax.set_ylabel("RMSE (SOH fraction)")
    ax.set_title("Batch 10 SOH RMSE by Normalized Cycle Stage", fontsize=12, pad=12)
    ax.set_ylim(0, float(np.nanmax(pivot.to_numpy(dtype=float))) * 1.18)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    out = figures_dir / "stage_wise_rmse_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    saved.append(out)

    late_rmse = (
        stage[stage["cycle_stage"] == "late"]
        .set_index("model")
        .loc[MODELS, "rmse"]
        .to_numpy(dtype=float)
    )
    tail_bias = (
        tail.set_index("model")
        .loc[MODELS, "signed_bias_pred_minus_true"]
        .to_numpy(dtype=float)
    )
    slope_frac = (
        tail.set_index("model")
        .loc[MODELS, "positive_predicted_slope_fraction"]
        .to_numpy(dtype=float)
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.0), sharey=True)
    labels = [MODEL_LABELS[m] for m in MODELS]
    y = np.arange(len(MODELS))
    colors = [MODEL_COLORS[m] for m in MODELS]

    axes[0].barh(y, tail_bias, color=colors)
    axes[0].axvline(0, color="#444444", linewidth=0.8)
    axes[0].set_title("Tail Signed Bias", fontsize=11)
    axes[0].set_xlabel("Mean pred - true")

    axes[1].barh(y, late_rmse, color=colors)
    axes[1].set_title("Late-Stage RMSE", fontsize=11)
    axes[1].set_xlabel("RMSE")

    axes[2].barh(y, slope_frac, color=colors)
    axes[2].set_title("Tail Positive-Slope Fraction", fontsize=11)
    axes[2].set_xlabel("Fraction")
    axes[2].set_xlim(0, max(0.05, float(np.nanmax(slope_frac)) * 1.25))

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()

    fig.suptitle("Batch 10 Tail and Late-Stage Diagnostics", y=1.01, fontsize=12)
    fig.tight_layout()
    out = figures_dir / "tail_bias_late_stage_diagnostics.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    saved.append(out)

    return saved


def format_float(value: float, digits: int = 6) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def markdown_table(frame: pd.DataFrame, columns: List[str], float_digits: int = 6) -> str:
    view = frame.loc[:, columns].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: format_float(x, float_digits))
    return view.to_markdown(index=False)


def write_report(
    output_dir: Path,
    prediction_source: str,
    overall: pd.DataFrame,
    stage: pd.DataFrame,
    soh_bin: pd.DataFrame,
    tail: pd.DataFrame,
    bootstrap: pd.DataFrame,
    deltas: pd.DataFrame,
    figure_paths: List[Path],
) -> Path:
    report_path = output_dir / "soh_stage_tail_bootstrap_v1_report.md"

    overall_cols = [
        "model",
        "rmse",
        "macro_rmse_mean_of_cell_rmse",
        "mae",
        "signed_bias_pred_minus_true",
        "sample_count",
        "cell_count",
    ]
    stage_rmse = (
        stage.pivot(index="model", columns="cycle_stage", values="rmse")
        .reindex(MODELS)
        .reset_index()
        .loc[:, ["model", "early", "mid", "late"]]
    )
    tail_cols = [
        "model",
        "rmse",
        "mae",
        "signed_bias_pred_minus_true",
        "max_abs_error",
        "monotonic_violation_count",
        "positive_predicted_slope_fraction",
    ]
    bootstrap_cols = [
        "model",
        "observed_macro_rmse",
        "macro_rmse_p2_5",
        "macro_rmse_p97_5",
        "observed_macro_mae",
        "macro_mae_p2_5",
        "macro_mae_p97_5",
    ]

    final_overall = overall.set_index("model").loc["final-retained-v1"]
    no_phys_overall = overall.set_index("model").loc["no-physical-base-v1"]
    no_mmd_overall = overall.set_index("model").loc["no-mmd-v1"]
    no_attention_overall = overall.set_index("model").loc["no-attention-v1"]
    final_tail = tail.set_index("model").loc["final-retained-v1"]
    no_phys_tail = tail.set_index("model").loc["no-physical-base-v1"]
    final_boot = bootstrap.set_index("model").loc["final-retained-v1"]

    no_phys_delta_rmse = no_phys_overall["rmse"] - final_overall["rmse"]
    no_mmd_delta_rmse = no_mmd_overall["rmse"] - final_overall["rmse"]
    no_attention_delta_rmse = no_attention_overall["rmse"] - final_overall["rmse"]
    no_phys_tail_delta = no_phys_tail["rmse"] - final_tail["rmse"]

    lines = [
        "# SOH Stage/Tail Bootstrap Diagnostic v1",
        "",
        "## Scope and Protocol",
        "",
        (
            "This package is a no-retraining diagnostic over saved Batch 10 "
            "target_test SOH predictions. It recomputes metrics from "
            f"`{prediction_source}` and reads existing JSON metrics only as run "
            "provenance checks."
        ),
        "",
        (
            "Protocol compliance: source-domain training remains Batch 1-8, "
            "target-domain development/adaptation remains Batch 9, and Batch 10 "
            "is used here only for blind-test reporting. No model selection, "
            "hyperparameter tuning, checkpoint selection, retraining, RUL, or EOL "
            "analysis is performed in this diagnostic."
        ),
        "",
        "## Diagnostic Definitions",
        "",
        (
            "Cycle stage uses per-cell min-max normalized cycle position: "
            "`(cycle - min_cycle) / (max_cycle - min_cycle)`. Stage bins are "
            "`early=[0,1/3]`, `mid=(1/3,2/3]`, and `late=(2/3,1]`."
        ),
        "",
        (
            "SOH bins are fixed before analysis as: `y_true <= 0.85`, "
            "`0.85 < y_true <= 0.90`, `0.90 < y_true <= 0.95`, "
            "`0.95 < y_true <= 1.00`, and `y_true > 1.00`."
        ),
        "",
        (
            "Tail cycles are defined as the last 20% per cell "
            "(`cycle_position_norm > 0.80`). Tail monotonicity uses a per-cell "
            f"trailing rolling mean of predictions with window={ROLLING_WINDOW_CYCLES} "
            "cycles and `min_periods=1`; adjacent smoothed increases greater than "
            f"{SLOPE_EPS:g} count as monotonic violations. Signed bias is "
            "`mean(prediction - y_true)`, so positive values indicate overprediction."
        ),
        "",
        (
            "The cell-level bootstrap uses 10000 resamples over the 8 Batch 10 "
            "cells. These are empirical resampling intervals for macro metrics, "
            "not calibrated predictive uncertainty."
        ),
        "",
        "## Key Numbers",
        "",
        markdown_table(overall, overall_cols, 6),
        "",
        (
            "The final retained model has pooled Batch 10 RMSE "
            f"{format_float(final_overall['rmse'])}, macro RMSE "
            f"{format_float(final_overall['macro_rmse_mean_of_cell_rmse'])}, "
            f"and MAE {format_float(final_overall['mae'])}. Removing the "
            f"low-order nonlinear degradation prior increases pooled RMSE by "
            f"{format_float(no_phys_delta_rmse)}. Removing MMD changes pooled "
            f"RMSE by {format_float(no_mmd_delta_rmse)}, while removing attention "
            f"changes it by {format_float(no_attention_delta_rmse)}."
        ),
        "",
        "## Stage-Wise RMSE",
        "",
        markdown_table(stage_rmse, ["model", "early", "mid", "late"], 6),
        "",
        "## Tail Diagnostics",
        "",
        markdown_table(tail, tail_cols, 6),
        "",
        (
            "In the last 20% of each cell, the final retained model has RMSE "
            f"{format_float(final_tail['rmse'])}, MAE {format_float(final_tail['mae'])}, "
            f"and signed bias {format_float(final_tail['signed_bias_pred_minus_true'])}. "
            f"The no-physical-base ablation increases tail RMSE by "
            f"{format_float(no_phys_tail_delta)}. This supports describing the cubic "
            "base only as a useful low-order nonlinear degradation prior that helps "
            "stabilize the tail; it should not be framed as a strong electrochemical "
            "mechanism."
        ),
        "",
        "## Cell Bootstrap",
        "",
        markdown_table(bootstrap, bootstrap_cols, 6),
        "",
        (
            "For the final retained model, the empirical cell-bootstrap interval is "
            f"{format_float(final_boot['macro_rmse_p2_5'])}-"
            f"{format_float(final_boot['macro_rmse_p97_5'])} for macro RMSE and "
            f"{format_float(final_boot['macro_mae_p2_5'])}-"
            f"{format_float(final_boot['macro_mae_p97_5'])} for macro MAE."
        ),
        "",
        "## Manuscript Implications",
        "",
        (
            "The diagnostic strengthens the SOH-only ablation story in a restrained "
            "way: the final retained model is best overall and the physical-base "
            "removal remains the clearest degradation, including in the tail. The "
            "MMD effect is small and mixed relative to the final model, so it should "
            "be described as a lightweight cross-domain regularizing/enhancement "
            "component rather than a dominant source of improvement. Attention shows "
            "a clearer but secondary contribution."
        ),
        "",
        (
            "Residual bands or intervals derived from these outputs should be called "
            "empirical. The bootstrap summarizes variability across the eight Batch "
            "10 cells and does not establish calibrated predictive uncertainty."
        ),
        "",
        "## Figures",
        "",
    ]
    for path in figure_paths:
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "## Output Tables",
            "",
            "- `overall_metrics_recomputed`: pooled and macro Batch 10 metrics recomputed from saved predictions.",
            "- `cycle_stage_metrics`: early/mid/late metrics by per-cell normalized cycle position.",
            "- `soh_bin_metrics`: fixed y_true SOH-bin metrics.",
            "- `tail_metrics`: last-20%-cycle metrics plus smoothed monotonicity diagnostics.",
            "- `cell_bootstrap_ci`: empirical cell-level bootstrap intervals over the 8 Batch 10 cells.",
            "- `final_vs_ablations_deltas`: ablation deltas relative to the final retained model.",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def autosize_workbook_columns(workbook_path: Path) -> None:
    try:
        from openpyxl import load_workbook
    except ImportError:
        return

    workbook = load_workbook(workbook_path)
    for sheet in workbook.worksheets:
        for column_cells in sheet.columns:
            values = [cell.value for cell in column_cells]
            max_len = max((len(str(v)) for v in values if v is not None), default=0)
            width = min(max(max_len + 2, 10), 60)
            sheet.column_dimensions[column_cells[0].column_letter].width = width
        sheet.freeze_panes = "A2"
    workbook.save(workbook_path)


def write_workbook(
    output_dir: Path,
    overall: pd.DataFrame,
    stage: pd.DataFrame,
    soh_bin: pd.DataFrame,
    tail: pd.DataFrame,
    bootstrap: pd.DataFrame,
    deltas: pd.DataFrame,
    per_cell: pd.DataFrame,
    provenance: pd.DataFrame,
    definitions: pd.DataFrame,
) -> Path:
    workbook_path = output_dir / "soh_stage_tail_bootstrap_v1_summary.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        overall.to_excel(writer, sheet_name="overall_metrics_recomputed", index=False)
        stage.to_excel(writer, sheet_name="cycle_stage_metrics", index=False)
        soh_bin.to_excel(writer, sheet_name="soh_bin_metrics", index=False)
        tail.to_excel(writer, sheet_name="tail_metrics", index=False)
        bootstrap.to_excel(writer, sheet_name="cell_bootstrap_ci", index=False)
        deltas.to_excel(writer, sheet_name="final_vs_ablations_deltas", index=False)
        per_cell.to_excel(writer, sheet_name="per_cell_metrics_recomputed", index=False)
        provenance.to_excel(writer, sheet_name="model_run_provenance", index=False)
        definitions.to_excel(writer, sheet_name="diagnostic_definitions", index=False)
    autosize_workbook_columns(workbook_path)
    return workbook_path


def run(args: argparse.Namespace) -> Dict[str, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_raw = read_predictions(Path(args.prediction_workbook), Path(args.prediction_csv))
    if not set(df_raw["batch"].unique()) == {10}:
        raise ValueError(f"Expected Batch 10 only; found batches {sorted(df_raw['batch'].unique())}")
    if df_raw["battery_index"].nunique() != 8:
        raise ValueError(
            f"Expected 8 Batch 10 cells; found {df_raw['battery_index'].nunique()}"
        )
    df = add_cycle_position_and_bins(df_raw)
    if df[MODELS + ["y_true"]].isna().any().any():
        raise ValueError("Predictions or labels contain NaN values.")

    provenance = load_json_provenance(WORKSPACE)
    per_cell = compute_per_cell_metrics(df)
    overall = compute_overall_metrics(df, per_cell, provenance)
    stage = compute_stage_metrics(df)
    soh_bin = compute_soh_bin_metrics(df)
    tail = compute_tail_metrics(df)
    bootstrap = compute_bootstrap_ci(per_cell)
    deltas = compute_final_vs_ablations_deltas(overall, stage, soh_bin, tail, bootstrap)
    definitions = definition_sheet(df_raw.attrs.get("prediction_source_path", "unknown"))

    figure_paths = make_figures(stage, tail, output_dir)
    workbook_path = write_workbook(
        output_dir,
        overall,
        stage,
        soh_bin,
        tail,
        bootstrap,
        deltas,
        per_cell,
        provenance,
        definitions,
    )
    report_path = write_report(
        output_dir,
        df_raw.attrs.get("prediction_source_path", "unknown"),
        overall,
        stage,
        soh_bin,
        tail,
        bootstrap,
        deltas,
        figure_paths,
    )

    print("Generated diagnostic package:")
    print(f"  workbook: {workbook_path}")
    print(f"  report:   {report_path}")
    for fig in figure_paths:
        print(f"  figure:   {fig}")
    print("\nOverall metrics:")
    print(
        overall[
            [
                "model",
                "rmse",
                "macro_rmse_mean_of_cell_rmse",
                "mae",
                "signed_bias_pred_minus_true",
            ]
        ].to_string(index=False)
    )
    return {
        "workbook": workbook_path,
        "report": report_path,
        **{f"figure_{i+1}": p for i, p in enumerate(figure_paths)},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="No-retraining SOH stage/tail/bootstrap diagnostics for Batch 10 predictions."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for workbook, figures, and report.",
    )
    parser.add_argument(
        "--prediction-workbook",
        default=str(DEFAULT_PREDICTION_WORKBOOK),
        help="Preferred workbook with target_test_predictions sheet.",
    )
    parser.add_argument(
        "--prediction-csv",
        default=str(DEFAULT_PREDICTION_CSV),
        help="Fallback CSV with target-test predictions.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
