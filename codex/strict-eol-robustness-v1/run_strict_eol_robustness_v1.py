from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover - scipy is optional for portability.
    stats = None


RUN_NAME = "strict-eol-robustness-v1"
BOOTSTRAP_SEED = 20260604
BOOTSTRAP_RESAMPLES = 20000

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
INPUT_DIR = PROJECT_ROOT / "codex" / "strict-eol-protocol-v1"
OUTPUT_DIR = SCRIPT_DIR

PREDICTIONS_CSV = INPUT_DIR / "strict-eol-protocol-v1_selected_test_predictions.csv"
CANDIDATES_CSV = INPUT_DIR / "strict-eol-protocol-v1_candidate_validation_summary.csv"

PER_CELL_CSV = OUTPUT_DIR / f"{RUN_NAME}_per_cell.csv"
LOO_CSV = OUTPUT_DIR / f"{RUN_NAME}_leave_one_cell_out.csv"
BOOTSTRAP_CSV = OUTPUT_DIR / f"{RUN_NAME}_bootstrap_metrics.csv"
SUMMARY_JSON = OUTPUT_DIR / f"{RUN_NAME}_summary.json"
REPORT_MD = OUTPUT_DIR / f"{RUN_NAME}_report.md"
SCATTER_PNG = OUTPUT_DIR / f"{RUN_NAME}_predicted_vs_true_eol.png"
ABS_ERROR_PNG = OUTPUT_DIR / f"{RUN_NAME}_per_cell_abs_error.png"
BOOTSTRAP_PNG = OUTPUT_DIR / f"{RUN_NAME}_bootstrap_rmse_distribution.png"


def require_columns(frame: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def rmse(errors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(errors))))


def mae(errors: np.ndarray) -> float:
    return float(np.mean(np.abs(errors)))


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    if stats is not None:
        result = stats.spearmanr(x, y)
        return float(result.statistic)
    x_rank = pd.Series(x).rank(method="average").to_numpy()
    y_rank = pd.Series(y).rank(method="average").to_numpy()
    return pearson_corr(x_rank, y_rank)


def ci(values: np.ndarray) -> dict[str, float]:
    q = np.percentile(values, [2.5, 50.0, 97.5])
    return {"p2_5": float(q[0]), "median": float(q[1]), "p97_5": float(q[2])}


def metric_block(errors: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    abs_errors = np.abs(errors)
    return {
        "n_cells": int(len(errors)),
        "rmse_cycles": rmse(errors),
        "mae_cycles": mae(errors),
        "median_absolute_error_cycles": float(np.median(abs_errors)),
        "max_absolute_error_cycles": float(np.max(abs_errors)),
        "bias_mean_error_cycles": float(np.mean(errors)),
        "pearson_true_pred": pearson_corr(y_true, y_pred),
        "spearman_true_pred": spearman_corr(y_true, y_pred),
    }


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(PREDICTIONS_CSV)
    candidates = pd.read_csv(CANDIDATES_CSV)

    require_columns(
        predictions,
        [
            "cell_id",
            "battery_index",
            "batch_id",
            "horizon",
            "eol_cycle",
            "selection_scope",
            "protocol",
            "model",
            "pred_eol_cycle",
        ],
        PREDICTIONS_CSV,
    )
    require_columns(
        candidates,
        [
            "protocol",
            "model",
            "horizon",
            "selection_set",
            "selection_method",
            "selection_eol_rmse",
            "selection_eol_mae",
        ],
        CANDIDATES_CSV,
    )
    return predictions, candidates


def select_batch9_candidate(candidates: pd.DataFrame) -> pd.Series:
    safe_candidates = candidates.copy()
    safe_candidates["selection_eol_rmse"] = pd.to_numeric(
        safe_candidates["selection_eol_rmse"], errors="coerce"
    )
    safe_candidates["selection_eol_mae"] = pd.to_numeric(
        safe_candidates["selection_eol_mae"], errors="coerce"
    )
    safe_candidates = safe_candidates.dropna(subset=["selection_eol_rmse"])
    if safe_candidates.empty:
        raise ValueError("No finite Batch 9 selection rows found.")
    ranked = safe_candidates.sort_values(
        ["selection_eol_rmse", "selection_eol_mae", "protocol", "model", "horizon"],
        ascending=[True, True, True, True, True],
        kind="mergesort",
    )
    return ranked.iloc[0]


def select_blind_predictions(
    predictions: pd.DataFrame, selected_candidate: pd.Series
) -> pd.DataFrame:
    global_scope = predictions[predictions["selection_scope"] == "global_best_by_batch9"].copy()
    if global_scope.empty:
        mask = (
            (predictions["protocol"] == selected_candidate["protocol"])
            & (predictions["model"] == selected_candidate["model"])
            & (pd.to_numeric(predictions["horizon"], errors="coerce") == int(selected_candidate["horizon"]))
        )
        global_scope = predictions[mask].copy()

    require_columns(global_scope, ["cell_id", "eol_cycle", "pred_eol_cycle"], PREDICTIONS_CSV)
    for col in ["batch_id", "horizon", "eol_cycle", "pred_eol_cycle"]:
        global_scope[col] = pd.to_numeric(global_scope[col], errors="raise")

    expected = {
        "protocol": str(selected_candidate["protocol"]),
        "model": str(selected_candidate["model"]),
        "horizon": int(selected_candidate["horizon"]),
    }
    actual_protocols = set(global_scope["protocol"].astype(str))
    actual_models = set(global_scope["model"].astype(str))
    actual_horizons = set(global_scope["horizon"].astype(int))
    if actual_protocols != {expected["protocol"]}:
        raise ValueError(f"Blind prediction protocol mismatch: {actual_protocols} vs {expected}")
    if actual_models != {expected["model"]}:
        raise ValueError(f"Blind prediction model mismatch: {actual_models} vs {expected}")
    if actual_horizons != {expected["horizon"]}:
        raise ValueError(f"Blind prediction horizon mismatch: {actual_horizons} vs {expected}")
    if set(global_scope["batch_id"].astype(int)) != {10}:
        raise ValueError("Selected blind predictions must contain Batch 10 only.")
    if global_scope["cell_id"].duplicated().any():
        duplicates = global_scope.loc[global_scope["cell_id"].duplicated(), "cell_id"].tolist()
        raise ValueError(f"Duplicate selected blind-test cell rows: {duplicates}")
    if len(global_scope) != 8:
        raise ValueError(f"Expected 8 Batch 10 cells, found {len(global_scope)}.")

    return global_scope.sort_values("cell_id", kind="mergesort").reset_index(drop=True)


def build_per_cell(frame: pd.DataFrame) -> pd.DataFrame:
    per_cell = frame.copy()
    per_cell["true_eol_cycle"] = per_cell["eol_cycle"].astype(float)
    per_cell["pred_eol_cycle"] = per_cell["pred_eol_cycle"].astype(float)
    per_cell["eol_error_cycles"] = per_cell["pred_eol_cycle"] - per_cell["true_eol_cycle"]
    per_cell["abs_eol_error_cycles"] = per_cell["eol_error_cycles"].abs()
    per_cell["observed_horizon_cycles"] = per_cell["horizon"].astype(int)
    per_cell["true_rul_at_horizon_cycles"] = (
        per_cell["true_eol_cycle"] - per_cell["observed_horizon_cycles"]
    )
    per_cell["pred_rul_at_horizon_cycles"] = (
        per_cell["pred_eol_cycle"] - per_cell["observed_horizon_cycles"]
    )
    per_cell["rul_error_at_horizon_cycles"] = (
        per_cell["pred_rul_at_horizon_cycles"] - per_cell["true_rul_at_horizon_cycles"]
    )
    per_cell["abs_rul_error_at_horizon_cycles"] = per_cell[
        "rul_error_at_horizon_cycles"
    ].abs()
    columns = [
        "cell_id",
        "battery_index",
        "batch_id",
        "selection_scope",
        "protocol",
        "model",
        "observed_horizon_cycles",
        "true_eol_cycle",
        "pred_eol_cycle",
        "eol_error_cycles",
        "abs_eol_error_cycles",
        "true_rul_at_horizon_cycles",
        "pred_rul_at_horizon_cycles",
        "rul_error_at_horizon_cycles",
        "abs_rul_error_at_horizon_cycles",
    ]
    return per_cell[columns]


def bootstrap_metrics(errors: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(errors)
    indices = rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))
    sampled = errors[indices]
    return pd.DataFrame(
        {
            "bootstrap_id": np.arange(BOOTSTRAP_RESAMPLES, dtype=int),
            "rmse_cycles": np.sqrt(np.mean(sampled * sampled, axis=1)),
            "mae_cycles": np.mean(np.abs(sampled), axis=1),
        }
    )


def leave_one_cell_out(per_cell: pd.DataFrame, full_metrics: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in per_cell.iterrows():
        subset = per_cell[per_cell["cell_id"] != row["cell_id"]]
        errors = subset["eol_error_cycles"].to_numpy(dtype=float)
        subset_rmse = rmse(errors)
        subset_mae = mae(errors)
        rows.append(
            {
                "left_out_cell_id": row["cell_id"],
                "left_out_abs_error_cycles": float(row["abs_eol_error_cycles"]),
                "n_cells_used": int(len(subset)),
                "rmse_cycles": subset_rmse,
                "mae_cycles": subset_mae,
                "delta_rmse_vs_full_cycles": subset_rmse - full_metrics["rmse_cycles"],
                "delta_mae_vs_full_cycles": subset_mae - full_metrics["mae_cycles"],
            }
        )
    return pd.DataFrame(rows)


def plot_scatter(per_cell: pd.DataFrame, metrics: dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    x = per_cell["true_eol_cycle"].to_numpy(dtype=float)
    y = per_cell["pred_eol_cycle"].to_numpy(dtype=float)
    ax.scatter(x, y, s=64, color="#276fbf", edgecolor="white", linewidth=0.9, zorder=3)
    lower = min(float(x.min()), float(y.min())) - 80.0
    upper = max(float(x.max()), float(y.max())) + 100.0
    ax.plot([lower, upper], [lower, upper], color="#444444", linewidth=1.1, linestyle="--")
    for _, row in per_cell.iterrows():
        x_offset = -36 if float(row["true_eol_cycle"]) > lower + 0.82 * (upper - lower) else 5
        y_offset = -14 if float(row["pred_eol_cycle"]) > lower + 0.82 * (upper - lower) else 5
        ax.annotate(
            row["cell_id"],
            (row["true_eol_cycle"], row["pred_eol_cycle"]),
            textcoords="offset points",
            xytext=(x_offset, y_offset),
            fontsize=8,
        )
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("True EOL cycle")
    ax.set_ylabel("Predicted EOL cycle")
    ax.set_title("Batch 10 blind-test EOL prediction")
    ax.text(
        0.03,
        0.97,
        f"RMSE={metrics['rmse_cycles']:.2f}, MAE={metrics['mae_cycles']:.2f} cycles",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#dddddd", "pad": 5},
    )
    ax.grid(True, color="#e6e6e6", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(SCATTER_PNG, dpi=220)
    plt.close(fig)


def plot_abs_errors(per_cell: pd.DataFrame, metrics: dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    labels = per_cell["cell_id"].tolist()
    values = per_cell["abs_eol_error_cycles"].to_numpy(dtype=float)
    colors = ["#c84c31" if v == values.max() else "#418f63" for v in values]
    ax.bar(labels, values, color=colors, edgecolor="#333333", linewidth=0.5)
    ax.axhline(
        metrics["mae_cycles"],
        color="#333333",
        linestyle="--",
        linewidth=1.1,
        label=f"MAE={metrics['mae_cycles']:.2f}",
    )
    ax.set_xlabel("Batch 10 cell")
    ax.set_ylabel("Absolute EOL/RUL error (cycles)")
    ax.set_title("Per-cell blind-test absolute error")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(ABS_ERROR_PNG, dpi=220)
    plt.close(fig)


def plot_bootstrap(bootstrap: pd.DataFrame, rmse_ci: dict[str, float], full_rmse: float) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.hist(
        bootstrap["rmse_cycles"],
        bins=40,
        color="#5f7fba",
        edgecolor="white",
        alpha=0.9,
    )
    ax.axvline(full_rmse, color="#222222", linewidth=1.3, label=f"Observed RMSE={full_rmse:.2f}")
    ax.axvline(rmse_ci["p2_5"], color="#b33a3a", linewidth=1.1, linestyle="--", label="95% CI")
    ax.axvline(rmse_ci["p97_5"], color="#b33a3a", linewidth=1.1, linestyle="--")
    ax.set_xlabel("Bootstrap RMSE (cycles)")
    ax.set_ylabel("Resamples")
    ax.set_title("Cell-level bootstrap RMSE distribution")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(BOOTSTRAP_PNG, dpi=220)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str], float_digits: int = 2) -> str:
    formatted = frame.loc[:, columns].copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]) or pd.api.types.is_numeric_dtype(
            formatted[col]
        ):
            formatted[col] = formatted[col].map(
                lambda value: f"{float(value):.{float_digits}f}"
            )
    return formatted.to_markdown(index=False)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return None if math.isnan(value) else value
    if isinstance(value, float):
        return None if math.isnan(value) else value
    return value


def write_report(
    generated_at: str,
    selected_candidate: pd.Series,
    candidate_top: pd.DataFrame,
    metrics: dict[str, float],
    bootstrap_summary: dict[str, dict[str, float]],
    per_cell: pd.DataFrame,
    loo: pd.DataFrame,
) -> None:
    max_error_row = per_cell.sort_values("abs_eol_error_cycles", ascending=False).iloc[0]
    loo_rmse_min = float(loo["rmse_cycles"].min())
    loo_rmse_max = float(loo["rmse_cycles"].max())
    loo_mae_min = float(loo["mae_cycles"].min())
    loo_mae_max = float(loo["mae_cycles"].max())

    candidate_columns = [
        "protocol",
        "model",
        "horizon",
        "selection_set",
        "selection_method",
        "selection_eol_rmse",
        "selection_eol_mae",
    ]
    per_cell_columns = [
        "cell_id",
        "true_eol_cycle",
        "pred_eol_cycle",
        "eol_error_cycles",
        "abs_eol_error_cycles",
        "true_rul_at_horizon_cycles",
        "pred_rul_at_horizon_cycles",
    ]
    loo_columns = [
        "left_out_cell_id",
        "left_out_abs_error_cycles",
        "rmse_cycles",
        "mae_cycles",
        "delta_rmse_vs_full_cycles",
    ]

    report = f"""# Strict EOL Robustness V1

Generated: {generated_at}

## Protocol Guard

- Fixed split: Batch 1-8 source, Batch 9 target adaptation/model selection, Batch 10 blind test.
- This script does not train, tune, or reselect a model. It only reports final blind-test statistics for the existing `strict-eol-protocol-v1` predictions.
- The selected candidate is recovered from `strict-eol-protocol-v1_candidate_validation_summary.csv`, which contains Batch 9 validation metrics only.
- Batch 10 predictions are filtered to `selection_scope == global_best_by_batch9`; they are not sorted or used to choose a model, horizon, feature set, or parameter.
- Selected Batch 9 path: `{selected_candidate['protocol']}` + `{selected_candidate['model']}` + {int(selected_candidate['horizon'])} observed cycles, selected by `{selected_candidate['selection_method']}` with Batch 9 RMSE {float(selected_candidate['selection_eol_rmse']):.2f} cycles.

Top Batch 9 validation rows:

{markdown_table(candidate_top, candidate_columns)}

## Main Metrics

Metrics use the eight Batch 10 blind-test cells. At a fixed observed horizon, EOL error and derived RUL error are numerically identical.

| Metric | Value |
|---|---:|
| n cells | {metrics['n_cells']} |
| RMSE | {metrics['rmse_cycles']:.2f} cycles |
| MAE | {metrics['mae_cycles']:.2f} cycles |
| Median absolute error | {metrics['median_absolute_error_cycles']:.2f} cycles |
| Max absolute error | {metrics['max_absolute_error_cycles']:.2f} cycles |
| Bias, mean signed error | {metrics['bias_mean_error_cycles']:.2f} cycles |
| Pearson true-vs-pred | {metrics['pearson_true_pred']:.3f} |
| Spearman true-vs-pred | {metrics['spearman_true_pred']:.3f} |

## Bootstrap CI

Cell-level bootstrap resampling was performed on the eight Batch 10 cells with replacement. This is final-result uncertainty reporting, not model selection.

| Metric | Resamples | 2.5% | Median | 97.5% |
|---|---:|---:|---:|---:|
| RMSE | {BOOTSTRAP_RESAMPLES} | {bootstrap_summary['rmse_cycles']['p2_5']:.2f} | {bootstrap_summary['rmse_cycles']['median']:.2f} | {bootstrap_summary['rmse_cycles']['p97_5']:.2f} |
| MAE | {BOOTSTRAP_RESAMPLES} | {bootstrap_summary['mae_cycles']['p2_5']:.2f} | {bootstrap_summary['mae_cycles']['median']:.2f} | {bootstrap_summary['mae_cycles']['p97_5']:.2f} |

## Per-cell Error

{markdown_table(per_cell, per_cell_columns)}

Largest absolute error: {max_error_row['cell_id']} with {float(max_error_row['abs_eol_error_cycles']):.2f} cycles. The blind-test result is not dominated by a single catastrophic cell, but the small n means each cell has visible influence.

Figures:

- `strict-eol-robustness-v1_predicted_vs_true_eol.png`
- `strict-eol-robustness-v1_per_cell_abs_error.png`
- `strict-eol-robustness-v1_bootstrap_rmse_distribution.png`

## Sensitivity

Leave-one-cell-out recomputation removes one Batch 10 cell at a time after the final prediction table is fixed.

{markdown_table(loo, loo_columns)}

LOO RMSE range: {loo_rmse_min:.2f} to {loo_rmse_max:.2f} cycles. LOO MAE range: {loo_mae_min:.2f} to {loo_mae_max:.2f} cycles.

## Paper-ready Wording

Under the fixed cross-batch protocol, Batch 1-8 were used as the source domain, Batch 9 was used for target-domain development and Batch-9-only model/horizon selection, and Batch 10 was reserved as the blind target test. The Batch-9-selected ExtraTrees baseline using the first 1000 observed cycles achieved an EOL/RUL RMSE of {metrics['rmse_cycles']:.2f} cycles and MAE of {metrics['mae_cycles']:.2f} cycles on the eight Batch 10 cells. Cell-level bootstrap resampling gave 95% confidence intervals of {bootstrap_summary['rmse_cycles']['p2_5']:.2f}-{bootstrap_summary['rmse_cycles']['p97_5']:.2f} cycles for RMSE and {bootstrap_summary['mae_cycles']['p2_5']:.2f}-{bootstrap_summary['mae_cycles']['p97_5']:.2f} cycles for MAE. Leave-one-cell-out analysis yielded RMSE values from {loo_rmse_min:.2f} to {loo_rmse_max:.2f} cycles, indicating that the reported blind-test performance is not explained by a single removed cell.

## Limitations

- Batch 10 contains only eight cells, so the bootstrap interval is wide and should be framed as cell-level test-set uncertainty.
- The bootstrap and LOO analyses do not measure retraining variability; they reuse the already selected final Batch 10 predictions.
- This remains a strict baseline/proof-of-concept, not the final proposed architecture.
- The protocol uses labeled Batch 9 target EOL values for supervised small-target adaptation and should not be described as unsupervised domain adaptation.
- The upstream strict baseline uses processed historical capacity/SOH features, including `y_nominal`; these must be framed as historical measured or estimated capacity/SOH information until a stricter raw-capacity or `dq` variant is promoted.
"""
    REPORT_MD.write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().replace(microsecond=0).isoformat()

    predictions, candidates = read_inputs()
    selected_candidate = select_batch9_candidate(candidates)
    selected_blind = select_blind_predictions(predictions, selected_candidate)
    per_cell = build_per_cell(selected_blind)

    errors = per_cell["eol_error_cycles"].to_numpy(dtype=float)
    y_true = per_cell["true_eol_cycle"].to_numpy(dtype=float)
    y_pred = per_cell["pred_eol_cycle"].to_numpy(dtype=float)
    metrics = metric_block(errors, y_true, y_pred)

    bootstrap = bootstrap_metrics(errors)
    bootstrap_summary = {
        "rmse_cycles": ci(bootstrap["rmse_cycles"].to_numpy(dtype=float)),
        "mae_cycles": ci(bootstrap["mae_cycles"].to_numpy(dtype=float)),
    }
    loo = leave_one_cell_out(per_cell, metrics)

    candidate_top = candidates.sort_values(
        ["selection_eol_rmse", "selection_eol_mae", "protocol", "model", "horizon"],
        ascending=[True, True, True, True, True],
        kind="mergesort",
    ).head(8)

    per_cell.to_csv(PER_CELL_CSV, index=False)
    loo.to_csv(LOO_CSV, index=False)
    bootstrap.to_csv(BOOTSTRAP_CSV, index=False)

    plot_scatter(per_cell, metrics)
    plot_abs_errors(per_cell, metrics)
    plot_bootstrap(bootstrap, bootstrap_summary["rmse_cycles"], metrics["rmse_cycles"])

    max_error_row = per_cell.sort_values("abs_eol_error_cycles", ascending=False).iloc[0]
    summary = {
        "run_name": RUN_NAME,
        "generated_at": generated_at,
        "input_files": {
            "predictions_csv": str(PREDICTIONS_CSV),
            "candidate_validation_summary_csv": str(CANDIDATES_CSV),
        },
        "protocol_guard": {
            "source_batches": "1-8",
            "target_adapt_batch": 9,
            "blind_test_batch": 10,
            "selection_source": "Batch 9 validation only",
            "batch10_use": "final blind-test reporting only",
            "no_training_or_reselection": True,
        },
        "selected_candidate_from_batch9": selected_candidate.to_dict(),
        "candidate_validation_top8_by_batch9": candidate_top.to_dict(orient="records"),
        "blind_test_metrics": metrics,
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "n_resamples": BOOTSTRAP_RESAMPLES,
            "unit": "Batch 10 cell",
            "ci_method": "percentile",
            "metrics": bootstrap_summary,
        },
        "per_cell": {
            "n_cells": int(len(per_cell)),
            "max_abs_error_cell_id": str(max_error_row["cell_id"]),
            "max_abs_error_cycles": float(max_error_row["abs_eol_error_cycles"]),
            "rows": per_cell.to_dict(orient="records"),
        },
        "leave_one_cell_out": {
            "rmse_min_cycles": float(loo["rmse_cycles"].min()),
            "rmse_max_cycles": float(loo["rmse_cycles"].max()),
            "mae_min_cycles": float(loo["mae_cycles"].min()),
            "mae_max_cycles": float(loo["mae_cycles"].max()),
            "rows": loo.to_dict(orient="records"),
        },
        "artifacts": {
            "report_md": str(REPORT_MD),
            "summary_json": str(SUMMARY_JSON),
            "per_cell_csv": str(PER_CELL_CSV),
            "leave_one_cell_out_csv": str(LOO_CSV),
            "bootstrap_metrics_csv": str(BOOTSTRAP_CSV),
            "scatter_png": str(SCATTER_PNG),
            "abs_error_png": str(ABS_ERROR_PNG),
            "bootstrap_rmse_png": str(BOOTSTRAP_PNG),
        },
    }

    SUMMARY_JSON.write_text(
        json.dumps(json_ready(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(
        generated_at=generated_at,
        selected_candidate=selected_candidate,
        candidate_top=candidate_top,
        metrics=metrics,
        bootstrap_summary=bootstrap_summary,
        per_cell=per_cell,
        loo=loo,
    )

    print(f"Wrote {REPORT_MD}")
    print(f"RMSE={metrics['rmse_cycles']:.2f} MAE={metrics['mae_cycles']:.2f}")
    print(
        "RMSE 95% CI="
        f"{bootstrap_summary['rmse_cycles']['p2_5']:.2f}-"
        f"{bootstrap_summary['rmse_cycles']['p97_5']:.2f}"
    )


if __name__ == "__main__":
    main()
