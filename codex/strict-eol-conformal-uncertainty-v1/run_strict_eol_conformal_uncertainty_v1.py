import importlib.util
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error


RUN_NAME = "strict-eol-conformal-uncertainty-v1"
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
STRICT_SCRIPT = ROOT / "codex" / "strict-eol-protocol-v1" / "strict_eol_protocol_v1.py"
STRICT_SELECTED_PREDICTIONS = (
    ROOT
    / "codex"
    / "strict-eol-protocol-v1"
    / "strict-eol-protocol-v1_selected_test_predictions.csv"
)

PROTOCOL = "source_plus_target_adapt_loocv"
MODEL_NAME = "extra_trees"
HORIZON = 1000
NOMINAL_LEVELS = [0.80, 0.90, 0.95]

REPORT_MD = OUT_DIR / f"{RUN_NAME}_report.md"
SUMMARY_JSON = OUT_DIR / f"{RUN_NAME}_summary.json"
BATCH9_RESIDUALS_CSV = OUT_DIR / "batch9_calibration_residuals.csv"
BATCH10_INTERVALS_CSV = OUT_DIR / "batch10_conformal_intervals.csv"


def load_strict_module():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("strict_eol_protocol_v1", STRICT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import strict protocol script: {STRICT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pct_label(level: float) -> str:
    return str(int(round(level * 100)))


def finite_sample_conformal_quantile(abs_residuals: np.ndarray, nominal: float) -> Dict[str, Any]:
    residuals = np.sort(np.asarray(abs_residuals, dtype=float))
    n = int(residuals.size)
    rank = int(math.ceil((n + 1) * nominal - 1e-12))
    if rank <= n:
        return {
            "nominal_coverage": float(nominal),
            "calibration_size": n,
            "order_stat_rank": rank,
            "finite_interval": True,
            "q_abs_error_cycles": float(residuals[rank - 1]),
            "finite_sample_effective_rank_coverage": float(rank / (n + 1)),
            "note": "finite split-conformal interval from Batch 9 calibration residuals",
        }
    return {
        "nominal_coverage": float(nominal),
        "calibration_size": n,
        "order_stat_rank": rank,
        "finite_interval": False,
        "q_abs_error_cycles": None,
        "finite_sample_effective_rank_coverage": 1.0,
        "note": (
            "standard finite-sample split conformal requires rank n+1, so the "
            "interval is unbounded with only 8 calibration cells"
        ),
    }


def build_interval_columns(df: pd.DataFrame, interval_specs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    coverage_rows: Dict[str, Dict[str, Any]] = {}
    for spec in interval_specs:
        label = pct_label(float(spec["nominal_coverage"]))
        finite = bool(spec["finite_interval"])
        q = spec["q_abs_error_cycles"]
        lower_col = f"conformal_{label}_lower"
        upper_col = f"conformal_{label}_upper"
        width_col = f"conformal_{label}_width"
        covered_col = f"conformal_{label}_covered"
        finite_col = f"conformal_{label}_finite_interval"

        if finite:
            q_float = float(q)
            df[lower_col] = df["pred_eol_cycle"] - q_float
            df[upper_col] = df["pred_eol_cycle"] + q_float
            df[width_col] = 2.0 * q_float
            df[covered_col] = df["abs_eol_error"] <= q_float + 1e-12
            df[finite_col] = True
            mean_width: Optional[float] = float(df[width_col].mean())
        else:
            df[lower_col] = -np.inf
            df[upper_col] = np.inf
            df[width_col] = np.inf
            df[covered_col] = True
            df[finite_col] = False
            mean_width = None

        coverage_rows[label] = {
            "nominal_coverage": float(spec["nominal_coverage"]),
            "finite_interval": finite,
            "q_abs_error_cycles": q,
            "empirical_coverage_on_batch10": float(df[covered_col].mean()),
            "covered_cells": int(df[covered_col].sum()),
            "n_batch10_cells": int(len(df)),
            "mean_interval_width_cycles": mean_width,
            "mean_interval_width_note": "finite" if finite else "unbounded/infinite",
            "order_stat_rank": int(spec["order_stat_rank"]),
            "finite_sample_effective_rank_coverage": float(
                spec["finite_sample_effective_rank_coverage"]
            ),
        }
    return coverage_rows


def add_tree_spread_diagnostics(
    df: pd.DataFrame,
    fitted_model,
    eval_x: pd.DataFrame,
    nominal_levels: List[float],
) -> Dict[str, Dict[str, Any]]:
    if not hasattr(fitted_model, "estimators_"):
        return {}

    tree_preds = np.vstack([tree.predict(eval_x.to_numpy()) for tree in fitted_model.estimators_])
    df["tree_pred_mean"] = tree_preds.mean(axis=0)
    df["tree_pred_std"] = tree_preds.std(axis=0)

    summary: Dict[str, Dict[str, Any]] = {}
    for level in nominal_levels:
        label = pct_label(level)
        tail = (1.0 - level) / 2.0
        lower = np.quantile(tree_preds, tail, axis=0)
        upper = np.quantile(tree_preds, 1.0 - tail, axis=0)
        covered = (df["eol_cycle"].to_numpy(dtype=float) >= lower) & (
            df["eol_cycle"].to_numpy(dtype=float) <= upper
        )
        df[f"tree_spread_{label}_lower"] = lower
        df[f"tree_spread_{label}_upper"] = upper
        df[f"tree_spread_{label}_width"] = upper - lower
        df[f"tree_spread_{label}_covered"] = covered
        summary[label] = {
            "nominal_central_tree_fraction": float(level),
            "empirical_coverage_on_batch10": float(np.mean(covered)),
            "covered_cells": int(np.sum(covered)),
            "n_batch10_cells": int(len(df)),
            "mean_interval_width_cycles": float(np.mean(upper - lower)),
            "note": "diagnostic only; tree spread is not conformal calibration",
        }
    return summary


def compare_to_strict_selected_predictions(batch10_df: pd.DataFrame) -> Dict[str, Any]:
    if not STRICT_SELECTED_PREDICTIONS.exists():
        return {"available": False, "note": "strict selected prediction CSV not found"}

    baseline = pd.read_csv(STRICT_SELECTED_PREDICTIONS)
    baseline = baseline[
        (baseline["selection_scope"] == "global_best_by_batch9")
        & (baseline["protocol"] == PROTOCOL)
        & (baseline["model"] == MODEL_NAME)
        & (baseline["horizon"] == HORIZON)
    ].copy()
    if baseline.empty:
        return {"available": False, "note": "matching strict selected predictions not found"}

    merged = batch10_df.merge(
        baseline[["cell_id", "pred_eol_cycle", "eol_cycle"]],
        on="cell_id",
        how="inner",
        suffixes=("_conformal_run", "_strict_baseline"),
    )
    if merged.empty:
        return {"available": False, "note": "no overlapping cells with strict selected predictions"}

    pred_diff = np.abs(
        merged["pred_eol_cycle_conformal_run"].to_numpy(dtype=float)
        - merged["pred_eol_cycle_strict_baseline"].to_numpy(dtype=float)
    )
    eol_diff = np.abs(
        merged["eol_cycle_conformal_run"].to_numpy(dtype=float)
        - merged["eol_cycle_strict_baseline"].to_numpy(dtype=float)
    )
    return {
        "available": True,
        "matched_cells": int(len(merged)),
        "max_abs_prediction_difference_cycles": float(np.max(pred_diff)),
        "max_abs_eol_label_difference_cycles": float(np.max(eol_diff)),
    }


def rounded(value: Any, digits: int = 2) -> str:
    if value is None:
        return "NA"
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def write_report(payload: Dict[str, Any]) -> None:
    calibration = payload["batch9_calibration"]
    batch10 = payload["batch10_metrics"]
    interval_summary = payload["conformal_interval_summary"]
    tree_summary = payload["tree_spread_diagnostic_summary"]
    consistency = payload["strict_baseline_consistency_check"]

    lines = [
        "# Strict EOL Conformal Uncertainty V1",
        "",
        f"- Finished at: {payload['finished_at']}",
        f"- Elapsed seconds: {payload['elapsed_seconds']:.1f}",
        f"- Data dir: `{payload['data_dir']}`",
        f"- Strict baseline script imported read-only: `{STRICT_SCRIPT}`",
        "",
        "## Fixed Setting",
        "",
        f"- Protocol: `{PROTOCOL}`.",
        f"- Model: `{MODEL_NAME}` with the strict baseline ExtraTrees parameters.",
        f"- Observed horizon: {HORIZON} cycles.",
        "- Source: Batch 1-8; calibration/adaptation: Batch 9; blind test: Batch 10.",
        "- Batch 9 leave-one-cell-out predictions are used only as calibration residuals.",
        "- The final point model is trained once on source plus all Batch 9 allowed data and then evaluated on Batch 10.",
        "",
        "## Point Prediction Check",
        "",
        "| Set | Cells | RMSE cycles | MAE cycles |",
        "|---|---:|---:|---:|",
        (
            f"| Batch 9 LOO calibration | {calibration['n_cells']} | "
            f"{calibration['rmse_cycles']:.2f} | {calibration['mae_cycles']:.2f} |"
        ),
        (
            f"| Batch 10 blind test | {batch10['n_cells']} | "
            f"{batch10['rmse_cycles']:.2f} | {batch10['mae_cycles']:.2f} |"
        ),
        "",
    ]

    if consistency.get("available"):
        lines.extend(
            [
                "Strict baseline consistency check:",
                "",
                (
                    f"- Matched Batch 10 cells: {consistency['matched_cells']}; "
                    f"max absolute prediction difference vs strict selected CSV: "
                    f"{consistency['max_abs_prediction_difference_cycles']:.6g} cycles."
                ),
                "",
            ]
        )
    else:
        lines.extend(["Strict baseline consistency check: not available.", ""])

    lines.extend(
        [
            "## Batch 9 Calibration Residuals",
            "",
            "| Cell | EOL | Predicted EOL | Error | Absolute error |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["batch9_residual_rows"]:
        lines.append(
            f"| {row['cell_id']} | {row['eol_cycle']:.0f} | {row['pred_eol_cycle']:.2f} | "
            f"{row['eol_error']:.2f} | {row['abs_eol_error']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Split-Conformal Intervals",
            "",
            "The split-conformal absolute-error radius uses the finite-sample order statistic "
            "`ceil((n_cal + 1) * nominal_coverage)` on the eight Batch 9 LOO residuals. "
            "With only eight calibration cells, 80% is finite at the maximum residual; "
            "90% and 95% require rank 9 and are therefore unbounded under the standard guarantee.",
            "",
            "| Nominal | Calibration rank | Radius cycles | Finite interval | Batch 10 coverage | Mean width cycles |",
            "|---:|---:|---:|---|---:|---:|",
        ]
    )
    for label in ["80", "90", "95"]:
        row = interval_summary[label]
        width = (
            f"{row['mean_interval_width_cycles']:.2f}"
            if row["mean_interval_width_cycles"] is not None
            else "unbounded"
        )
        radius = (
            f"{row['q_abs_error_cycles']:.2f}"
            if row["q_abs_error_cycles"] is not None
            else "unbounded"
        )
        lines.append(
            f"| {label}% | {row['order_stat_rank']} | {radius} | "
            f"{'yes' if row['finite_interval'] else 'no'} | "
            f"{row['empirical_coverage_on_batch10']:.3f} | {width} |"
        )

    lines.extend(
        [
            "",
            "## Batch 10 Per-Cell 80% Conformal Intervals",
            "",
            "| Cell | EOL | Prediction | Abs error | 80% lower | 80% upper | Covered |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["batch10_interval_rows"]:
        lines.append(
            f"| {row['cell_id']} | {row['eol_cycle']:.0f} | {row['pred_eol_cycle']:.2f} | "
            f"{row['abs_eol_error']:.2f} | {row['conformal_80_lower']:.2f} | "
            f"{row['conformal_80_upper']:.2f} | {str(row['conformal_80_covered']).lower()} |"
        )

    lines.extend(
        [
            "",
            "## ExtraTrees Tree-Spread Diagnostic",
            "",
            "Tree-spread intervals are included only as an ensemble-spread diagnostic. "
            "They are not calibrated conformal intervals and should not be used as formal uncertainty evidence.",
            "",
            "| Central tree fraction | Batch 10 coverage | Mean width cycles |",
            "|---:|---:|---:|",
        ]
    )
    for label in ["80", "90", "95"]:
        row = tree_summary[label]
        lines.append(
            f"| {label}% | {row['empirical_coverage_on_batch10']:.3f} | "
            f"{row['mean_interval_width_cycles']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation And Caveats",
            "",
            "- The 80% conformal interval is finite but conservative because the finite-sample rank is 8/9, using the largest Batch 9 residual as the radius.",
            "- The 90% and 95% conformal intervals are mathematically unbounded with only eight calibration cells; any finite max-residual substitute would not carry the standard split-conformal guarantee.",
            "- Batch 10 coverage is reported as a diagnostic only because Batch 10 has eight cells and was not used to tune the model or interval width.",
            "- This run does not replace the selected strict ExtraTrees baseline and does not tune model, horizon, or features.",
            "- The setting remains supervised small labeled target-domain adaptation because labeled Batch 9 EOL values are used.",
            "- `y_nominal`-derived features still require cautious wording as historical measured/estimated capacity-SOH features.",
            "",
            "## Output Files",
            "",
            f"- Script: `{OUT_DIR / 'run_strict_eol_conformal_uncertainty_v1.py'}`",
            f"- Batch 9 calibration residuals: `{BATCH9_RESIDUALS_CSV}`",
            f"- Batch 10 conformal intervals: `{BATCH10_INTERVALS_CSV}`",
            f"- Summary JSON: `{SUMMARY_JSON}`",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    strict = load_strict_module()

    splits = {split: strict.load_split(split) for split in ["source", "target_adapt", "target_test"]}
    frames = {
        split: strict.build_cell_features(x, meta, HORIZON)
        for split, (x, meta) in splits.items()
    }

    model_lookup = {name: model for name, model in strict.model_zoo()}
    selected_model = model_lookup[MODEL_NAME]

    calibration_row, batch9_pred = strict.validate_candidate(
        model_name=MODEL_NAME,
        model=selected_model,
        protocol=PROTOCOL,
        source_df=frames["source"],
        target_adapt_df=frames["target_adapt"],
    )
    batch9_pred["protocol"] = PROTOCOL
    batch9_pred["model"] = MODEL_NAME
    batch9_pred["eol_error"] = batch9_pred["pred_eol_cycle"] - batch9_pred["eol_cycle"]
    batch9_pred["abs_eol_error"] = np.abs(batch9_pred["eol_error"])
    batch9_pred = batch9_pred.sort_values("cell_id").reset_index(drop=True)
    batch9_pred.to_csv(BATCH9_RESIDUALS_CSV, index=False, encoding="utf-8-sig")

    train_df = pd.concat([frames["source"], frames["target_adapt"]], ignore_index=True)
    eval_df = frames["target_test"].copy()
    cols = strict.feature_columns(train_df)
    train_x, train_y, eval_x, eval_y = strict.prepare_train_eval(train_df, eval_df, cols)
    fitted = clone(selected_model)
    fitted.fit(train_x, train_y)
    pred = np.asarray(fitted.predict(eval_x), dtype=float)

    batch10_pred = eval_df[["cell_id", "battery_index", "batch_id", "horizon", "eol_cycle"]].copy()
    batch10_pred["protocol"] = PROTOCOL
    batch10_pred["model"] = MODEL_NAME
    batch10_pred["pred_eol_cycle"] = pred
    batch10_pred["eol_error"] = batch10_pred["pred_eol_cycle"] - batch10_pred["eol_cycle"]
    batch10_pred["abs_eol_error"] = np.abs(batch10_pred["eol_error"])
    batch10_pred = batch10_pred.sort_values("cell_id").reset_index(drop=True)

    interval_specs = [
        finite_sample_conformal_quantile(batch9_pred["abs_eol_error"].to_numpy(dtype=float), level)
        for level in NOMINAL_LEVELS
    ]
    conformal_summary = build_interval_columns(batch10_pred, interval_specs)
    tree_summary = add_tree_spread_diagnostics(batch10_pred, fitted, eval_x, NOMINAL_LEVELS)
    batch10_pred.to_csv(BATCH10_INTERVALS_CSV, index=False, encoding="utf-8-sig")

    consistency = compare_to_strict_selected_predictions(batch10_pred)

    payload: Dict[str, Any] = {
        "run_name": RUN_NAME,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": time.time() - started,
        "data_dir": str(strict.DATA_DIR),
        "strict_script": str(STRICT_SCRIPT),
        "fixed_setting": {
            "protocol": PROTOCOL,
            "model": MODEL_NAME,
            "horizon_cycles": HORIZON,
            "source": "Batch 1-8",
            "target_adapt_calibration": "Batch 9 leave-one-cell-out",
            "target_test": "Batch 10 blind test",
            "nominal_coverages": NOMINAL_LEVELS,
        },
        "forbidden_model_inputs": sorted(strict.FORBIDDEN_MODEL_INPUTS),
        "feature_columns_used": cols,
        "n_feature_columns": int(len(cols)),
        "batch9_calibration": {
            "n_cells": int(batch9_pred["cell_id"].nunique()),
            "rmse_cycles": float(strict.rmse(batch9_pred["eol_cycle"], batch9_pred["pred_eol_cycle"])),
            "mae_cycles": float(mean_absolute_error(batch9_pred["eol_cycle"], batch9_pred["pred_eol_cycle"])),
            "mean_abs_residual_cycles": float(batch9_pred["abs_eol_error"].mean()),
            "median_abs_residual_cycles": float(batch9_pred["abs_eol_error"].median()),
            "max_abs_residual_cycles": float(batch9_pred["abs_eol_error"].max()),
            "strict_validation_row": calibration_row,
        },
        "conformal_quantile_specs": interval_specs,
        "conformal_interval_summary": conformal_summary,
        "batch10_metrics": {
            "n_cells": int(batch10_pred["cell_id"].nunique()),
            "rmse_cycles": float(strict.rmse(eval_y, pred)),
            "mae_cycles": float(mean_absolute_error(eval_y, pred)),
            "mean_abs_error_cycles": float(batch10_pred["abs_eol_error"].mean()),
            "max_abs_error_cycles": float(batch10_pred["abs_eol_error"].max()),
            "bias_pred_minus_true_cycles": float(batch10_pred["eol_error"].mean()),
        },
        "tree_spread_diagnostic_summary": tree_summary,
        "strict_baseline_consistency_check": consistency,
        "outputs": {
            "script": str(OUT_DIR / "run_strict_eol_conformal_uncertainty_v1.py"),
            "report": str(REPORT_MD),
            "summary_json": str(SUMMARY_JSON),
            "batch9_calibration_residuals_csv": str(BATCH9_RESIDUALS_CSV),
            "batch10_conformal_intervals_csv": str(BATCH10_INTERVALS_CSV),
        },
        "caveats": [
            "Only eight Batch 9 calibration cells are available.",
            "Standard split conformal gives a finite 80% interval but unbounded 90% and 95% intervals at n_cal=8.",
            "Batch 10 empirical coverage is a diagnostic over eight blind-test cells, not a tuning target.",
            "ExtraTrees tree-spread intervals are diagnostic only and are not conformal calibration.",
            "The run keeps the selected strict baseline model, horizon, and features fixed.",
        ],
    }

    SUMMARY_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_report(
        {
            **payload,
            "batch9_residual_rows": batch9_pred.to_dict(orient="records"),
            "batch10_interval_rows": batch10_pred.to_dict(orient="records"),
        }
    )

    print(
        json.dumps(
            {
                "report": str(REPORT_MD),
                "batch9_rmse_cycles": payload["batch9_calibration"]["rmse_cycles"],
                "batch10_rmse_cycles": payload["batch10_metrics"]["rmse_cycles"],
                "conformal_interval_summary": conformal_summary,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
