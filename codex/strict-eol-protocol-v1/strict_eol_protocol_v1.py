import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVR


RUN_NAME = "strict-eol-protocol-v1"
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent

HORIZONS = [50, 100, 200, 300, 500, 800, 1000]
MIN_OBSERVED_CYCLES = 5

FORBIDDEN_MODEL_INPUTS = {
    "eol_cycle",
    "rul_cycle",
    "observed_ratio",
    "full_cell_length",
    "n_cycles_total",
    "last_observed_cycle",
    "n_observed",
    "cell_id",
    "battery_index",
    "batch_id",
    "horizon",
    "split",
}

REPORT_MD = OUT_DIR / f"{RUN_NAME}_report.md"
SUMMARY_JSON = OUT_DIR / f"{RUN_NAME}_summary.json"
CANDIDATE_CSV = OUT_DIR / f"{RUN_NAME}_candidate_validation_summary.csv"
SELECTED_TEST_CSV = OUT_DIR / f"{RUN_NAME}_selected_test_summary.csv"
PREDICTIONS_CSV = OUT_DIR / f"{RUN_NAME}_selected_test_predictions.csv"
FEATURE_AUDIT_CSV = OUT_DIR / f"{RUN_NAME}_feature_audit.csv"


class ConstantEOLRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, strategy: str = "median") -> None:
        self.strategy = strategy

    def fit(self, X: pd.DataFrame, y: Iterable[float]):
        y_arr = np.asarray(list(y), dtype=float)
        if self.strategy == "mean":
            self.constant_ = float(np.mean(y_arr))
        elif self.strategy == "median":
            self.constant_ = float(np.median(y_arr))
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.constant_, dtype=float)


def locate_data_dir() -> Path:
    candidates = [
        child / "processed_clean"
        for child in ROOT.iterdir()
        if child.is_dir()
        and child.name.lower().startswith("python")
        and (child / "processed_clean" / "clean_source_meta.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError("Could not locate processed_clean data directory.")
    return candidates[0]


DATA_DIR = locate_data_dir()


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def load_split(split: str) -> Tuple[np.ndarray, pd.DataFrame]:
    x = np.load(DATA_DIR / f"clean_{split}_x.npy").astype(np.float32)
    meta = pd.read_csv(DATA_DIR / f"clean_{split}_meta.csv")
    meta["split"] = split
    return x, meta


def curve_stats(values: np.ndarray, prefix: str) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        names = ["first", "last", "mean", "std", "min", "max", "delta"]
        return {f"{prefix}_{name}": np.nan for name in names}
    return {
        f"{prefix}_first": float(values[0]),
        f"{prefix}_last": float(values[-1]),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_delta": float(values[-1] - values[0]),
    }


def add_quantiles(row: Dict[str, float], values: np.ndarray, prefix: str) -> None:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        for q in [5, 25, 50, 75, 95]:
            row[f"{prefix}_q{q}"] = np.nan
        return
    for q, value in zip([5, 25, 50, 75, 95], np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])):
        row[f"{prefix}_q{q}"] = float(value)


def latest_soh_at_or_before(observed: pd.DataFrame, cycle: int) -> float:
    eligible = observed[observed["cycle"] <= cycle]
    if eligible.empty:
        return np.nan
    return float(eligible.sort_values("cycle")["y_nominal"].iloc[-1])


def build_cell_features(x: np.ndarray, meta: pd.DataFrame, horizon: int) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for battery_index, group in meta.groupby("battery_index", sort=True):
        group = group.sort_values("cycle")
        eol_cycle = float((group["cycle"] + group["rul_cycle"]).median())
        if eol_cycle <= horizon:
            continue

        observed = group[group["cycle"] <= horizon].copy()
        if len(observed) < MIN_OBSERVED_CYCLES:
            continue

        row_indices = observed.index.to_numpy(dtype=int)
        cycles = observed["cycle"].to_numpy(dtype=float)
        soh = observed["y_nominal"].to_numpy(dtype=float)

        row: Dict[str, float] = {
            "split": str(group["split"].iloc[0]),
            "battery_index": int(battery_index),
            "cell_id": str(group["cell_id"].iloc[0]),
            "batch_id": int(group["batch_id"].iloc[0]),
            "horizon": int(horizon),
            "n_observed": int(len(observed)),
            "last_observed_cycle": float(cycles[-1]),
            "eol_cycle": eol_cycle,
        }

        row.update(curve_stats(soh, "soh"))
        add_quantiles(row, soh, "soh")

        for anchor in [10, 20, 50, 100, 200, 300, 500, 800, 1000]:
            value = latest_soh_at_or_before(observed, anchor)
            row[f"soh_at_{anchor}"] = value
            if not np.isnan(value):
                row[f"soh_drop_1_to_{anchor}"] = float(value - soh[0])
            else:
                row[f"soh_drop_1_to_{anchor}"] = np.nan

        for window in [20, 50, 100, 200, 500]:
            local = observed.tail(window)
            if len(local) >= MIN_OBSERVED_CYCLES:
                c = local["cycle"].to_numpy(dtype=float)
                y = local["y_nominal"].to_numpy(dtype=float)
                slope, intercept = np.polyfit(c / 1000.0, y, 1)
                row[f"soh_slope_last_{window}"] = float(slope)
                row[f"soh_intercept_last_{window}"] = float(intercept)
                row[f"soh_drop_last_{window}"] = float(y[-1] - y[0])
            else:
                row[f"soh_slope_last_{window}"] = np.nan
                row[f"soh_intercept_last_{window}"] = np.nan
                row[f"soh_drop_last_{window}"] = np.nan

        for degree in [1, 2, 3, 4]:
            if len(cycles) >= degree + 2:
                coeffs = np.polyfit(cycles / 1000.0, soh, degree)
                for idx, value in enumerate(coeffs):
                    row[f"soh_poly{degree}_coef_{idx}"] = float(value)
            else:
                for idx in range(degree + 1):
                    row[f"soh_poly{degree}_coef_{idx}"] = np.nan

        x_observed = x[row_indices]
        x_mean_curve = np.mean(x_observed, axis=0)
        x_last_curve = x_observed[-1]
        x_delta_curve = x_observed[-1] - x_observed[0]
        for prefix, arr in [
            ("x_mean_curve", x_mean_curve),
            ("x_last_curve", x_last_curve),
            ("x_delta_curve", x_delta_curve),
        ]:
            row.update(curve_stats(arr, prefix))
            add_quantiles(row, arr, prefix)

        if len(row_indices) >= 100:
            diff_100_10 = x[row_indices[99]] - x[row_indices[9]]
            row.update(curve_stats(diff_100_10, "x_cycle100_minus_10"))
            add_quantiles(row, diff_100_10, "x_cycle100_minus_10")
            row["x_cycle100_minus_10_logvar"] = float(np.log(np.var(diff_100_10) + 1e-12))
        else:
            row.update(curve_stats(np.asarray([]), "x_cycle100_minus_10"))
            add_quantiles(row, np.asarray([]), "x_cycle100_minus_10")
            row["x_cycle100_minus_10_logvar"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def feature_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for col in df.columns:
        if col in FORBIDDEN_MODEL_INPUTS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    leaked = sorted(set(cols) & FORBIDDEN_MODEL_INPUTS)
    if leaked:
        raise RuntimeError(f"Forbidden features reached model inputs: {leaked}")
    return cols


def prepare_train_eval(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    cols: List[str],
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    train_x = train_df[cols].replace([np.inf, -np.inf], np.nan)
    eval_x = eval_df[cols].replace([np.inf, -np.inf], np.nan)
    med = train_x.median(numeric_only=True)
    train_x = train_x.fillna(med).fillna(0.0)
    eval_x = eval_x.fillna(med).fillna(0.0)
    train_y = train_df["eol_cycle"].to_numpy(dtype=float)
    eval_y = eval_df["eol_cycle"].to_numpy(dtype=float)
    return train_x, train_y, eval_x, eval_y


def model_zoo() -> List[Tuple[str, object]]:
    return [
        ("source_median_eol", ConstantEOLRegressor(strategy="median")),
        ("source_mean_eol", ConstantEOLRegressor(strategy="mean")),
        ("ridge", make_pipeline(StandardScaler(), Ridge(alpha=50.0))),
        ("elastic_net", make_pipeline(StandardScaler(), ElasticNet(alpha=0.05, l1_ratio=0.2, max_iter=5000, random_state=42))),
        ("svr_rbf", make_pipeline(RobustScaler(), SVR(C=300.0, epsilon=40.0, gamma="scale"))),
        ("kernel_ridge_rbf", make_pipeline(StandardScaler(), KernelRidge(alpha=30.0, kernel="rbf", gamma=0.03))),
        ("extra_trees", ExtraTreesRegressor(n_estimators=220, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=1)),
        ("random_forest", RandomForestRegressor(n_estimators=220, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=1)),
        ("gbr", GradientBoostingRegressor(n_estimators=120, learning_rate=0.03, max_depth=2, random_state=42, subsample=0.8)),
    ]


def predict_with_fit(model, train_df: pd.DataFrame, eval_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    cols = feature_columns(train_df)
    train_x, train_y, eval_x, eval_y = prepare_train_eval(train_df, eval_df, cols)
    fitted = clone(model)
    fitted.fit(train_x, train_y)
    pred = fitted.predict(eval_x)
    return eval_y, np.asarray(pred, dtype=float)


def validate_candidate(
    model_name: str,
    model,
    protocol: str,
    source_df: pd.DataFrame,
    target_adapt_df: pd.DataFrame,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    pred_rows: List[pd.DataFrame] = []

    if protocol == "source_only":
        y_true, pred = predict_with_fit(model, source_df, target_adapt_df)
        pred_df = target_adapt_df[["cell_id", "battery_index", "batch_id", "horizon", "eol_cycle"]].copy()
        pred_df["pred_eol_cycle"] = pred
        pred_rows.append(pred_df)
    elif protocol == "source_plus_target_adapt_loocv":
        for heldout_cell in sorted(target_adapt_df["cell_id"].unique()):
            heldout_df = target_adapt_df[target_adapt_df["cell_id"] == heldout_cell]
            target_train_df = target_adapt_df[target_adapt_df["cell_id"] != heldout_cell]
            train_df = pd.concat([source_df, target_train_df], ignore_index=True)
            y_true, pred = predict_with_fit(model, train_df, heldout_df)
            pred_df = heldout_df[["cell_id", "battery_index", "batch_id", "horizon", "eol_cycle"]].copy()
            pred_df["pred_eol_cycle"] = pred
            pred_rows.append(pred_df)
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    all_pred = pd.concat(pred_rows, ignore_index=True)
    row = {
        "protocol": protocol,
        "model": model_name,
        "horizon": int(target_adapt_df["horizon"].iloc[0]),
        "selection_set": "target_adapt",
        "selection_method": "direct_batch9_eval" if protocol == "source_only" else "batch9_leave_one_cell_out",
        "selection_eol_rmse": rmse(all_pred["eol_cycle"], all_pred["pred_eol_cycle"]),
        "selection_eol_mae": float(mean_absolute_error(all_pred["eol_cycle"], all_pred["pred_eol_cycle"])),
        "n_source_cells": int(source_df["cell_id"].nunique()),
        "n_target_adapt_cells": int(target_adapt_df["cell_id"].nunique()),
    }
    return row, all_pred


def final_test_for_selection(
    selection_row: pd.Series,
    model_lookup: Dict[str, object],
    frames_by_horizon: Dict[int, Dict[str, pd.DataFrame]],
    selection_scope: str,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    horizon = int(selection_row["horizon"])
    protocol = str(selection_row["protocol"])
    model_name = str(selection_row["model"])
    frames = frames_by_horizon[horizon]

    if protocol == "source_only":
        train_df = frames["source"]
    elif protocol == "source_plus_target_adapt_loocv":
        train_df = pd.concat([frames["source"], frames["target_adapt"]], ignore_index=True)
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    target_test_df = frames["target_test"]
    y_true, pred = predict_with_fit(model_lookup[model_name], train_df, target_test_df)
    pred_df = target_test_df[["cell_id", "battery_index", "batch_id", "horizon", "eol_cycle"]].copy()
    pred_df["selection_scope"] = selection_scope
    pred_df["protocol"] = protocol
    pred_df["model"] = model_name
    pred_df["pred_eol_cycle"] = pred
    pred_df["eol_error"] = pred_df["pred_eol_cycle"] - pred_df["eol_cycle"]
    pred_df["abs_eol_error"] = np.abs(pred_df["eol_error"])

    result = {
        "selection_scope": selection_scope,
        "protocol": protocol,
        "model": model_name,
        "horizon": horizon,
        "selection_eol_rmse": float(selection_row["selection_eol_rmse"]),
        "selection_eol_mae": float(selection_row["selection_eol_mae"]),
        "target_test_eol_rmse": rmse(y_true, pred),
        "target_test_eol_mae": float(mean_absolute_error(y_true, pred)),
        "target_test_n_cells": int(target_test_df["cell_id"].nunique()),
    }
    return result, pred_df


def build_feature_audit(frames_by_horizon: Dict[int, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    all_cols = set()
    present_horizons: Dict[str, set] = {}
    present_splits: Dict[str, set] = {}
    used_horizons: Dict[str, set] = {}
    used_splits: Dict[str, set] = {}

    for horizon, split_frames in frames_by_horizon.items():
        for split, df in split_frames.items():
            model_cols_for_frame = set(feature_columns(df))
            for col in df.columns:
                all_cols.add(col)
                present_horizons.setdefault(col, set()).add(horizon)
                present_splits.setdefault(col, set()).add(split)
            for col in model_cols_for_frame:
                used_horizons.setdefault(col, set()).add(horizon)
                used_splits.setdefault(col, set()).add(split)

    rows = []
    for col in sorted(all_cols | FORBIDDEN_MODEL_INPUTS):
        model_cols = set(used_horizons.get(col, set()))
        if col in FORBIDDEN_MODEL_INPUTS:
            decision = "excluded"
            risk = "high_if_used" if col not in {"split"} else "metadata"
        elif model_cols:
            decision = "model_input"
            if col.startswith("soh"):
                risk = "low_if_historical_measured_or_estimated_soh"
            elif col.startswith("x_"):
                risk = "low_if_processed_curve_uses_only_observed_cycles"
            else:
                risk = "low"
        else:
            decision = "not_numeric_or_unused"
            risk = "not_used"
        rows.append(
            {
                "feature": col,
                "present_in_raw_feature_table": bool(col in all_cols),
                "present_horizons": ";".join(str(x) for x in sorted(present_horizons.get(col, set()))),
                "present_splits": ";".join(sorted(present_splits.get(col, set()))),
                "used_as_model_input": bool(model_cols),
                "used_horizons": ";".join(str(x) for x in sorted(used_horizons.get(col, set()))),
                "used_splits": ";".join(sorted(used_splits.get(col, set()))),
                "decision": decision,
                "online_availability_note": risk,
            }
        )
    return pd.DataFrame(rows)


def write_report(payload: Dict[str, object]) -> None:
    selected = pd.DataFrame(payload["selected_test_rows"])
    candidates = pd.DataFrame(payload["candidate_rows"]).sort_values("selection_eol_rmse")
    global_row = selected[selected["selection_scope"] == "global_best_by_batch9"].iloc[0]

    lines = [
        "# Strict EOL Protocol V1",
        "",
        f"- Finished at: {payload['finished_at']}",
        f"- Elapsed seconds: {payload['elapsed_seconds']:.1f}",
        f"- Data dir: `{payload['data_dir']}`",
        "",
        "## Protocol",
        "",
        "- Source domain: Batch 1-8.",
        "- Target development/adaptation set: Batch 9.",
        "- Blind test set: Batch 10.",
        "- Candidate model and horizon selection uses Batch 9 only.",
        "- `source_plus_target_adapt_loocv` uses leave-one-Batch-9-cell-out validation, so the held-out target cell is not used for supervised adaptation.",
        "- Batch 10 is evaluated only after Batch-9-based selection; candidate rows are not sorted by Batch 10.",
        "",
        "## Global Selection",
        "",
        "| Scope | Protocol | Model | Horizon | Batch 9 RMSE | Batch 10 RMSE | Batch 10 MAE |",
        "|---|---|---|---:|---:|---:|---:|",
        (
            f"| {global_row['selection_scope']} | {global_row['protocol']} | {global_row['model']} | "
            f"{int(global_row['horizon'])} | {global_row['selection_eol_rmse']:.2f} | "
            f"{global_row['target_test_eol_rmse']:.2f} | {global_row['target_test_eol_mae']:.2f} |"
        ),
        "",
        "## Selected Baselines",
        "",
        "Each row below selected its horizon by Batch 9 validation before Batch 10 evaluation.",
        "",
        "| Scope | Protocol | Model | Horizon | Batch 9 RMSE | Batch 10 RMSE | Batch 10 MAE |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for _, row in selected.sort_values(["selection_scope", "protocol", "model"]).iterrows():
        lines.append(
            f"| {row['selection_scope']} | {row['protocol']} | {row['model']} | "
            f"{int(row['horizon'])} | {row['selection_eol_rmse']:.2f} | "
            f"{row['target_test_eol_rmse']:.2f} | {row['target_test_eol_mae']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Batch 9 Candidate Ranking",
            "",
            "This table is validation-only and is safe for model/horizon selection.",
            "",
            "| Protocol | Model | Horizon | Batch 9 RMSE | Batch 9 MAE | Selection method |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for _, row in candidates.head(20).iterrows():
        lines.append(
            f"| {row['protocol']} | {row['model']} | {int(row['horizon'])} | "
            f"{row['selection_eol_rmse']:.2f} | {row['selection_eol_mae']:.2f} | {row['selection_method']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The reported EOL RMSE in cycles is numerically the same as RUL RMSE at a fixed observation cycle when RUL is derived as predicted EOL minus current cycle.",
            "- This run is stricter than the earlier EOL fingerprint pilot because Batch 10 is not used for selecting the model or horizon.",
            "- `y_nominal` features must be described as historical measured/estimated capacity-SOH information. A later raw-capacity or `dq` feature audit is still needed.",
            "- The feature audit covers every evaluated horizon and split.",
            "- The ExtraTrees/GBR family remains a strong baseline. It is not yet the final publishable architecture.",
            "",
            "## Output Files",
            "",
            f"- Candidate validation table: `{CANDIDATE_CSV}`",
            f"- Selected Batch 10 table: `{SELECTED_TEST_CSV}`",
            f"- Selected predictions: `{PREDICTIONS_CSV}`",
            f"- Feature audit: `{FEATURE_AUDIT_CSV}`",
            f"- JSON summary: `{SUMMARY_JSON}`",
        ]
    )

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.time()
    splits = {split: load_split(split) for split in ["source", "target_adapt", "target_test"]}
    frames_by_horizon: Dict[int, Dict[str, pd.DataFrame]] = {}
    candidate_rows: List[Dict[str, float]] = []
    model_pairs = model_zoo()
    model_lookup = {name: model for name, model in model_pairs}

    for horizon in HORIZONS:
        frames_by_horizon[horizon] = {
            split: build_cell_features(x, meta, horizon)
            for split, (x, meta) in splits.items()
        }
        for protocol in ["source_only", "source_plus_target_adapt_loocv"]:
            for model_name, model in model_pairs:
                row, _ = validate_candidate(
                    model_name=model_name,
                    model=model,
                    protocol=protocol,
                    source_df=frames_by_horizon[horizon]["source"],
                    target_adapt_df=frames_by_horizon[horizon]["target_adapt"],
                )
                candidate_rows.append(row)

    candidate_df = pd.DataFrame(candidate_rows).sort_values("selection_eol_rmse")
    candidate_df.to_csv(CANDIDATE_CSV, index=False, encoding="utf-8-sig")

    selected_refs: List[Tuple[str, pd.Series]] = []
    selected_refs.append(("global_best_by_batch9", candidate_df.iloc[0]))
    for (protocol, model_name), group in candidate_df.groupby(["protocol", "model"], sort=True):
        best = group.sort_values("selection_eol_rmse").iloc[0]
        selected_refs.append((f"best_horizon_for_{protocol}_{model_name}", best))

    seen_keys = set()
    selected_test_rows: List[Dict[str, float]] = []
    selected_pred_frames: List[pd.DataFrame] = []
    for scope, row in selected_refs:
        key = (scope, str(row["protocol"]), str(row["model"]), int(row["horizon"]))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        result, pred_df = final_test_for_selection(row, model_lookup, frames_by_horizon, scope)
        selected_test_rows.append(result)
        selected_pred_frames.append(pred_df)

    selected_test_df = pd.DataFrame(selected_test_rows)
    selected_predictions = pd.concat(selected_pred_frames, ignore_index=True)
    selected_test_df.to_csv(SELECTED_TEST_CSV, index=False, encoding="utf-8-sig")
    selected_predictions.to_csv(PREDICTIONS_CSV, index=False, encoding="utf-8-sig")

    feature_audit = build_feature_audit(frames_by_horizon)
    feature_audit.to_csv(FEATURE_AUDIT_CSV, index=False, encoding="utf-8-sig")

    payload = {
        "run_name": RUN_NAME,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": time.time() - started,
        "data_dir": str(DATA_DIR),
        "horizons": HORIZONS,
        "forbidden_model_inputs": sorted(FORBIDDEN_MODEL_INPUTS),
        "candidate_rows": candidate_rows,
        "selected_test_rows": selected_test_rows,
        "outputs": {
            "candidate_validation_summary": str(CANDIDATE_CSV),
            "selected_test_summary": str(SELECTED_TEST_CSV),
            "selected_test_predictions": str(PREDICTIONS_CSV),
            "feature_audit": str(FEATURE_AUDIT_CSV),
            "report": str(REPORT_MD),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)

    global_row = selected_test_df[selected_test_df["selection_scope"] == "global_best_by_batch9"].iloc[0].to_dict()
    print(json.dumps({"global_best_by_batch9": global_row, "report": str(REPORT_MD)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
