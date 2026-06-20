import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVR
import warnings


RUN_NAME = "strict-raw-provenance-v3"
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = ROOT / "our_data"
STRICT_REFERENCE_RMSE = 56.19
FIXED_HORIZON = 1000
SLOPE_WINDOWS = [20, 50, 100, 200, 500]
ANCHOR_CYCLES = [10, 50, 100, 200, 300, 500, 800, 1000]
MIN_TREND_POINTS = 5

RAW_SERIES_COLUMNS = [
    "dq_raw",
    "cap_max_mAh",
    "cap_min_mAh",
    "cap_span_mAh",
    "discharge_capacity_span_mAh",
    "charge_capacity_span_mAh",
    "time_total_s",
    "discharge_total_duration_s",
    "charge_total_duration_s",
    "status_discharge_row_fraction",
    "status_charge_row_fraction",
    "voltage_mean_V",
    "voltage_std_V",
    "voltage_min_V",
    "voltage_max_V",
    "current_abs_mean_mA",
    "discharge_current_abs_mean_mA",
    "charge_current_abs_mean_mA",
]

REPORT_MD = OUT_DIR / "strict-raw-provenance-v3_report.md"
SUMMARY_JSON = OUT_DIR / "strict-raw-provenance-v3_summary.json"
FEATURE_AUDIT_CSV = OUT_DIR / "raw_feature_audit_v3.csv"
MODEL_RESULTS_CSV = OUT_DIR / "raw_model_results_v3.csv"
SELECTED_FEATURES_CSV = OUT_DIR / "selected_raw_features_v3.csv"

FORBIDDEN_MODEL_INPUTS = {
    "batch_id",
    "battery_index",
    "cell_id",
    "eol_cycle",
    "full_cell_length",
    "horizon",
    "identifier",
    "last_observed_cycle",
    "n_cycles_total",
    "n_observed",
    "observed_ratio",
    "pkl_file",
    "rul_cycle",
    "split",
}


class Strategy:
    def __init__(self, name: str, kind: str, family: str = "fused", k: int = 0) -> None:
        self.name = name
        self.kind = kind
        self.family = family
        self.k = k
        self.components = 0


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def split_for_batch(batch_id: int) -> str:
    if 1 <= batch_id <= 8:
        return "source"
    if batch_id == 9:
        return "target_adapt"
    if batch_id == 10:
        return "target_test"
    raise ValueError(f"Unexpected batch_id={batch_id}")


def batch_from_cell_id(cell_id: str) -> int:
    return int(str(cell_id).split("-", 1)[0])


def safe_float(value) -> float:
    try:
        out = float(value)
    except Exception:
        return np.nan
    return out if np.isfinite(out) else np.nan


def finite_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def finite_xy(x_values: Iterable[float], y_values: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(list(x_values), dtype=float)
    y = np.asarray(list(y_values), dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def masked_values(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if values.size == 0 or mask.size == 0 or not np.any(mask):
        return np.asarray([], dtype=float)
    out = values[mask]
    return out[np.isfinite(out)]


def array_span(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.max(values) - np.min(values))


def array_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else np.nan


def array_stat(values: np.ndarray, stat: str) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    if stat == "min":
        return float(np.min(values))
    if stat == "max":
        return float(np.max(values))
    if stat == "std":
        return float(np.std(values))
    raise ValueError(stat)


def duration_from_time(times: np.ndarray) -> float:
    times = times[np.isfinite(times)]
    if times.size == 0:
        return np.nan
    return float(np.max(times) - np.min(times))


def cycle_scalar_summary(df: pd.DataFrame, dq_value: float) -> Dict[str, float]:
    if df.empty:
        return {col: np.nan for col in RAW_SERIES_COLUMNS}

    status = df["Status"].astype(str).to_numpy(dtype=str)
    status_lower = np.char.lower(status)
    discharge_mask = np.char.find(status_lower, "discharge") >= 0
    charge_mask = (np.char.find(status_lower, "charge") >= 0) & ~discharge_mask

    cap = df["Capacity (mAh)"].to_numpy(dtype=float)
    times = df["Time (s)"].to_numpy(dtype=float)
    voltage = df["Voltage (V)"].to_numpy(dtype=float)
    current = df["Current (mA)"].to_numpy(dtype=float)
    n_rows = max(len(df), 1)

    discharge_cap = masked_values(cap, discharge_mask)
    charge_cap = masked_values(cap, charge_mask)
    discharge_times = masked_values(times, discharge_mask)
    charge_times = masked_values(times, charge_mask)
    discharge_current = masked_values(current, discharge_mask)
    charge_current = masked_values(current, charge_mask)

    return {
        "dq_raw": safe_float(dq_value),
        "cap_max_mAh": array_stat(cap, "max"),
        "cap_min_mAh": array_stat(cap, "min"),
        "cap_span_mAh": array_span(cap),
        "discharge_capacity_span_mAh": array_span(discharge_cap),
        "charge_capacity_span_mAh": array_span(charge_cap),
        "time_total_s": duration_from_time(times),
        "discharge_total_duration_s": duration_from_time(discharge_times),
        "charge_total_duration_s": duration_from_time(charge_times),
        "status_discharge_row_fraction": float(np.mean(discharge_mask)) if n_rows else np.nan,
        "status_charge_row_fraction": float(np.mean(charge_mask)) if n_rows else np.nan,
        "voltage_mean_V": array_mean(voltage),
        "voltage_std_V": array_stat(voltage, "std"),
        "voltage_min_V": array_stat(voltage, "min"),
        "voltage_max_V": array_stat(voltage, "max"),
        "current_abs_mean_mA": array_mean(np.abs(current)),
        "discharge_current_abs_mean_mA": array_mean(np.abs(discharge_current)),
        "charge_current_abs_mean_mA": array_mean(np.abs(charge_current)),
    }


def add_history_summary(row: Dict[str, float], cycles: np.ndarray, values: Iterable[float], prefix: str) -> None:
    arr_all = np.asarray(list(values), dtype=float)
    arr = arr_all[np.isfinite(arr_all)]
    summary_keys = ["first", "last", "mean", "std", "min", "max", "q05", "q50", "q95", "delta"]
    if arr.size == 0:
        for key in summary_keys:
            row[f"{prefix}_{key}"] = np.nan
        row[f"{prefix}_rel_delta"] = np.nan
        row[f"{prefix}_log_var"] = np.nan
    else:
        q05, q50, q95 = np.quantile(arr, [0.05, 0.50, 0.95])
        row[f"{prefix}_first"] = float(arr[0])
        row[f"{prefix}_last"] = float(arr[-1])
        row[f"{prefix}_mean"] = float(np.mean(arr))
        row[f"{prefix}_std"] = float(np.std(arr))
        row[f"{prefix}_min"] = float(np.min(arr))
        row[f"{prefix}_max"] = float(np.max(arr))
        row[f"{prefix}_q05"] = float(q05)
        row[f"{prefix}_q50"] = float(q50)
        row[f"{prefix}_q95"] = float(q95)
        row[f"{prefix}_delta"] = float(arr[-1] - arr[0])
        denom = abs(float(arr[0])) if abs(float(arr[0])) > 1e-12 else np.nan
        row[f"{prefix}_rel_delta"] = float((arr[-1] - arr[0]) / denom) if np.isfinite(denom) else np.nan
        row[f"{prefix}_log_var"] = float(np.log(np.var(arr) + 1e-12))

    x, y = finite_xy(cycles, arr_all)
    if len(x) >= MIN_TREND_POINTS and np.nanstd(y) > 0:
        coef = np.polyfit(x / 1000.0, y, 1)
        row[f"{prefix}_slope_all"] = float(coef[0])
        residual = y - np.polyval(coef, x / 1000.0)
        row[f"{prefix}_trend_resid_std"] = float(np.std(residual))
    else:
        row[f"{prefix}_slope_all"] = np.nan
        row[f"{prefix}_trend_resid_std"] = np.nan

    diffs = np.diff(y)
    if diffs.size:
        row[f"{prefix}_first_diff_mean"] = float(np.mean(diffs))
        row[f"{prefix}_first_diff_std"] = float(np.std(diffs))
        row[f"{prefix}_first_diff_min"] = float(np.min(diffs))
        row[f"{prefix}_first_diff_max"] = float(np.max(diffs))
        row[f"{prefix}_positive_diff_fraction"] = float(np.mean(diffs > 0))
    else:
        row[f"{prefix}_first_diff_mean"] = np.nan
        row[f"{prefix}_first_diff_std"] = np.nan
        row[f"{prefix}_first_diff_min"] = np.nan
        row[f"{prefix}_first_diff_max"] = np.nan
        row[f"{prefix}_positive_diff_fraction"] = np.nan

    for window in SLOPE_WINDOWS:
        local_x = x[-window:]
        local_y = y[-window:]
        if len(local_x) >= MIN_TREND_POINTS and np.nanstd(local_y) > 0:
            coef = np.polyfit(local_x / 1000.0, local_y, 1)
            row[f"{prefix}_slope_last_{window}"] = float(coef[0])
            row[f"{prefix}_drop_last_{window}"] = float(local_y[-1] - local_y[0])
        else:
            row[f"{prefix}_slope_last_{window}"] = np.nan
            row[f"{prefix}_drop_last_{window}"] = np.nan

    first_value = arr_all[0] if len(arr_all) else np.nan
    for anchor in ANCHOR_CYCLES:
        idx = anchor - 1
        value = arr_all[idx] if 0 <= idx < len(arr_all) else np.nan
        row[f"{prefix}_at_cycle_{anchor:04d}"] = value
        row[f"{prefix}_delta_cycle_0001_to_{anchor:04d}"] = (
            float(value - first_value) if np.isfinite(value) and np.isfinite(first_value) else np.nan
        )


def derive_eol_cycle(cell: Dict[str, Dict]) -> Tuple[int, bool, str]:
    data = cell.get("data", {})
    max_cycle = int(max(data.keys())) if data else 0
    raw_rul = cell.get("rul", {})
    if isinstance(raw_rul, dict) and raw_rul:
        first_cycle = int(min(raw_rul.keys()))
        eol = int(first_cycle + int(raw_rul[first_cycle]) - 1)
        return eol, bool(eol == max_cycle), "raw_rul_first_cycle_plus_cycle_minus_1"
    return max_cycle, True, "max_raw_data_cycle_no_raw_rul"


def build_cell_row(cell_id: str, cell: Dict[str, Dict]) -> Tuple[Dict[str, float] | None, Dict[str, object]]:
    batch_id = batch_from_cell_id(cell_id)
    split = split_for_batch(batch_id)
    data = cell.get("data", {})
    dq = cell.get("dq", {})
    eol_cycle, target_consistent, target_note = derive_eol_cycle(cell)
    audit = {
        "cell_id": cell_id,
        "batch_id": batch_id,
        "split": split,
        "eol_cycle": eol_cycle,
        "has_raw_dq": isinstance(dq, dict),
        "has_raw_rul": isinstance(cell.get("rul", {}), dict),
        "n_data_cycles": len(data) if hasattr(data, "__len__") else 0,
        "n_dq_cycles": len(dq) if hasattr(dq, "__len__") else 0,
        "target_consistent_with_max_data_cycle": target_consistent,
        "target_note": target_note,
        "parsed_for_horizon": False,
    }
    if eol_cycle <= FIXED_HORIZON:
        audit["skip_reason"] = "eol_not_beyond_fixed_horizon"
        return None, audit
    if not all(cycle in data for cycle in range(1, FIXED_HORIZON + 1)):
        audit["skip_reason"] = "missing_observed_cycle_before_horizon"
        return None, audit

    history = {col: [] for col in RAW_SERIES_COLUMNS}
    for cycle in range(1, FIXED_HORIZON + 1):
        dq_value = dq.get(cycle, np.nan) if isinstance(dq, dict) else np.nan
        summary = cycle_scalar_summary(data[cycle], dq_value)
        for col in RAW_SERIES_COLUMNS:
            history[col].append(summary[col])

    cycles = np.arange(1, FIXED_HORIZON + 1, dtype=float)
    row: Dict[str, float] = {
        "split": split,
        "batch_id": int(batch_id),
        "cell_id": str(cell_id),
        "horizon": int(FIXED_HORIZON),
        "n_observed": int(FIXED_HORIZON),
        "last_observed_cycle": int(FIXED_HORIZON),
        "eol_cycle": float(eol_cycle),
        "rul_cycle": float(eol_cycle - FIXED_HORIZON),
    }
    for col in RAW_SERIES_COLUMNS:
        add_history_summary(row, cycles, history[col], col)
    audit["parsed_for_horizon"] = True
    audit["skip_reason"] = ""
    return row, audit


def load_raw_feature_matrix_light(raw_data_dir: Path) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    rows: List[Dict[str, float]] = []
    cell_audit: List[Dict[str, object]] = []
    pkl_files = sorted(raw_data_dir.glob("*.pkl"), key=lambda p: [int(x) for x in p.stem.split("-")])
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found under {raw_data_dir}")
    for idx, pkl_path in enumerate(pkl_files, start=1):
        print(f"[{idx:02d}/{len(pkl_files):02d}] parsing {pkl_path.name}", flush=True)
        with pkl_path.open("rb") as fh:
            payload = pickle.load(fh)
        for cell_id, cell in payload.items():
            row, audit = build_cell_row(str(cell_id), cell)
            audit["pkl_file"] = pkl_path.name
            cell_audit.append(audit)
            if row is not None:
                rows.append(row)
    feature_df = pd.DataFrame(rows)
    if feature_df.empty:
        raise RuntimeError("Raw PKL lightweight parser produced no fixed-horizon feature rows.")
    return feature_df, cell_audit


def starts_with_feature_prefix(feature: str, prefix: str) -> bool:
    return feature == prefix or feature.startswith(f"{prefix}_")


def feature_family(feature: str) -> str:
    if feature.startswith("dq_") or feature.startswith("qv_curve_"):
        return "dq"
    if (
        feature.startswith("cap_")
        or feature.startswith("discharge_capacity_")
        or feature.startswith("charge_capacity_")
        or "capacity_span" in feature
    ):
        return "capacity"
    if (
        feature.startswith("time_")
        or "duration" in feature
        or feature.startswith("status_")
        or "_duration_" in feature
    ):
        return "time_status"
    if (
        "voltage" in feature
        or feature.startswith("discharge_v_at_qfrac_")
        or feature.startswith("charge_v_at_qfrac_")
        or "current" in feature
        or feature.endswith("_mA")
        or feature.endswith("_V")
    ):
        return "voltage_current"
    return "other"


def source_field_note(feature: str) -> str:
    if feature in FORBIDDEN_MODEL_INPUTS:
        return "metadata_or_label_excluded"
    if feature.startswith("dq_raw"):
        return "raw pkl cell['dq'][cycle], cycles 1..1000 only"
    if feature_family(feature) == "capacity":
        return "raw per-cycle data['Capacity (mAh)'], cycles 1..1000 only"
    if feature_family(feature) == "time_status":
        return "raw per-cycle data['Time (s)'] and data['Status'], cycles 1..1000 only"
    if feature_family(feature) == "voltage_current":
        return "raw per-cycle data['Voltage (V)'] / data['Current (mA)'], cycles 1..1000 only"
    return "derived statistic from raw observed cycle records"


def is_allowed_family_feature(feature: str) -> bool:
    return feature_family(feature) in {"dq", "capacity", "time_status", "voltage_current"}


def base_feature_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for col in df.columns:
        if col in FORBIDDEN_MODEL_INPUTS:
            continue
        if not is_allowed_family_feature(col):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    leaked = sorted(set(cols) & FORBIDDEN_MODEL_INPUTS)
    if leaked:
        raise RuntimeError(f"Forbidden features reached model inputs: {leaked}")
    return cols


def family_filter(cols: List[str], family: str) -> List[str]:
    if family == "fused":
        return list(cols)
    return [col for col in cols if feature_family(col) == family]


def clean_xy(
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


def corr_select(train_x: pd.DataFrame, y: np.ndarray, cols: List[str], k: int) -> List[str]:
    y_centered = y - np.mean(y)
    y_std = float(np.std(y_centered)) or 1.0
    scores = []
    arr = train_x.to_numpy(dtype=float)
    for idx, _ in enumerate(cols):
        x = arr[:, idx]
        x_centered = x - np.mean(x)
        denom = (float(np.std(x_centered)) or 0.0) * y_std
        score = 0.0 if denom == 0.0 else abs(float(np.mean(x_centered * y_centered) / denom))
        scores.append(score if np.isfinite(score) else 0.0)
    order = np.argsort(scores)[::-1][: min(k, len(cols))]
    return [cols[int(i)] for i in order]


def selected_columns_for_strategy(strategy: Strategy, train_x: pd.DataFrame, y: np.ndarray, cols: List[str]) -> List[str]:
    family_cols = family_filter(cols, strategy.family)
    if not family_cols:
        raise ValueError(f"No features for family={strategy.family}")
    if strategy.kind == "family":
        return family_cols
    if strategy.kind == "corr_topk":
        return corr_select(train_x[family_cols], y, family_cols, strategy.k)
    raise ValueError(f"Unknown strategy kind={strategy.kind}")


def transform_for_strategy(
    strategy: Strategy,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    base_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    raw_train_x, train_y, raw_eval_x, eval_y = clean_xy(train_df, eval_df, base_cols)
    selected = selected_columns_for_strategy(strategy, raw_train_x, train_y, base_cols)
    train_x = raw_train_x[selected].to_numpy(dtype=float)
    eval_x = raw_eval_x[selected].to_numpy(dtype=float)
    return train_x, train_y, eval_x, eval_y, selected


def strategy_list() -> List[Strategy]:
    return [
        Strategy("family_capacity_only", "family", family="capacity"),
        Strategy("family_dq_only", "family", family="dq"),
        Strategy("family_time_status_only", "family", family="time_status"),
        Strategy("family_voltage_current_only", "family", family="voltage_current"),
        Strategy("family_fused", "family", family="fused"),
        Strategy("corr_fused_top20", "corr_topk", family="fused", k=20),
        Strategy("corr_fused_top50", "corr_topk", family="fused", k=50),
        Strategy("corr_fused_top100", "corr_topk", family="fused", k=100),
        Strategy("corr_capacity_top20", "corr_topk", family="capacity", k=20),
        Strategy("corr_dq_top20", "corr_topk", family="dq", k=20),
        Strategy("corr_time_status_top20", "corr_topk", family="time_status", k=20),
    ]


def model_zoo() -> List[Tuple[str, object]]:
    return [
        ("ridge", make_pipeline(StandardScaler(), Ridge(alpha=50.0))),
        ("elastic_net", make_pipeline(StandardScaler(), ElasticNet(alpha=0.05, l1_ratio=0.2, max_iter=10000, random_state=42))),
        ("extra_trees", ExtraTreesRegressor(n_estimators=80, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=1)),
        ("random_forest", RandomForestRegressor(n_estimators=80, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=1)),
        ("svr_rbf", make_pipeline(RobustScaler(), SVR(C=300.0, epsilon=40.0, gamma="scale"))),
    ]


def allow_model_for_strategy(model_name: str, strategy: Strategy) -> bool:
    if model_name != "svr_rbf":
        return True
    return strategy.kind == "corr_topk"


def fit_predict(
    model,
    strategy: Strategy,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    base_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    train_x, train_y, eval_x, eval_y, selected = transform_for_strategy(strategy, train_df, eval_df, base_cols)
    fitted = clone(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        fitted.fit(train_x, train_y)
    pred = np.asarray(fitted.predict(eval_x), dtype=float)
    pred = np.maximum(pred, float(FIXED_HORIZON))
    return eval_y, pred, selected


def validate_candidate(
    model_name: str,
    model,
    strategy: Strategy,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    base_cols: List[str],
    protocol: str,
) -> Dict[str, object]:
    pred_frames = []
    feature_counts = []
    if protocol == "source_only":
        _, pred, selected = fit_predict(model, strategy, source_df, target_df, base_cols)
        pred_df = target_df[["cell_id", "batch_id", "horizon", "eol_cycle", "rul_cycle"]].copy()
        pred_df["pred_eol_cycle"] = pred
        pred_frames.append(pred_df)
        feature_counts.append(len(selected))
    elif protocol == "source_plus_target_adapt_loocv":
        for heldout_cell in sorted(target_df["cell_id"].unique()):
            heldout = target_df[target_df["cell_id"] == heldout_cell]
            target_train = target_df[target_df["cell_id"] != heldout_cell]
            train_df = pd.concat([source_df, target_train], ignore_index=True)
            _, pred, selected = fit_predict(model, strategy, train_df, heldout, base_cols)
            pred_df = heldout[["cell_id", "batch_id", "horizon", "eol_cycle", "rul_cycle"]].copy()
            pred_df["pred_eol_cycle"] = pred
            pred_frames.append(pred_df)
            feature_counts.append(len(selected))
    else:
        raise ValueError(protocol)

    all_pred = pd.concat(pred_frames, ignore_index=True)
    all_pred["pred_rul_cycle"] = all_pred["pred_eol_cycle"] - float(FIXED_HORIZON)
    score_rmse = rmse(all_pred["eol_cycle"], all_pred["pred_eol_cycle"])
    score_mae = float(mean_absolute_error(all_pred["eol_cycle"], all_pred["pred_eol_cycle"]))
    return {
        "selection_rank": np.nan,
        "is_selected_by_batch9": False,
        "protocol": protocol,
        "model": model_name,
        "horizon": FIXED_HORIZON,
        "feature_strategy": strategy.name,
        "strategy_kind": strategy.kind,
        "feature_family_scope": strategy.family,
        "k": strategy.k,
        "selection_set": "Batch9",
        "selection_method": "direct_batch9_eval" if protocol == "source_only" else "batch9_leave_one_cell_out",
        "selection_eol_rmse": score_rmse,
        "selection_eol_mae": score_mae,
        "selection_rul_rmse": score_rmse,
        "selection_rul_mae": score_mae,
        "target_test_eol_rmse": np.nan,
        "target_test_eol_mae": np.nan,
        "target_test_rul_rmse": np.nan,
        "target_test_rul_mae": np.nan,
        "beats_strict_reference_rmse_56_19": np.nan,
        "n_features_median": int(np.median(feature_counts)),
        "n_features_min": int(np.min(feature_counts)),
        "n_features_max": int(np.max(feature_counts)),
        "n_source_cells": int(source_df["cell_id"].nunique()),
        "n_target_adapt_cells": int(target_df["cell_id"].nunique()),
        "target_test_n_cells": np.nan,
        "n_train_cells_for_final": np.nan,
        "batch10_role": "not_evaluated_for_nonselected_candidate",
    }


def final_test(
    selection_row: pd.Series,
    model_lookup: Dict[str, object],
    strategy_lookup: Dict[str, Strategy],
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_cols: List[str],
) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    model_name = str(selection_row["model"])
    protocol = str(selection_row["protocol"])
    strategy = strategy_lookup[str(selection_row["feature_strategy"])]
    train_df = source_df if protocol == "source_only" else pd.concat([source_df, target_df], ignore_index=True)
    y_true, pred, selected = fit_predict(model_lookup[model_name], strategy, train_df, test_df, base_cols)

    pred_df = test_df[["cell_id", "batch_id", "horizon", "eol_cycle", "rul_cycle"]].copy()
    pred_df["protocol"] = protocol
    pred_df["model"] = model_name
    pred_df["feature_strategy"] = strategy.name
    pred_df["pred_eol_cycle"] = pred
    pred_df["pred_rul_cycle"] = pred_df["pred_eol_cycle"] - float(FIXED_HORIZON)
    pred_df["eol_error"] = pred_df["pred_eol_cycle"] - pred_df["eol_cycle"]
    pred_df["rul_error"] = pred_df["pred_rul_cycle"] - pred_df["rul_cycle"]
    pred_df["abs_eol_error"] = np.abs(pred_df["eol_error"])
    pred_df["abs_rul_error"] = np.abs(pred_df["rul_error"])

    test_rmse = rmse(y_true, pred)
    test_mae = float(mean_absolute_error(y_true, pred))
    selected_features = pd.DataFrame(
        {
            "rank": np.arange(1, len(selected) + 1),
            "feature": selected,
            "feature_family": [feature_family(feature) for feature in selected],
            "source_field_note": [source_field_note(feature) for feature in selected],
            "provenance": "computed from raw our_data/*.pkl observed cycles 1..1000 only",
            "feature_strategy": strategy.name,
            "model": model_name,
            "horizon": FIXED_HORIZON,
            "selection_basis": "selected/fitted using source plus Batch9 target-adapt labels; Batch10 not used",
        }
    )
    result = {
        "selection_scope": "global_best_by_batch9",
        "protocol": protocol,
        "model": model_name,
        "horizon": FIXED_HORIZON,
        "feature_strategy": strategy.name,
        "strategy_kind": strategy.kind,
        "feature_family_scope": strategy.family,
        "n_features": int(len(selected)),
        "selection_eol_rmse": float(selection_row["selection_eol_rmse"]),
        "selection_eol_mae": float(selection_row["selection_eol_mae"]),
        "selection_rul_rmse": float(selection_row["selection_rul_rmse"]),
        "selection_rul_mae": float(selection_row["selection_rul_mae"]),
        "target_test_eol_rmse": test_rmse,
        "target_test_eol_mae": test_mae,
        "target_test_rul_rmse": test_rmse,
        "target_test_rul_mae": test_mae,
        "beats_strict_reference_rmse_56_19": bool(test_rmse < STRICT_REFERENCE_RMSE),
        "target_test_n_cells": int(test_df["cell_id"].nunique()),
        "n_train_cells_for_final": int(train_df["cell_id"].nunique()),
    }
    return result, pred_df, selected_features


def build_audit(feature_df: pd.DataFrame, base_cols: List[str], selected_features: pd.DataFrame) -> pd.DataFrame:
    selected = set(selected_features["feature"].astype(str))
    all_cols = sorted(set(feature_df.columns) | FORBIDDEN_MODEL_INPUTS)
    rows = []
    for col in all_cols:
        present = col in feature_df.columns
        forbidden = col in FORBIDDEN_MODEL_INPUTS
        candidate_pool = col in base_cols
        selected_final = col in selected
        fam = feature_family(col)
        if forbidden:
            decision = "excluded_forbidden_metadata_or_label"
        elif selected_final:
            decision = "selected_final_model_input"
        elif candidate_pool:
            decision = "allowed_candidate_feature_not_selected"
        else:
            decision = "not_used"
        rows.append(
            {
                "feature": col,
                "present_in_raw_feature_table": present,
                "used_in_model_candidate_pool": candidate_pool,
                "selected_by_final_batch9_model": selected_final,
                "decision": decision,
                "feature_family": fam,
                "source_field_note": source_field_note(col),
                "provenance_from_raw_observed_records": candidate_pool or selected_final,
                "forbidden_input": forbidden,
                "online_availability_note": (
                    "excluded from model input"
                    if forbidden
                    else "computed only from cycles 1..1000; no future-cycle aggregate"
                    if candidate_pool
                    else "not used as model input"
                ),
            }
        )
    return pd.DataFrame(rows)


def family_counts(cols: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for col in cols:
        fam = feature_family(col)
        out[fam] = out.get(fam, 0) + 1
    return dict(sorted(out.items()))


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    return value


def run_candidates(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    base_cols: List[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    strategies = strategy_list()
    models = model_zoo()
    for strategy in strategies:
        print(f"validating strategy={strategy.name}", flush=True)
        for model_name, model in models:
            if not allow_model_for_strategy(model_name, strategy):
                continue
            for protocol in ["source_only", "source_plus_target_adapt_loocv"]:
                rows.append(validate_candidate(model_name, model, strategy, source_df, target_df, base_cols, protocol))
    candidate_df = pd.DataFrame(rows).sort_values(["selection_eol_rmse", "selection_eol_mae"]).reset_index(drop=True)
    candidate_df["selection_rank"] = np.arange(1, len(candidate_df) + 1)
    return candidate_df


def write_report(payload: Dict[str, object], candidate_df: pd.DataFrame, selected_features: pd.DataFrame) -> None:
    selected = payload["selected_test_row"]
    beats = "yes" if selected["beats_strict_reference_rmse_56_19"] else "no"
    interpretation = (
        "The selected raw-only diagnostic is numerically stronger than the 56.19-cycle reference, but this should be treated as one strict diagnostic run rather than a broad raw-only superiority claim."
        if selected["beats_strict_reference_rmse_56_19"]
        else "The selected raw-only diagnostic is weaker than the 56.19-cycle strict processed-feature reference; it is provenance evidence, not a replacement result."
    )
    lines = [
        "# Strict Raw-Feature Provenance V3",
        "",
        f"- Finished at: {payload['finished_at']}",
        f"- Elapsed seconds: {payload['elapsed_seconds']:.1f}",
        f"- Raw data directory: `{payload['raw_data_dir']}`",
        f"- Fixed observation horizon: {FIXED_HORIZON} cycles",
        "- Feature source: direct `our_data/*.pkl` parse. The script does not load processed `clean_*_x.npy`, `clean_*_meta.csv`, or `y_nominal` as model inputs.",
        "",
        "## Protocol",
        "",
        "- Batch 1-8 are source cells.",
        "- Batch 9 is the labeled target development/adaptation set and the only set used for model/feature-strategy selection.",
        "- Batch 10 is blind target test and is evaluated only for the global Batch9-selected configuration.",
        "- The 1000-cycle horizon is fixed as a provenance diagnostic aligned with the current strict 1000-cycle main result; it is not selected using Batch10.",
        "- EOL/RUL labels, full cell length, observed ratio, identifiers, horizon, observed-count fields, and future cycles are excluded from model inputs.",
        "",
        "## Selected Batch9 Configuration",
        "",
        "| Strategy | Features | Model | Protocol | Batch9 RMSE/MAE | Batch10 RMSE/MAE | Beats 56.19? |",
        "|---|---:|---|---|---:|---:|---|",
        (
            f"| {selected['feature_strategy']} | {selected['n_features']} | {selected['model']} | {selected['protocol']} | "
            f"{selected['selection_eol_rmse']:.2f}/{selected['selection_eol_mae']:.2f} | "
            f"{selected['target_test_eol_rmse']:.2f}/{selected['target_test_eol_mae']:.2f} | {beats} |"
        ),
        "",
        "## Interpretation",
        "",
        f"- {interpretation}",
        "- RUL is derived after EOL prediction as `predicted_EOL - 1000`; therefore EOL and RUL errors are numerically identical at this fixed horizon.",
        "- Batch9 labels make this supervised small labeled target-domain adaptation, not unsupervised domain adaptation.",
        "",
        "## Raw Feature Provenance",
        "",
        "| Family | Candidate features | Selected features | Raw source |",
        "|---|---:|---:|---|",
    ]
    candidate_counts = payload["candidate_feature_family_counts"]
    selected_counts = payload["selected_feature_family_counts"]
    for fam in ["capacity", "dq", "time_status", "voltage_current"]:
        lines.append(
            f"| {fam} | {candidate_counts.get(fam, 0)} | {selected_counts.get(fam, 0)} | {payload['family_source_notes'].get(fam, '')} |"
        )
    lines.extend(
        [
            "",
            "## Batch9 Candidate Ranking",
            "",
            "| Rank | Strategy | Model | Protocol | Batch9 RMSE | Batch9 MAE | Features |",
            "|---:|---|---|---|---:|---:|---:|",
        ]
    )
    for _, row in candidate_df.head(30).iterrows():
        lines.append(
            f"| {int(row['selection_rank'])} | {row['feature_strategy']} | {row['model']} | {row['protocol']} | "
            f"{row['selection_eol_rmse']:.2f} | {row['selection_eol_mae']:.2f} | {int(row['n_features_median'])} |"
        )
    lines.extend(
        [
            "",
            "## Selected Raw Features",
            "",
            "| Rank | Feature | Family | Source note |",
            "|---:|---|---|---|",
        ]
    )
    for _, row in selected_features.head(40).iterrows():
        lines.append(f"| {int(row['rank'])} | `{row['feature']}` | {row['feature_family']} | {row['source_field_note']} |")
    if len(selected_features) > 40:
        lines.append(f"| ... | {len(selected_features) - 40} additional features | ... | see CSV |")
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- Script: `{Path(__file__).resolve()}`",
            f"- Report: `{REPORT_MD}`",
            f"- Summary JSON: `{SUMMARY_JSON}`",
            f"- Feature audit CSV: `{FEATURE_AUDIT_CSV}`",
            f"- Model results CSV: `{MODEL_RESULTS_CSV}`",
            f"- Selected raw features CSV: `{SELECTED_FEATURES_CSV}`",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    feature_df, cell_audit = load_raw_feature_matrix_light(RAW_DATA_DIR)
    horizon_df = feature_df[feature_df["horizon"] == FIXED_HORIZON].copy().reset_index(drop=True)
    source_df = horizon_df[horizon_df["split"] == "source"].copy().reset_index(drop=True)
    target_df = horizon_df[horizon_df["split"] == "target_adapt"].copy().reset_index(drop=True)
    test_df = horizon_df[horizon_df["split"] == "target_test"].copy().reset_index(drop=True)
    if source_df.empty or target_df.empty or test_df.empty:
        raise RuntimeError("Fixed 1000-cycle horizon did not produce all three protocol splits.")

    base_cols = base_feature_columns(horizon_df)
    candidate_df = run_candidates(source_df, target_df, base_cols)
    selected_row = candidate_df.iloc[0]
    strategy_lookup = {strategy.name: strategy for strategy in strategy_list()}
    model_lookup = {name: model for name, model in model_zoo()}
    selected_result, pred_df, selected_features = final_test(
        selection_row=selected_row,
        model_lookup=model_lookup,
        strategy_lookup=strategy_lookup,
        source_df=source_df,
        target_df=target_df,
        test_df=test_df,
        base_cols=base_cols,
    )

    selected_mask = candidate_df["selection_rank"] == int(selected_row["selection_rank"])
    candidate_df.loc[selected_mask, "is_selected_by_batch9"] = True
    candidate_df["beats_strict_reference_rmse_56_19"] = candidate_df["beats_strict_reference_rmse_56_19"].astype(object)
    for metric in [
        "target_test_eol_rmse",
        "target_test_eol_mae",
        "target_test_rul_rmse",
        "target_test_rul_mae",
        "beats_strict_reference_rmse_56_19",
        "target_test_n_cells",
        "n_train_cells_for_final",
    ]:
        candidate_df.loc[selected_mask, metric] = selected_result[metric]
    candidate_df.loc[selected_mask, "batch10_role"] = "blind_test_evaluated_once_for_batch9_selected_candidate"
    candidate_df.to_csv(MODEL_RESULTS_CSV, index=False, encoding="utf-8-sig")

    selected_features.to_csv(SELECTED_FEATURES_CSV, index=False, encoding="utf-8-sig")
    audit_df = build_audit(horizon_df, base_cols, selected_features)
    audit_df.to_csv(FEATURE_AUDIT_CSV, index=False, encoding="utf-8-sig")

    payload = {
        "run_name": RUN_NAME,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": time.time() - started,
        "raw_data_dir": str(RAW_DATA_DIR),
        "helper_source_scripts": [],
        "parse_mode": "direct_lightweight_raw_pkl_parse_in_v3_script",
        "processed_inputs_used": False,
        "processed_input_statement": "No processed clean_*_x.npy, clean_*_meta.csv, clean_*_rul.npy, or y_nominal fields are loaded or used as model inputs.",
        "fixed_horizon": FIXED_HORIZON,
        "selection_rule": "Select the global best candidate by Batch9 EOL RMSE, then Batch9 EOL MAE; evaluate Batch10 once for that selected candidate.",
        "protocol_splits": {
            "source_batches": "1-8",
            "target_development_batch": 9,
            "blind_test_batch": 10,
            "source_cells": int(source_df["cell_id"].nunique()),
            "target_adapt_cells": int(target_df["cell_id"].nunique()),
            "target_test_cells": int(test_df["cell_id"].nunique()),
        },
        "raw_schema_summary": {
            "pkl_files": int(len({row.get("pkl_file") for row in cell_audit})),
            "cells": int(horizon_df["cell_id"].nunique()),
            "cells_with_dq_model_features": int(horizon_df.groupby("cell_id")["dq_raw_last"].apply(lambda s: bool(s.notna().any())).sum())
            if "dq_raw_last" in horizon_df
            else 0,
            "cells_with_raw_rul_target": int(horizon_df["cell_id"].nunique()),
            "raw_cycle_frame_columns": ["Status", "Cycle number", "Current (mA)", "Voltage (V)", "Capacity (mAh)", "Time (s)"],
        },
        "target_source_note": "Raw rul is used only to derive supervised EOL/RUL targets; target columns are excluded from model inputs.",
        "forbidden_model_inputs": sorted(FORBIDDEN_MODEL_INPUTS),
        "candidate_feature_count": int(len(base_cols)),
        "candidate_feature_family_counts": family_counts(base_cols),
        "selected_feature_family_counts": family_counts(selected_features["feature"].astype(str).tolist()),
        "family_source_notes": {
            "capacity": "raw Capacity (mAh) summaries/trends from observed cycle records",
            "dq": "raw cell['dq'] sequence summaries/trends if present",
            "time_status": "raw Time (s) duration and Status-phase summaries from observed records",
            "voltage_current": "raw Voltage (V), Current (mA), and capacity-voltage summaries from observed records",
        },
        "selected_test_row": selected_result,
        "candidate_rows": candidate_df.to_dict(orient="records"),
        "batch10_selected_predictions": pred_df.to_dict(orient="records"),
        "cell_parse_audit": cell_audit,
        "outputs": {
            "script": str(Path(__file__).resolve()),
            "report": str(REPORT_MD),
            "summary_json": str(SUMMARY_JSON),
            "raw_feature_audit": str(FEATURE_AUDIT_CSV),
            "raw_model_results": str(MODEL_RESULTS_CSV),
            "selected_raw_features": str(SELECTED_FEATURES_CSV),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_report(payload, candidate_df, selected_features)

    print(
        json.dumps(
            {
                "selected": selected_result,
                "elapsed_seconds": payload["elapsed_seconds"],
                "outputs": payload["outputs"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
