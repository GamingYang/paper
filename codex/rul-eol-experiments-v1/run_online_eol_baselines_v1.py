import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


RUN_NAME = "online-eol-baselines-v1"
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = next(
    child / "processed_clean"
    for child in ROOT.iterdir()
    if child.is_dir()
    and child.name.lower().startswith("python")
    and (child / "processed_clean" / "clean_source_meta.csv").exists()
)
OUT_DIR = Path(__file__).resolve().parent
SUMMARY_CSV = OUT_DIR / f"{RUN_NAME}_summary.csv"
DETAIL_CSV = OUT_DIR / f"{RUN_NAME}_target_test_predictions.csv"
REPORT_JSON = OUT_DIR / f"{RUN_NAME}_summary.json"
REPORT_MD = OUT_DIR / f"{RUN_NAME}_report.md"

EOL_SOH = 0.80


def load_meta(split: str) -> pd.DataFrame:
    meta = pd.read_csv(DATA_DIR / f"clean_{split}_meta.csv")
    meta["split"] = split
    return meta


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(pred))))


def macro_rmse(df: pd.DataFrame) -> float:
    return float(np.mean([rmse(g["rul_cycle"], g["pred_rul"]) for _, g in df.groupby("battery_index")]))


def fit_poly_eol(cycles: np.ndarray, soh: np.ndarray, degree: int) -> Optional[float]:
    if len(cycles) < degree + 2:
        return None
    x = cycles.astype(float)
    y = soh.astype(float)
    try:
        coef = np.polyfit(x, y, degree)
        poly = np.poly1d(coef)
    except np.linalg.LinAlgError:
        return None

    current = float(x[-1])
    shifted = coef.copy()
    shifted[-1] -= EOL_SOH
    roots = np.roots(shifted)
    real_roots = np.real(roots[np.isreal(roots)])
    future_roots = real_roots[(real_roots >= current) & (real_roots <= 4000.0)]
    if future_roots.size == 0:
        return None
    return float(np.min(future_roots))


def linear_window_eol(cycles: np.ndarray, soh: np.ndarray, window: int) -> Optional[float]:
    if len(cycles) < max(10, min(window, 10)):
        return None
    x = cycles[-window:].astype(float)
    y = soh[-window:].astype(float)
    if len(x) < 3:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    if slope >= -1e-7:
        return None
    eol = (EOL_SOH - intercept) / slope
    if not np.isfinite(eol):
        return None
    return float(max(eol, cycles[-1]))


def robust_recent_slope_eol(cycles: np.ndarray, soh: np.ndarray, windows: List[int]) -> Optional[float]:
    preds = []
    for window in windows:
        pred = linear_window_eol(cycles, soh, window)
        if pred is not None:
            preds.append(pred)
    if not preds:
        return None
    return float(np.median(preds))


def source_prior_lifetime(source_meta: pd.DataFrame, kind: str) -> float:
    lifetimes = source_meta.groupby("battery_index")["cycle"].max().to_numpy(dtype=float)
    if kind == "mean":
        return float(np.mean(lifetimes))
    if kind == "median":
        return float(np.median(lifetimes))
    raise ValueError(kind)


def predict_split(meta: pd.DataFrame, source_meta: pd.DataFrame, method: str, min_history: int, prior_kind: str) -> pd.DataFrame:
    prior_eol = source_prior_lifetime(source_meta, prior_kind)
    rows = []
    for _, group in meta.groupby("battery_index"):
        group = group.sort_values("cycle").copy()
        cycles_all = group["cycle"].to_numpy(dtype=float)
        soh_all = group["y_nominal"].to_numpy(dtype=float)
        true_rul_all = group["rul_cycle"].to_numpy(dtype=float)
        cached_predictions: Dict[int, Optional[float]] = {}
        for idx in range(len(group)):
            current = cycles_all[idx]
            hist_cycles = cycles_all[: idx + 1]
            hist_soh = soh_all[: idx + 1]
            pred_eol = None
            # Fit only every 5 cycles for polynomial methods and reuse the nearest
            # previous fit. This keeps the online protocol causal while making the
            # ablation practical to run.
            cache_key = idx if not method.startswith("poly") else (idx // 5) * 5
            if cache_key in cached_predictions:
                pred_eol = cached_predictions[cache_key]
            elif len(hist_cycles) >= min_history:
                if method == "linear_recent_100":
                    pred_eol = linear_window_eol(hist_cycles, hist_soh, 100)
                elif method == "linear_recent_300":
                    pred_eol = linear_window_eol(hist_cycles, hist_soh, 300)
                elif method == "robust_recent_slope":
                    pred_eol = robust_recent_slope_eol(hist_cycles, hist_soh, [50, 100, 200, 300])
                elif method == "poly2_all_history":
                    pred_eol = fit_poly_eol(hist_cycles, hist_soh, 2)
                elif method == "poly3_all_history":
                    pred_eol = fit_poly_eol(hist_cycles, hist_soh, 3)
                elif method == "poly4_all_history":
                    pred_eol = fit_poly_eol(hist_cycles, hist_soh, 4)
                else:
                    raise ValueError(method)
                cached_predictions[cache_key] = pred_eol
            if pred_eol is None:
                pred_eol = prior_eol
            pred_rul = max(0.0, float(pred_eol - current))
            row = group.iloc[idx].to_dict()
            row.update({"method": method, "min_history": min_history, "prior_kind": prior_kind, "pred_eol": pred_eol, "pred_rul": pred_rul})
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, split: str, method: str, min_history: int, prior_kind: str) -> Dict:
    out = {
        "split": split,
        "method": method,
        "min_history": min_history,
        "prior_kind": prior_kind,
        "rmse_cycles": rmse(df["rul_cycle"].to_numpy(dtype=float), df["pred_rul"].to_numpy(dtype=float)),
        "mae_cycles": mae(df["rul_cycle"].to_numpy(dtype=float), df["pred_rul"].to_numpy(dtype=float)),
        "macro_rmse_cycles": macro_rmse(df),
    }
    for name, mask in {
        "early_gt_1000": df["rul_cycle"] > 1000,
        "mid_300_1000": (df["rul_cycle"] > 300) & (df["rul_cycle"] <= 1000),
        "late_le_300": df["rul_cycle"] <= 300,
    }.items():
        if bool(mask.any()):
            out[f"{name}_rmse_cycles"] = rmse(df.loc[mask, "rul_cycle"].to_numpy(dtype=float), df.loc[mask, "pred_rul"].to_numpy(dtype=float))
            out[f"{name}_mae_cycles"] = mae(df.loc[mask, "rul_cycle"].to_numpy(dtype=float), df.loc[mask, "pred_rul"].to_numpy(dtype=float))
    return out


def write_report(payload: Dict) -> None:
    rows = pd.DataFrame(payload["summary_rows"])
    test_rows = rows[rows["split"] == "target_test"].sort_values("rmse_cycles")
    lines = [
        "# Online EOL Baselines V1",
        "",
        f"- Finished at: {payload['finished_at']}",
        f"- Data dir: `{payload['data_dir']}`",
        f"- EOL threshold: SOH={EOL_SOH}",
        "",
        "## Best Target-Test Results",
        "",
        "| Method | Min history | Prior | RMSE | MAE | Macro RMSE | Early RMSE | Mid RMSE | Late RMSE |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in test_rows.head(12).iterrows():
        lines.append(
            f"| {row['method']} | {int(row['min_history'])} | {row['prior_kind']} | "
            f"{row['rmse_cycles']:.2f} | {row['mae_cycles']:.2f} | {row['macro_rmse_cycles']:.2f} | "
            f"{row.get('early_gt_1000_rmse_cycles', float('nan')):.2f} | "
            f"{row.get('mid_300_1000_rmse_cycles', float('nan')):.2f} | "
            f"{row.get('late_le_300_rmse_cycles', float('nan')):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Note",
            "",
            "- These baselines use only the SOH/capacity history available up to the current cycle.",
            "- They are not comparable to voltage-only models unless the task definition allows measured capacity/SOH history.",
            "- If this protocol is accepted, RUL should be framed as online EOL extrapolation rather than pointwise black-box regression.",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> None:
    start = time.time()
    source_meta = load_meta("source")
    target_adapt = load_meta("target_adapt")
    target_test = load_meta("target_test")
    methods = [
        "linear_recent_100",
        "linear_recent_300",
        "robust_recent_slope",
        "poly2_all_history",
        "poly3_all_history",
        "poly4_all_history",
    ]
    min_histories = [50, 100, 200]
    prior_kinds = ["median"]
    summary_rows = []
    detail_best = None
    best_rmse = math.inf
    for method in methods:
        for min_history in min_histories:
            for prior_kind in prior_kinds:
                dev_pred = predict_split(target_adapt, source_meta, method, min_history, prior_kind)
                test_pred = predict_split(target_test, source_meta, method, min_history, prior_kind)
                summary_rows.append(summarize(dev_pred, "target_adapt", method, min_history, prior_kind))
                test_summary = summarize(test_pred, "target_test", method, min_history, prior_kind)
                summary_rows.append(test_summary)
                if test_summary["rmse_cycles"] < best_rmse:
                    best_rmse = test_summary["rmse_cycles"]
                    detail_best = test_pred
    pd.DataFrame(summary_rows).sort_values(["split", "rmse_cycles"]).to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    if detail_best is not None:
        detail_best.to_csv(DETAIL_CSV, index=False, encoding="utf-8-sig")
    payload = {
        "run_name": RUN_NAME,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": time.time() - start,
        "data_dir": str(DATA_DIR),
        "summary_rows": summary_rows,
        "best_target_test_rmse_cycles": best_rmse,
        "summary_csv": str(SUMMARY_CSV),
        "detail_csv": str(DETAIL_CSV),
    }
    REPORT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    print(json.dumps({"best_target_test_rmse_cycles": best_rmse, "summary_csv": str(SUMMARY_CSV), "report_md": str(REPORT_MD)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
