import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVR


RUN_NAME = "cell-level-eol-fingerprint-v1"
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
PRED_CSV = OUT_DIR / f"{RUN_NAME}_test_cell_predictions.csv"
REPORT_MD = OUT_DIR / f"{RUN_NAME}_report.md"
REPORT_JSON = OUT_DIR / f"{RUN_NAME}_summary.json"


def load_split(split: str) -> Tuple[np.ndarray, pd.DataFrame]:
    x = np.load(DATA_DIR / f"clean_{split}_x.npy").astype(np.float32)
    meta = pd.read_csv(DATA_DIR / f"clean_{split}_meta.csv")
    meta["split"] = split
    return x, meta


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


def make_curve_stats(values: np.ndarray, prefix: str) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {f"{prefix}_{name}": np.nan for name in ["first", "last", "mean", "std", "min", "max", "delta"]}
    return {
        f"{prefix}_first": float(values[0]),
        f"{prefix}_last": float(values[-1]),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_delta": float(values[-1] - values[0]),
    }


def cell_features_for_horizon(x: np.ndarray, meta: pd.DataFrame, horizon: int) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    x_df = pd.DataFrame({"row_idx": np.arange(len(meta)), "battery_index": meta["battery_index"].to_numpy(), "cycle": meta["cycle"].to_numpy()})
    for bid, group in meta.groupby("battery_index"):
        group = group.sort_values("cycle")
        observed = group[group["cycle"] <= min(horizon, group["cycle"].max())].copy()
        if len(observed) < 5:
            continue
        row_indices = observed.index.to_numpy(dtype=int)
        cycles = observed["cycle"].to_numpy(dtype=float)
        soh = observed["y_nominal"].to_numpy(dtype=float)
        eol = float(group["cycle"].max())
        row: Dict[str, float] = {
            "split": str(group["split"].iloc[0]),
            "battery_index": int(bid),
            "cell_id": str(group["cell_id"].iloc[0]),
            "batch_id": int(group["batch_id"].iloc[0]),
            "horizon": int(horizon),
            "n_observed": int(len(observed)),
            "eol_cycle": eol,
            "last_observed_cycle": float(cycles[-1]),
        }
        row.update(make_curve_stats(soh, "soh"))
        for window in [20, 50, 100, 200]:
            local = observed.tail(window)
            if len(local) >= 5:
                c = local["cycle"].to_numpy(dtype=float)
                y = local["y_nominal"].to_numpy(dtype=float)
                slope, intercept = np.polyfit(c, y, 1)
                row[f"soh_slope_last_{window}"] = float(slope)
                row[f"soh_intercept_last_{window}"] = float(intercept)
                row[f"soh_drop_last_{window}"] = float(y[-1] - y[0])
            else:
                row[f"soh_slope_last_{window}"] = np.nan
                row[f"soh_intercept_last_{window}"] = np.nan
                row[f"soh_drop_last_{window}"] = np.nan
        for degree in [1, 2, 3, 4]:
            if len(cycles) >= degree + 2:
                coef = np.polyfit(cycles / 3000.0, soh, degree)
                for idx, value in enumerate(coef):
                    row[f"soh_poly{degree}_coef_{idx}"] = float(value)
            else:
                for idx in range(degree + 1):
                    row[f"soh_poly{degree}_coef_{idx}"] = np.nan
        x_obs = x[row_indices]
        x_mean_curve = np.mean(x_obs, axis=0)
        x_last = x_obs[-1]
        x_delta = x_obs[-1] - x_obs[0]
        for prefix, arr in [("x_mean_curve", x_mean_curve), ("x_last_curve", x_last), ("x_delta_curve", x_delta)]:
            row.update(make_curve_stats(arr, prefix))
            quant = np.quantile(arr, [0.05, 0.25, 0.50, 0.75, 0.95])
            for q, value in zip([5, 25, 50, 75, 95], quant):
                row[f"{prefix}_q{q}"] = float(value)
        if len(row_indices) >= 100:
            diff_100 = x[row_indices[min(99, len(row_indices) - 1)]] - x[row_indices[9]]
            row.update(make_curve_stats(diff_100, "x_cycle100_minus_10"))
            row["x_cycle100_minus_10_logvar"] = float(np.log(np.var(diff_100) + 1e-12))
        else:
            for key, value in make_curve_stats(np.asarray([]), "x_cycle100_minus_10").items():
                row[key] = value
            row["x_cycle100_minus_10_logvar"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    drop_cols = ["split", "battery_index", "cell_id", "batch_id", "eol_cycle"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    y = df["eol_cycle"].to_numpy(dtype=float)
    return X, y, feature_cols


def evaluate_model(name: str, model, train_df: pd.DataFrame, dev_df: pd.DataFrame, test_df: pd.DataFrame, protocol: str) -> Tuple[Dict, pd.DataFrame]:
    train_x, train_y, feature_cols = prepare_xy(train_df)
    dev_x = dev_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    test_x = test_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    med = train_x.median(numeric_only=True)
    dev_x = dev_x.fillna(med).fillna(0.0)
    test_x = test_x.fillna(med).fillna(0.0)
    model.fit(train_x, train_y)
    dev_pred = model.predict(dev_x)
    test_pred = model.predict(test_x)
    row = {
        "model": name,
        "protocol": protocol,
        "horizon": int(test_df["horizon"].iloc[0]),
        "target_adapt_eol_rmse": rmse(dev_df["eol_cycle"].to_numpy(dtype=float), dev_pred),
        "target_adapt_eol_mae": float(mean_absolute_error(dev_df["eol_cycle"].to_numpy(dtype=float), dev_pred)),
        "target_test_eol_rmse": rmse(test_df["eol_cycle"].to_numpy(dtype=float), test_pred),
        "target_test_eol_mae": float(mean_absolute_error(test_df["eol_cycle"].to_numpy(dtype=float), test_pred)),
    }
    pred_df = test_df[["cell_id", "battery_index", "batch_id", "horizon", "eol_cycle", "last_observed_cycle"]].copy()
    pred_df["model"] = name
    pred_df["protocol"] = protocol
    pred_df["pred_eol_cycle"] = test_pred
    pred_df["abs_error"] = np.abs(pred_df["eol_cycle"] - pred_df["pred_eol_cycle"])
    return row, pred_df


def model_zoo() -> List[Tuple[str, object]]:
    return [
        ("ridge", make_pipeline(StandardScaler(), Ridge(alpha=50.0))),
        ("elastic_net", make_pipeline(StandardScaler(), ElasticNet(alpha=0.05, l1_ratio=0.2, max_iter=5000, random_state=42))),
        ("svr_rbf", make_pipeline(RobustScaler(), SVR(C=300.0, epsilon=40.0, gamma="scale"))),
        ("kernel_ridge_rbf", make_pipeline(StandardScaler(), KernelRidge(alpha=30.0, kernel="rbf", gamma=0.03))),
        ("extra_trees", ExtraTreesRegressor(n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=1)),
        ("random_forest", RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=1)),
        ("gbr", GradientBoostingRegressor(n_estimators=120, learning_rate=0.03, max_depth=2, random_state=42, subsample=0.8)),
    ]


def write_report(payload: Dict) -> None:
    df = pd.DataFrame(payload["summary_rows"]).sort_values("target_test_eol_rmse")
    lines = [
        "# Cell-Level EOL Fingerprint V1",
        "",
        f"- Finished at: {payload['finished_at']}",
        f"- Data dir: `{payload['data_dir']}`",
        "",
        "## Best Results",
        "",
        "| Model | Protocol | Horizon | Dev EOL RMSE | Test EOL RMSE | Test EOL MAE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in df.head(20).iterrows():
        lines.append(
            f"| {row['model']} | {row['protocol']} | {int(row['horizon'])} | "
            f"{row['target_adapt_eol_rmse']:.2f} | {row['target_test_eol_rmse']:.2f} | {row['target_test_eol_mae']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Cell-level EOL error is numerically equal to pointwise RUL error when RUL is computed as predicted EOL minus current cycle.",
            "- This protocol uses early observed SOH/capacity history and processed voltage-curve fingerprints.",
            "- The `source_plus_target_adapt` protocol is supervised small-target adaptation, not unsupervised domain adaptation.",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> None:
    start = time.time()
    splits = {}
    for split in ["source", "target_adapt", "target_test"]:
        splits[split] = load_split(split)
    horizons = [50, 100, 200, 300, 500, 800, 1000]
    summary_rows: List[Dict] = []
    pred_frames: List[pd.DataFrame] = []
    for horizon in horizons:
        feature_frames = {}
        for split, (x, meta) in splits.items():
            feature_frames[split] = cell_features_for_horizon(x, meta, horizon)
        for train_name, train_df in [
            ("source_only", feature_frames["source"]),
            ("source_plus_target_adapt", pd.concat([feature_frames["source"], feature_frames["target_adapt"]], ignore_index=True)),
        ]:
            for model_name, model in model_zoo():
                row, pred_df = evaluate_model(
                    model_name,
                    model,
                    train_df,
                    feature_frames["target_adapt"],
                    feature_frames["target_test"],
                    train_name,
                )
                summary_rows.append(row)
                pred_frames.append(pred_df)
    summary_df = pd.DataFrame(summary_rows).sort_values("target_test_eol_rmse")
    pred_df = pd.concat(pred_frames, ignore_index=True)
    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    pred_df.to_csv(PRED_CSV, index=False, encoding="utf-8-sig")
    payload = {
        "run_name": RUN_NAME,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": time.time() - start,
        "data_dir": str(DATA_DIR),
        "summary_rows": summary_rows,
        "summary_csv": str(SUMMARY_CSV),
        "pred_csv": str(PRED_CSV),
    }
    REPORT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    print(
        json.dumps(
            {
                "best_target_test_eol_rmse": float(summary_df.iloc[0]["target_test_eol_rmse"]),
                "best": summary_df.iloc[0].to_dict(),
                "report_md": str(REPORT_MD),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
