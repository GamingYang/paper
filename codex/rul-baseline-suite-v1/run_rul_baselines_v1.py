import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RUN_NAME = "rul-baseline-suite-v1"
SEED = 42
GLOBAL_MAX_CYCLE = 3000.0

SCRIPT_PATH = Path(__file__).resolve()
RUN_DIR = SCRIPT_PATH.parent
ROOT = RUN_DIR.parents[1]
RESULT_CSV = RUN_DIR / f"{RUN_NAME}_summary.csv"
RESULT_JSON = RUN_DIR / f"{RUN_NAME}_summary.json"
REPORT_MD = RUN_DIR / f"{RUN_NAME}_report.md"


def find_data_dir(root: Path) -> Path:
    for child in root.iterdir():
        candidate = child / "processed_clean"
        if child.is_dir() and child.name.lower().startswith("python") and (candidate / "clean_source_x.npy").exists():
            return candidate
    raise FileNotFoundError("processed_clean data directory not found")


DATA_DIR = find_data_dir(ROOT)


def load_split(split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    x = np.load(DATA_DIR / f"clean_{split}_x.npy").astype(np.float32)
    p = np.load(DATA_DIR / f"clean_{split}_p.npy").astype(np.float32).reshape(-1)
    rul = np.load(DATA_DIR / f"clean_{split}_rul.npy").astype(np.float32).reshape(-1)
    bidx = np.load(DATA_DIR / f"clean_{split}_battery_index.npy").astype(np.int64).reshape(-1)
    meta = pd.read_csv(DATA_DIR / f"clean_{split}_meta.csv")
    return x, p, rul, bidx, meta


def make_features(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    diff = np.diff(x, axis=1)
    quantiles = np.quantile(x, [0.10, 0.25, 0.50, 0.75, 0.90], axis=1).T.astype(np.float32)
    t = np.linspace(-0.5, 0.5, x.shape[1], dtype=np.float32)
    slope = (x * t).mean(axis=1) / (t * t).mean()
    return np.column_stack(
        [
            p / GLOBAL_MAX_CYCLE,
            p,
            np.log1p(p),
            x.mean(axis=1),
            x.std(axis=1),
            x.min(axis=1),
            x.max(axis=1),
            x[:, 0],
            x[:, -1],
            x[:, -1] - x[:, 0],
            slope,
            diff.mean(axis=1),
            diff.std(axis=1),
            diff.min(axis=1),
            diff.max(axis=1),
            quantiles,
        ]
    ).astype(np.float32)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true).reshape(-1) - np.asarray(y_pred).reshape(-1)) ** 2)))


def macro_rmse(y_true: np.ndarray, y_pred: np.ndarray, bidx: np.ndarray) -> float:
    vals = []
    for b in np.unique(bidx):
        mask = bidx == b
        vals.append(rmse(y_true[mask], y_pred[mask]))
    return float(np.mean(vals))


def evaluate(name: str, protocol: str, dev_true: np.ndarray, dev_pred: np.ndarray, dev_bidx: np.ndarray,
             test_true: np.ndarray, test_pred: np.ndarray, test_bidx: np.ndarray) -> Dict:
    dev_pred = np.maximum(dev_pred, 0.0)
    test_pred = np.maximum(test_pred, 0.0)
    return {
        "model": name,
        "protocol": protocol,
        "target_adapt_rmse_cycles": rmse(dev_true, dev_pred),
        "target_adapt_mae_cycles": float(mean_absolute_error(dev_true, dev_pred)),
        "target_adapt_macro_rmse_cycles": macro_rmse(dev_true, dev_pred, dev_bidx),
        "target_test_rmse_cycles": rmse(test_true, test_pred),
        "target_test_mae_cycles": float(mean_absolute_error(test_true, test_pred)),
        "target_test_macro_rmse_cycles": macro_rmse(test_true, test_pred, test_bidx),
    }


def cell_lifetimes(meta: pd.DataFrame) -> np.ndarray:
    return np.asarray([group["cycle"].max() for _, group in meta.groupby("battery_index")], dtype=float)


def write_report(payload: Dict) -> None:
    rows = payload["results"]
    df = pd.DataFrame(rows).sort_values("target_test_rmse_cycles")
    lines = [
        "# RUL baseline suite v1",
        "",
        f"- Finished at: {payload['finished_at']}",
        f"- Elapsed seconds: {payload['elapsed_seconds']:.1f}",
        f"- Data dir: `{payload['data_dir']}`",
        "",
        "## Result Summary",
        "",
        "| Model | Protocol | Dev RMSE | Test RMSE | Test MAE | Test Macro RMSE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['model']} | {row['protocol']} | "
            f"{row['target_adapt_rmse_cycles']:.2f} | {row['target_test_rmse_cycles']:.2f} | "
            f"{row['target_test_mae_cycles']:.2f} | {row['target_test_macro_rmse_cycles']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The naive source-domain median EOL baseline is already strong because RUL is structurally close to EOL minus current cycle.",
            "- ExtraTrees with engineered partial-curve statistics improves over the quick neural RUL run, but it remains far from a publishable RUL result.",
            "- Full target-adapt supervision is reported as a small-labeled-target upper protocol, not as an unsupervised domain adaptation result.",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> None:
    start = time.time()
    s_x, s_p, s_y, s_b, s_meta = load_split("source")
    ta_x, ta_p, ta_y, ta_b, ta_meta = load_split("target_adapt")
    tt_x, tt_p, tt_y, tt_b, tt_meta = load_split("target_test")

    s_life = cell_lifetimes(s_meta)
    ta_life = cell_lifetimes(ta_meta)

    results: List[Dict] = []
    for name, eol in [
        ("mean_eol_source", float(np.mean(s_life))),
        ("median_eol_source", float(np.median(s_life))),
        ("mean_eol_source_plus_target_adapt", float(np.mean(np.r_[s_life, ta_life]))),
        ("median_eol_source_plus_target_adapt", float(np.median(np.r_[s_life, ta_life]))),
    ]:
        protocol = "source_only" if "target_adapt" not in name else "source_plus_labeled_target_adapt_eol_prior"
        results.append(evaluate(name, protocol, ta_y, eol - ta_p, ta_b, tt_y, eol - tt_p, tt_b))

    coeff = np.polyfit(s_p / GLOBAL_MAX_CYCLE, s_y / GLOBAL_MAX_CYCLE, 3)
    results.append(
        evaluate(
            "cubic_cycle_to_rul_source",
            "source_only",
            ta_y,
            np.polyval(coeff, ta_p / GLOBAL_MAX_CYCLE) * GLOBAL_MAX_CYCLE,
            ta_b,
            tt_y,
            np.polyval(coeff, tt_p / GLOBAL_MAX_CYCLE) * GLOBAL_MAX_CYCLE,
            tt_b,
        )
    )

    fs = make_features(s_x, s_p)
    fta = make_features(ta_x, ta_p)
    ftt = make_features(tt_x, tt_p)

    etr_direct = ExtraTreesRegressor(
        n_estimators=120,
        max_depth=20,
        min_samples_leaf=8,
        random_state=SEED,
        n_jobs=1,
    )
    etr_direct.fit(fs, s_y)
    dev_pred = etr_direct.predict(fta)
    test_pred = etr_direct.predict(ftt)
    results.append(evaluate("extra_trees_direct_rul_source", "source_only", ta_y, dev_pred, ta_b, tt_y, test_pred, tt_b))

    calibrator = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    calibrator.fit(np.column_stack([dev_pred, ta_p]), ta_y)
    results.append(
        evaluate(
            "extra_trees_direct_rul_source_linear_calibrated",
            "source_plus_full_labeled_target_adapt_calibration",
            ta_y,
            calibrator.predict(np.column_stack([dev_pred, ta_p])),
            ta_b,
            tt_y,
            calibrator.predict(np.column_stack([test_pred, tt_p])),
            tt_b,
        )
    )

    etr_direct_adapt = ExtraTreesRegressor(
        n_estimators=120,
        max_depth=20,
        min_samples_leaf=8,
        random_state=SEED,
        n_jobs=1,
    )
    etr_direct_adapt.fit(np.vstack([fs, fta]), np.r_[s_y, ta_y])
    results.append(
        evaluate(
            "extra_trees_direct_rul_source_plus_full_target_adapt",
            "source_plus_full_labeled_target_adapt_training",
            ta_y,
            etr_direct_adapt.predict(fta),
            ta_b,
            tt_y,
            etr_direct_adapt.predict(ftt),
            tt_b,
        )
    )

    elapsed = time.time() - start
    payload = {
        "run_name": RUN_NAME,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed,
        "data_dir": str(DATA_DIR),
        "source_lifetime_mean": float(np.mean(s_life)),
        "source_lifetime_median": float(np.median(s_life)),
        "target_adapt_lifetimes": [int(x) for x in ta_life.tolist()],
        "target_test_lifetimes": [int(x) for x in cell_lifetimes(tt_meta).tolist()],
        "results": results,
    }
    pd.DataFrame(results).sort_values("target_test_rmse_cycles").to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")
    RESULT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    print(json.dumps({"elapsed_seconds": elapsed, "result_csv": str(RESULT_CSV), "report_md": str(REPORT_MD)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
