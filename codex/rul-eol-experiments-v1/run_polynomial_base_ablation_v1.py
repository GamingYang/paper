import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


RUN_NAME = "polynomial-base-ablation-v1"
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = next(
    child / "processed_clean"
    for child in ROOT.iterdir()
    if child.is_dir()
    and child.name.lower().startswith("python")
    and (child / "processed_clean" / "clean_source_p.npy").exists()
)
OUT_DIR = Path(__file__).resolve().parent
SUMMARY_CSV = OUT_DIR / f"{RUN_NAME}_summary.csv"
REPORT_MD = OUT_DIR / f"{RUN_NAME}_report.md"
REPORT_JSON = OUT_DIR / f"{RUN_NAME}_summary.json"
GLOBAL_MAX_CYCLE = 3000.0


def load(split: str):
    p = np.load(DATA_DIR / f"clean_{split}_p.npy").reshape(-1).astype(float)
    soh = np.load(DATA_DIR / f"clean_{split}_y_nominal.npy").reshape(-1).astype(float)
    rul = np.load(DATA_DIR / f"clean_{split}_rul.npy").reshape(-1).astype(float)
    bidx = np.load(DATA_DIR / f"clean_{split}_battery_index.npy").reshape(-1).astype(int)
    return p, soh, rul, bidx


def rmse(y, pred) -> float:
    y = np.asarray(y)
    pred = np.asarray(pred)
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def mae(y, pred) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(pred))))


def macro_rmse(y, pred, bidx) -> float:
    return float(np.mean([rmse(y[bidx == b], pred[bidx == b]) for b in np.unique(bidx)]))


def eval_poly(task: str, degree: int, source, target_adapt, target_test) -> List[Dict]:
    s_p, s_soh, s_rul, _ = source
    y_source = s_soh if task == "soh" else s_rul / GLOBAL_MAX_CYCLE
    coef = np.polyfit(s_p / GLOBAL_MAX_CYCLE, y_source, degree)
    rows = []
    for split, data in [("target_adapt", target_adapt), ("target_test", target_test)]:
        p, soh, rul, bidx = data
        true = soh if task == "soh" else rul / GLOBAL_MAX_CYCLE
        pred = np.polyval(coef, p / GLOBAL_MAX_CYCLE)
        if task == "rul":
            true_eval = true * GLOBAL_MAX_CYCLE
            pred_eval = np.maximum(pred, 0.0) * GLOBAL_MAX_CYCLE
            unit = "cycles"
        else:
            true_eval = true
            pred_eval = pred
            unit = "soh_ratio"
        rows.append(
            {
                "task": task,
                "degree": degree,
                "split": split,
                "rmse": rmse(true_eval, pred_eval),
                "mae": mae(true_eval, pred_eval),
                "macro_rmse": macro_rmse(true_eval, pred_eval, bidx),
                "unit": unit,
                "coefficients": json.dumps([float(x) for x in coef]),
            }
        )
    return rows


def write_report(rows: List[Dict]) -> None:
    df = pd.DataFrame(rows)
    lines = [
        "# Polynomial Base Ablation V1",
        "",
        f"- Finished at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Data dir: `{DATA_DIR}`",
        "",
        "## SOH Base-Only Target-Test",
        "",
        "| Degree | RMSE | MAE | Macro RMSE |",
        "|---:|---:|---:|---:|",
    ]
    soh = df[(df["task"] == "soh") & (df["split"] == "target_test")].sort_values("degree")
    for _, row in soh.iterrows():
        lines.append(f"| {int(row['degree'])} | {row['rmse']:.6f} | {row['mae']:.6f} | {row['macro_rmse']:.6f} |")
    lines.extend(["", "## RUL Base-Only Target-Test", "", "| Degree | RMSE cycles | MAE cycles | Macro RMSE cycles |", "|---:|---:|---:|---:|"])
    rul = df[(df["task"] == "rul") & (df["split"] == "target_test")].sort_values("degree")
    for _, row in rul.iterrows():
        lines.append(f"| {int(row['degree'])} | {row['rmse']:.2f} | {row['mae']:.2f} | {row['macro_rmse']:.2f} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a base-only ablation. It tests the global degradation prior, not the full residual neural model.",
            "- If degree 4/5 improves only marginally while adding oscillation risk, degree 3 is easier to justify as the lowest flexible nonlinear prior.",
            "- A journal version should still include a full-model base-degree ablation for degree 2/3/4.",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> None:
    source = load("source")
    target_adapt = load("target_adapt")
    target_test = load("target_test")
    rows = []
    for task in ["soh", "rul"]:
        for degree in [1, 2, 3, 4, 5]:
            rows.extend(eval_poly(task, degree, source, target_adapt, target_test))
    pd.DataFrame(rows).to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    REPORT_JSON.write_text(json.dumps({"rows": rows, "summary_csv": str(SUMMARY_CSV)}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(rows)
    print(json.dumps({"summary_csv": str(SUMMARY_CSV), "report_md": str(REPORT_MD)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
