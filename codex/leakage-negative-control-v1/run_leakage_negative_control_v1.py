from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # Keep the audit runnable in minimal Python runtimes.
    plt = None


ROOT = Path(r"C:\Users\ROG\Desktop\paper")
STRICT_SCRIPT = ROOT / "codex" / "strict-eol-protocol-v1" / "strict_eol_protocol_v1.py"
OUT = ROOT / "codex" / "leakage-negative-control-v1"
RUN = "leakage-negative-control-v1"

HORIZONS = [50, 100, 200, 300, 500, 800, 1000]
VALID_FAMILIES = {
    "strict_fused_all": lambda cols: [c for c in cols],
    "soh_capacity_only": lambda cols: [c for c in cols if c.startswith("soh")],
    "x_curve_only": lambda cols: [c for c in cols if c.startswith("x_")],
}


def load_strict_module():
    spec = importlib.util.spec_from_file_location("strict_eol_protocol_v1", STRICT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {STRICT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rmse(y_true: Iterable[float], pred: Iterable[float]) -> float:
    y = np.asarray(list(y_true), dtype=float)
    p = np.asarray(list(pred), dtype=float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def sanitize_xy(train_df: pd.DataFrame, eval_df: pd.DataFrame, cols: list[str]):
    x_train = train_df[cols].replace([np.inf, -np.inf], np.nan)
    x_eval = eval_df[cols].replace([np.inf, -np.inf], np.nan)
    med = x_train.median(numeric_only=True)
    x_train = x_train.fillna(med).fillna(0.0)
    x_eval = x_eval.fillna(med).fillna(0.0)
    y_train = train_df["eol_cycle"].to_numpy(float)
    y_eval = eval_df["eol_cycle"].to_numpy(float)
    return x_train, y_train, x_eval, y_eval


def predict_model(model, train_df: pd.DataFrame, eval_df: pd.DataFrame, cols: list[str]):
    x_train, y_train, x_eval, y_eval = sanitize_xy(train_df, eval_df, cols)
    fitted = clone(model)
    fitted.fit(x_train, y_train)
    pred = fitted.predict(x_eval)
    return y_eval, np.asarray(pred, dtype=float)


def batch9_loocv(model, source_df: pd.DataFrame, target_df: pd.DataFrame, cols: list[str]):
    rows = []
    for heldout_cell in sorted(target_df["cell_id"].unique()):
        heldout = target_df[target_df["cell_id"] == heldout_cell]
        target_train = target_df[target_df["cell_id"] != heldout_cell]
        train = pd.concat([source_df, target_train], ignore_index=True)
        y, pred = predict_model(model, train, heldout, cols)
        part = heldout[["cell_id", "battery_index", "batch_id", "horizon", "eol_cycle"]].copy()
        part["pred_eol_cycle"] = pred
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def final_test(model, source_df: pd.DataFrame, target_df: pd.DataFrame, test_df: pd.DataFrame, cols: list[str]):
    train = pd.concat([source_df, target_df], ignore_index=True)
    y, pred = predict_model(model, train, test_df, cols)
    out = test_df[["cell_id", "battery_index", "batch_id", "horizon", "eol_cycle"]].copy()
    out["pred_eol_cycle"] = pred
    out["eol_error"] = out["pred_eol_cycle"] - out["eol_cycle"]
    out["abs_eol_error"] = out["eol_error"].abs()
    return out


def add_invalid_leak_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["invalid_observed_ratio_true_eol"] = out["horizon"] / out["eol_cycle"]
    out["invalid_remaining_cycles_at_horizon"] = out["eol_cycle"] - out["horizon"]
    return out


def summarize_predictions(df: pd.DataFrame) -> dict[str, float]:
    return {
        "rmse_cycles": rmse(df["eol_cycle"], df["pred_eol_cycle"]),
        "mae_cycles": float(mean_absolute_error(df["eol_cycle"], df["pred_eol_cycle"])),
        "max_abs_error_cycles": float(df["abs_eol_error"].max() if "abs_eol_error" in df else np.abs(df["pred_eol_cycle"] - df["eol_cycle"]).max()),
        "n_cells": int(df["cell_id"].nunique()),
    }


def main() -> None:
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    strict = load_strict_module()
    splits = {split: strict.load_split(split) for split in ["source", "target_adapt", "target_test"]}
    model = ExtraTreesRegressor(n_estimators=220, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=1)
    leak_model = make_pipeline(StandardScaler(), Ridge(alpha=1e-6))

    rows = []
    selected_rows = []
    prediction_tables = []
    feature_audit_rows = []

    for horizon in HORIZONS:
        valid_frames = {
            split: strict.build_cell_features(x, meta, horizon)
            for split, (x, meta) in splits.items()
        }
        frames = {split: add_invalid_leak_features(df) for split, df in valid_frames.items()}

        # Strict valid families are computed before injecting invalid_* stress-test
        # fields. This is the key guard that the negative-control fields cannot
        # contaminate valid feature-family rows.
        strict_cols = strict.feature_columns(valid_frames["source"])
        for family_name, selector in VALID_FAMILIES.items():
            cols = selector(strict_cols)
            if not cols:
                continue
            pred9 = batch9_loocv(model, valid_frames["source"], valid_frames["target_adapt"], cols)
            s9 = summarize_predictions(pred9)
            test_pred = final_test(model, valid_frames["source"], valid_frames["target_adapt"], valid_frames["target_test"], cols)
            s10 = summarize_predictions(test_pred)
            rows.append(
                {
                    "family": family_name,
                    "model": "extra_trees",
                    "horizon": horizon,
                    "validity": "valid_protocol_feature_family",
                    "n_features": len(cols),
                    "batch9_rmse": s9["rmse_cycles"],
                    "batch9_mae": s9["mae_cycles"],
                    "batch10_rmse_report_only": s10["rmse_cycles"],
                    "batch10_mae_report_only": s10["mae_cycles"],
                    "selection_rule": "Batch9 LOOCV only; Batch10 report-only",
                }
            )
            for c in cols:
                feature_audit_rows.append({"family": family_name, "horizon": horizon, "feature": c, "used": True, "validity": "valid"})

        invalid_cols = strict_cols + ["invalid_observed_ratio_true_eol", "invalid_remaining_cycles_at_horizon"]
        pred9_leak = batch9_loocv(leak_model, frames["source"], frames["target_adapt"], invalid_cols)
        s9_leak = summarize_predictions(pred9_leak)
        pred10_leak = final_test(leak_model, frames["source"], frames["target_adapt"], frames["target_test"], invalid_cols)
        s10_leak = summarize_predictions(pred10_leak)
        rows.append(
            {
                "family": "invalid_observed_ratio_or_endpoint_leak",
                "model": "ridge_leakage_stress_test",
                "horizon": horizon,
                "validity": "invalid_negative_control_do_not_use",
                "n_features": len(invalid_cols),
                "batch9_rmse": s9_leak["rmse_cycles"],
                "batch9_mae": s9_leak["mae_cycles"],
                "batch10_rmse_report_only": s10_leak["rmse_cycles"],
                "batch10_mae_report_only": s10_leak["mae_cycles"],
                "selection_rule": "Invalid leakage stress test; proves blacklist importance",
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(OUT / f"{RUN}_horizon_family_leakage_stress_results.csv", index=False)
    pd.DataFrame(feature_audit_rows).to_csv(OUT / f"{RUN}_feature_family_used_features.csv", index=False)

    valid = result[result["validity"] == "valid_protocol_feature_family"].copy()
    selected = (
        valid.sort_values("batch9_rmse")
        .groupby("family", as_index=False)
        .first()
        .sort_values("batch9_rmse")
    )
    selected.to_csv(OUT / f"{RUN}_selected_by_batch9_valid_families.csv", index=False)

    plot_files: list[str] = []
    leak_df = result[result["family"] == "invalid_observed_ratio_or_endpoint_leak"].sort_values("horizon")
    if plt is not None:
        fig_df = valid.copy()
        plt.figure(figsize=(8, 4.8))
        for family, group in fig_df.groupby("family"):
            group = group.sort_values("horizon")
            plt.plot(group["horizon"], group["batch9_rmse"], marker="o", label=family)
        plt.xlabel("Observed horizon (cycles)")
        plt.ylabel("Batch 9 LOOCV RMSE (cycles)")
        plt.title("Validation-only horizon sensitivity by feature family")
        plt.legend()
        plt.tight_layout()
        p1 = OUT / f"{RUN}_horizon_feature_family_batch9_rmse.png"
        plt.savefig(p1, dpi=180)
        plot_files.append(str(p1.name))
        plt.close()

        plt.figure(figsize=(8, 4.8))
        plt.plot(leak_df["horizon"], leak_df["batch9_rmse"], marker="o", label="Invalid leakage stress test")
        plt.axhline(valid["batch9_rmse"].min(), linestyle="--", color="gray", label="Best valid Batch9 RMSE")
        plt.xlabel("Observed horizon (cycles)")
        plt.ylabel("Batch 9 RMSE (cycles)")
        plt.title("Invalid leakage negative control")
        plt.legend()
        plt.tight_layout()
        p2 = OUT / f"{RUN}_invalid_leakage_negative_control.png"
        plt.savefig(p2, dpi=180)
        plot_files.append(str(p2.name))
        plt.close()

    forbidden = sorted(strict.FORBIDDEN_MODEL_INPUTS)
    strict_cols = strict.feature_columns(strict.build_cell_features(*splits["source"], 1000))
    blacklist_intersection = sorted(set(forbidden) & set(strict_cols))
    payload = {
        "run": RUN,
        "finished_at": pd.Timestamp.now().isoformat(),
        "elapsed_seconds": round(time.time() - start, 2),
        "protocol": {
            "source": "Batch 1-8",
            "target_development_adaptation": "Batch 9",
            "blind_test": "Batch 10 report-only after Batch9 selection",
        },
        "forbidden_model_inputs": forbidden,
        "blacklist_intersection_with_strict_cols_at_1000": blacklist_intersection,
        "valid_family_selected_by_batch9": selected.to_dict(orient="records"),
        "invalid_negative_control": result[result["validity"] == "invalid_negative_control_do_not_use"].to_dict(orient="records"),
        "plot_files": plot_files,
    }
    (OUT / f"{RUN}_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"# {RUN}",
        "",
        f"- Finished at: {payload['finished_at']}",
        f"- Elapsed seconds: {payload['elapsed_seconds']}",
        "- Purpose: supplementary feature-family / horizon / leakage negative-control audit.",
        "- Protocol: Batch 1-8 source, Batch 9 validation/adaptation, Batch 10 report-only after Batch9 selection.",
        "- This is not a new model-selection run for the main paper; it is reviewer-defense material.",
        "",
        "## Valid Feature-Family Selection by Batch 9",
        "",
        "| Family | Selected horizon | Features | Batch9 RMSE | Batch9 MAE | Batch10 RMSE report-only | Batch10 MAE report-only |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in selected.iterrows():
        lines.append(
            f"| {row['family']} | {int(row['horizon'])} | {int(row['n_features'])} | "
            f"{row['batch9_rmse']:.2f} | {row['batch9_mae']:.2f} | "
            f"{row['batch10_rmse_report_only']:.2f} | {row['batch10_mae_report_only']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Leakage Blacklist Check",
            "",
            f"- Forbidden inputs: `{', '.join(forbidden)}`.",
            f"- Intersection between forbidden inputs and strict model columns at 1000 cycles: `{blacklist_intersection}`.",
            "- Direct endpoint, RUL, full life length, observed-ratio, horizon, observed-count, cell id and batch id fields are not allowed in the valid feature families.",
            "",
            "## Invalid Leakage Negative Control",
            "",
            "The invalid negative control injects fields derived from true EOL, including `invalid_observed_ratio_true_eol = horizon / eol_cycle` and `invalid_remaining_cycles_at_horizon = eol_cycle - horizon`. These fields are deliberately invalid and must never be used as model inputs. Their purpose is to demonstrate why the blacklist is necessary.",
            "",
            "| Horizon | Invalid Batch9 RMSE | Invalid Batch10 RMSE report-only |",
            "|---:|---:|---:|",
        ]
    )
    for _, row in leak_df.iterrows():
        lines.append(
            f"| {int(row['horizon'])} | {row['batch9_rmse']:.4f} | {row['batch10_rmse_report_only']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Manuscript-Safe Interpretation",
            "",
            "- Safe: report valid feature-family and horizon sensitivity as Batch9 validation-only evidence.",
            "- Safe: state that endpoint-derived fields collapse the task and are therefore explicitly blacklisted.",
            "- Unsafe: use invalid leakage negative-control numbers as performance evidence.",
            "- Unsafe: select any horizon or feature family by Batch10 report-only metrics.",
            "",
            "## Output Files",
            "",
            f"- `{RUN}_horizon_family_leakage_stress_results.csv`",
            f"- `{RUN}_selected_by_batch9_valid_families.csv`",
            f"- `{RUN}_feature_family_used_features.csv`",
            f"- plot files: `{plot_files if plot_files else 'not generated; matplotlib unavailable'}`",
            f"- `{RUN}_summary.json`",
        ]
    )
    (OUT / f"{RUN}_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(OUT / f"{RUN}_report.md")


if __name__ == "__main__":
    main()
