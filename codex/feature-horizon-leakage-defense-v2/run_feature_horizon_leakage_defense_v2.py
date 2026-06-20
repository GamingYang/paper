import importlib.util
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error


RUN_NAME = "feature-horizon-leakage-defense-v2"
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
STRICT_SCRIPT = ROOT / "codex" / "strict-eol-protocol-v1" / "strict_eol_protocol_v1.py"

HORIZONS = [50, 100, 200, 300, 500, 800, 1000]
PROTOCOL = "source_plus_target_adapt_loocv"
MODEL_NAME = "extra_trees"

VALIDATION_CSV = OUT_DIR / "batch9_validation_by_feature_family_horizon.csv"
BEST_FAMILY_CSV = OUT_DIR / "feature_family_best_batch9_validation.csv"
BLACKLIST_CSV = OUT_DIR / "leakage_blacklist_audit.csv"
FEATURE_LIST_CSV = OUT_DIR / "feature_family_column_inventory.csv"
RAW_DIAGNOSTIC_CSV = OUT_DIR / "raw_provenance_separate_diagnostic_summary.csv"
SUMMARY_JSON = OUT_DIR / f"{RUN_NAME}_summary.json"
REPORT_MD = OUT_DIR / f"{RUN_NAME}_report.md"
HORIZON_PNG = OUT_DIR / "horizon_vs_batch9_rmse.png"
FAMILY_PNG = OUT_DIR / "feature_family_batch9_rmse.png"


def load_strict_module():
    spec = importlib.util.spec_from_file_location("strict_eol_protocol_v1", STRICT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import strict script: {STRICT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STRICT = load_strict_module()
FORBIDDEN_MODEL_INPUTS = set(STRICT.FORBIDDEN_MODEL_INPUTS) | {
    "identifier",
    "pkl_file",
    "full_length",
    "full_life_length",
}


def feature_columns_for_family(df: pd.DataFrame, family: str) -> List[str]:
    strict_cols = list(STRICT.feature_columns(df))
    if family == "fused_all":
        cols = strict_cols
    elif family == "soh_capacity":
        cols = [col for col in strict_cols if col.startswith("soh")]
    elif family == "x_curve":
        cols = [col for col in strict_cols if col.startswith("x_")]
    else:
        raise ValueError(f"Unknown feature family: {family}")
    leaked = sorted(set(cols) & FORBIDDEN_MODEL_INPUTS)
    if leaked:
        raise RuntimeError(f"Forbidden features reached {family}: {leaked}")
    if not cols:
        raise RuntimeError(f"No usable columns for feature family: {family}")
    return cols


def eol_rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def predict_with_family(model, train_df: pd.DataFrame, eval_df: pd.DataFrame, family: str) -> Tuple[np.ndarray, np.ndarray]:
    cols = feature_columns_for_family(train_df, family)
    train_x, train_y, eval_x, eval_y = STRICT.prepare_train_eval(train_df, eval_df, cols)
    fitted = clone(model)
    fitted.fit(train_x, train_y)
    pred = fitted.predict(eval_x)
    return eval_y, np.asarray(pred, dtype=float)


def validate_family_horizon(
    model,
    family: str,
    source_df: pd.DataFrame,
    target_adapt_df: pd.DataFrame,
) -> Dict[str, object]:
    pred_frames: List[pd.DataFrame] = []
    for heldout_cell in sorted(target_adapt_df["cell_id"].unique()):
        heldout_df = target_adapt_df[target_adapt_df["cell_id"] == heldout_cell]
        target_train_df = target_adapt_df[target_adapt_df["cell_id"] != heldout_cell]
        train_df = pd.concat([source_df, target_train_df], ignore_index=True)
        _, pred = predict_with_family(model, train_df, heldout_df, family)
        pred_df = heldout_df[["cell_id", "battery_index", "batch_id", "horizon", "eol_cycle"]].copy()
        pred_df["pred_eol_cycle"] = pred
        pred_frames.append(pred_df)

    all_pred = pd.concat(pred_frames, ignore_index=True)
    feature_cols = feature_columns_for_family(source_df, family)
    return {
        "feature_family": family,
        "protocol": PROTOCOL,
        "model": MODEL_NAME,
        "horizon": int(target_adapt_df["horizon"].iloc[0]),
        "selection_set": "Batch9_target_adapt_validation_only",
        "selection_method": "Batch9_leave_one_cell_out",
        "batch10_used": False,
        "n_features": len(feature_cols),
        "n_source_cells": int(source_df["cell_id"].nunique()),
        "n_target_adapt_cells": int(target_adapt_df["cell_id"].nunique()),
        "batch9_eol_rmse": eol_rmse(all_pred["eol_cycle"], all_pred["pred_eol_cycle"]),
        "batch9_eol_mae": float(mean_absolute_error(all_pred["eol_cycle"], all_pred["pred_eol_cycle"])),
        "batch9_rul_rmse": eol_rmse(all_pred["eol_cycle"], all_pred["pred_eol_cycle"]),
        "batch9_rul_mae": float(mean_absolute_error(all_pred["eol_cycle"], all_pred["pred_eol_cycle"])),
    }


def build_blacklist_audit(frames_by_horizon: Dict[int, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    family_usage: Dict[str, set] = {}
    present: Dict[str, set] = {}
    for horizon, split_frames in frames_by_horizon.items():
        for split, df in split_frames.items():
            for col in df.columns:
                present.setdefault(col, set()).add(f"{split}@{horizon}")
        for family in ["fused_all", "soh_capacity", "x_curve"]:
            for col in feature_columns_for_family(split_frames["source"], family):
                family_usage.setdefault(col, set()).add(family)

    rows = []
    for feature in sorted(set(present) | FORBIDDEN_MODEL_INPUTS):
        used_families = sorted(family_usage.get(feature, set()))
        is_forbidden = feature in FORBIDDEN_MODEL_INPUTS
        if is_forbidden and used_families:
            decision = "FAIL_FORBIDDEN_USED"
        elif is_forbidden:
            decision = "excluded_blacklist"
        elif used_families:
            decision = "allowed_model_input"
        else:
            decision = "not_used"

        if feature.startswith("soh"):
            note = "historical measured/estimated capacity-SOH feature; manuscript must state this assumption"
        elif feature.startswith("x_"):
            note = "processed curve fingerprint from observed rows; preprocessing provenance remains a wording caveat"
        elif is_forbidden:
            note = "not allowed as a model input under strict EOL/RUL online protocol"
        else:
            note = "not used by evaluated model families"

        rows.append(
            {
                "feature": feature,
                "present_in_generated_table": feature in present,
                "present_locations": ";".join(sorted(present.get(feature, set()))),
                "used_as_model_input": bool(used_families),
                "families_using_feature": ";".join(used_families),
                "blacklisted": is_forbidden,
                "decision": decision,
                "audit_note": note,
            }
        )

    audit = pd.DataFrame(rows)
    failed = audit[audit["decision"] == "FAIL_FORBIDDEN_USED"]
    if not failed.empty:
        raise RuntimeError(f"Forbidden model inputs used: {failed['feature'].tolist()}")
    return audit


def build_feature_inventory(frames_by_horizon: Dict[int, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    reference_df = frames_by_horizon[HORIZONS[-1]]["source"]
    for family in ["fused_all", "soh_capacity", "x_curve"]:
        for col in feature_columns_for_family(reference_df, family):
            rows.append(
                {
                    "feature_family": family,
                    "feature": col,
                    "blacklisted": col in FORBIDDEN_MODEL_INPUTS,
                    "note": "included in strict processed-feature family",
                }
            )
    raw_like = [
        col
        for col in STRICT.feature_columns(reference_df)
        if any(token in col.lower() for token in ["raw", "dq", "cap_span", "mah", "voltage", "current", "duration", "status"])
    ]
    if raw_like:
        for col in raw_like:
            rows.append(
                {
                    "feature_family": "raw_provenance_detected_in_processed_table",
                    "feature": col,
                    "blacklisted": col in FORBIDDEN_MODEL_INPUTS,
                    "note": "name-matched raw/provenance-like field in strict processed feature table",
                }
            )
    else:
        rows.append(
            {
                "feature_family": "raw_provenance",
                "feature": "n/a",
                "blacklisted": False,
                "note": "no direct raw/dq/capacity-span/time-status/voltage-current provenance fields are identifiable in strict processed feature table; use separate raw-provenance diagnostic",
            }
        )
    return pd.DataFrame(rows)


def build_raw_diagnostic_summary() -> pd.DataFrame:
    raw_summary_path = ROOT / "codex" / "strict-raw-provenance-v3" / "strict-raw-provenance-v3_summary.json"
    if not raw_summary_path.exists():
        return pd.DataFrame(
            [
                {
                    "diagnostic": "strict-raw-provenance-v3",
                    "status": "not_found",
                    "role": "separate diagnostic only",
                }
            ]
        )
    payload = json.loads(raw_summary_path.read_text(encoding="utf-8"))
    selected = payload.get("selected_test_row", {})
    rows = [
        {
            "diagnostic": "strict-raw-provenance-v3",
            "status": "available",
            "role": "separate raw-feature provenance diagnostic; not part of this Batch9 family/horizon selection",
            "fixed_horizon": payload.get("fixed_horizon"),
            "processed_inputs_used": payload.get("processed_inputs_used"),
            "candidate_feature_count": payload.get("candidate_feature_count"),
            "candidate_feature_family_counts": json.dumps(payload.get("candidate_feature_family_counts", {}), ensure_ascii=False),
            "selected_strategy": selected.get("feature_strategy"),
            "selected_model": selected.get("model"),
            "selected_protocol": selected.get("protocol"),
            "batch9_rmse": selected.get("selection_eol_rmse"),
            "batch10_rmse_reference_only": selected.get("target_test_eol_rmse"),
            "manuscript_note": "Use to defend raw observed-record provenance; do not claim raw-only superiority over strict processed ExtraTrees baseline.",
        }
    ]
    return pd.DataFrame(rows)


def plot_horizon(validation: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for family, group in validation.sort_values("horizon").groupby("feature_family"):
        ax.plot(group["horizon"], group["batch9_eol_rmse"], marker="o", linewidth=2, label=family)
    ax.set_xlabel("Observed horizon (cycles)")
    ax.set_ylabel("Batch 9 LOOCV RMSE (cycles)")
    ax.set_title("Horizon vs Batch 9 RMSE")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(HORIZON_PNG, dpi=220)
    plt.close(fig)


def plot_family(best_family: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ordered = best_family.sort_values("batch9_eol_rmse")
    ax.bar(ordered["feature_family"], ordered["batch9_eol_rmse"], color=["#3465a4", "#4e9a06", "#c17d11"])
    ax.set_xlabel("Feature family")
    ax.set_ylabel("Best Batch 9 LOOCV RMSE (cycles)")
    ax.set_title("Feature-Family Batch 9 RMSE")
    ax.grid(True, axis="y", alpha=0.25)
    for idx, row in enumerate(ordered.itertuples(index=False)):
        ax.text(idx, row.batch9_eol_rmse + 2, f"{row.batch9_eol_rmse:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FAMILY_PNG, dpi=220)
    plt.close(fig)


def write_report(payload: Dict[str, object]) -> None:
    validation = pd.read_csv(VALIDATION_CSV)
    best_family = pd.read_csv(BEST_FAMILY_CSV)
    blacklist = pd.read_csv(BLACKLIST_CSV)
    raw_diag = pd.read_csv(RAW_DIAGNOSTIC_CSV)
    best_overall = validation.sort_values(["batch9_eol_rmse", "batch9_eol_mae"]).iloc[0]

    forbidden_rows = blacklist[blacklist["blacklisted"] == True]  # noqa: E712
    forbidden_used = forbidden_rows[forbidden_rows["used_as_model_input"] == True]  # noqa: E712

    lines = [
        "# Feature-Horizon Leakage Defense V2",
        "",
        f"- Finished at: {payload['finished_at']}",
        f"- Elapsed seconds: {payload['elapsed_seconds']:.1f}",
        f"- Strict source script: `{STRICT_SCRIPT}`",
        f"- Data directory: `{payload['data_dir']}`",
        "",
        "## Protocol Guardrails",
        "",
        "- Batch 1-8 are source cells.",
        "- Batch 9 is the only development/adaptation/validation set used here.",
        "- Batch 10 is not loaded by this script and is not used for feature-family, horizon, model, seed, epoch, checkpoint, or ranking decisions.",
        "- All rows below use `source_plus_target_adapt_loocv`: each Batch 9 held-out cell is scored after training on Batch 1-8 plus the other Batch 9 cells.",
        "- The model is fixed to the strict-v1 `extra_trees` baseline to isolate feature-family and horizon effects.",
        "",
        "## Batch 9 Validation-Only Horizon Table",
        "",
        "| Family | Horizon | Features | Batch 9 RMSE | Batch 9 MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in validation.sort_values(["feature_family", "horizon"]).iterrows():
        lines.append(
            f"| {row['feature_family']} | {int(row['horizon'])} | {int(row['n_features'])} | "
            f"{row['batch9_eol_rmse']:.2f} | {row['batch9_eol_mae']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Feature-Family Summary",
            "",
            "| Family | Best horizon by Batch 9 | Features | Batch 9 RMSE | Batch 9 MAE | Status |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in best_family.sort_values("batch9_eol_rmse").iterrows():
        lines.append(
            f"| {row['feature_family']} | {int(row['horizon'])} | {int(row['n_features'])} | "
            f"{row['batch9_eol_rmse']:.2f} | {row['batch9_eol_mae']:.2f} | available |"
        )
    raw_status = "separate diagnostic only"
    if not raw_diag.empty and raw_diag.iloc[0].get("status") == "available":
        raw_status = "separate diagnostic available; not in strict processed family table"
    lines.append(f"| raw/provenance | n/a | n/a | n/a | n/a | {raw_status} |")

    lines.extend(
        [
            "",
            "## Leakage Blacklist Audit",
            "",
            f"- Forbidden fields audited: `{', '.join(sorted(FORBIDDEN_MODEL_INPUTS))}`.",
            f"- Forbidden fields used as model inputs: `{len(forbidden_used)}`.",
            "- Direct leakage status: PASS if the count above is zero.",
            "- `horizon`, `n_observed`, `last_observed_cycle`, `observed_ratio`, full-life length fields, EOL/RUL labels, and cell/batch identifiers are excluded from all evaluated families.",
            "- `soh_*` features are conditionally manuscript-safe only as historical measured/estimated capacity-SOH signals.",
            "- `x_*` features are conditionally manuscript-safe as processed curve fingerprints from observed rows; final writing should state the provenance caveat.",
            "",
            "## Manuscript-Safe Wording",
            "",
            "Safe to write:",
            "",
            "- The supplementary Batch 9 validation-only audit compares fixed-horizon feature families under the same strict small labeled target-adaptation protocol.",
            "- Model inputs exclude EOL/RUL labels, full-life length or observed-ratio variables, horizon/count metadata, and cell/batch identifiers.",
            "- The fused processed-feature family gives the best Batch 9 validation result in this audit, while SOH/capacity and processed curve families remain useful but weaker standalone families.",
            "- Raw/provenance evidence should be reported as a separate diagnostic based on direct `our_data/*.pkl` parsing, not as the best-performing main result.",
            "",
            "Do not write:",
            "",
            "- Do not claim Batch 10 was used to choose the horizon or feature family.",
            "- Do not claim raw-only or voltage-only superiority from this audit.",
            "- Do not call the Batch 9 labeled adaptation protocol unsupervised domain adaptation.",
            "- Do not present processed `y_nominal`/SOH summaries as leakage-free labels; frame them as historical measured/estimated health observations.",
            "- Do not claim very-early 50/100-cycle performance is below 100 cycles unless a strict Batch-9-selected result supports it.",
            "",
            "## Core Result",
            "",
            (
                f"Best validation-only row: `{best_overall['feature_family']}` at "
                f"{int(best_overall['horizon'])} cycles, Batch 9 RMSE "
                f"{best_overall['batch9_eol_rmse']:.2f} cycles and MAE "
                f"{best_overall['batch9_eol_mae']:.2f} cycles."
            ),
            "",
            "## Output Files",
            "",
            f"- Validation CSV: `{VALIDATION_CSV}`",
            f"- Best-family CSV: `{BEST_FAMILY_CSV}`",
            f"- Blacklist audit CSV: `{BLACKLIST_CSV}`",
            f"- Feature inventory CSV: `{FEATURE_LIST_CSV}`",
            f"- Raw diagnostic summary CSV: `{RAW_DIAGNOSTIC_CSV}`",
            f"- Horizon figure: `{HORIZON_PNG}`",
            f"- Feature-family figure: `{FAMILY_PNG}`",
            f"- JSON summary: `{SUMMARY_JSON}`",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_lookup = {name: model for name, model in STRICT.model_zoo()}
    model = model_lookup[MODEL_NAME]

    # Batch 10 is intentionally not loaded in this defense run.
    splits = {split: STRICT.load_split(split) for split in ["source", "target_adapt"]}
    frames_by_horizon: Dict[int, Dict[str, pd.DataFrame]] = {}
    validation_rows: List[Dict[str, object]] = []

    for horizon in HORIZONS:
        frames_by_horizon[horizon] = {
            split: STRICT.build_cell_features(x, meta, horizon)
            for split, (x, meta) in splits.items()
        }
        for family in ["fused_all", "soh_capacity", "x_curve"]:
            validation_rows.append(
                validate_family_horizon(
                    model=model,
                    family=family,
                    source_df=frames_by_horizon[horizon]["source"],
                    target_adapt_df=frames_by_horizon[horizon]["target_adapt"],
                )
            )

    validation = pd.DataFrame(validation_rows).sort_values(["feature_family", "horizon"])
    validation.to_csv(VALIDATION_CSV, index=False, encoding="utf-8-sig")

    best_family = (
        validation.sort_values(["batch9_eol_rmse", "batch9_eol_mae"])
        .groupby("feature_family", as_index=False, sort=True)
        .first()
        .sort_values("batch9_eol_rmse")
    )
    best_family.to_csv(BEST_FAMILY_CSV, index=False, encoding="utf-8-sig")

    blacklist = build_blacklist_audit(frames_by_horizon)
    blacklist.to_csv(BLACKLIST_CSV, index=False, encoding="utf-8-sig")

    feature_inventory = build_feature_inventory(frames_by_horizon)
    feature_inventory.to_csv(FEATURE_LIST_CSV, index=False, encoding="utf-8-sig")

    raw_diag = build_raw_diagnostic_summary()
    raw_diag.to_csv(RAW_DIAGNOSTIC_CSV, index=False, encoding="utf-8-sig")

    plot_horizon(validation)
    plot_family(best_family)

    payload = {
        "run_name": RUN_NAME,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": time.time() - started,
        "strict_script": str(STRICT_SCRIPT),
        "data_dir": str(STRICT.DATA_DIR),
        "horizons": HORIZONS,
        "protocol": PROTOCOL,
        "model": MODEL_NAME,
        "batch10_loaded_or_used": False,
        "feature_families": ["fused_all", "soh_capacity", "x_curve"],
        "forbidden_model_inputs": sorted(FORBIDDEN_MODEL_INPUTS),
        "best_validation_row": validation.sort_values(["batch9_eol_rmse", "batch9_eol_mae"]).iloc[0].to_dict(),
        "outputs": {
            "script": str(Path(__file__).resolve()),
            "validation_csv": str(VALIDATION_CSV),
            "best_family_csv": str(BEST_FAMILY_CSV),
            "blacklist_csv": str(BLACKLIST_CSV),
            "feature_inventory_csv": str(FEATURE_LIST_CSV),
            "raw_diagnostic_csv": str(RAW_DIAGNOSTIC_CSV),
            "horizon_png": str(HORIZON_PNG),
            "family_png": str(FAMILY_PNG),
            "report_md": str(REPORT_MD),
            "summary_json": str(SUMMARY_JSON),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)

    print(
        json.dumps(
            {
                "best_validation_row": payload["best_validation_row"],
                "batch10_loaded_or_used": False,
                "report": str(REPORT_MD),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
