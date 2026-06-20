import importlib.util
import itertools
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, median_absolute_error


RUN_NAME = "target-label-budget-sensitivity-v1"
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
STRICT_SCRIPT = ROOT / "codex" / "strict-eol-protocol-v1" / "strict_eol_protocol_v1.py"
STRICT_SELECTED_SUMMARY = (
    ROOT
    / "codex"
    / "strict-eol-protocol-v1"
    / "strict-eol-protocol-v1_selected_test_summary.csv"
)
STRICT_SELECTED_PREDICTIONS = (
    ROOT
    / "codex"
    / "strict-eol-protocol-v1"
    / "strict-eol-protocol-v1_selected_test_predictions.csv"
)

MODEL_NAME = "extra_trees"
HORIZON = 1000
BUDGETS = [0, 1, 2, 4, 8]
K4_RANDOM_SEED = 20260604

REPORT_MD = OUT_DIR / f"{RUN_NAME}_report.md"
SUMMARY_JSON = OUT_DIR / f"{RUN_NAME}_summary.json"
SUBSET_RESULTS_CSV = OUT_DIR / "budget_subset_results.csv"
AGGREGATE_RESULTS_CSV = OUT_DIR / "budget_aggregate_results.csv"
PLOT_PATH = OUT_DIR / "target_label_budget_rmse_v1.png"


def load_strict_module():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("strict_eol_protocol_v1", STRICT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import strict protocol script: {STRICT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [clean_for_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_for_json(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def subset_id(k: int, subset_cells: Sequence[str], index: int) -> str:
    if k == 0:
        return "k0_source_only"
    if k == 8:
        return "k8_all_batch9"
    return f"k{k}_subset_{index:03d}__" + "_".join(str(cell) for cell in subset_cells)


def predefined_subsets(batch9_cells: Sequence[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cells = tuple(sorted(str(cell) for cell in batch9_cells))
    subset_specs: List[Dict[str, Any]] = []
    strategy = {
        "budgets": BUDGETS,
        "batch9_cells": list(cells),
        "k1_strategy": "all combinations",
        "k2_strategy": "all combinations",
        "k4_strategy": "all 70 combinations" if len(cells) == 8 else "deterministic sample if required",
        "k4_random_seed_if_sampling_needed": K4_RANDOM_SEED,
        "k8_strategy": "all Batch 9 target-adapt cells",
    }

    for k in BUDGETS:
        if k == 0:
            combos = [tuple()]
            combo_strategy = "source_only"
        elif k == 4:
            all_combos = list(itertools.combinations(cells, k))
            if len(all_combos) <= 70:
                combos = all_combos
                combo_strategy = "all_combinations"
            else:
                rng = np.random.default_rng(K4_RANDOM_SEED)
                indices = sorted(rng.choice(len(all_combos), size=70, replace=False).tolist())
                combos = [all_combos[i] for i in indices]
                combo_strategy = "deterministic_sample_70"
        else:
            combos = list(itertools.combinations(cells, k))
            combo_strategy = "all_combinations"

        for index, combo in enumerate(combos, start=1):
            subset_specs.append(
                {
                    "budget_k": k,
                    "subset_index_within_budget": index,
                    "subset_count_for_budget": len(combos),
                    "subset_id": subset_id(k, combo, index),
                    "target_label_cells": ";".join(combo),
                    "target_label_cell_tuple": combo,
                    "subset_generation_strategy": combo_strategy,
                }
            )
    return subset_specs, strategy


def evaluate_subset(
    strict,
    model,
    source_df: pd.DataFrame,
    target_adapt_df: pd.DataFrame,
    target_test_df: pd.DataFrame,
    spec: Dict[str, Any],
) -> Dict[str, Any]:
    subset_cells = tuple(spec["target_label_cell_tuple"])
    if subset_cells:
        target_train_df = target_adapt_df[target_adapt_df["cell_id"].isin(subset_cells)]
        train_df = pd.concat([source_df, target_train_df], ignore_index=True)
        protocol = f"source_plus_k_batch9_labels"
    else:
        train_df = source_df.copy()
        protocol = "source_only"

    cols = strict.feature_columns(train_df)
    train_x, train_y, eval_x, eval_y = strict.prepare_train_eval(train_df, target_test_df, cols)
    fitted = clone(model)
    fitted.fit(train_x, train_y)
    pred = np.asarray(fitted.predict(eval_x), dtype=float)
    errors = pred - eval_y
    abs_errors = np.abs(errors)

    return {
        "budget_k": int(spec["budget_k"]),
        "subset_index_within_budget": int(spec["subset_index_within_budget"]),
        "subset_count_for_budget": int(spec["subset_count_for_budget"]),
        "subset_id": spec["subset_id"],
        "target_label_cells": spec["target_label_cells"],
        "subset_generation_strategy": spec["subset_generation_strategy"],
        "protocol": protocol,
        "model": MODEL_NAME,
        "horizon": HORIZON,
        "n_source_cells": int(source_df["cell_id"].nunique()),
        "n_target_label_cells": int(len(subset_cells)),
        "n_train_cells": int(train_df["cell_id"].nunique()),
        "target_test_n_cells": int(target_test_df["cell_id"].nunique()),
        "target_test_eol_rmse": rmse(eval_y, pred),
        "target_test_eol_mae": float(mean_absolute_error(eval_y, pred)),
        "target_test_eol_medae": float(median_absolute_error(eval_y, pred)),
        "target_test_eol_max_abs_error": float(np.max(abs_errors)),
        "target_test_eol_bias_pred_minus_true": float(np.mean(errors)),
        "target_test_eol_error_std": float(np.std(errors)),
        "predictions_by_cell": json.dumps(
            {
                str(cell): float(value)
                for cell, value in zip(target_test_df["cell_id"].astype(str).tolist(), pred)
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "abs_errors_by_cell": json.dumps(
            {
                str(cell): float(value)
                for cell, value in zip(target_test_df["cell_id"].astype(str).tolist(), abs_errors)
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    }


def aggregate_results(subset_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for k, group in subset_df.groupby("budget_k", sort=True):
        rmse_values = group["target_test_eol_rmse"].to_numpy(dtype=float)
        mae_values = group["target_test_eol_mae"].to_numpy(dtype=float)
        rows.append(
            {
                "budget_k": int(k),
                "n_subsets": int(len(group)),
                "subset_generation_strategy": ";".join(sorted(group["subset_generation_strategy"].unique())),
                "rmse_mean": float(np.mean(rmse_values)),
                "rmse_std": float(np.std(rmse_values, ddof=1)) if len(rmse_values) > 1 else np.nan,
                "rmse_min": float(np.min(rmse_values)),
                "rmse_q25": float(np.quantile(rmse_values, 0.25)),
                "rmse_median": float(np.quantile(rmse_values, 0.50)),
                "rmse_q75": float(np.quantile(rmse_values, 0.75)),
                "rmse_max": float(np.max(rmse_values)),
                "mae_mean": float(np.mean(mae_values)),
                "mae_std": float(np.std(mae_values, ddof=1)) if len(mae_values) > 1 else np.nan,
                "mae_min": float(np.min(mae_values)),
                "mae_q25": float(np.quantile(mae_values, 0.25)),
                "mae_median": float(np.quantile(mae_values, 0.50)),
                "mae_q75": float(np.quantile(mae_values, 0.75)),
                "mae_max": float(np.max(mae_values)),
                "note": "sensitivity aggregate only; not used for Batch10-based model selection",
            }
        )
    return pd.DataFrame(rows)


def strict_reference_values() -> Dict[str, Any]:
    reference: Dict[str, Any] = {
        "strict_selected_summary_available": STRICT_SELECTED_SUMMARY.exists(),
        "strict_selected_predictions_available": STRICT_SELECTED_PREDICTIONS.exists(),
    }
    if STRICT_SELECTED_SUMMARY.exists():
        summary = pd.read_csv(STRICT_SELECTED_SUMMARY)
        source = summary[
            (summary["protocol"] == "source_only")
            & (summary["model"] == MODEL_NAME)
            & (summary["horizon"] == HORIZON)
        ]
        full = summary[
            (summary["protocol"] == "source_plus_target_adapt_loocv")
            & (summary["model"] == MODEL_NAME)
            & (summary["horizon"] == HORIZON)
        ]
        if not source.empty:
            row = source.sort_values("selection_scope").iloc[0]
            reference["strict_source_only_extra_trees_1000"] = {
                "target_test_eol_rmse": float(row["target_test_eol_rmse"]),
                "target_test_eol_mae": float(row["target_test_eol_mae"]),
                "selection_scope": str(row["selection_scope"]),
            }
        if not full.empty:
            row = full.sort_values("selection_scope").iloc[0]
            reference["strict_full_batch9_extra_trees_1000"] = {
                "target_test_eol_rmse": float(row["target_test_eol_rmse"]),
                "target_test_eol_mae": float(row["target_test_eol_mae"]),
                "selection_scope": str(row["selection_scope"]),
            }
    return reference


def add_consistency_checks(subset_df: pd.DataFrame, reference: Dict[str, Any]) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    k0 = subset_df[subset_df["budget_k"] == 0]
    k8 = subset_df[subset_df["budget_k"] == 8]

    if "strict_source_only_extra_trees_1000" in reference and not k0.empty:
        ref = reference["strict_source_only_extra_trees_1000"]
        row = k0.iloc[0]
        checks["k0_source_only_vs_strict_summary"] = {
            "rmse_difference_cycles": float(row["target_test_eol_rmse"] - ref["target_test_eol_rmse"]),
            "mae_difference_cycles": float(row["target_test_eol_mae"] - ref["target_test_eol_mae"]),
            "reproduces_strict_summary": bool(
                np.isclose(row["target_test_eol_rmse"], ref["target_test_eol_rmse"])
                and np.isclose(row["target_test_eol_mae"], ref["target_test_eol_mae"])
            ),
        }

    if "strict_full_batch9_extra_trees_1000" in reference and not k8.empty:
        ref = reference["strict_full_batch9_extra_trees_1000"]
        row = k8.iloc[0]
        checks["k8_full_batch9_vs_strict_summary"] = {
            "rmse_difference_cycles": float(row["target_test_eol_rmse"] - ref["target_test_eol_rmse"]),
            "mae_difference_cycles": float(row["target_test_eol_mae"] - ref["target_test_eol_mae"]),
            "reproduces_strict_summary": bool(
                np.isclose(row["target_test_eol_rmse"], ref["target_test_eol_rmse"])
                and np.isclose(row["target_test_eol_mae"], ref["target_test_eol_mae"])
            ),
        }

    if STRICT_SELECTED_PREDICTIONS.exists() and not k8.empty:
        baseline = pd.read_csv(STRICT_SELECTED_PREDICTIONS)
        baseline = baseline[
            (baseline["selection_scope"] == "global_best_by_batch9")
            & (baseline["protocol"] == "source_plus_target_adapt_loocv")
            & (baseline["model"] == MODEL_NAME)
            & (baseline["horizon"] == HORIZON)
        ]
        if not baseline.empty:
            preds = json.loads(k8.iloc[0]["predictions_by_cell"])
            diffs = []
            for _, row in baseline.iterrows():
                cell = str(row["cell_id"])
                if cell in preds:
                    diffs.append(abs(float(preds[cell]) - float(row["pred_eol_cycle"])))
            checks["k8_full_batch9_vs_strict_predictions"] = {
                "matched_cells": int(len(diffs)),
                "max_abs_prediction_difference_cycles": float(max(diffs)) if diffs else None,
                "reproduces_strict_predictions": bool(diffs and max(diffs) <= 1e-9),
            }
    return checks


def create_plot(subset_df: pd.DataFrame, aggregate_df: pd.DataFrame) -> Dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"created": False, "path": None, "reason": f"matplotlib unavailable: {exc}"}

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    budgets = sorted(subset_df["budget_k"].unique())
    data = [
        subset_df[subset_df["budget_k"] == k]["target_test_eol_rmse"].to_numpy(dtype=float)
        for k in budgets
    ]
    ax.boxplot(data, positions=budgets, widths=0.45, showfliers=False)
    for k, values in zip(budgets, data):
        if len(values) == 1:
            jitter = np.array([0.0])
        else:
            jitter = np.linspace(-0.12, 0.12, len(values))
        ax.scatter(np.full(len(values), k) + jitter, values, s=16, alpha=0.55, color="#2f6f9f")
    ax.plot(
        aggregate_df["budget_k"],
        aggregate_df["rmse_median"],
        color="#b6463a",
        marker="o",
        linewidth=1.6,
        label="Median RMSE",
    )
    ax.set_xlabel("Number of labeled Batch 9 target cells used for adaptation")
    ax.set_ylabel("Batch 10 EOL RMSE (cycles)")
    ax.set_title("Target-label budget sensitivity, ExtraTrees at 1000 cycles")
    ax.set_xticks(budgets)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PLOT_PATH)
    plt.close(fig)
    return {"created": True, "path": str(PLOT_PATH)}


def format_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "NA"
    try:
        val = float(value)
    except Exception:
        return str(value)
    if math.isnan(val) or math.isinf(val):
        return "NA"
    return f"{val:.{digits}f}"


def write_report(payload: Dict[str, Any]) -> None:
    aggregate = pd.DataFrame(payload["aggregate_rows"])
    subset = pd.DataFrame(payload["subset_rows"])
    checks = payload["consistency_checks"]

    lines = [
        "# Target-Label Budget Sensitivity V1",
        "",
        f"- Finished at: {payload['finished_at']}",
        f"- Elapsed seconds: {payload['elapsed_seconds']:.1f}",
        f"- Data dir: `{payload['data_dir']}`",
        f"- Strict baseline script imported read-only: `{payload['strict_script']}`",
        "",
        "## Fixed Protocol",
        "",
        "- EOL-first target: cell-level EOL cycle; RUL at a fixed observation cycle is derived as predicted EOL minus current cycle.",
        "- Source domain: Batch 1-8.",
        "- Target labels: predefined subsets of labeled Batch 9 cells.",
        "- Blind test: Batch 10 only.",
        "- Model/horizon/features are fixed before this sensitivity run: ExtraTrees, 1000 observed cycles, strict feature blacklist.",
        "- Batch 10 results are reported for all predefined subsets as sensitivity evidence only; no subset is selected by Batch 10.",
        "- Because Batch 9 labels are used, the setting is supervised small labeled target-domain adaptation, not unsupervised domain adaptation.",
        "",
        "## Subset Budget Design",
        "",
        "| Budget k | Subsets evaluated | Strategy |",
        "|---:|---:|---|",
    ]
    for _, row in aggregate.sort_values("budget_k").iterrows():
        lines.append(
            f"| {int(row['budget_k'])} | {int(row['n_subsets'])} | {row['subset_generation_strategy']} |"
        )

    lines.extend(
        [
            "",
            "## Aggregate Batch 10 Sensitivity",
            "",
            "| k labeled Batch 9 cells | Subsets | RMSE mean | RMSE median | RMSE range | MAE mean | MAE median | MAE range |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in aggregate.sort_values("budget_k").iterrows():
        lines.append(
            f"| {int(row['budget_k'])} | {int(row['n_subsets'])} | "
            f"{format_number(row['rmse_mean'])} | {format_number(row['rmse_median'])} | "
            f"{format_number(row['rmse_min'])}-{format_number(row['rmse_max'])} | "
            f"{format_number(row['mae_mean'])} | {format_number(row['mae_median'])} | "
            f"{format_number(row['mae_min'])}-{format_number(row['mae_max'])} |"
        )

    k0 = subset[subset["budget_k"] == 0].iloc[0]
    k8 = subset[subset["budget_k"] == 8].iloc[0]
    lines.extend(
        [
            "",
            "## Anchor Rows",
            "",
            "| Row | k | Target labels | Batch 10 RMSE | Batch 10 MAE |",
            "|---|---:|---|---:|---:|",
            (
                f"| Source-only anchor | 0 | none | "
                f"{k0['target_test_eol_rmse']:.2f} | {k0['target_test_eol_mae']:.2f} |"
            ),
            (
                f"| Full Batch 9 adaptation anchor | 8 | all Batch 9 cells | "
                f"{k8['target_test_eol_rmse']:.2f} | {k8['target_test_eol_mae']:.2f} |"
            ),
            "",
        ]
    )

    if "k8_full_batch9_vs_strict_predictions" in checks:
        pred_check = checks["k8_full_batch9_vs_strict_predictions"]
        lines.extend(
            [
                "Strict baseline reproduction:",
                "",
                (
                    f"- k=8 matched {pred_check['matched_cells']} Batch 10 cells against the strict selected "
                    f"prediction CSV; max absolute prediction difference = "
                    f"{format_number(pred_check['max_abs_prediction_difference_cycles'], 6)} cycles."
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation",
            "",
        ]
    )
    source_rmse = float(k0["target_test_eol_rmse"])
    full_rmse = float(k8["target_test_eol_rmse"])
    delta = full_rmse - source_rmse
    if full_rmse < source_rmse:
        lines.append(
            f"- Full labeled Batch 9 adaptation improves the fixed ExtraTrees 1000-cycle anchor from {source_rmse:.2f} to {full_rmse:.2f} RMSE cycles."
        )
    else:
        lines.append(
            f"- Full labeled Batch 9 adaptation does not improve over source-only in this fixed run: {source_rmse:.2f} vs {full_rmse:.2f} RMSE cycles."
        )
    lines.extend(
        [
            f"- The k=8 minus k=0 RMSE difference is {delta:.2f} cycles; lower is better.",
            "- The k=1, k=2, and k=4 rows show substantial subset dependence, so few-label results should be discussed as sensitivity evidence rather than as selected models.",
            "- Any low-error few-label subset is not a replacement for the main strict baseline because this task predefines and reports all subsets instead of selecting by Batch 10.",
            "- This supports a cautious small-labeled-target adaptation narrative only when paired with the fixed protocol caveats.",
            "",
            "## Caveats",
            "",
            "- Batch 10 has only eight cells, so the aggregate ranges are sensitive to individual target-test cells.",
            "- Batch 9 has only eight labeled cells; k=1 and k=2 are especially unstable by design.",
            "- The experiment uses historical measured/estimated SOH-capacity features from `y_nominal`; wording must keep the online-availability caveat.",
            "- This is not unsupervised domain adaptation and not Batch10-based model selection.",
            "- The main point result remains the pre-existing strict ExtraTrees full-Batch-9 anchor.",
            "",
            "## Output Files",
            "",
            f"- Script: `{OUT_DIR / 'run_target_label_budget_sensitivity_v1.py'}`",
            f"- Subset results: `{SUBSET_RESULTS_CSV}`",
            f"- Aggregate results: `{AGGREGATE_RESULTS_CSV}`",
            f"- Summary JSON: `{SUMMARY_JSON}`",
        ]
    )
    if payload["plot"]["created"]:
        lines.append(f"- Optional plot: `{payload['plot']['path']}`")
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
    source_df = frames["source"]
    target_adapt_df = frames["target_adapt"]
    target_test_df = frames["target_test"]

    batch9_cells = sorted(str(cell) for cell in target_adapt_df["cell_id"].unique())
    subset_specs, subset_strategy = predefined_subsets(batch9_cells)
    model_lookup = {name: model for name, model in strict.model_zoo()}
    model = model_lookup[MODEL_NAME]

    subset_rows: List[Dict[str, Any]] = []
    for spec in subset_specs:
        subset_rows.append(evaluate_subset(strict, model, source_df, target_adapt_df, target_test_df, spec))

    subset_df = pd.DataFrame(subset_rows).sort_values(["budget_k", "subset_index_within_budget"])
    aggregate_df = aggregate_results(subset_df)
    subset_df.to_csv(SUBSET_RESULTS_CSV, index=False, encoding="utf-8-sig")
    aggregate_df.to_csv(AGGREGATE_RESULTS_CSV, index=False, encoding="utf-8-sig")

    reference = strict_reference_values()
    consistency = add_consistency_checks(subset_df, reference)
    plot = create_plot(subset_df, aggregate_df)

    payload: Dict[str, Any] = {
        "run_name": RUN_NAME,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": time.time() - started,
        "data_dir": str(strict.DATA_DIR),
        "strict_script": str(STRICT_SCRIPT),
        "fixed_setting": {
            "target": "EOL-first cell-level EOL cycle",
            "model": MODEL_NAME,
            "horizon_cycles": HORIZON,
            "budgets": BUDGETS,
            "source": "Batch 1-8",
            "target_label_budget_source": "Batch 9 labeled target-adapt cells",
            "target_test": "Batch 10 blind test",
            "batch10_selection_allowed": False,
            "domain_adaptation_wording": "supervised small labeled target-domain adaptation",
        },
        "forbidden_model_inputs": sorted(strict.FORBIDDEN_MODEL_INPUTS),
        "n_feature_columns": int(len(strict.feature_columns(pd.concat([source_df, target_adapt_df], ignore_index=True)))),
        "subset_strategy": subset_strategy,
        "strict_reference_values": reference,
        "consistency_checks": consistency,
        "subset_rows": subset_df.to_dict(orient="records"),
        "aggregate_rows": aggregate_df.to_dict(orient="records"),
        "plot": plot,
        "outputs": {
            "script": str(OUT_DIR / "run_target_label_budget_sensitivity_v1.py"),
            "report": str(REPORT_MD),
            "summary_json": str(SUMMARY_JSON),
            "budget_subset_results_csv": str(SUBSET_RESULTS_CSV),
            "budget_aggregate_results_csv": str(AGGREGATE_RESULTS_CSV),
            "plot": str(PLOT_PATH) if plot["created"] else None,
        },
        "caveats": [
            "Batch 10 is reported for all predefined subsets as sensitivity evidence only.",
            "No subset is selected as a new model based on Batch 10.",
            "The setting is supervised small labeled target-domain adaptation, not unsupervised domain adaptation.",
            "The main point result remains strict-eol-protocol-v1 full Batch 9 ExtraTrees at 1000 cycles.",
        ],
    }

    payload = clean_for_json(payload)
    SUMMARY_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)

    print(
        json.dumps(
            {
                "report": str(REPORT_MD),
                "n_subset_rows": int(len(subset_df)),
                "aggregate": aggregate_df[
                    ["budget_k", "n_subsets", "rmse_mean", "rmse_median", "rmse_min", "rmse_max", "mae_mean"]
                ].to_dict(orient="records"),
                "consistency_checks": clean_for_json(consistency),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
