"""Final-control runs for adaptive-attention-v1.

This wrapper keeps the frozen adaptive-attention-v1 implementation untouched and
adds only manuscript-control variants needed for Section 4.5:
adaptive MMD, static MMD, no MMD, and no physical base. Batch10 remains a
blind-test reporting split.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"C:\Users\ROG\Desktop\paper")
CODEX_DIR = ROOT / "codex"
BASE_DIR = CODEX_DIR / "adaptive-attention-v1"
RUN_SCRIPT = BASE_DIR / "run_adaptive_attention_v1.py"
OUT_DIR = CODEX_DIR / "adaptive-attention-final-controls-v1"
PROPOSED_SUMMARY = BASE_DIR / "adaptive_attention_v1_multi_seed_summary.csv"


CONTROL_VARIANTS: Dict[str, Dict] = {
    "static-mmd": {
        "model": "adaptive-attention-static-mmd",
        "label": "static MMD",
        "category": "mmd_control",
        "use_physical_base": True,
        "use_mmd": True,
        "mmd_mode": "static",
        "description": "Adaptive attention with a static MMD penalty.",
    },
    "no-mmd": {
        "model": "adaptive-attention-no-mmd",
        "label": "w/o adaptive MMD",
        "category": "mmd_control",
        "use_physical_base": True,
        "use_mmd": False,
        "mmd_mode": "none",
        "description": "Adaptive attention without the MMD alignment penalty.",
    },
    "no-physical-base": {
        "model": "adaptive-attention-no-physical-base",
        "label": "w/o physical base",
        "category": "structure_control",
        "use_physical_base": False,
        "use_mmd": True,
        "mmd_mode": "adaptive",
        "description": "Adaptive attention without the cubic degradation base.",
    },
}


def load_runner():
    spec = importlib.util.spec_from_file_location("adaptive_attention_v1_runner", RUN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import runner from {RUN_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_existing_proposed(seeds: List[int]) -> pd.DataFrame:
    if not PROPOSED_SUMMARY.exists():
        raise FileNotFoundError(f"Missing proposed summary: {PROPOSED_SUMMARY}")
    df = pd.read_csv(PROPOSED_SUMMARY)
    df = df[df["seed"].isin(seeds)].copy()
    df["label"] = "Proposed"
    df["category"] = "full_model"
    df["mmd_mode"] = "adaptive"
    df["use_physical_base"] = True
    df["source_status"] = "reused"
    return df


def result_from_existing_json(run_dir: Path, model_name: str) -> Dict:
    payload = json.loads((run_dir / f"{model_name}_best_metrics.json").read_text(encoding="utf-8"))
    return payload["result"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adaptive-attention-v1 final-control variants.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    parser.add_argument("--variants", nargs="+", default=list(CONTROL_VARIANTS.keys()))
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    unknown = [v for v in args.variants if v not in CONTROL_VARIANTS]
    if unknown:
        raise ValueError(f"Unknown control variants: {unknown}")

    mod = load_runner()
    mod.OUT_DIR = OUT_DIR
    mod.EPOCHS = args.epochs
    mod.LOG_EVERY = args.log_every
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    class StaticMMD(mod.LifetimeAdaptiveMMD):
        def forward(self, sf, tf, s_base, t_base):
            if sf.size(0) == 0 or tf.size(0) == 0:
                return torch.tensor(0.0, device=sf.device)
            return self.compute_mmd(sf, tf)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Output dir: {OUT_DIR}", flush=True)
    print(f"Seeds: {args.seeds}", flush=True)
    print(f"Variants: {args.variants}", flush=True)
    print(f"Epochs: {args.epochs}", flush=True)

    data = {
        "source": mod.load_split("source", mod.LABEL_MODE),
        "target_adapt": mod.load_split("target_adapt", mod.LABEL_MODE),
        "target_test": mod.load_split("target_test", mod.LABEL_MODE),
    }
    s_x, s_y, s_p, _, _ = data["source"]
    coeffs = np.polyfit((s_p.flatten() / mod.GLOBAL_MAX_CYCLE), s_y.flatten(), 3)

    all_rows = []
    proposed = load_existing_proposed(args.seeds)
    all_rows.extend(proposed.to_dict(orient="records"))

    adaptive_mmd_cls = mod.LifetimeAdaptiveMMD
    for variant_id in args.variants:
        meta = CONTROL_VARIANTS[variant_id]
        if meta["mmd_mode"] == "static":
            mod.LifetimeAdaptiveMMD = StaticMMD
        else:
            mod.LifetimeAdaptiveMMD = adaptive_mmd_cls
        for seed in args.seeds:
            model_name = f"{meta['model']}-seed{seed}"
            run_dir = OUT_DIR / model_name
            best_json = run_dir / f"{model_name}_best_metrics.json"
            if best_json.exists() and not args.force:
                result = result_from_existing_json(run_dir, model_name)
                result["reused_existing_run"] = True
                print(f"SKIP existing: {model_name}", flush=True)
            else:
                result = mod.train_variant(meta, seed, data, coeffs, device)
            result.update(
                {
                    "label": meta["label"],
                    "category": meta["category"],
                    "mmd_mode": meta["mmd_mode"],
                    "use_physical_base": meta["use_physical_base"],
                    "source_status": "new_run" if not result.get("reused_existing_run") else "reused",
                }
            )
            all_rows.append(result)

    mod.LifetimeAdaptiveMMD = adaptive_mmd_cls

    summary_df = pd.DataFrame(all_rows)
    summary_csv = OUT_DIR / "adaptive_attention_final_controls_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_df.to_excel(OUT_DIR / "adaptive_attention_final_controls_summary.xlsx", index=False)

    agg_rows = []
    for label, group in summary_df.groupby("label", sort=False):
        agg_rows.append(
            {
                "label": label,
                "category": group["category"].iloc[0],
                "n_seeds": int(group["seed"].nunique()),
                "mean_rmse": float(group["target_test_rmse"].mean()),
                "std_rmse": float(group["target_test_rmse"].std(ddof=1)),
                "mean_macro_rmse": float(group["target_test_macro_rmse"].mean()),
                "std_macro_rmse": float(group["target_test_macro_rmse"].std(ddof=1)),
                "mean_mae": float(group["target_test_mae"].mean()),
                "std_mae": float(group["target_test_mae"].std(ddof=1)),
                "mean_early_rmse": float(group["target_test_early_rmse"].mean()),
                "mean_mid_rmse": float(group["target_test_mid_rmse"].mean()),
                "mean_late_rmse": float(group["target_test_late_rmse"].mean()),
            }
        )
    agg_df = pd.DataFrame(agg_rows)
    agg_csv = OUT_DIR / "adaptive_attention_final_controls_aggregate.csv"
    agg_df.to_csv(agg_csv, index=False, encoding="utf-8-sig")
    agg_df.to_excel(OUT_DIR / "adaptive_attention_final_controls_aggregate.xlsx", index=False)

    log = [
        "# Adaptive Attention Final Controls",
        "",
        "Protocol: Batch1-8 source training, Batch9 target development/adaptation and checkpoint selection, Batch10 blind-test reporting only.",
        "",
        f"Seeds: {', '.join(map(str, args.seeds))}",
        f"Epochs: {args.epochs}",
        "",
        "## Aggregate",
        "",
        agg_df.to_string(index=False),
        "",
        "## Per Seed",
        "",
        summary_df[
            [
                "label",
                "seed",
                "best_epoch",
                "target_test_rmse",
                "target_test_macro_rmse",
                "target_test_mae",
                "target_test_early_rmse",
                "target_test_mid_rmse",
                "target_test_late_rmse",
                "source_status",
            ]
        ].to_string(index=False),
        "",
        "Interpretation note: MMD controls are auxiliary regularization controls; they should not be written as the dominant contribution.",
    ]
    (OUT_DIR / "ADAPTIVE_ATTENTION_FINAL_CONTROLS_RUN_LOG_20260729.md").write_text(
        "\n".join(log), encoding="utf-8"
    )

    print(f"Saved summary: {summary_csv}", flush=True)
    print(f"Saved aggregate: {agg_csv}", flush=True)
    print(agg_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
