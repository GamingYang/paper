from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(r"C:\Users\ROG\Desktop\paper")
DATA_DIR = ROOT / "PYthon训练代码" / "processed_clean"
OUT_DIR = ROOT / "codex" / "advanced-attention-candidates-v1"
FIG_DIR = OUT_DIR / "figures"

FINAL_SCRIPT = ROOT / "codex" / "final-retained-v1" / "train_final_retained_scheme_a.py"
FINAL_CKPT = ROOT / "codex" / "final-retained-v1" / "final-retained-v1_best_by_target_adapt.pt"
ADAPTIVE_SCRIPT = ROOT / "codex" / "adaptive-attention-v1" / "run_adaptive_attention_v1.py"
ADAPTIVE_CKPT = ROOT / "codex" / "adaptive-attention-v1" / "adaptive-attention-v1-seed42" / "adaptive-attention-v1-seed42_best.pt"
CANDIDATE_SCRIPT = OUT_DIR / "run_advanced_attention_candidates_v1.py"

V1_DIR = ROOT / "codex" / "adaptive-attention-v1"
V1_COMPARISON = V1_DIR / "adaptive_attention_v1_comparison_metrics.csv"
V1_SUMMARY = V1_DIR / "adaptive_attention_v1_multi_seed_summary.csv"
V1_STAGE42 = V1_DIR / "adaptive_attention_v1_seed42_stage_metrics.csv"
CAND_SUMMARY = OUT_DIR / "advanced_attention_candidates_v1_summary.csv"

GLOBAL_MAX_CYCLE = 3000.0
EVAL_BATCH_SIZE = 1024
SEEDS = [42, 7, 123]

METHOD_LABELS = {
    "Original attention": "Original",
    "Adaptive curve attention": "Adaptive v1",
    "adaptive-attention-v1": "Adaptive v1",
    "multi-expert-gated-attention-v1": "Multi-expert gate",
    "stage-aware-branch-attention-v1": "Stage-aware branch",
}

METHOD_ORDER = ["Original", "Adaptive v1", "Multi-expert gate", "Stage-aware branch"]
COLORS = {
    "Original": "#9CA3AF",
    "Adaptive v1": "#1B3D6E",
    "Multi-expert gate": "#C2410C",
    "Stage-aware branch": "#2A9D8F",
}
STAGE_COLORS = {"Early": "#2A9D8F", "Middle": "#E9C46A", "Late": "#C2410C"}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_split(split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.load(DATA_DIR / f"clean_{split}_x.npy").astype(np.float32)
    y = np.load(DATA_DIR / f"clean_{split}_y_nominal.npy").astype(np.float32)
    p = np.load(DATA_DIR / f"clean_{split}_p.npy").astype(np.float32)
    bidx = np.load(DATA_DIR / f"clean_{split}_battery_index.npy").astype(np.int64)
    return x, y, p, bidx


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true.reshape(-1) - y_pred.reshape(-1)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true.reshape(-1) - y_pred.reshape(-1))))


def stage_masks(p_norm: np.ndarray) -> Dict[str, np.ndarray]:
    p_flat = p_norm.reshape(-1)
    return {
        "Early": p_flat < 1.0 / 3.0,
        "Middle": (p_flat >= 1.0 / 3.0) & (p_flat < 2.0 / 3.0),
        "Late": p_flat >= 2.0 / 3.0,
    }


def paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.linewidth": 1.0,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.unicode_minus": False,
        }
    )


def style_axes(ax, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, axis="y", color="#E5E7EB", linestyle="--", linewidth=0.7)
        ax.set_axisbelow(True)


def load_metric_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    v1_comp = pd.read_csv(V1_COMPARISON)
    base_rows = []
    for _, row in v1_comp.iterrows():
        label = METHOD_LABELS.get(row["variant"], row["variant"])
        if label not in ["Original", "Adaptive v1"]:
            continue
        base_rows.append(
            {
                "seed": int(row["seed"]),
                "method": label,
                "target_test_rmse": float(row["target_test_rmse"]),
                "target_test_macro_rmse": float(row["target_test_macro_rmse"]),
                "target_test_mae": float(row["target_test_mae"]),
                "best_epoch": int(row["best_epoch"]),
            }
        )
    cand = pd.read_csv(CAND_SUMMARY)
    cand_rows = []
    for _, row in cand.iterrows():
        cand_rows.append(
            {
                "seed": int(row["seed"]),
                "method": METHOD_LABELS[row["variant"]],
                "target_test_rmse": float(row["target_test_rmse"]),
                "target_test_macro_rmse": float(row["target_test_macro_rmse"]),
                "target_test_mae": float(row["target_test_mae"]),
                "target_test_early_rmse": float(row["target_test_early_rmse"]),
                "target_test_mid_rmse": float(row["target_test_mid_rmse"]),
                "target_test_late_rmse": float(row["target_test_late_rmse"]),
                "best_epoch": int(row["best_epoch"]),
            }
        )
    metrics = pd.DataFrame(base_rows + cand_rows)
    metrics["method"] = pd.Categorical(metrics["method"], METHOD_ORDER, ordered=True)
    metrics = metrics.sort_values(["method", "seed"]).reset_index(drop=True)
    metrics.to_csv(OUT_DIR / "advanced_attention_full_comparison_metrics.csv", index=False, encoding="utf-8-sig")

    agg_rows = []
    for method, group in metrics.groupby("method", observed=True):
        agg_rows.append(
            {
                "method": method,
                "n_seeds": int(group["seed"].nunique()),
                "mean_rmse": float(group["target_test_rmse"].mean()),
                "std_rmse": float(group["target_test_rmse"].std(ddof=1)),
                "mean_macro_rmse": float(group["target_test_macro_rmse"].mean()),
                "std_macro_rmse": float(group["target_test_macro_rmse"].std(ddof=1)),
                "mean_mae": float(group["target_test_mae"].mean()),
                "std_mae": float(group["target_test_mae"].std(ddof=1)),
            }
        )
    aggregate = pd.DataFrame(agg_rows)
    aggregate.to_csv(OUT_DIR / "advanced_attention_full_aggregate.csv", index=False, encoding="utf-8-sig")

    v1_summary = pd.read_csv(V1_SUMMARY)
    stage_mean_rows = []
    for _, row in v1_summary.iterrows():
        stage_mean_rows.extend(
            [
                {"seed": int(row["seed"]), "method": "Adaptive v1", "stage": "Early", "rmse": float(row["target_test_early_rmse"])},
                {"seed": int(row["seed"]), "method": "Adaptive v1", "stage": "Middle", "rmse": float(row["target_test_mid_rmse"])},
                {"seed": int(row["seed"]), "method": "Adaptive v1", "stage": "Late", "rmse": float(row["target_test_late_rmse"])},
            ]
        )
    for _, row in cand.iterrows():
        label = METHOD_LABELS[row["variant"]]
        stage_mean_rows.extend(
            [
                {"seed": int(row["seed"]), "method": label, "stage": "Early", "rmse": float(row["target_test_early_rmse"])},
                {"seed": int(row["seed"]), "method": label, "stage": "Middle", "rmse": float(row["target_test_mid_rmse"])},
                {"seed": int(row["seed"]), "method": label, "stage": "Late", "rmse": float(row["target_test_late_rmse"])},
            ]
        )
    stage_mean = pd.DataFrame(stage_mean_rows)
    stage_mean.to_csv(OUT_DIR / "advanced_attention_stage_rmse_three_seed.csv", index=False, encoding="utf-8-sig")

    stage42_rows = []
    v1_stage42 = pd.read_csv(V1_STAGE42)
    for _, row in v1_stage42.iterrows():
        method = "Original" if row["method"] == "Original attention" else "Adaptive v1"
        stage42_rows.append({"method": method, "stage": row["stage"], "rmse": float(row["rmse"]), "mae": float(row["mae"])})
    for _, row in cand[cand["seed"].astype(int) == 42].iterrows():
        label = METHOD_LABELS[row["variant"]]
        stage42_rows.extend(
            [
                {"method": label, "stage": "Early", "rmse": float(row["target_test_early_rmse"]), "mae": np.nan},
                {"method": label, "stage": "Middle", "rmse": float(row["target_test_mid_rmse"]), "mae": np.nan},
                {"method": label, "stage": "Late", "rmse": float(row["target_test_late_rmse"]), "mae": np.nan},
            ]
        )
    stage42 = pd.DataFrame(stage42_rows)
    stage42.to_csv(OUT_DIR / "advanced_attention_stage_rmse_seed42.csv", index=False, encoding="utf-8-sig")
    return metrics, aggregate, stage_mean, stage42


def collect_predictions_and_attention(model: torch.nn.Module, dataset, device: torch.device) -> Dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    captured: Dict[str, List[np.ndarray]] = {
        "attn": [],
        "temperature": [],
        "gate": [],
        "expert_attention": [],
        "expert_temperature": [],
        "stage_embedding": [],
        "gamma": [],
        "beta": [],
    }

    def hook(_module, _inputs, output):
        if not isinstance(output, tuple):
            return
        if len(output) > 1 and torch.is_tensor(output[1]):
            captured["attn"].append(output[1].detach().cpu().numpy().squeeze(-1))
        if len(output) > 2 and torch.is_tensor(output[2]):
            captured["temperature"].append(output[2].detach().cpu().numpy().reshape(-1))
        if len(output) > 3 and isinstance(output[3], dict):
            for key in ["gate", "expert_attention", "expert_temperature", "stage_embedding", "gamma", "beta"]:
                val = output[3].get(key)
                if torch.is_tensor(val):
                    captured[key].append(val.detach().cpu().numpy())

    handle = model.attention.register_forward_hook(hook)
    model.eval()
    preds, trues, bidxs, pvals = [], [], [], []
    with torch.no_grad():
        for x, y, p, b in loader:
            x = x.float().to(device)
            p = p.float().to(device)
            out, _, _ = model(x, p)
            preds.append(out.cpu().numpy().reshape(-1))
            trues.append(y.numpy().reshape(-1))
            bidxs.append(b.numpy().reshape(-1))
            pvals.append(p.cpu().numpy().reshape(-1))
    handle.remove()
    result = {
        "pred": np.concatenate(preds),
        "true": np.concatenate(trues),
        "bidx": np.concatenate(bidxs),
        "p_norm": np.concatenate(pvals),
    }
    for key, vals in captured.items():
        if vals:
            axis = 0
            result[key] = np.concatenate(vals, axis=axis)
    return result


def load_seed42_models() -> Dict[str, Dict[str, np.ndarray]]:
    final_mod = load_module(FINAL_SCRIPT, "final_retained_v1_mod")
    adaptive_mod = load_module(ADAPTIVE_SCRIPT, "adaptive_attention_v1_mod")
    candidate_mod = load_module(CANDIDATE_SCRIPT, "advanced_attention_candidates_v1_mod")

    source_x, source_y, source_p, _ = load_split("source")
    target_x, target_y, target_p, target_bidx = load_split("target_test")
    coeffs = np.polyfit((source_p.flatten() / GLOBAL_MAX_CYCLE), source_y.flatten(), 3)

    device = torch.device("cpu")
    out: Dict[str, Dict[str, np.ndarray]] = {}

    original_ckpt = torch.load(FINAL_CKPT, map_location=device, weights_only=False)
    original_coeffs = torch.tensor(original_ckpt["base_coeffs"], dtype=torch.float32, device=device)
    original_model = final_mod.ResidualCRNN(original_coeffs).to(device)
    original_model.load_state_dict(original_ckpt["model_state_dict"])
    original_ds = final_mod.HybridDataset(target_x, target_y, target_p, target_bidx)
    out["Original"] = collect_predictions_and_attention(original_model, original_ds, device)

    adaptive_ckpt = torch.load(ADAPTIVE_CKPT, map_location=device, weights_only=False)
    adaptive_coeffs = torch.tensor(coeffs, dtype=torch.float32, device=device)
    adaptive_model = adaptive_mod.ResidualCRNN(adaptive_coeffs, use_physical_base=True).to(device)
    adaptive_model.load_state_dict(adaptive_ckpt["model_state_dict"])
    adaptive_ds = adaptive_mod.HybridDataset(target_x, target_y, target_p, target_bidx)
    out["Adaptive v1"] = collect_predictions_and_attention(adaptive_model, adaptive_ds, device)

    for method, model_name, attn_type in [
        ("Multi-expert gate", "multi-expert-gated-attention-v1-seed42", "multi_expert"),
        ("Stage-aware branch", "stage-aware-branch-attention-v1-seed42", "stage_branch"),
    ]:
        ckpt_path = OUT_DIR / model_name / f"{model_name}_best.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        candidate_coeffs = torch.tensor(ckpt["base_coeffs"], dtype=torch.float32, device=device)
        model = candidate_mod.ResidualCRNN(candidate_coeffs, attention_type=attn_type).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        ds = candidate_mod.HybridDataset(target_x, target_y, target_p, target_bidx)
        out[method] = collect_predictions_and_attention(model, ds, device)

    pred_df = pd.DataFrame(
        {
            "battery_index": target_bidx.reshape(-1).astype(int),
            "cycle": target_p.reshape(-1).astype(float),
            "p_norm": target_p.reshape(-1).astype(float) / GLOBAL_MAX_CYCLE,
            "true_soh": target_y.reshape(-1).astype(float),
            "original_pred": out["Original"]["pred"],
            "adaptive_v1_pred": out["Adaptive v1"]["pred"],
            "multi_expert_pred": out["Multi-expert gate"]["pred"],
            "stage_branch_pred": out["Stage-aware branch"]["pred"],
        }
    )
    pred_df.to_csv(OUT_DIR / "advanced_attention_seed42_target_test_predictions.csv", index=False, encoding="utf-8-sig")
    return out


def plot_metric_bars(aggregate: pd.DataFrame) -> None:
    metrics = [
        ("mean_rmse", "std_rmse", "RMSE"),
        ("mean_macro_rmse", "std_macro_rmse", "Macro RMSE"),
        ("mean_mae", "std_mae", "MAE"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.8))
    for ax, (mean_col, std_col, title) in zip(axes, metrics):
        df = aggregate.set_index("method").loc[METHOD_ORDER]
        vals = df[mean_col].to_numpy() * 100
        errs = df[std_col].to_numpy() * 100
        x = np.arange(len(METHOD_ORDER))
        bars = ax.bar(
            x,
            vals,
            yerr=errs,
            capsize=3,
            color=[COLORS[m] for m in METHOD_ORDER],
            edgecolor="#1F2937",
            linewidth=0.7,
        )
        bars[2].set_hatch("//")
        bars[3].set_hatch("\\\\")
        ax.set_xticks(x)
        ax.set_xticklabels(METHOD_ORDER, rotation=25, ha="right")
        ax.set_ylabel(f"{title} (%)")
        ax.set_title(title, loc="left", fontweight="bold")
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(vals) * 0.025, f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)
        style_axes(ax)
    fig.suptitle("Three-seed blind-test error summary", x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "fig01_three_seed_error_summary.png")
    plt.close(fig)


def plot_seedwise_rmse(metrics: pd.DataFrame) -> None:
    pivot = metrics.pivot(index="seed", columns="method", values="target_test_rmse").loc[SEEDS, METHOD_ORDER]
    x = np.arange(len(SEEDS))
    width = 0.18
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for idx, method in enumerate(METHOD_ORDER):
        offset = (idx - 1.5) * width
        bars = ax.bar(
            x + offset,
            pivot[method].to_numpy() * 100,
            width,
            label=method,
            color=COLORS[method],
            edgecolor="#1F2937",
            linewidth=0.6,
        )
        if method == "Multi-expert gate":
            for b in bars:
                b.set_hatch("//")
        if method == "Stage-aware branch":
            for b in bars:
                b.set_hatch("\\\\")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in SEEDS])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Batch10 RMSE (%)")
    ax.set_title("Seed-wise RMSE comparison", loc="left", fontweight="bold")
    ax.legend(ncol=2, frameon=True, edgecolor="#DDDDDD")
    style_axes(ax)
    fig.savefig(FIG_DIR / "fig02_seedwise_rmse_comparison.png")
    plt.close(fig)


def plot_stage_seed42(stage42: pd.DataFrame) -> None:
    stages = ["Early", "Middle", "Late"]
    x = np.arange(len(stages))
    width = 0.18
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for idx, method in enumerate(METHOD_ORDER):
        vals = stage42[stage42["method"] == method].set_index("stage").loc[stages, "rmse"].to_numpy() * 100
        bars = ax.bar(
            x + (idx - 1.5) * width,
            vals,
            width,
            label=method,
            color=COLORS[method],
            edgecolor="#1F2937",
            linewidth=0.6,
        )
        if idx >= 2:
            for b in bars:
                b.set_hatch("//" if idx == 2 else "\\\\")
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("RMSE (%)")
    ax.set_title("Seed42 error by degradation stage", loc="left", fontweight="bold")
    ax.legend(ncol=2, frameon=True, edgecolor="#DDDDDD")
    style_axes(ax)
    fig.savefig(FIG_DIR / "fig03_seed42_stage_rmse_comparison.png")
    plt.close(fig)


def plot_stage_mean(stage_mean: pd.DataFrame) -> None:
    stages = ["Early", "Middle", "Late"]
    methods = ["Adaptive v1", "Multi-expert gate", "Stage-aware branch"]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6), sharey=False)
    for ax, stage in zip(axes, stages):
        df = stage_mean[stage_mean["stage"] == stage]
        vals = [df[df["method"] == m]["rmse"].mean() * 100 for m in methods]
        errs = [df[df["method"] == m]["rmse"].std(ddof=1) * 100 for m in methods]
        x = np.arange(len(methods))
        bars = ax.bar(
            x,
            vals,
            yerr=errs,
            capsize=3,
            color=[COLORS[m] for m in methods],
            edgecolor="#1F2937",
            linewidth=0.7,
        )
        bars[1].set_hatch("//")
        bars[2].set_hatch("\\\\")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=25, ha="right")
        ax.set_ylabel("RMSE (%)")
        ax.set_title(stage, loc="left", fontweight="bold")
        style_axes(ax)
    fig.suptitle("Three-seed stage-wise RMSE", x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "fig04_three_seed_stage_rmse.png")
    plt.close(fig)


def plot_tradeoff(metrics: pd.DataFrame) -> None:
    cand = pd.read_csv(CAND_SUMMARY)
    v1 = pd.read_csv(V1_SUMMARY)
    rows = []
    for _, row in v1.iterrows():
        rows.append({"method": "Adaptive v1", "seed": int(row["seed"]), "rmse": float(row["target_test_rmse"]), "late_rmse": float(row["target_test_late_rmse"])})
    for _, row in cand.iterrows():
        rows.append({"method": METHOD_LABELS[row["variant"]], "seed": int(row["seed"]), "rmse": float(row["target_test_rmse"]), "late_rmse": float(row["target_test_late_rmse"])})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "advanced_attention_overall_late_tradeoff.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for method in ["Adaptive v1", "Multi-expert gate", "Stage-aware branch"]:
        g = df[df["method"] == method]
        ax.scatter(
            g["rmse"] * 100,
            g["late_rmse"] * 100,
            s=70,
            label=method,
            color=COLORS[method],
            edgecolor="#111827",
            linewidth=0.7,
            marker="s" if method == "Multi-expert gate" else ("^" if method == "Stage-aware branch" else "o"),
        )
        for _, row in g.iterrows():
            ax.text(row["rmse"] * 100 + 0.025, row["late_rmse"] * 100 + 0.025, str(int(row["seed"])), fontsize=8)
    ax.set_xlabel("Overall Batch10 RMSE (%)")
    ax.set_ylabel("Late-stage RMSE (%)")
    ax.set_title("Overall-late accuracy tradeoff", loc="left", fontweight="bold")
    ax.legend(frameon=True, edgecolor="#DDDDDD")
    style_axes(ax)
    fig.savefig(FIG_DIR / "fig05_overall_late_tradeoff.png")
    plt.close(fig)


def attention_summary(data_by_method: Dict[str, Dict[str, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for method, data in data_by_method.items():
        masks = stage_masks(data["p_norm"])
        for stage, mask in masks.items():
            attn = data["attn"][mask]
            entropy = -np.sum(attn * np.log(np.clip(attn, 1e-12, None)), axis=1)
            row = {
                "method": method,
                "stage": stage,
                "n_samples": int(mask.sum()),
                "mean_entropy": float(entropy.mean()),
                "mean_max_weight": float(attn.max(axis=1).mean()),
                "mean_center_of_mass": float((attn * np.arange(1, attn.shape[1] + 1)).sum(axis=1).mean()),
                "mean_temperature": float(data["temperature"][mask].mean()) if "temperature" in data else np.nan,
            }
            if "gate" in data:
                gate = data["gate"][mask]
                for idx in range(gate.shape[1]):
                    row[f"gate_expert_{idx + 1}"] = float(gate[:, idx].mean())
            if "gamma" in data:
                row["mean_abs_gamma"] = float(np.abs(data["gamma"][mask]).mean())
                row["mean_abs_beta"] = float(np.abs(data["beta"][mask]).mean())
                row["mean_stage_embedding_norm"] = float(np.linalg.norm(data["stage_embedding"][mask], axis=1).mean())
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "advanced_attention_seed42_interpretability_summary.csv", index=False, encoding="utf-8-sig")
    return df


def plot_attention_profiles(data_by_method: Dict[str, Dict[str, np.ndarray]]) -> None:
    stages = ["Early", "Middle", "Late"]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.6), sharey=True)
    axes = axes.ravel()
    for ax, method in zip(axes, METHOD_ORDER):
        data = data_by_method[method]
        x = np.arange(1, data["attn"].shape[1] + 1)
        masks = stage_masks(data["p_norm"])
        for stage in stages:
            mean_attn = data["attn"][masks[stage]].mean(axis=0)
            ax.plot(x, mean_attn, linewidth=1.7, color=STAGE_COLORS[stage], label=stage)
        ax.set_title(method, loc="left", fontweight="bold")
        ax.set_xlabel("Temporal token")
        ax.grid(True, color="#E5E7EB", linestyle=":", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Mean attention weight")
    axes[2].set_ylabel("Mean attention weight")
    axes[1].legend(frameon=True, edgecolor="#DDDDDD")
    fig.suptitle("Seed42 attention profiles by degradation stage", x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / "fig06_attention_profiles_seed42.png")
    plt.close(fig)


def plot_gate_and_branch(data_by_method: Dict[str, Dict[str, np.ndarray]], interp: pd.DataFrame) -> None:
    stages = ["Early", "Middle", "Late"]
    multi = data_by_method["Multi-expert gate"]
    masks = stage_masks(multi["p_norm"])
    gate_mat = np.vstack([multi["gate"][masks[s]].mean(axis=0) for s in stages])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(stages))
    width = 0.24
    expert_labels = ["Expert 1", "Expert 2", "Expert 3"]
    expert_colors = ["#A8C8E8", "#F4A261", "#C2410C"]
    for idx in range(3):
        ax.bar(x + (idx - 1) * width, gate_mat[:, idx], width, label=expert_labels[idx], color=expert_colors[idx], edgecolor="#1F2937", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("Mean gate weight")
    ax.set_title("Multi-expert gate allocation, seed42", loc="left", fontweight="bold")
    ax.legend(frameon=True, edgecolor="#DDDDDD")
    style_axes(ax)
    fig.savefig(FIG_DIR / "fig07_multi_expert_gate_stage_seed42.png")
    plt.close(fig)

    branch_rows = interp[interp["method"] == "Stage-aware branch"].set_index("stage").loc[stages]
    fig, ax1 = plt.subplots(figsize=(6.6, 3.8))
    x = np.arange(len(stages))
    bars = ax1.bar(x - 0.16, branch_rows["mean_abs_gamma"], 0.32, label="|gamma|", color="#2A9D8F", edgecolor="#1F2937", linewidth=0.6)
    ax1.bar(x + 0.16, branch_rows["mean_abs_beta"], 0.32, label="|beta|", color="#E9C46A", edgecolor="#1F2937", linewidth=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages)
    ax1.set_ylabel("Mean modulation magnitude")
    ax1.set_title("Stage-branch modulation strength, seed42", loc="left", fontweight="bold")
    ax2 = ax1.twinx()
    ax2.plot(x, branch_rows["mean_temperature"], color="#C2410C", marker="o", linewidth=1.5, label="Temperature")
    ax2.set_ylabel("Mean temperature")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=True, edgecolor="#DDDDDD", loc="upper left")
    style_axes(ax1)
    ax2.spines["top"].set_visible(False)
    fig.savefig(FIG_DIR / "fig08_stage_branch_modulation_seed42.png")
    plt.close(fig)


def plot_training_curves() -> None:
    paths = {
        "Adaptive v1": V1_DIR / "adaptive-attention-v1-seed42" / "adaptive-attention-v1-seed42_history.csv",
        "Multi-expert gate": OUT_DIR / "multi-expert-gated-attention-v1-seed42" / "multi-expert-gated-attention-v1-seed42_history.csv",
        "Stage-aware branch": OUT_DIR / "stage-aware-branch-attention-v1-seed42" / "stage-aware-branch-attention-v1-seed42_history.csv",
    }
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for method, path in paths.items():
        df = pd.read_csv(path)
        col = "ta_rmse"
        ax.plot(df["epoch"], df[col] * 100, label=method, color=COLORS[method], linewidth=1.4)
        best_idx = df[col].idxmin()
        ax.scatter(df.loc[best_idx, "epoch"], df.loc[best_idx, col] * 100, color=COLORS[method], edgecolor="#111827", s=32, zorder=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Batch9 target-adapt RMSE (%)")
    ax.set_title("Seed42 checkpoint-selection curves", loc="left", fontweight="bold")
    ax.legend(frameon=True, edgecolor="#DDDDDD")
    style_axes(ax)
    fig.savefig(FIG_DIR / "fig09_seed42_training_curves.png")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paper_style()
    metrics, aggregate, stage_mean, stage42 = load_metric_tables()
    plot_metric_bars(aggregate)
    plot_seedwise_rmse(metrics)
    plot_stage_seed42(stage42)
    plot_stage_mean(stage_mean)
    plot_tradeoff(metrics)
    data_by_method = load_seed42_models()
    interp = attention_summary(data_by_method)
    plot_attention_profiles(data_by_method)
    plot_gate_and_branch(data_by_method, interp)
    plot_training_curves()
    print("Saved figures:")
    for fig in sorted(FIG_DIR.glob("*.png")):
        print(fig)
    print("\nAggregate:")
    print(aggregate.to_string(index=False))
    print("\nSeed42 interpretability summary:")
    print(interp.to_string(index=False))


if __name__ == "__main__":
    main()
