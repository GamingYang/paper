from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(r"C:\Users\ROG\Desktop\paper")
DATA_DIR = ROOT / "PYthon\u8bad\u7ec3\u4ee3\u7801" / "processed_clean"
OUT_DIR = ROOT / "codex" / "adaptive-attention-v1"
FIG_DIR = OUT_DIR / "figures"

FINAL_SCRIPT = ROOT / "codex" / "final-retained-v1" / "train_final_retained_scheme_a.py"
ADAPTIVE_SCRIPT = OUT_DIR / "run_adaptive_attention_v1.py"

FINAL_CKPT = ROOT / "codex" / "final-retained-v1" / "final-retained-v1_best_by_target_adapt.pt"
ADAPTIVE_CKPT = OUT_DIR / "adaptive-attention-v1-seed42" / "adaptive-attention-v1-seed42_best.pt"

SUPP_DIR = ROOT / "codex" / "supplementary-experiments"
FINAL_SEED42 = SUPP_DIR / "seed42_summary.csv"
FINAL_MULTI = SUPP_DIR / "multi_seed_summary.csv"
ADAPTIVE_MULTI = OUT_DIR / "adaptive_attention_v1_multi_seed_summary.csv"

GLOBAL_MAX_CYCLE = 3000.0
EVAL_BATCH_SIZE = 1024


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


def collect_predictions_and_attention(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    captured: Dict[str, List[np.ndarray]] = {"attn": [], "temperature": []}

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            attn = output[1]
            captured["attn"].append(attn.detach().cpu().numpy().squeeze(-1))
            if len(output) > 2:
                captured["temperature"].append(output[2].detach().cpu().numpy().reshape(-1))

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
    out = {
        "pred": np.concatenate(preds),
        "true": np.concatenate(trues),
        "bidx": np.concatenate(bidxs),
        "p_norm": np.concatenate(pvals),
        "attn": np.concatenate(captured["attn"], axis=0),
    }
    if captured["temperature"]:
        out["temperature"] = np.concatenate(captured["temperature"])
    return out


def build_metric_table() -> pd.DataFrame:
    seed42 = pd.read_csv(FINAL_SEED42)
    final_42 = seed42.loc[seed42["model"] == "final-retained"].copy()
    final_42["variant"] = "Original attention"

    final_multi = pd.read_csv(FINAL_MULTI)
    final_multi = final_multi.loc[final_multi["variant"] == "final-retained"].copy()
    final_multi["variant"] = "Original attention"

    adaptive = pd.read_csv(ADAPTIVE_MULTI).copy()
    adaptive["variant"] = "Adaptive curve attention"

    cols = [
        "seed",
        "variant",
        "best_epoch",
        "target_test_rmse",
        "target_test_macro_rmse",
        "target_test_mae",
        "target_test_mape",
    ]
    combined = pd.concat([final_42[cols], final_multi[cols], adaptive[cols]], ignore_index=True)
    combined = combined.sort_values(["seed", "variant"]).reset_index(drop=True)
    combined.to_csv(OUT_DIR / "adaptive_attention_v1_comparison_metrics.csv", index=False, encoding="utf-8-sig")
    return combined


def paper_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.linewidth": 1.1,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def plot_seed_rmse(metrics: pd.DataFrame):
    pivot = metrics.pivot(index="seed", columns="variant", values="target_test_rmse").loc[[42, 7, 123]]
    original = pivot["Original attention"].to_numpy()
    adaptive = pivot["Adaptive curve attention"].to_numpy()
    seeds = [str(s) for s in pivot.index]
    x = np.arange(len(seeds))
    width = 0.34

    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    b1 = ax.bar(x - width / 2, original * 100, width, label="Original attention", color="#A8C8E8", edgecolor="#1F2937", linewidth=0.7)
    b2 = ax.bar(x + width / 2, adaptive * 100, width, label="Adaptive curve attention", color="#1B3D6E", edgecolor="#1F2937", linewidth=0.7)
    for i, (o, a) in enumerate(zip(original, adaptive)):
        rel = (o - a) / o * 100.0
        ax.text(x[i], max(o, a) * 100 + 0.035, f"-{rel:.1f}%", ha="center", va="bottom", color="#B42318", fontsize=8.5, fontweight="bold")
    ax.axhline(original.mean() * 100, color="#777777", linestyle="--", linewidth=0.9, label="Original mean")
    ax.axhline(adaptive.mean() * 100, color="#1B3D6E", linestyle=":", linewidth=1.2, label="Adaptive mean")
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Batch10 RMSE (%)")
    ax.set_title("Three-seed blind-test RMSE comparison", loc="left", fontweight="bold")
    ax.legend(ncol=2, frameon=True, edgecolor="#DDDDDD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bars in [b1, b2]:
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7.5)
    fig.savefig(FIG_DIR / "fig01_three_seed_rmse_comparison.png")
    plt.close(fig)


def plot_stage_rmse(stage_df: pd.DataFrame):
    stages = ["Early", "Middle", "Late"]
    original = stage_df.loc[stage_df["method"] == "Original attention"].set_index("stage").loc[stages, "rmse"].to_numpy()
    adaptive = stage_df.loc[stage_df["method"] == "Adaptive curve attention"].set_index("stage").loc[stages, "rmse"].to_numpy()
    x = np.arange(len(stages))
    width = 0.34

    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    ax.bar(x - width / 2, original * 100, width, label="Original attention", color="#D1D5DB", edgecolor="#374151", linewidth=0.7, hatch="//")
    ax.bar(x + width / 2, adaptive * 100, width, label="Adaptive curve attention", color="#1B3D6E", edgecolor="#111827", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("RMSE (%)")
    ax.set_title("Seed42 error by degradation stage", loc="left", fontweight="bold")
    ax.legend(frameon=True, edgecolor="#DDDDDD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, (o, a) in enumerate(zip(original, adaptive)):
        delta = (o - a) / o * 100.0
        label = f"{delta:+.1f}%"
        color = "#067647" if delta > 0 else "#B42318"
        ax.text(x[i], max(o, a) * 100 + 0.045, label, ha="center", va="bottom", fontsize=8.5, color=color, fontweight="bold")
    fig.savefig(FIG_DIR / "fig02_seed42_stage_rmse.png")
    plt.close(fig)


def plot_per_cell_rmse(pred_df: pd.DataFrame):
    rows = []
    for b, g in pred_df.groupby("battery_index"):
        rows.append(
            {
                "battery_index": int(b),
                "original_rmse": rmse(g["true_soh"].to_numpy(), g["original_pred"].to_numpy()),
                "adaptive_rmse": rmse(g["true_soh"].to_numpy(), g["adaptive_pred"].to_numpy()),
            }
        )
    cell_df = pd.DataFrame(rows)
    cell_df.to_csv(OUT_DIR / "adaptive_attention_v1_seed42_per_cell_rmse.csv", index=False, encoding="utf-8-sig")

    x = np.arange(len(cell_df))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(x - width / 2, cell_df["original_rmse"] * 100, width, label="Original attention", color="#A8C8E8", edgecolor="#1F2937", linewidth=0.7)
    ax.bar(x + width / 2, cell_df["adaptive_rmse"] * 100, width, label="Adaptive curve attention", color="#1B3D6E", edgecolor="#1F2937", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(cell_df["battery_index"].astype(str))
    ax.set_xlabel("Batch10 cell index")
    ax.set_ylabel("RMSE (%)")
    ax.set_title("Seed42 per-cell blind-test RMSE", loc="left", fontweight="bold")
    ax.legend(frameon=True, edgecolor="#DDDDDD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(FIG_DIR / "fig03_seed42_per_cell_rmse.png")
    plt.close(fig)


def plot_prediction_curves(pred_df: pd.DataFrame):
    batteries = sorted(pred_df["battery_index"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(11.6, 5.8), sharey=True)
    axes = axes.ravel()
    for ax, b in zip(axes, batteries):
        g = pred_df[pred_df["battery_index"] == b].sort_values("cycle")
        ax.plot(g["cycle"], g["true_soh"], color="#111827", linewidth=1.4, label="True SOH")
        ax.plot(g["cycle"], g["original_pred"], color="#7C9DC9", linewidth=1.0, label="Original")
        ax.plot(g["cycle"], g["adaptive_pred"], color="#C2410C", linewidth=1.0, label="Adaptive")
        ax.set_title(f"Cell {b}", fontsize=9.5, fontweight="bold")
        ax.grid(True, color="#E5E7EB", linestyle=":", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(loc="lower left", frameon=True, edgecolor="#DDDDDD", fontsize=8)
    for ax in axes[4:]:
        ax.set_xlabel("Cycle")
    for ax in axes[::4]:
        ax.set_ylabel("SOH")
    fig.suptitle("Seed42 Batch10 prediction curves", x=0.02, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / "fig04_seed42_prediction_curves.png")
    plt.close(fig)


def plot_attention_profiles(original: Dict[str, np.ndarray], adaptive: Dict[str, np.ndarray]):
    stages = ["Early", "Middle", "Late"]
    x = np.arange(1, original["attn"].shape[1] + 1)
    colors = {"Early": "#2A9D8F", "Middle": "#E9C46A", "Late": "#C2410C"}

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), sharey=True)
    for ax, data, title in zip(axes, [original, adaptive], ["Original attention", "Adaptive curve attention"]):
        masks = stage_masks(data["p_norm"])
        for stage in stages:
            m = masks[stage]
            mean_attn = data["attn"][m].mean(axis=0)
            ax.plot(x, mean_attn, linewidth=1.8, label=stage, color=colors[stage])
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Temporal token after CNN-LSTM")
        ax.grid(True, color="#E5E7EB", linestyle=":", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Mean attention weight")
    axes[1].legend(frameon=True, edgecolor="#DDDDDD")
    fig.suptitle("Seed42 attention profiles across degradation stages", x=0.02, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "fig05_seed42_attention_stage_profiles.png")
    plt.close(fig)


def plot_attention_heatmap(original: Dict[str, np.ndarray], adaptive: Dict[str, np.ndarray]):
    stages = ["Early", "Middle", "Late"]
    rows = []
    labels = []
    for method, data in [("Original", original), ("Adaptive", adaptive)]:
        masks = stage_masks(data["p_norm"])
        for stage in stages:
            rows.append(data["attn"][masks[stage]].mean(axis=0))
            labels.append(f"{method} {stage}")
    mat = np.vstack(rows)
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(0, mat.shape[1], 4))
    ax.set_xticklabels([str(i + 1) for i in range(0, mat.shape[1], 4)])
    ax.set_xlabel("Temporal token")
    ax.set_title("Mean attention map by stage and method", loc="left", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Attention weight")
    fig.savefig(FIG_DIR / "fig06_seed42_attention_heatmap.png")
    plt.close(fig)


def plot_training_curve():
    original = pd.read_csv(ROOT / "codex" / "final-retained-v1" / "final-retained-v1_history.csv")
    adaptive = pd.read_csv(OUT_DIR / "adaptive-attention-v1-seed42" / "adaptive-attention-v1-seed42_history.csv")
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.plot(original["epoch"], original["target_adapt_pooled_rmse"] * 100, color="#7C9DC9", linewidth=1.4, label="Original attention")
    ax.plot(adaptive["epoch"], adaptive["ta_rmse"] * 100, color="#C2410C", linewidth=1.4, label="Adaptive curve attention")
    ax.scatter([int(original.loc[original["target_adapt_pooled_rmse"].idxmin(), "epoch"])], [original["target_adapt_pooled_rmse"].min() * 100], color="#1B3D6E", s=28, zorder=4)
    ax.scatter([int(adaptive.loc[adaptive["ta_rmse"].idxmin(), "epoch"])], [adaptive["ta_rmse"].min() * 100], color="#C2410C", s=28, zorder=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Batch9 target-adapt RMSE (%)")
    ax.set_title("Seed42 checkpoint-selection curve", loc="left", fontweight="bold")
    ax.grid(True, color="#E5E7EB", linestyle=":", linewidth=0.7)
    ax.legend(frameon=True, edgecolor="#DDDDDD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(FIG_DIR / "fig07_seed42_target_adapt_training_curve.png")
    plt.close(fig)


def save_attention_summary(original: Dict[str, np.ndarray], adaptive: Dict[str, np.ndarray]):
    rows = []
    for method, data in [("Original attention", original), ("Adaptive curve attention", adaptive)]:
        masks = stage_masks(data["p_norm"])
        for stage, m in masks.items():
            attn = data["attn"][m]
            entropy = -np.sum(attn * np.log(np.clip(attn, 1e-12, None)), axis=1)
            rows.append(
                {
                    "method": method,
                    "stage": stage,
                    "n_samples": int(m.sum()),
                    "mean_entropy": float(entropy.mean()),
                    "mean_max_weight": float(attn.max(axis=1).mean()),
                    "mean_center_of_mass": float((attn * np.arange(1, attn.shape[1] + 1)).sum(axis=1).mean()),
                    "mean_temperature": float(np.mean(data["temperature"][m])) if "temperature" in data else np.nan,
                }
            )
    pd.DataFrame(rows).to_csv(OUT_DIR / "adaptive_attention_v1_seed42_attention_stage_summary.csv", index=False, encoding="utf-8-sig")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paper_style()

    metrics = build_metric_table()
    plot_seed_rmse(metrics)
    plot_training_curve()

    final_mod = load_module(FINAL_SCRIPT, "final_retained_v1_mod")
    adaptive_mod = load_module(ADAPTIVE_SCRIPT, "adaptive_attention_v1_mod")

    source_x, source_y, source_p, _ = load_split("source")
    target_x, target_y, target_p, target_bidx = load_split("target_test")
    coeffs = np.polyfit((source_p.flatten() / GLOBAL_MAX_CYCLE), source_y.flatten(), 3)

    device = torch.device("cpu")
    original_ckpt = torch.load(FINAL_CKPT, map_location=device, weights_only=False)
    original_coeffs = torch.tensor(original_ckpt["base_coeffs"], dtype=torch.float32, device=device)
    original_model = final_mod.ResidualCRNN(original_coeffs).to(device)
    original_model.load_state_dict(original_ckpt["model_state_dict"])

    adaptive_ckpt = torch.load(ADAPTIVE_CKPT, map_location=device, weights_only=False)
    adaptive_coeffs = torch.tensor(coeffs, dtype=torch.float32, device=device)
    adaptive_model = adaptive_mod.ResidualCRNN(adaptive_coeffs, use_physical_base=True).to(device)
    adaptive_model.load_state_dict(adaptive_ckpt["model_state_dict"])

    ds_final = final_mod.HybridDataset(target_x, target_y, target_p, target_bidx)
    ds_adaptive = adaptive_mod.HybridDataset(target_x, target_y, target_p, target_bidx)
    original = collect_predictions_and_attention(original_model, ds_final, device)
    adaptive = collect_predictions_and_attention(adaptive_model, ds_adaptive, device)

    pred_df = pd.DataFrame(
        {
            "battery_index": target_bidx.reshape(-1).astype(int),
            "cycle": target_p.reshape(-1).astype(float),
            "p_norm": target_p.reshape(-1).astype(float) / GLOBAL_MAX_CYCLE,
            "true_soh": target_y.reshape(-1).astype(float),
            "original_pred": original["pred"],
            "adaptive_pred": adaptive["pred"],
            "original_abs_error": np.abs(target_y.reshape(-1) - original["pred"]),
            "adaptive_abs_error": np.abs(target_y.reshape(-1) - adaptive["pred"]),
        }
    )
    pred_df.to_csv(OUT_DIR / "adaptive_attention_v1_seed42_target_test_predictions.csv", index=False, encoding="utf-8-sig")

    stage_rows = []
    masks = stage_masks(pred_df["p_norm"].to_numpy())
    for stage, m in masks.items():
        for method, col in [("Original attention", "original_pred"), ("Adaptive curve attention", "adaptive_pred")]:
            stage_rows.append(
                {
                    "method": method,
                    "stage": stage,
                    "n_samples": int(m.sum()),
                    "rmse": rmse(pred_df.loc[m, "true_soh"].to_numpy(), pred_df.loc[m, col].to_numpy()),
                    "mae": mae(pred_df.loc[m, "true_soh"].to_numpy(), pred_df.loc[m, col].to_numpy()),
                }
            )
    stage_df = pd.DataFrame(stage_rows)
    stage_df.to_csv(OUT_DIR / "adaptive_attention_v1_seed42_stage_metrics.csv", index=False, encoding="utf-8-sig")

    plot_stage_rmse(stage_df)
    plot_per_cell_rmse(pred_df)
    plot_prediction_curves(pred_df)
    plot_attention_profiles(original, adaptive)
    plot_attention_heatmap(original, adaptive)
    save_attention_summary(original, adaptive)

    print("Saved figures:")
    for p in sorted(FIG_DIR.glob("*.png")):
        print(p)
    print("\nThree-seed metrics:")
    print(metrics.to_string(index=False))
    print("\nSeed42 stage metrics:")
    print(stage_df.to_string(index=False))


if __name__ == "__main__":
    main()
