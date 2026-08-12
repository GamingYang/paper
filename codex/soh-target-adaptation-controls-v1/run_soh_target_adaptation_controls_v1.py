import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


EXPERIMENT_NAME = "soh-target-adaptation-controls-v1"
SCRIPT_PATH = Path(__file__).resolve()
RUN_DIR = SCRIPT_PATH.parent
ROOT = RUN_DIR.parents[1]
FIGURE_DIR = RUN_DIR / "figures"
PLOT_SKILL_DIR = ROOT / "paper-plot-skills-main" / "plot-from-data"
SOH_LIB_PATH = ROOT / "codex" / "soh-physical-degree-ablation-v1" / "run_soh_physical_degree_ablation_v1.py"

LABEL_MODE = "nominal"
DATA_PROTOCOL = "processed_clean_fixed_split"
SEED = 42
DEGREE = 3
EPOCHS = 150
BATCH_SIZE = 128
EVAL_BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
TARGET_MMD_WEIGHT = 0.00005
LR_DECAY_EPOCH = 100
LR_DECAY_FACTOR = 0.1


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    protocol: str
    k_target_cells: int
    target_cells: Tuple[int, ...]
    selection_split: str
    uses_target_mmd: bool
    uses_target_checkpoint_selection: bool


def load_soh_lib():
    if not SOH_LIB_PATH.exists():
        raise FileNotFoundError(f"Missing SOH library script: {SOH_LIB_PATH}")
    spec = importlib.util.spec_from_file_location("soh_degree_lib", SOH_LIB_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import SOH library script: {SOH_LIB_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["soh_degree_lib"] = module
    spec.loader.exec_module(module)
    return module


SOH = load_soh_lib()
DATA_DIR = SOH.DATA_DIR


def load_split(split_name: str):
    return SOH.load_split(split_name, LABEL_MODE)


def set_seed(seed: int) -> None:
    SOH.set_seed(seed)


def subset_arrays_by_cells(
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    bidx: np.ndarray,
    batch: np.ndarray,
    cells: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bidx_flat = bidx.reshape(-1)
    batch_flat = batch.reshape(-1)
    mask = np.isin(bidx_flat, np.asarray(cells, dtype=np.int64))
    return x[mask], y[mask], p[mask], bidx_flat[mask], batch_flat[mask]


def invert_cells(all_cells: Sequence[int], keepout_cells: Sequence[int]) -> List[int]:
    keepout = set(int(c) for c in keepout_cells)
    return [int(c) for c in all_cells if int(c) not in keepout]


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    bidx: np.ndarray,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    seed: int,
) -> DataLoader:
    dataset = SOH.HybridDataset(x, y, p, bidx)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        generator=generator if shuffle else None,
    )


def next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffers = sum(b.numel() for b in model.buffers())
    return {"total_params": int(total), "trainable_params": int(trainable), "buffer_params": int(buffers)}


def benchmark_forward(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    repeats: int = 3,
) -> Dict[str, float]:
    model.eval()
    sample_count = len(loader.dataset)
    if sample_count == 0:
        return {"inference_seconds": float("nan"), "ms_per_sample": float("nan"), "samples_per_second": float("nan")}

    with torch.no_grad():
        for x, _, p, _ in loader:
            x = x.float().to(device)
            p = p.float().to(device)
            model(x, p)
            break
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed_values = []
    for _ in range(repeats):
        start = time.perf_counter()
        with torch.no_grad():
            for x, _, p, _ in loader:
                x = x.float().to(device)
                p = p.float().to(device)
                model(x, p)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_values.append(time.perf_counter() - start)

    elapsed = float(np.mean(elapsed_values))
    return {
        "inference_seconds": elapsed,
        "ms_per_sample": float(elapsed * 1000.0 / sample_count),
        "samples_per_second": float(sample_count / elapsed) if elapsed > 0 else float("inf"),
    }


def fixed_source_validation_cells(source_cells: Sequence[int], n_val: int = 8) -> Tuple[int, ...]:
    rng = np.random.default_rng(SEED)
    values = np.asarray(sorted(int(c) for c in source_cells), dtype=np.int64)
    order = rng.permutation(values)
    return tuple(sorted(int(c) for c in order[:n_val]))


def fixed_target_order(target_cells: Sequence[int]) -> Tuple[int, ...]:
    rng = np.random.default_rng(SEED)
    values = np.asarray(sorted(int(c) for c in target_cells), dtype=np.int64)
    order = rng.permutation(values)
    return tuple(int(c) for c in order)


def build_plan(target_cells: Sequence[int], plan: str) -> Tuple[List[RunConfig], Tuple[int, ...]]:
    target_order = fixed_target_order(target_cells)
    if plan == "single":
        configs = [
            RunConfig(
                run_id="k0_source_only",
                protocol="source_only_no_target_adaptation",
                k_target_cells=0,
                target_cells=tuple(),
                selection_split="source_internal_validation",
                uses_target_mmd=False,
                uses_target_checkpoint_selection=False,
            )
        ]
        for k in [1, 2, 4, len(target_order)]:
            cells = tuple(sorted(target_order[:k]))
            suffix = "all" if k == len(target_order) else f"{k}"
            configs.append(
                RunConfig(
                    run_id=f"k{k}_target_subset_{suffix}",
                    protocol="source_plus_fixed_batch9_subset_mmd_and_selection",
                    k_target_cells=k,
                    target_cells=cells,
                    selection_split="batch9_target_subset",
                    uses_target_mmd=True,
                    uses_target_checkpoint_selection=True,
                )
            )
        return configs, target_order

    if plan == "expanded":
        configs = [
            RunConfig(
                run_id="k0_source_only",
                protocol="source_only_no_target_adaptation",
                k_target_cells=0,
                target_cells=tuple(),
                selection_split="source_internal_validation",
                uses_target_mmd=False,
                uses_target_checkpoint_selection=False,
            )
        ]
        sorted_cells = tuple(sorted(int(c) for c in target_cells))
        subset_sets = {
            1: [(c,) for c in sorted_cells],
            2: [tuple(sorted((sorted_cells[i], sorted_cells[(i + 1) % len(sorted_cells)]))) for i in range(len(sorted_cells))],
            4: [
                tuple(sorted(sorted_cells[:4])),
                tuple(sorted(sorted_cells[4:])),
                tuple(sorted(sorted_cells[::2])),
                tuple(sorted(sorted_cells[1::2])),
            ],
            len(sorted_cells): [sorted_cells],
        }
        seen = set()
        for k, subsets in subset_sets.items():
            for rep_idx, cells in enumerate(subsets, start=1):
                if cells in seen:
                    continue
                seen.add(cells)
                suffix = "all" if k == len(sorted_cells) else f"rep{rep_idx}"
                configs.append(
                    RunConfig(
                        run_id=f"k{k}_{suffix}",
                        protocol="source_plus_fixed_batch9_subset_mmd_and_selection",
                        k_target_cells=k,
                        target_cells=tuple(int(c) for c in cells),
                        selection_split="batch9_target_subset",
                        uses_target_mmd=True,
                        uses_target_checkpoint_selection=True,
                    )
                )
        return configs, target_order

    raise ValueError("plan must be 'single' or 'expanded'")


def metric_row(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {
        f"{prefix}_pooled_rmse": float(metrics["pooled_rmse"]),
        f"{prefix}_macro_rmse": float(metrics["macro_rmse"]),
        f"{prefix}_mae": float(metrics["mae"]),
        f"{prefix}_mape_percent": float(metrics["mape_percent"]),
    }


def train_one_config(
    cfg: RunConfig,
    arrays: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    source_val_cells: Tuple[int, ...],
    reference_steps_per_epoch: int,
    device: torch.device,
    epochs: int,
    max_train_batches: Optional[int],
    log_every: int,
    smoke: bool,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    set_seed(SEED)
    run_path = RUN_DIR / cfg.run_id
    run_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_path / f"{cfg.run_id}_best.pt"
    history_csv = run_path / f"{cfg.run_id}_history.csv"

    s_x, s_y, s_p, s_bidx, s_batch = arrays["source"]
    ta_x, ta_y, ta_p, ta_bidx, ta_batch = arrays["target_adapt"]
    tt_x, tt_y, tt_p, tt_bidx, tt_batch = arrays["target_test"]

    if cfg.k_target_cells == 0:
        source_train_cells = invert_cells(sorted(np.unique(s_bidx).tolist()), source_val_cells)
        tr_x, tr_y, tr_p, tr_bidx, tr_batch = subset_arrays_by_cells(s_x, s_y, s_p, s_bidx, s_batch, source_train_cells)
        val_x, val_y, val_p, val_bidx, val_batch = subset_arrays_by_cells(s_x, s_y, s_p, s_bidx, s_batch, source_val_cells)
        sel_x, sel_y, sel_p, sel_bidx = val_x, val_y, val_p, val_bidx
        target_train_loader = None
        target_subset_summary = {"target_subset_samples": 0, "target_subset_cells": 0}
    else:
        tr_x, tr_y, tr_p, tr_bidx, tr_batch = s_x, s_y, s_p, s_bidx, s_batch
        tgt_x, tgt_y, tgt_p, tgt_bidx, tgt_batch = subset_arrays_by_cells(
            ta_x, ta_y, ta_p, ta_bidx, ta_batch, cfg.target_cells
        )
        sel_x, sel_y, sel_p, sel_bidx = tgt_x, tgt_y, tgt_p, tgt_bidx
        target_train_loader = make_loader(tgt_x, tgt_y, tgt_p, tgt_bidx, BATCH_SIZE, True, True, SEED + cfg.k_target_cells)
        target_subset_summary = {
            "target_subset_samples": int(len(tgt_x)),
            "target_subset_cells": int(len(np.unique(tgt_bidx))),
        }
        if len(target_train_loader) == 0:
            raise RuntimeError(f"Target subset for {cfg.run_id} has no complete training batch.")

    coeffs = np.polyfit((tr_p.flatten() / SOH.GLOBAL_MAX_CYCLE), tr_y.flatten(), DEGREE)
    base_coeffs = torch.tensor(coeffs, dtype=torch.float32, device=device)
    model = SOH.ResidualCRNN(base_coeffs).to(device)
    params = count_parameters(model)

    source_train_loader = make_loader(tr_x, tr_y, tr_p, tr_bidx, BATCH_SIZE, True, True, SEED)
    selection_loader = make_loader(sel_x, sel_y, sel_p, sel_bidx, EVAL_BATCH_SIZE, False, False, SEED)
    full_target_adapt_eval = make_loader(ta_x, ta_y, ta_p, ta_bidx, EVAL_BATCH_SIZE, False, False, SEED)
    target_test_eval = make_loader(tt_x, tt_y, tt_p, tt_bidx, EVAL_BATCH_SIZE, False, False, SEED)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_mse = nn.MSELoss()
    criterion_mmd = SOH.LifetimeAdaptiveMMD().to(device)

    max_steps = reference_steps_per_epoch
    if max_train_batches is not None and max_train_batches > 0:
        max_steps = min(max_steps, max_train_batches)
    if max_steps <= 0:
        raise RuntimeError("No training steps are available.")

    best_selection_rmse = float("inf")
    best_selection_metrics = None
    best_epoch = -1
    history_rows: List[Dict] = []
    source_iter = iter(source_train_loader)
    target_iter = iter(target_train_loader) if target_train_loader is not None else None
    started = time.perf_counter()
    started_at = datetime.now().isoformat(timespec="seconds")

    print(
        f"RUN {cfg.run_id}: protocol={cfg.protocol} k={cfg.k_target_cells} "
        f"target_cells={list(cfg.target_cells)} epochs={epochs} steps_per_epoch={max_steps}",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        if epoch == LR_DECAY_EPOCH + 1:
            for param_group in optimizer.param_groups:
                param_group["lr"] = LEARNING_RATE * LR_DECAY_FACTOR

        if cfg.uses_target_mmd and epoch > 20 and epoch <= 120:
            mmd_weight = TARGET_MMD_WEIGHT
            stage = "MMD Guide"
        elif cfg.uses_target_mmd and epoch <= 20:
            mmd_weight = 0.0
            stage = "MSE Warmup"
        elif cfg.uses_target_mmd:
            mmd_weight = 0.0
            stage = "Pure MSE Sprint"
        else:
            mmd_weight = 0.0
            stage = "Source-only MSE"

        batch_losses, mse_losses, mmd_losses = [], [], []
        for _ in range(max_steps):
            (sx, sy, sp, _), source_iter = next_batch(source_iter, source_train_loader)
            sx = sx.float().to(device)
            sy = sy.float().to(device)
            sp = sp.float().to(device)
            optimizer.zero_grad()
            pred_s, feat_s, base_s = model(sx, sp)
            loss_mse = criterion_mse(pred_s, sy)
            if cfg.uses_target_mmd and mmd_weight > 0 and target_train_loader is not None:
                (tx, _, tp, _), target_iter = next_batch(target_iter, target_train_loader)
                tx = tx.float().to(device)
                tp = tp.float().to(device)
                _, feat_t, base_t = model(tx, tp)
                loss_mmd = criterion_mmd(feat_s, feat_t, base_s, base_t)
            else:
                loss_mmd = torch.tensor(0.0, device=device)
            loss = loss_mse + mmd_weight * loss_mmd
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))
            mse_losses.append(float(loss_mse.item()))
            mmd_losses.append(float(loss_mmd.item()))

        selection_metrics, _ = SOH.evaluate(model, selection_loader, device)
        if selection_metrics["pooled_rmse"] < best_selection_rmse:
            best_selection_rmse = selection_metrics["pooled_rmse"]
            best_selection_metrics = selection_metrics
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "base_coeffs": coeffs,
                    "degree": DEGREE,
                    "selection_split": cfg.selection_split,
                    "selection_metrics": selection_metrics,
                    "config": cfg.__dict__,
                    "protocol": cfg.protocol,
                },
                checkpoint_path,
            )

        row = {
            "run_id": cfg.run_id,
            "epoch": epoch,
            "stage": stage,
            "train_loss": float(np.mean(batch_losses)),
            "train_mse": float(np.mean(mse_losses)),
            "train_mmd": float(np.mean(mmd_losses)),
            "mmd_weight": float(mmd_weight),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "selection_pooled_rmse": selection_metrics["pooled_rmse"],
            "selection_macro_rmse": selection_metrics["macro_rmse"],
            "selection_mae": selection_metrics["mae"],
            "selection_mape_percent": selection_metrics["mape_percent"],
            "best_epoch": best_epoch,
            "best_selection_rmse": best_selection_rmse,
        }
        history_rows.append(row)
        if epoch == 1 or epoch == epochs or epoch % log_every == 0:
            pd.DataFrame(history_rows).to_csv(history_csv, index=False, encoding="utf-8-sig")
            print(
                f"  epoch {epoch:03d}/{epochs} stage={stage} "
                f"selection_rmse={selection_metrics['pooled_rmse']:.6f} "
                f"best={best_selection_rmse:.6f}@{best_epoch}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    selection_metrics, selection_cells = SOH.evaluate(model, selection_loader, device)
    full_target_adapt_metrics, full_target_adapt_cells = SOH.evaluate(model, full_target_adapt_eval, device)
    target_test_metrics, target_test_cells = SOH.evaluate(model, target_test_eval, device)
    inference = benchmark_forward(model, target_test_eval, device, repeats=2 if smoke else 3)

    ckpt_mb = checkpoint_path.stat().st_size / (1024 * 1024) if checkpoint_path.exists() else float("nan")
    summary = {
        "run_id": cfg.run_id,
        "experiment": EXPERIMENT_NAME,
        "protocol": cfg.protocol,
        "data_protocol": DATA_PROTOCOL,
        "label_mode": LABEL_MODE,
        "degree": DEGREE,
        "seed": SEED,
        "epochs": epochs,
        "steps_per_epoch": int(max_steps),
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "train_wall_seconds": float(elapsed),
        "selection_split": cfg.selection_split,
        "uses_target_mmd": cfg.uses_target_mmd,
        "uses_target_checkpoint_selection": cfg.uses_target_checkpoint_selection,
        "k_target_cells": cfg.k_target_cells,
        "target_cells": ",".join(str(c) for c in cfg.target_cells),
        "source_val_cells_for_k0": ",".join(str(c) for c in source_val_cells) if cfg.k_target_cells == 0 else "",
        "best_epoch": int(best_epoch),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_size_mb": float(ckpt_mb),
        "base_coeffs": json.dumps([float(v) for v in coeffs]),
        **target_subset_summary,
        **params,
        **inference,
        **metric_row("selection", selection_metrics),
        **metric_row("batch9_full_diagnostic", full_target_adapt_metrics),
        **metric_row("batch10_blind", target_test_metrics),
    }
    history_df = pd.DataFrame(history_rows)
    selection_cells = selection_cells.assign(run_id=cfg.run_id, split=cfg.selection_split)
    full_target_adapt_cells = full_target_adapt_cells.assign(run_id=cfg.run_id, split="target_adapt_full_diagnostic")
    target_test_cells = target_test_cells.assign(run_id=cfg.run_id, split="target_test_blind")

    payload = {
        "summary": summary,
        "history": history_rows,
        "selection_cells": selection_cells.to_dict(orient="records"),
        "target_adapt_full_cells": full_target_adapt_cells.to_dict(orient="records"),
        "target_test_cells": target_test_cells.to_dict(orient="records"),
    }
    (run_path / f"{cfg.run_id}_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    history_df.to_csv(history_csv, index=False, encoding="utf-8-sig")
    return summary, history_df, full_target_adapt_cells, target_test_cells, payload


def aggregate_by_budget(overall: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, group in overall.groupby("k_target_cells", sort=True):
        rows.append(
            {
                "k_target_cells": int(k),
                "n_runs": int(len(group)),
                "batch10_rmse_mean": float(group["batch10_blind_pooled_rmse"].mean()),
                "batch10_rmse_std": float(group["batch10_blind_pooled_rmse"].std(ddof=1)) if len(group) > 1 else 0.0,
                "batch10_macro_rmse_mean": float(group["batch10_blind_macro_rmse"].mean()),
                "batch10_mae_mean": float(group["batch10_blind_mae"].mean()),
                "batch9_full_rmse_mean": float(group["batch9_full_diagnostic_pooled_rmse"].mean()),
                "selection_rmse_mean": float(group["selection_pooled_rmse"].mean()),
                "train_wall_seconds_mean": float(group["train_wall_seconds"].mean()),
                "ms_per_sample_mean": float(group["ms_per_sample"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("k_target_cells")


def load_reference_rows() -> pd.DataFrame:
    rows = []
    final_json = ROOT / "codex" / "final-retained-v1" / "final-retained-v1_best_metrics.json"
    if final_json.exists():
        payload = json.loads(final_json.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        metrics = {row["split"]: row for row in payload.get("metrics", [])}
        tt = metrics.get("target_test", {})
        ta = metrics.get("target_adapt", {})
        ckpt_path = Path(summary.get("checkpoint_path", ""))
        rows.append(
            {
                "reference_model": "archived_final_retained_v1",
                "selection_split": "batch9_target_adapt",
                "target_cells": "61,62,63,64,65,66,67,68",
                "best_epoch": summary.get("best_epoch_selected_by_target_adapt", np.nan),
                "batch9_pooled_rmse": ta.get("pooled_rmse", np.nan),
                "batch10_pooled_rmse": tt.get("pooled_rmse", np.nan),
                "batch10_macro_rmse": tt.get("macro_rmse", np.nan),
                "batch10_mae": tt.get("mae", np.nan),
                "checkpoint_size_mb": ckpt_path.stat().st_size / (1024 * 1024) if ckpt_path.exists() else np.nan,
                "note": "archived main result; not rerun by this script",
            }
        )
    return pd.DataFrame(rows)


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "DejaVu Serif", "Times New Roman"],
            "axes.unicode_minus": False,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "legend.fontsize": 8.8,
            "hatch.color": "white",
            "hatch.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png = FIGURE_DIR / f"{name}.png"
    pdf = FIGURE_DIR / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_budget_curve(budget: pd.DataFrame) -> None:
    setup_plot_style()
    x = budget["k_target_cells"].to_numpy(dtype=float)
    y = budget["batch10_rmse_mean"].to_numpy(dtype=float)
    std = budget["batch10_rmse_std"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 3.45))
    color = "#3A8B3A"
    ax.fill_between(x, y - std, y + std, color=color, alpha=0.15, linewidth=0)
    ax.plot(x, y, color=color, lw=1.9, marker="o", markersize=5.0, label="Batch 10 RMSE")
    ax.set_xlabel("Number of Batch 9 target-development cells")
    ax.set_ylabel("Blind-test RMSE")
    ax.set_title("Target-domain budget sensitivity", fontsize=11.5, pad=8)
    ax.set_xticks(x)
    ax.set_xlim(float(np.min(x)) - 0.4, float(np.max(x)) + 0.4)
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    y_pad = max(0.00020, 0.12 * (y_max - y_min if y_max > y_min else y_max))
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.grid(False)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_linewidth(0.9)
    ax.tick_params(direction="out", length=3.0, width=0.8, labelsize=9.5)
    ax.legend(framealpha=0.0, loc="upper right", fontsize=9.0)
    x_left = float(np.min(x))
    for xi, yi in zip(x, y):
        offset = max(0.000055, 0.025 * (y_max - y_min if y_max > y_min else y_max))
        ha = "left" if abs(float(xi) - x_left) < 1e-9 else "center"
        x_text = xi + 0.10 if ha == "left" else xi
        if yi + offset > y_max + 0.55 * y_pad:
            ax.text(x_text, yi - offset, f"{yi:.4f}", ha=ha, va="top", fontsize=8.3)
        else:
            ax.text(x_text, yi + offset, f"{yi:.4f}", ha=ha, va="bottom", fontsize=8.3)
    save_figure(fig, "soh_batch9_budget_sensitivity_line_confidence_band_style")


def plot_source_vs_budget_bars(budget: pd.DataFrame) -> None:
    setup_plot_style()
    labels = [f"k={int(k)}" for k in budget["k_target_cells"]]
    x = np.arange(len(labels))
    rmse = budget["batch10_rmse_mean"].to_numpy(dtype=float)
    mae = budget["batch10_mae_mean"].to_numpy(dtype=float)
    colors = ["#D3D3D3", "#A9A9A9", "#8FB8D8", "#E8845A", "#C0392B"][: len(labels)]
    fig, ax = plt.subplots(figsize=(6.0, 3.55))
    width = 0.36
    bars1 = ax.bar(x - width / 2, rmse, width=width, color=colors, edgecolor="white", linewidth=0.8, label="RMSE")
    bars2 = ax.bar(
        x + width / 2,
        mae,
        width=width,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.55,
        hatch="//",
        label="MAE",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Batch 10 error")
    ax.set_title("Source-only vs target-adapted controls", fontsize=11.5)
    ax.set_ylim(0, max(float(np.max(rmse)), float(np.max(mae))) * 1.18)
    ax.yaxis.grid(True, color="#EBEBEB", linewidth=0.7, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_linewidth(0.9)
    ax.tick_params(length=0, labelsize=9.5)
    ax.legend(framealpha=1.0, facecolor="white", edgecolor="#C8C8C8", fontsize=8.8, loc="upper right")
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.00025, f"{h:.4f}", ha="center", va="bottom", fontsize=7.6)
    save_figure(fig, "soh_source_only_vs_target_budget_bar_grouped_hatch_style")


def plot_efficiency_bars(overall: pd.DataFrame) -> None:
    setup_plot_style()
    labels = overall["run_id"].tolist()
    short_labels = [label.replace("_target_subset", "").replace("k0_source_only", "k0") for label in labels]
    x = np.arange(len(labels))
    train_min = overall["train_wall_seconds"].to_numpy(dtype=float) / 60.0
    ms = overall["ms_per_sample"].to_numpy(dtype=float)
    fig, ax1 = plt.subplots(figsize=(7.2, 3.8))
    ax2 = ax1.twinx()
    bars = ax1.bar(x - 0.18, train_min, width=0.36, color="#A9A9A9", edgecolor="white", linewidth=0.8, label="Training time")
    line = ax2.plot(x + 0.18, ms, marker="o", lw=1.8, color="#C0392B", label="Inference ms/sample")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, rotation=25, ha="right")
    ax1.set_ylabel("Training time (min)")
    ax2.set_ylabel("Inference latency (ms/sample)")
    ax1.set_title("Model complexity and efficiency diagnostics", fontsize=11.5)
    ax1.set_ylim(0, max(train_min) * 1.15)
    ax2.set_ylim(0, max(ms) * 1.25)
    ax1.yaxis.grid(True, color="#EBEBEB", linewidth=0.7, linestyle="--", zorder=0)
    for side, spine in ax1.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_linewidth(0.9)
    ax2.spines["top"].set_visible(False)
    ax1.tick_params(length=0, labelsize=8.8)
    ax2.tick_params(length=0, labelsize=8.8)
    handles = [bars, line[0]]
    labels_leg = ["Training time", "Inference ms/sample"]
    ax1.legend(handles, labels_leg, framealpha=1.0, facecolor="white", edgecolor="#C8C8C8", fontsize=8.8, loc="upper right")
    save_figure(fig, "soh_complexity_efficiency_bar_grouped_hatch_style")


def write_outputs(
    overall_rows: List[Dict],
    histories: List[pd.DataFrame],
    target_adapt_cells: List[pd.DataFrame],
    target_test_cells: List[pd.DataFrame],
    arrays: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    source_val_cells: Tuple[int, ...],
    target_order: Tuple[int, ...],
    plan: str,
    epochs: int,
    smoke: bool,
) -> None:
    overall = pd.DataFrame(overall_rows)
    history_all = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    ta_cells = pd.concat(target_adapt_cells, ignore_index=True) if target_adapt_cells else pd.DataFrame()
    tt_cells = pd.concat(target_test_cells, ignore_index=True) if target_test_cells else pd.DataFrame()
    budget = aggregate_by_budget(overall)
    references = load_reference_rows()

    split_rows = []
    for split, (x, y, p, bidx, batch) in arrays.items():
        split_rows.append(SOH.split_summary(split, x, y, p, bidx, batch))
    protocol_rows = [
        {"item": "fixed_split", "value": "Batch 1-8 source; Batch 9 target-development/adaptation; Batch 10 blind-test only"},
        {"item": "source_only_selection", "value": f"Cell-level source internal validation cells: {','.join(map(str, source_val_cells))}"},
        {"item": "target_budget_order", "value": ",".join(str(c) for c in target_order)},
        {"item": "target_budget_rule", "value": "For k>0, use the first k cells in the fixed target order; Batch 10 is not used for subset selection."},
        {"item": "plot_skill", "value": str(PLOT_SKILL_DIR)},
        {"item": "plot_styles", "value": "line_confidence_band; bar_grouped_hatch"},
        {"item": "degree_prior_wording", "value": "Degree-3 base is a low-order nonlinear degradation prior, not a strong electrochemical mechanism."},
        {"item": "mmd_wording", "value": "MMD is a lightweight alignment regularizer and should not be overclaimed."},
    ]

    overall_csv = RUN_DIR / "soh_target_adaptation_controls_v1_overall.csv"
    budget_csv = RUN_DIR / "soh_target_budget_summary.csv"
    complexity_csv = RUN_DIR / "soh_complexity_efficiency_table.csv"
    xlsx_path = RUN_DIR / "soh_target_adaptation_controls_v1_summary.xlsx"
    overall.to_csv(overall_csv, index=False, encoding="utf-8-sig")
    budget.to_csv(budget_csv, index=False, encoding="utf-8-sig")
    complexity_cols = [
        "run_id",
        "k_target_cells",
        "total_params",
        "trainable_params",
        "buffer_params",
        "checkpoint_size_mb",
        "train_wall_seconds",
        "inference_seconds",
        "ms_per_sample",
        "samples_per_second",
        "best_epoch",
        "steps_per_epoch",
    ]
    overall[complexity_cols].to_csv(complexity_csv, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        overall.to_excel(writer, index=False, sheet_name="overall_runs")
        budget.to_excel(writer, index=False, sheet_name="budget_summary")
        overall[complexity_cols].to_excel(writer, index=False, sheet_name="complexity_efficiency")
        references.to_excel(writer, index=False, sheet_name="archived_references")
        history_all.to_excel(writer, index=False, sheet_name="history_all")
        ta_cells.to_excel(writer, index=False, sheet_name="batch9_full_cells")
        tt_cells.to_excel(writer, index=False, sheet_name="batch10_cells")
        pd.DataFrame(split_rows).to_excel(writer, index=False, sheet_name="splits")
        pd.DataFrame(protocol_rows).to_excel(writer, index=False, sheet_name="protocol_notes")

    plot_budget_curve(budget)
    plot_source_vs_budget_bars(budget)
    plot_efficiency_bars(overall)

    summary_payload = {
        "experiment": EXPERIMENT_NAME,
        "status": "success",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "plan": plan,
        "smoke": smoke,
        "epochs": epochs,
        "data_dir": str(DATA_DIR),
        "plot_skill_dir": str(PLOT_SKILL_DIR),
        "outputs": {
            "overall_csv": str(overall_csv),
            "budget_csv": str(budget_csv),
            "complexity_csv": str(complexity_csv),
            "summary_xlsx": str(xlsx_path),
            "figures": [str(p) for p in sorted(FIGURE_DIR.glob("*.png"))],
        },
        "budget_summary": budget.to_dict(orient="records"),
        "overall": overall.to_dict(orient="records"),
        "archived_references": references.to_dict(orient="records"),
        "protocol_notes": protocol_rows,
    }
    (RUN_DIR / "soh_target_adaptation_controls_v1_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(overall, budget, references, protocol_rows, xlsx_path)


def fmt(value: float, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def write_report(
    overall: pd.DataFrame,
    budget: pd.DataFrame,
    references: pd.DataFrame,
    protocol_rows: List[Dict],
    xlsx_path: Path,
) -> None:
    lines = [
        "# SOH Target-Adaptation Controls v1",
        "",
        "This package adds SOH-only source-only / no-target-adaptation control experiments, Batch 9 target-development cell-budget sensitivity, and a model complexity/efficiency table.",
        "",
        "## Protocol Guardrails",
        "",
        "- Source domain: Batch 1-8.",
        "- Target-development/adaptation domain: Batch 9.",
        "- Blind-test domain: Batch 10.",
        "- Batch 10 is not used for training, MMD alignment, checkpoint selection, target subset selection, or hyperparameter tuning.",
        "- The cubic base is treated as a low-order nonlinear degradation prior, not a strong electrochemical mechanism model.",
        "- MMD is treated as a lightweight alignment regularizer; this experiment does not turn it into the main contribution.",
        "",
        "## Key Outputs",
        "",
        f"- Summary workbook: `{xlsx_path}`",
        f"- Overall CSV: `{RUN_DIR / 'soh_target_adaptation_controls_v1_overall.csv'}`",
        f"- Target-budget CSV: `{RUN_DIR / 'soh_target_budget_summary.csv'}`",
        f"- Complexity CSV: `{RUN_DIR / 'soh_complexity_efficiency_table.csv'}`",
        f"- Figures: `{FIGURE_DIR}`",
        "",
        "## Budget Summary",
        "",
        "| k Batch9 cells | n runs | Batch10 RMSE mean | Batch10 MAE mean | Batch9 full RMSE mean |",
        "|---:|---:|---:|---:|---:|",
    ]
    for _, row in budget.iterrows():
        lines.append(
            "| {k} | {n} | {rmse} | {mae} | {b9} |".format(
                k=int(row["k_target_cells"]),
                n=int(row["n_runs"]),
                rmse=fmt(row["batch10_rmse_mean"], 6),
                mae=fmt(row["batch10_mae_mean"], 6),
                b9=fmt(row["batch9_full_rmse_mean"], 6),
            )
        )

    lines.extend(
        [
            "",
            "## Run-Level Results",
            "",
            "| run | protocol | target cells | selection | Batch10 RMSE | Batch10 macro RMSE | Batch10 MAE | best epoch |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in overall.iterrows():
        lines.append(
            "| {run} | {protocol} | {cells} | {sel} | {rmse} | {macro} | {mae} | {epoch} |".format(
                run=row["run_id"],
                protocol=row["protocol"],
                cells=row["target_cells"] if str(row["target_cells"]) else "-",
                sel=row["selection_split"],
                rmse=fmt(row["batch10_blind_pooled_rmse"], 6),
                macro=fmt(row["batch10_blind_macro_rmse"], 6),
                mae=fmt(row["batch10_blind_mae"], 6),
                epoch=int(row["best_epoch"]),
            )
        )

    lines.extend(
        [
            "",
            "## Complexity And Efficiency",
            "",
            "| run | trainable params | checkpoint MB | train min | inference ms/sample | samples/s |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in overall.iterrows():
        lines.append(
            "| {run} | {params} | {ckpt} | {train} | {ms} | {sps} |".format(
                run=row["run_id"],
                params=int(row["trainable_params"]),
                ckpt=fmt(row["checkpoint_size_mb"], 3),
                train=fmt(float(row["train_wall_seconds"]) / 60.0, 2),
                ms=fmt(row["ms_per_sample"], 5),
                sps=fmt(row["samples_per_second"], 1),
            )
        )

    if len(references) > 0:
        ref = references.iloc[0]
        lines.extend(
            [
                "",
                "## Archived Final-Retained Reference",
                "",
                "The archived final-retained-v1 row is listed for manuscript cross-checking only; it was not rerun or reselected here.",
                "",
                f"- Batch 9 RMSE: {fmt(ref['batch9_pooled_rmse'], 6)}",
                f"- Batch 10 pooled RMSE: {fmt(ref['batch10_pooled_rmse'], 6)}",
                f"- Batch 10 macro RMSE: {fmt(ref['batch10_macro_rmse'], 6)}",
                f"- Batch 10 MAE: {fmt(ref['batch10_mae'], 6)}",
            ]
        )

    lines.extend(
        [
            "",
            "## Plot Skill Use",
            "",
            f"- Skill directory used: `{PLOT_SKILL_DIR}`",
            "- Styles used: `line_confidence_band` for target-budget curve and `bar_grouped_hatch` for grouped bars.",
            "- Generated numerical figures are data-derived Matplotlib outputs; they are not AI-generated replacement images.",
            "",
            "## Manuscript Interpretation",
            "",
            "- Use k=0 as the clean source-only / no-target-adaptation control because it selects checkpoints only on source internal validation.",
            "- Use the k>0 rows as fixed nested target-development budget sensitivity, not as Batch 10-driven model selection.",
            "- Because the default `single` plan uses one predefined Batch 9 subset per budget, do not claim monotonic or distribution-level label-budget improvement unless the `expanded` plan is run.",
            "- If the k=8 rerun differs from archived final-retained-v1, report it as a same-protocol rerun and keep final-retained-v1 as the locked archived main result unless the whole protocol is formally restarted.",
            "- Do not claim that MMD is the dominant source of improvement from this table alone.",
        ]
    )

    (RUN_DIR / "soh_target_adaptation_controls_v1_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--plan", choices=["single", "expanded"], default="single")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = int(args.epochs)
    max_train_batches = int(args.max_train_batches) if int(args.max_train_batches) > 0 else None

    if args.smoke:
        epochs = min(epochs, 2)
        max_train_batches = 2

    arrays = {
        "source": load_split("source"),
        "target_adapt": load_split("target_adapt"),
        "target_test": load_split("target_test"),
    }
    s_x, s_y, s_p, s_bidx, s_batch = arrays["source"]
    ta_x, ta_y, ta_p, ta_bidx, ta_batch = arrays["target_adapt"]

    source_cells = tuple(sorted(int(c) for c in np.unique(s_bidx)))
    target_cells = tuple(sorted(int(c) for c in np.unique(ta_bidx)))
    source_val_cells = fixed_source_validation_cells(source_cells, n_val=min(8, max(1, len(source_cells) - 1)))
    configs, target_order = build_plan(target_cells, args.plan)
    reference_steps_per_epoch = len(
        make_loader(ta_x, ta_y, ta_p, ta_bidx, BATCH_SIZE, True, True, SEED)
    )
    if reference_steps_per_epoch <= 0:
        raise RuntimeError("Batch 9 target_adapt has no complete reference training batches.")

    print(f"EXPERIMENT={EXPERIMENT_NAME}", flush=True)
    print(f"python={sys.executable}", flush=True)
    print(f"device={device}", flush=True)
    print(f"data_dir={DATA_DIR}", flush=True)
    print(f"plot_skill_dir={PLOT_SKILL_DIR}", flush=True)
    print(f"plan={args.plan} epochs={epochs} smoke={args.smoke}", flush=True)
    print(f"source_val_cells={list(source_val_cells)}", flush=True)
    print(f"target_order={list(target_order)}", flush=True)

    overall_rows: List[Dict] = []
    histories: List[pd.DataFrame] = []
    target_adapt_cells: List[pd.DataFrame] = []
    target_test_cells: List[pd.DataFrame] = []

    for cfg in configs:
        summary, history, ta_cells, tt_cells, _ = train_one_config(
            cfg=cfg,
            arrays=arrays,
            source_val_cells=source_val_cells,
            reference_steps_per_epoch=reference_steps_per_epoch,
            device=device,
            epochs=epochs,
            max_train_batches=max_train_batches,
            log_every=max(1, int(args.log_every)),
            smoke=bool(args.smoke),
        )
        overall_rows.append(summary)
        histories.append(history)
        target_adapt_cells.append(ta_cells)
        target_test_cells.append(tt_cells)

    write_outputs(
        overall_rows=overall_rows,
        histories=histories,
        target_adapt_cells=target_adapt_cells,
        target_test_cells=target_test_cells,
        arrays=arrays,
        source_val_cells=source_val_cells,
        target_order=target_order,
        plan=args.plan,
        epochs=epochs,
        smoke=bool(args.smoke),
    )
    print("DONE", RUN_DIR, flush=True)


if __name__ == "__main__":
    main()
