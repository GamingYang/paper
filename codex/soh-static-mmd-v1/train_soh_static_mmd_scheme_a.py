import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


MODEL_NAME = "soh-static-mmd"
VERSION = "v1"
RUN_NAME = f"{MODEL_NAME}-{VERSION}"

SCRIPT_PATH = Path(__file__).resolve()
RUN_DIR = SCRIPT_PATH.parent
ROOT = RUN_DIR.parents[1]
DATA_DIR = ROOT / "PYthon训练代码" / "processed_clean"

LABEL_MODE = "nominal"
DATA_PROTOCOL = "processed_clean_fixed_split"
SEED = 42
EPOCHS = 150
BATCH_SIZE = 128
EVAL_BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
GLOBAL_MAX_CYCLE = 3000.0
TARGET_MMD_WEIGHT = 0.00005
LAMBDA_SMOOTH = 0.0
LR_DECAY_EPOCH = 100
LR_DECAY_FACTOR = 0.1

CHECKPOINT_PATH = RUN_DIR / f"{RUN_NAME}_best_by_target_adapt.pt"
BEST_JSON = RUN_DIR / f"{RUN_NAME}_best_metrics.json"
RESULT_XLSX = RUN_DIR / f"{RUN_NAME}_results.xlsx"
SUMMARY_XLSX = RUN_DIR / f"{RUN_NAME}_summary.xlsx"
HISTORY_CSV = RUN_DIR / f"{RUN_NAME}_history.csv"
SUMMARY_CSV = RUN_DIR / f"{RUN_NAME}_summary.csv"
REPORT_MD = RUN_DIR / f"{RUN_NAME}_report.md"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return np.load(path)


def load_split(split_name: str, label_mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = load_npy(DATA_DIR / f"clean_{split_name}_x.npy").astype(np.float32)
    p = load_npy(DATA_DIR / f"clean_{split_name}_p.npy").astype(np.float32)
    bidx = load_npy(DATA_DIR / f"clean_{split_name}_battery_index.npy").astype(np.int64)
    batch = load_npy(DATA_DIR / f"clean_{split_name}_batch.npy").astype(np.int64)
    if label_mode == "nominal":
        y = load_npy(DATA_DIR / f"clean_{split_name}_y_nominal.npy").astype(np.float32)
    elif label_mode == "relative":
        y = load_npy(DATA_DIR / f"clean_{split_name}_y_relative.npy").astype(np.float32)
    else:
        raise ValueError("label_mode must be 'nominal' or 'relative'")
    return x, y, p, bidx, batch


class HybridDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, p: np.ndarray, bidx: np.ndarray):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        self.p = (p.astype(np.float32) / GLOBAL_MAX_CYCLE).reshape(-1, 1)
        self.bidx = bidx.astype(np.int64)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.p[idx], self.bidx[idx]


class LifetimeAdaptiveMMD(nn.Module):
    def __init__(self, kernel_mul: float = 2.0, kernel_num: int = 5, lambda_scale: float = 5.5):
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.lambda_scale = lambda_scale

    def gaussian_kernel(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_samples = int(source.size(0)) + int(target.size(0))
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
        total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
        l2_distance = ((total0 - total1) ** 2).sum(2)
        bandwidth = torch.sum(l2_distance.detach()) / max(n_samples**2 - n_samples, 1)
        bandwidth = torch.clamp(bandwidth, min=1e-6)
        bandwidth = bandwidth / (self.kernel_mul ** (self.kernel_num // 2))
        bandwidth_list = [bandwidth * (self.kernel_mul**i) for i in range(self.kernel_num)]
        return sum(torch.exp(-l2_distance / bw) for bw in bandwidth_list)

    def compute_mmd(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if source.size(0) == 0 or target.size(0) == 0:
            return torch.tensor(0.0, device=source.device)
        batch_size_s = int(source.size(0))
        kernels = self.gaussian_kernel(source, target)
        xx = kernels[:batch_size_s, :batch_size_s]
        yy = kernels[batch_size_s:, batch_size_s:]
        xy = kernels[:batch_size_s, batch_size_s:]
        return torch.mean(xx) + torch.mean(yy) - 2 * torch.mean(xy)

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        source_base_soh: torch.Tensor,
        target_base_soh: torch.Tensor,
    ) -> torch.Tensor:
        if source_features.size(0) == 0 or target_features.size(0) == 0:
            return torch.tensor(0.0, device=source_features.device)
        return self.compute_mmd(source_features, target_features)


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, lstm_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_weights = F.softmax(self.attention(lstm_outputs), dim=1)
        context_vector = torch.sum(attn_weights * lstm_outputs, dim=1)
        return context_vector, attn_weights


class ResidualCRNN(nn.Module):
    def __init__(self, base_coeffs: torch.Tensor):
        super().__init__()
        self.base_coeffs = base_coeffs
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        self.attention = TemporalAttention(hidden_dim=64)
        self.phy = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU())
        self.fc_soh = nn.Sequential(nn.Linear(72, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a, b, c, d = self.base_coeffs
        base_soh = a * (p**3) + b * (p**2) + c * p + d
        out = self.cnn(x.unsqueeze(1)).permute(0, 2, 1)
        lstm_out, _ = self.lstm(out)
        context_feat, _ = self.attention(lstm_out)
        fused_features = torch.cat((context_feat, self.phy(p)), dim=1)
        residual_soh = self.fc_soh(fused_features)
        pred_soh = base_soh + residual_soh
        return pred_soh, fused_features, base_soh


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean((y_true.reshape(-1) - y_pred.reshape(-1)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true.reshape(-1) - y_pred.reshape(-1))))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.clip(np.abs(y_true.reshape(-1)), 1e-8, None)
    return float(np.mean(np.abs((y_true.reshape(-1) - y_pred.reshape(-1)) / denom)) * 100)


def macro_rmse_by_battery(y_true: np.ndarray, y_pred: np.ndarray, battery_idx: np.ndarray) -> float:
    vals = []
    for b in np.unique(battery_idx):
        mask = battery_idx == b
        if np.any(mask):
            vals.append(rmse(y_true[mask], y_pred[mask]))
    return float(np.mean(vals)) if vals else float("nan")


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[Dict[str, float], pd.DataFrame]:
    model.eval()
    preds_all, trues_all, bidx_all = [], [], []
    with torch.no_grad():
        for x, y, p, b in loader:
            x = x.float().to(device)
            p = p.float().to(device)
            out, _, _ = model(x, p)
            preds_all.append(out.cpu().numpy().reshape(-1))
            trues_all.append(y.numpy().reshape(-1))
            bidx_all.append(b.numpy().reshape(-1))
    y_pred = np.concatenate(preds_all)
    y_true = np.concatenate(trues_all)
    bidx = np.concatenate(bidx_all)
    metrics = {
        "pooled_rmse": rmse(y_true, y_pred),
        "macro_rmse": macro_rmse_by_battery(y_true, y_pred, bidx),
        "mae": mae(y_true, y_pred),
        "mape_percent": mape(y_true, y_pred),
    }
    rows = []
    for b in np.unique(bidx):
        mask = bidx == b
        rows.append(
            {
                "battery_index": int(b),
                "rmse": rmse(y_true[mask], y_pred[mask]),
                "mae": mae(y_true[mask], y_pred[mask]),
                "n_points": int(np.sum(mask)),
            }
        )
    return metrics, pd.DataFrame(rows)


def split_summary(name: str, x: np.ndarray, y: np.ndarray, p: np.ndarray, bidx: np.ndarray, batch: np.ndarray) -> Dict:
    return {
        "split": name,
        "samples": int(len(x)),
        "cells": int(len(np.unique(bidx))),
        "batches": ",".join(map(str, sorted(np.unique(batch).tolist()))),
        "x_shape": "x".join(map(str, x.shape)),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "cycle_min": float(np.min(p)),
        "cycle_max": float(np.max(p)),
    }


def save_outputs(
    summary: Dict,
    split_rows: List[Dict],
    history_rows: List[Dict],
    target_adapt_metrics: Dict,
    target_test_metrics: Dict,
    target_adapt_per_battery: pd.DataFrame,
    target_test_per_battery: pd.DataFrame,
) -> None:
    metrics_rows = [
        {"split": "target_adapt", **target_adapt_metrics, "role": "model_selection"},
        {"split": "target_test", **target_test_metrics, "role": "final_blind_test_once"},
    ]
    summary_rows = [
        {
            "model": RUN_NAME,
            "best_target_adapt_rmse": target_adapt_metrics["pooled_rmse"],
            "final_target_test_rmse": target_test_metrics["pooled_rmse"],
        }
    ]
    with pd.ExcelWriter(RESULT_XLSX, engine="openpyxl") as writer:
        pd.DataFrame([summary]).to_excel(writer, index=False, sheet_name="config")
        pd.DataFrame(split_rows).to_excel(writer, index=False, sheet_name="splits")
        pd.DataFrame(history_rows).to_excel(writer, index=False, sheet_name="history")
        pd.DataFrame(metrics_rows).to_excel(writer, index=False, sheet_name="metrics")
        target_adapt_per_battery.to_excel(writer, index=False, sheet_name="target_adapt_cells")
        target_test_per_battery.to_excel(writer, index=False, sheet_name="target_test_cells")
    pd.DataFrame(summary_rows).to_excel(SUMMARY_XLSX, index=False, sheet_name="summary")
    pd.DataFrame(metrics_rows).to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    payload = {
        "summary": summary,
        "splits": split_rows,
        "history": history_rows,
        "metrics": metrics_rows,
        "target_adapt_per_battery": target_adapt_per_battery.to_dict(orient="records"),
        "target_test_per_battery": target_test_per_battery.to_dict(orient="records"),
        "compact_summary": summary_rows,
    }
    BEST_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_archived_metrics(path: Path) -> Dict[str, Dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["split"]: row for row in payload["metrics"]}


def write_report(summary: Dict, target_adapt_metrics: Dict, target_test_metrics: Dict) -> None:
    final_retained = load_archived_metrics(ROOT / "codex" / "final-retained-v1" / "final-retained-v1_best_metrics.json")
    no_mmd = load_archived_metrics(ROOT / "codex" / "no-mmd-v1" / "no-mmd-v1_best_metrics.json")

    def metric_line(name: str, metrics: Dict[str, float]) -> str:
        return (
            f"| {name} | {metrics['pooled_rmse']:.9f} | {metrics['macro_rmse']:.9f} | "
            f"{metrics['mae']:.9f} | {metrics['mape_percent']:.9f} |"
        )

    report = [
        f"# {RUN_NAME} Static MMD SOH Control",
        "",
        "## Protocol",
        "",
        "- SOH-only experiment on `processed_clean_fixed_split`.",
        "- Batch 1-8 source data are used for supervised training.",
        "- Batch 9 (`target_adapt`) is used for MMD adaptation and checkpoint/model selection.",
        "- Batch 10 (`target_test`) is used only once for final reporting after selection.",
        "- Batch 10 is not used for training, tuning, checkpoint selection, hyperparameter choice, seed selection, or figure selection.",
        "- Network structure, seed, epochs, MSE warmup/MMD Guide/Pure MSE Sprint schedule, and `TARGET_MMD_WEIGHT=5e-05` follow `final-retained-v1`.",
        "- Static MMD change: `MMD_loss = compute_mmd(source_features, target_features)`; the exponential SOH-dependent `dynamic_weight` is removed.",
        "",
        "## Outputs",
        "",
        f"- Training script: `{SCRIPT_PATH}`",
        f"- Summary JSON: `{BEST_JSON}`",
        f"- Results XLSX: `{RESULT_XLSX}`",
        f"- Summary XLSX: `{SUMMARY_XLSX}`",
        f"- Summary CSV: `{SUMMARY_CSV}`",
        f"- History CSV: `{HISTORY_CSV}`",
        f"- Report: `{REPORT_MD}`",
        f"- Checkpoint: `{CHECKPOINT_PATH}`",
        "",
        "## Static MMD Metrics",
        "",
        "| split | pooled RMSE | macro RMSE | MAE | MAPE (%) |",
        "|---|---:|---:|---:|---:|",
        metric_line("target_adapt", target_adapt_metrics),
        metric_line("Batch10 target_test", target_test_metrics),
        "",
        "## Archived Comparison",
        "",
        "| model | Batch10 pooled RMSE | Batch10 macro RMSE | Batch10 MAE | Batch10 MAPE (%) |",
        "|---|---:|---:|---:|---:|",
        metric_line("static MMD", target_test_metrics),
        metric_line("final-retained-v1", final_retained["target_test"]),
        metric_line("no-mmd-v1", no_mmd["target_test"]),
        "",
        "## Short Interpretation",
        "",
        (
            f"- Static MMD Batch10 pooled RMSE is {target_test_metrics['pooled_rmse']:.9f}, "
            f"versus archived final-retained-v1 {final_retained['target_test']['pooled_rmse']:.9f} "
            f"and archived no-mmd-v1 {no_mmd['target_test']['pooled_rmse']:.9f}."
        ),
        (
            f"- target_adapt pooled RMSE is {target_adapt_metrics['pooled_rmse']:.9f}; "
            "this is the only split used for checkpoint/model selection."
        ),
    ]
    REPORT_MD.write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    required = [
        DATA_DIR / "clean_source_x.npy",
        DATA_DIR / "clean_source_y_nominal.npy",
        DATA_DIR / "clean_target_adapt_x.npy",
        DATA_DIR / "clean_target_adapt_y_nominal.npy",
        DATA_DIR / "clean_target_test_x.npy",
        DATA_DIR / "clean_target_test_y_nominal.npy",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("processed_clean is incomplete: " + "; ".join(missing))

    print(f"RUN={RUN_NAME}", flush=True)
    print(f"python={sys.executable}", flush=True)
    print(f"device={device}", flush=True)
    print(f"data_dir={DATA_DIR}", flush=True)

    s_x, s_y, s_p, s_bidx, s_batch = load_split("source", LABEL_MODE)
    ta_x, ta_y, ta_p, ta_bidx, ta_batch = load_split("target_adapt", LABEL_MODE)
    tt_x, tt_y, tt_p, tt_bidx, tt_batch = load_split("target_test", LABEL_MODE)

    coeffs = np.polyfit((s_p.flatten() / GLOBAL_MAX_CYCLE), s_y.flatten(), 3)
    base_coeffs = torch.tensor(coeffs, dtype=torch.float32, device=device)
    print(f"base_coeffs={coeffs}", flush=True)

    ds_s = HybridDataset(s_x, s_y, s_p, s_bidx)
    ds_ta = HybridDataset(ta_x, ta_y, ta_p, ta_bidx)
    ds_tt = HybridDataset(tt_x, tt_y, tt_p, tt_bidx)
    loader_s = DataLoader(ds_s, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    loader_ta = DataLoader(ds_ta, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    loader_ta_eval = DataLoader(ds_ta, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    loader_tt_eval = DataLoader(ds_tt, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    model = ResidualCRNN(base_coeffs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_mse = nn.MSELoss()
    criterion_mmd = LifetimeAdaptiveMMD().to(device)

    best_target_adapt_rmse = float("inf")
    best_epoch = -1
    best_target_adapt_metrics = None
    history_rows: List[Dict] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        if epoch == LR_DECAY_EPOCH + 1:
            for param_group in optimizer.param_groups:
                param_group["lr"] = LEARNING_RATE * LR_DECAY_FACTOR

        if epoch <= 20:
            mmd_weight = 0.0
            stage = "MSE Warmup"
        elif epoch <= 120:
            mmd_weight = TARGET_MMD_WEIGHT
            stage = "MMD Guide"
        else:
            mmd_weight = 0.0
            stage = "Pure MSE Sprint"

        batch_losses, mse_losses, mmd_losses = [], [], []
        for (sx, sy, sp, _), (tx, _, tp, _) in zip(loader_s, loader_ta):
            sx = sx.float().to(device)
            sy = sy.float().to(device)
            sp = sp.float().to(device)
            tx = tx.float().to(device)
            tp = tp.float().to(device)
            optimizer.zero_grad()
            pred_s, feat_s, base_s = model(sx, sp)
            _, feat_t, base_t = model(tx, tp)
            loss_mse = criterion_mse(pred_s, sy)
            loss_mmd = criterion_mmd(feat_s, feat_t, base_s, base_t) if mmd_weight > 0 else torch.tensor(0.0, device=device)
            loss = loss_mse + mmd_weight * loss_mmd
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))
            mse_losses.append(float(loss_mse.item()))
            mmd_losses.append(float(loss_mmd.item()))

        target_adapt_metrics, _ = evaluate(model, loader_ta_eval, device)
        if target_adapt_metrics["pooled_rmse"] < best_target_adapt_rmse:
            best_target_adapt_rmse = target_adapt_metrics["pooled_rmse"]
            best_epoch = epoch
            best_target_adapt_metrics = target_adapt_metrics
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "base_coeffs": coeffs,
                    "selection_split": "target_adapt",
                    "target_adapt_metrics": target_adapt_metrics,
                    "config": {
                        "model_name": MODEL_NAME,
                        "version": VERSION,
                        "label_mode": LABEL_MODE,
                        "data_protocol": DATA_PROTOCOL,
                        "seed": SEED,
                        "epochs": EPOCHS,
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "target_mmd_weight": TARGET_MMD_WEIGHT,
                        "lambda_smooth": LAMBDA_SMOOTH,
                    },
                },
                CHECKPOINT_PATH,
            )

        row = {
            "epoch": epoch,
            "stage": stage,
            "train_loss": float(np.mean(batch_losses)),
            "train_mse": float(np.mean(mse_losses)),
            "train_mmd": float(np.mean(mmd_losses)),
            "mmd_weight": float(mmd_weight),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "target_adapt_pooled_rmse": target_adapt_metrics["pooled_rmse"],
            "target_adapt_macro_rmse": target_adapt_metrics["macro_rmse"],
            "target_adapt_mae": target_adapt_metrics["mae"],
            "target_adapt_mape_percent": target_adapt_metrics["mape_percent"],
            "best_epoch": best_epoch,
            "best_target_adapt_rmse": best_target_adapt_rmse,
        }
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(HISTORY_CSV, index=False, encoding="utf-8-sig")
        print(
            f"Epoch {epoch:03d}/{EPOCHS} [{stage}] "
            f"loss={row['train_loss']:.6f} "
            f"target_adapt_rmse={row['target_adapt_pooled_rmse']:.5f} "
            f"best={best_target_adapt_rmse:.5f}@{best_epoch}",
            flush=True,
        )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    target_adapt_metrics, target_adapt_per_battery = evaluate(model, loader_ta_eval, device)
    target_test_metrics, target_test_per_battery = evaluate(model, loader_tt_eval, device)

    split_rows = [
        split_summary("source", s_x, s_y, s_p, s_bidx, s_batch),
        split_summary("target_adapt", ta_x, ta_y, ta_p, ta_bidx, ta_batch),
        split_summary("target_test", tt_x, tt_y, tt_p, tt_bidx, tt_batch),
    ]
    summary = {
        "model": RUN_NAME,
        "status": "success",
        "protocol": "Scheme A: source supervised training + target_adapt MMD/model selection + target_test final once",
        "mmd_variant": "static_mmd_no_dynamic_weight",
        "batch10_usage": "final_report_only_not_training_tuning_checkpoint_model_seed_or_figure_selection",
        "python_executable": sys.executable,
        "device": str(device),
        "data_dir": str(DATA_DIR),
        "label_mode": LABEL_MODE,
        "data_protocol": DATA_PROTOCOL,
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "target_mmd_weight": TARGET_MMD_WEIGHT,
        "lambda_smooth": LAMBDA_SMOOTH,
        "best_epoch_selected_by_target_adapt": best_epoch,
        "best_target_adapt_rmse": best_target_adapt_metrics["pooled_rmse"] if best_target_adapt_metrics else None,
        "final_target_test_rmse": target_test_metrics["pooled_rmse"],
        "base_coeffs": json.dumps([float(x) for x in coeffs]),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "result_xlsx": str(RESULT_XLSX),
        "summary_xlsx": str(SUMMARY_XLSX),
        "summary_csv": str(SUMMARY_CSV),
        "report_md": str(REPORT_MD),
    }
    save_outputs(
        summary,
        split_rows,
        history_rows,
        target_adapt_metrics,
        target_test_metrics,
        target_adapt_per_battery,
        target_test_per_battery,
    )
    write_report(summary, target_adapt_metrics, target_test_metrics)
    print("FINAL_METRICS", json.dumps({
        "model": RUN_NAME,
        "best_epoch": best_epoch,
        "best_target_adapt_rmse": best_target_adapt_metrics["pooled_rmse"],
        "final_target_test_rmse": target_test_metrics["pooled_rmse"],
        "target_test_macro_rmse": target_test_metrics["macro_rmse"],
        "target_test_mae": target_test_metrics["mae"],
        "target_test_mape_percent": target_test_metrics["mape_percent"],
        "summary_json": str(BEST_JSON),
        "summary_csv": str(SUMMARY_CSV),
        "report_md": str(REPORT_MD),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
