import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset


EXPERIMENT_NAME = "soh-physical-degree-ablation-v1"
SCRIPT_PATH = Path(__file__).resolve()
RUN_DIR = SCRIPT_PATH.parent
ROOT = RUN_DIR.parents[1]

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


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def required_data_files(data_dir: Path) -> List[Path]:
    return [
        data_dir / "clean_source_x.npy",
        data_dir / "clean_source_y_nominal.npy",
        data_dir / "clean_target_adapt_x.npy",
        data_dir / "clean_target_adapt_y_nominal.npy",
        data_dir / "clean_target_test_x.npy",
        data_dir / "clean_target_test_y_nominal.npy",
    ]


def resolve_data_dir(root: Path) -> Path:
    candidates = []
    for child in root.iterdir():
        candidate = child / "processed_clean"
        if candidate.is_dir():
            candidates.append(candidate)
    valid = [path for path in candidates if all(file.exists() for file in required_data_files(path))]
    if len(valid) == 1:
        return valid[0]
    if len(valid) > 1:
        valid_sorted = sorted(valid, key=lambda path: str(path))
        return valid_sorted[0]
    searched = "; ".join(str(path) for path in candidates) if candidates else str(root)
    raise FileNotFoundError(f"Could not locate processed_clean with required files under: {searched}")


DATA_DIR = resolve_data_dir(ROOT)


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
        current_soh = torch.clamp(torch.mean(target_base_soh), max=1.0)
        dynamic_weight = torch.exp(self.lambda_scale * (1.0 - current_soh))
        return dynamic_weight * self.compute_mmd(source_features, target_features)


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
        self.register_buffer("base_coeffs", base_coeffs.detach().clone().float())
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

    def polynomial_base(self, p: torch.Tensor) -> torch.Tensor:
        base = torch.zeros_like(p)
        for coeff in self.base_coeffs:
            base = base * p + coeff
        return base

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_soh = self.polynomial_base(p)
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


def maybe_subset(dataset: Dataset, max_samples: Optional[int]) -> Dataset:
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max_samples)))


def create_loaders(
    s_x: np.ndarray,
    s_y: np.ndarray,
    s_p: np.ndarray,
    s_bidx: np.ndarray,
    ta_x: np.ndarray,
    ta_y: np.ndarray,
    ta_p: np.ndarray,
    ta_bidx: np.ndarray,
    tt_x: np.ndarray,
    tt_y: np.ndarray,
    tt_p: np.ndarray,
    tt_bidx: np.ndarray,
    max_smoke_samples: Optional[int] = None,
) -> Dict[str, DataLoader]:
    ds_s = HybridDataset(s_x, s_y, s_p, s_bidx)
    ds_ta = HybridDataset(ta_x, ta_y, ta_p, ta_bidx)
    ds_tt = HybridDataset(tt_x, tt_y, tt_p, tt_bidx)

    ds_s_train = maybe_subset(ds_s, max_smoke_samples)
    ds_ta_train = maybe_subset(ds_ta, max_smoke_samples)
    ds_ta_eval = maybe_subset(ds_ta, max_smoke_samples)
    ds_tt_eval = maybe_subset(ds_tt, max_smoke_samples)

    return {
        "source_train": DataLoader(ds_s_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True),
        "target_adapt_train": DataLoader(ds_ta_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True),
        "target_adapt_eval": DataLoader(ds_ta_eval, batch_size=EVAL_BATCH_SIZE, shuffle=False),
        "target_test_eval": DataLoader(ds_tt_eval, batch_size=EVAL_BATCH_SIZE, shuffle=False),
    }


def coeff_columns(degree: int, coeffs: Iterable[float]) -> Dict[str, float]:
    values = list(coeffs)
    row = {}
    for idx, coeff in enumerate(values):
        power = degree - idx
        row[f"coeff_p{power}"] = float(coeff)
    return row


def save_degree_outputs(
    degree_dir: Path,
    degree: int,
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
    result_xlsx = degree_dir / f"degree_{degree}_results.xlsx"
    with pd.ExcelWriter(result_xlsx, engine="openpyxl") as writer:
        pd.DataFrame([summary]).to_excel(writer, index=False, sheet_name="config")
        pd.DataFrame(split_rows).to_excel(writer, index=False, sheet_name="splits")
        pd.DataFrame(history_rows).to_excel(writer, index=False, sheet_name="history")
        pd.DataFrame(metrics_rows).to_excel(writer, index=False, sheet_name="metrics")
        target_adapt_per_battery.to_excel(writer, index=False, sheet_name="target_adapt_cells")
        target_test_per_battery.to_excel(writer, index=False, sheet_name="target_test_cells")
    payload = {
        "summary": summary,
        "splits": split_rows,
        "history": history_rows,
        "metrics": metrics_rows,
        "target_adapt_per_battery": target_adapt_per_battery.to_dict(orient="records"),
        "target_test_per_battery": target_test_per_battery.to_dict(orient="records"),
    }
    (degree_dir / f"degree_{degree}_best_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def train_one_degree(
    degree: int,
    s_x: np.ndarray,
    s_y: np.ndarray,
    s_p: np.ndarray,
    s_bidx: np.ndarray,
    s_batch: np.ndarray,
    ta_x: np.ndarray,
    ta_y: np.ndarray,
    ta_p: np.ndarray,
    ta_bidx: np.ndarray,
    ta_batch: np.ndarray,
    tt_x: np.ndarray,
    tt_y: np.ndarray,
    tt_p: np.ndarray,
    tt_bidx: np.ndarray,
    tt_batch: np.ndarray,
    device: torch.device,
    output_dir: Path,
    epochs: int,
    max_train_batches: Optional[int],
    max_smoke_samples: Optional[int],
    log_every: int,
    smoke: bool,
) -> Dict:
    set_seed(SEED)
    degree_dir = output_dir / f"degree_{degree}"
    degree_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = degree_dir / f"degree_{degree}_best_by_target_adapt.pt"
    history_csv = degree_dir / f"degree_{degree}_history.csv"

    coeffs = np.polyfit((s_p.flatten() / GLOBAL_MAX_CYCLE), s_y.flatten(), degree)
    base_coeffs = torch.tensor(coeffs, dtype=torch.float32, device=device)
    loaders = create_loaders(s_x, s_y, s_p, s_bidx, ta_x, ta_y, ta_p, ta_bidx, tt_x, tt_y, tt_p, tt_bidx, max_smoke_samples)

    model = ResidualCRNN(base_coeffs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_mse = nn.MSELoss()
    criterion_mmd = LifetimeAdaptiveMMD().to(device)

    best_target_adapt_rmse = float("inf")
    best_epoch = -1
    best_target_adapt_metrics = None
    history_rows: List[Dict] = []
    started_at = datetime.now().isoformat(timespec="seconds")

    print(
        f"DEGREE={degree} coeffs={json.dumps([float(x) for x in coeffs])} "
        f"epochs={epochs} output={degree_dir}",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
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
        for batch_index, ((sx, sy, sp, _), (tx, _, tp, _)) in enumerate(
            zip(loaders["source_train"], loaders["target_adapt_train"]), start=1
        ):
            if max_train_batches is not None and max_train_batches > 0 and batch_index > max_train_batches:
                break
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

        if not batch_losses:
            raise RuntimeError("No training batches were produced. Check batch size or smoke-sample settings.")

        target_adapt_metrics, _ = evaluate(model, loaders["target_adapt_eval"], device)
        if target_adapt_metrics["pooled_rmse"] < best_target_adapt_rmse:
            best_target_adapt_rmse = target_adapt_metrics["pooled_rmse"]
            best_epoch = epoch
            best_target_adapt_metrics = target_adapt_metrics
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "degree": degree,
                    "base_coeffs": coeffs,
                    "selection_split": "target_adapt",
                    "target_adapt_metrics": target_adapt_metrics,
                    "config": {
                        "experiment": EXPERIMENT_NAME,
                        "label_mode": LABEL_MODE,
                        "data_protocol": DATA_PROTOCOL,
                        "seed": SEED,
                        "epochs": epochs,
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "target_mmd_weight": TARGET_MMD_WEIGHT,
                        "lambda_smooth": LAMBDA_SMOOTH,
                        "max_train_batches": max_train_batches,
                        "max_smoke_samples": max_smoke_samples,
                        "smoke": smoke,
                    },
                },
                checkpoint_path,
            )

        row = {
            "degree": degree,
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
        pd.DataFrame(history_rows).to_csv(history_csv, index=False, encoding="utf-8-sig")
        if epoch == 1 or epoch == epochs or epoch % max(log_every, 1) == 0:
            print(
                f"Degree {degree} Epoch {epoch:03d}/{epochs} [{stage}] "
                f"loss={row['train_loss']:.6f} "
                f"target_adapt_rmse={row['target_adapt_pooled_rmse']:.6f} "
                f"best={best_target_adapt_rmse:.6f}@{best_epoch}",
                flush=True,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    target_adapt_metrics, target_adapt_per_battery = evaluate(model, loaders["target_adapt_eval"], device)
    target_test_metrics, target_test_per_battery = evaluate(model, loaders["target_test_eval"], device)

    split_rows = [
        split_summary("source", s_x, s_y, s_p, s_bidx, s_batch),
        split_summary("target_adapt", ta_x, ta_y, ta_p, ta_bidx, ta_batch),
        split_summary("target_test", tt_x, tt_y, tt_p, tt_bidx, tt_batch),
    ]
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "success",
        "degree": degree,
        "protocol": "Source supervised residual training + Batch 9 target_adapt MMD/model selection + Batch 10 blind-test reporting once after checkpoint is fixed",
        "python_executable": sys.executable,
        "device": str(device),
        "data_dir": str(DATA_DIR),
        "label_mode": LABEL_MODE,
        "data_protocol": DATA_PROTOCOL,
        "seed": SEED,
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "target_mmd_weight": TARGET_MMD_WEIGHT,
        "lambda_smooth": LAMBDA_SMOOTH,
        "max_train_batches": max_train_batches,
        "max_smoke_samples": max_smoke_samples,
        "smoke": smoke,
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "best_epoch_selected_by_batch9_target_adapt": best_epoch,
        "batch9_target_adapt_rmse": target_adapt_metrics["pooled_rmse"],
        "batch9_target_adapt_macro_rmse": target_adapt_metrics["macro_rmse"],
        "batch9_target_adapt_mae": target_adapt_metrics["mae"],
        "batch9_target_adapt_mape_percent": target_adapt_metrics["mape_percent"],
        "batch10_target_test_pooled_rmse": target_test_metrics["pooled_rmse"],
        "batch10_target_test_macro_rmse": target_test_metrics["macro_rmse"],
        "batch10_target_test_mae": target_test_metrics["mae"],
        "batch10_target_test_mape_percent": target_test_metrics["mape_percent"],
        "base_coeffs": json.dumps([float(x) for x in coeffs]),
        "checkpoint_path": str(checkpoint_path),
        "history_csv": str(history_csv),
        "result_xlsx": str(degree_dir / f"degree_{degree}_results.xlsx"),
    }
    summary.update(coeff_columns(degree, coeffs))
    save_degree_outputs(
        degree_dir,
        degree,
        summary,
        split_rows,
        history_rows,
        target_adapt_metrics,
        target_test_metrics,
        target_adapt_per_battery,
        target_test_per_battery,
    )
    print(
        "FINAL_DEGREE_METRICS "
        + json.dumps(
            {
                "degree": degree,
                "best_epoch": best_epoch,
                "batch9_target_adapt_rmse": target_adapt_metrics["pooled_rmse"],
                "batch10_target_test_pooled_rmse": target_test_metrics["pooled_rmse"],
                "batch10_target_test_macro_rmse": target_test_metrics["macro_rmse"],
                "batch10_target_test_mae": target_test_metrics["mae"],
                "batch10_target_test_mape_percent": target_test_metrics["mape_percent"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    return {
        "summary": summary,
        "target_adapt_per_battery": target_adapt_per_battery.assign(degree=degree),
        "target_test_per_battery": target_test_per_battery.assign(degree=degree),
        "history": pd.DataFrame(history_rows),
    }


def collect_existing_degree(output_dir: Path, degree: int) -> Optional[Dict]:
    metrics_path = output_dir / f"degree_{degree}" / f"degree_{degree}_best_metrics.json"
    if not metrics_path.exists():
        return None
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "summary": payload["summary"],
        "target_adapt_per_battery": pd.DataFrame(payload["target_adapt_per_battery"]).assign(degree=degree),
        "target_test_per_battery": pd.DataFrame(payload["target_test_per_battery"]).assign(degree=degree),
        "history": pd.DataFrame(payload["history"]),
    }


def protocol_notes_rows(degrees: List[int], smoke: bool, output_dir: Path) -> List[Dict[str, str]]:
    return [
        {"item": "task_scope", "note": "SOH-only physical/degradation-prior degree ablation for the independent SOH paper."},
        {"item": "data_protocol", "note": "processed_clean_fixed_split is preserved: Batch 1-8 source, Batch 9 target_adapt, Batch 10 blind-test."},
        {"item": "source_training", "note": "Polynomial/degradation-prior coefficients are fitted on source training data only."},
        {"item": "model_selection", "note": "For each degree, checkpoint selection uses Batch 9 target-domain development RMSE."},
        {"item": "blind_test", "note": "Batch 10 is evaluated only after each degree checkpoint is fixed; it is reported separately and not used for degree/model selection."},
        {"item": "final_model_identity", "note": "This is a formal ablation/diagnostic and does not replace final-retained-v1 unless the parent workflow later decides so."},
        {"item": "mmd_scope", "note": "MMD settings are inherited as a lightweight domain-alignment component; this ablation does not claim MMD as the dominant source of improvement."},
        {"item": "prior_wording", "note": "The cubic base, where discussed, should be described as a compact low-order nonlinear degradation prior, not as a strong electrochemical mechanism."},
        {"item": "degrees_requested", "note": ",".join(map(str, degrees))},
        {"item": "smoke", "note": str(smoke)},
        {"item": "output_dir", "note": str(output_dir)},
    ]


def build_summary_tables(results: List[Dict], degrees: List[int], smoke: bool, output_dir: Path) -> Dict[str, pd.DataFrame]:
    overall_rows = [result["summary"] for result in results]
    per_degree_overall = pd.DataFrame(overall_rows)
    desired_cols = [
        "degree",
        "base_coeffs",
        "best_epoch_selected_by_batch9_target_adapt",
        "batch9_target_adapt_rmse",
        "batch9_target_adapt_mae",
        "batch9_target_adapt_mape_percent",
        "batch10_target_test_pooled_rmse",
        "batch10_target_test_macro_rmse",
        "batch10_target_test_mae",
        "batch10_target_test_mape_percent",
    ]
    coeff_cols = sorted([col for col in per_degree_overall.columns if col.startswith("coeff_p")], reverse=True)
    leading_cols = [col for col in desired_cols if col in per_degree_overall.columns] + coeff_cols
    remaining_cols = [col for col in per_degree_overall.columns if col not in leading_cols]
    per_degree_overall = per_degree_overall[leading_cols + remaining_cols].sort_values("degree")

    cells = []
    for result in results:
        df = result["target_test_per_battery"].copy()
        df["split"] = "target_test"
        cells.append(df)
    per_degree_cells = pd.concat(cells, ignore_index=True)
    per_degree_cells = per_degree_cells[["degree", "split", "battery_index", "rmse", "mae", "n_points"]].sort_values(
        ["degree", "battery_index"]
    )

    batch9_ranking = per_degree_overall.sort_values(
        ["batch9_target_adapt_rmse", "degree"], ascending=[True, True]
    ).reset_index(drop=True)
    batch9_ranking.insert(0, "batch9_rank", range(1, len(batch9_ranking) + 1))
    batch10_reporting = per_degree_overall.sort_values(
        ["batch10_target_test_pooled_rmse", "degree"], ascending=[True, True]
    ).reset_index(drop=True)
    batch10_reporting.insert(0, "batch10_report_order", range(1, len(batch10_reporting) + 1))

    protocol_notes = pd.DataFrame(protocol_notes_rows(degrees, smoke, output_dir))

    return {
        "per_degree_overall": per_degree_overall,
        "per_degree_cells": per_degree_cells,
        "batch9_ranking": batch9_ranking,
        "batch10_reporting": batch10_reporting,
        "protocol_notes": protocol_notes,
    }


def write_summary_workbook(tables: Dict[str, pd.DataFrame], output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in tables.items():
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    return path


def plot_figures(tables: Dict[str, pd.DataFrame], output_dir: Path) -> List[Path]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    overall = tables["per_degree_overall"].sort_values("degree")
    paths = []

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    ax.plot(overall["degree"], overall["batch9_target_adapt_rmse"], marker="o", label="Batch 9 RMSE (selection)")
    ax.plot(overall["degree"], overall["batch10_target_test_pooled_rmse"], marker="s", label="Batch 10 pooled RMSE (reporting)")
    ax.set_xlabel("Polynomial prior degree")
    ax.set_ylabel("RMSE")
    ax.set_title("SOH prior degree ablation")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = figures_dir / "degree_vs_batch9_batch10_rmse.png"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    cells = tables["per_degree_cells"].copy()
    pivot = cells.pivot(index="battery_index", columns="degree", values="rmse").sort_index()
    fig_width = max(6.4, 1.1 * len(pivot.columns) + 3.2)
    fig_height = max(4.2, 0.42 * len(pivot.index) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=160)
    image = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(col) for col in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(idx) for idx in pivot.index])
    ax.set_xlabel("Polynomial prior degree")
    ax.set_ylabel("Batch 10 battery index")
    ax.set_title("Batch 10 per-cell RMSE by prior degree")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{value:.4f}", ha="center", va="center", color="white" if value > np.nanmean(pivot.values) else "black", fontsize=7)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("RMSE")
    fig.tight_layout()
    path = figures_dir / "batch10_per_battery_rmse_heatmap_by_degree.png"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)
    return paths


def near_best(value: float, best: float, tolerance: float = 0.00025) -> bool:
    return value <= best + tolerance


def write_markdown_report(tables: Dict[str, pd.DataFrame], output_dir: Path, workbook_path: Path, figure_paths: List[Path]) -> Path:
    overall = tables["per_degree_overall"].sort_values("degree").copy()
    batch9 = tables["batch9_ranking"].copy()
    batch10 = tables["batch10_reporting"].copy()
    best_batch9 = batch9.iloc[0]
    best_batch10 = batch10.iloc[0]
    cubic_rows = overall[overall["degree"] == 3]
    cubic_text = ""
    if not cubic_rows.empty:
        cubic = cubic_rows.iloc[0]
        if int(best_batch9["degree"]) == 3 or near_best(
            float(cubic["batch9_target_adapt_rmse"]), float(best_batch9["batch9_target_adapt_rmse"])
        ):
            cubic_text = (
                "Degree 3 is best or near-best on the Batch 9 development criterion, so the result supports retaining "
                "a compact low-order nonlinear degradation prior as a reasonable complexity choice."
            )
        else:
            cubic_text = (
                "Degree 3 is not the Batch 9 winner in this run. The manuscript should therefore avoid claiming "
                "degree-3 optimality; at most, the cubic prior can be described as a compact complexity/oscillation "
                "compromise if supported by the broader retained-model evidence."
            )
    else:
        cubic_text = "Degree 3 was not included in this run, so no cubic-prior conclusion is made."

    metric_lines = []
    for _, row in overall.iterrows():
        metric_lines.append(
            "| {degree} | {epoch} | {b9:.6f} | {b10:.6f} | {macro:.6f} | {mae:.6f} | {mape:.4f} |".format(
                degree=int(row["degree"]),
                epoch=int(row["best_epoch_selected_by_batch9_target_adapt"]),
                b9=float(row["batch9_target_adapt_rmse"]),
                b10=float(row["batch10_target_test_pooled_rmse"]),
                macro=float(row["batch10_target_test_macro_rmse"]),
                mae=float(row["batch10_target_test_mae"]),
                mape=float(row["batch10_target_test_mape_percent"]),
            )
        )

    report = f"""# SOH Physical Degree Ablation v1

## Scope And Protocol

This run is a formal SOH-only ablation/diagnostic for the physical/degradation-prior polynomial degree in the full residual temporal model. It preserves `processed_clean_fixed_split`: Batch 1-8 are the source domain, Batch 9 is target-domain development/adaptation, and Batch 10 is blind-test reporting only.

For each degree, the prior coefficients were fitted on source training data only. The residual CRNN/LSTM/Attention model, seed, optimizer schedule, nominal SOH labels, lightweight MMD settings, and metrics follow the retained Scheme A script. Checkpoints were selected by Batch 9 target-adapt RMSE. Batch 10 metrics were computed after each degree checkpoint was fixed and were not used for tuning, degree selection, or final-model identity changes.

## Overall Results

| Degree | Best epoch by Batch 9 | Batch 9 RMSE | Batch 10 pooled RMSE | Batch 10 macro RMSE | Batch 10 MAE | Batch 10 MAPE (%) |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(metric_lines)}

Batch 9 ranking selects degree {int(best_batch9["degree"])} with RMSE {float(best_batch9["batch9_target_adapt_rmse"]):.6f}. Batch 10 reporting is separate: the lowest blind-test pooled RMSE in this diagnostic is degree {int(best_batch10["degree"])} with RMSE {float(best_batch10["batch10_target_test_pooled_rmse"]):.6f}. This Batch 10 ordering is reported for transparency only and must not be used to choose the degree.

## Interpretation

{cubic_text}

The ablation should be read as evidence about the polynomial prior's complexity, not as a claim that a specific polynomial degree encodes a strong electrochemical mechanism. In particular, the cubic base should be phrased as a compact low-order nonlinear degradation prior. MMD is inherited from the retained model as a lightweight cross-domain alignment component; these results do not by themselves isolate or overclaim the effect of MMD.

This diagnostic does not alter the established `final-retained-v1` identity. Any manuscript decision to replace or revise that model should be made by the parent workflow after comparing this diagnostic with the existing retained-model evidence.

## Artifacts

- Summary workbook: `{workbook_path.name}`
- Main figure: `{figure_paths[0].relative_to(output_dir).as_posix() if figure_paths else "not generated"}`
- Per-cell heatmap: `{figure_paths[1].relative_to(output_dir).as_posix() if len(figure_paths) > 1 else "not generated"}`
"""
    path = output_dir / "soh_physical_degree_ablation_v1_report.md"
    path.write_text(report, encoding="utf-8")
    return path


def write_progress_file(output_dir: Path, completed: List[int], requested: List[int], error: Optional[str] = None) -> None:
    payload = {
        "experiment": EXPERIMENT_NAME,
        "completed_degrees": completed,
        "requested_degrees": requested,
        "remaining_degrees": [degree for degree in requested if degree not in completed],
        "error": error,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (output_dir / "progress.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SOH physical/degradation-prior degree ablation.")
    parser.add_argument("--degrees", nargs="+", type=int, default=[1, 2, 3, 4, 5], help="Polynomial prior degrees to run.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs per degree.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Optional cap on training batches per epoch.")
    parser.add_argument("--max-smoke-samples", type=int, default=None, help="Optional cap on samples per split for smoke checks.")
    parser.add_argument("--log-every", type=int, default=1, help="Epoch print frequency.")
    parser.add_argument("--smoke", action="store_true", help="Write outputs to smoke_test and default to a tiny run if caps are omitted.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse completed per-degree JSON files when present.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    degrees = sorted(dict.fromkeys(args.degrees))
    if any(degree < 0 for degree in degrees):
        raise ValueError("Polynomial degree must be non-negative.")
    output_dir = RUN_DIR / "smoke_test" if args.smoke else RUN_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    epochs = args.epochs
    max_train_batches = args.max_train_batches
    max_smoke_samples = args.max_smoke_samples
    if args.smoke:
        epochs = min(epochs, 2)
        max_train_batches = max_train_batches or 3
        max_smoke_samples = max_smoke_samples or 512

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    missing = [str(path) for path in required_data_files(DATA_DIR) if not path.exists()]
    if missing:
        raise FileNotFoundError("processed_clean is incomplete: " + "; ".join(missing))

    print(f"EXPERIMENT={EXPERIMENT_NAME}", flush=True)
    print(f"python={sys.executable}", flush=True)
    print(f"device={device}", flush=True)
    print(f"data_dir={DATA_DIR}", flush=True)
    print(f"degrees={degrees}", flush=True)

    s_x, s_y, s_p, s_bidx, s_batch = load_split("source", LABEL_MODE)
    ta_x, ta_y, ta_p, ta_bidx, ta_batch = load_split("target_adapt", LABEL_MODE)
    tt_x, tt_y, tt_p, tt_bidx, tt_batch = load_split("target_test", LABEL_MODE)

    results = []
    completed: List[int] = []
    write_progress_file(output_dir, completed, degrees)
    try:
        for degree in degrees:
            if args.skip_existing:
                existing = collect_existing_degree(output_dir, degree)
                if existing is not None:
                    print(f"SKIP_EXISTING degree={degree}", flush=True)
                    results.append(existing)
                    completed.append(degree)
                    write_progress_file(output_dir, completed, degrees)
                    continue
            result = train_one_degree(
                degree,
                s_x,
                s_y,
                s_p,
                s_bidx,
                s_batch,
                ta_x,
                ta_y,
                ta_p,
                ta_bidx,
                ta_batch,
                tt_x,
                tt_y,
                tt_p,
                tt_bidx,
                tt_batch,
                device,
                output_dir,
                epochs,
                max_train_batches,
                max_smoke_samples,
                args.log_every,
                args.smoke,
            )
            results.append(result)
            completed.append(degree)
            write_progress_file(output_dir, completed, degrees)
    except Exception as exc:
        write_progress_file(output_dir, completed, degrees, error=repr(exc))
        raise

    tables = build_summary_tables(results, degrees, args.smoke, output_dir)
    workbook_name = "soh_physical_degree_ablation_v1_summary.xlsx" if not args.smoke else "smoke_soh_physical_degree_ablation_v1_summary.xlsx"
    workbook_path = write_summary_workbook(tables, output_dir, workbook_name)
    figure_paths = plot_figures(tables, output_dir)
    report_path = write_markdown_report(tables, output_dir, workbook_path, figure_paths)
    print(
        "ABLATION_COMPLETE "
        + json.dumps(
            {
                "completed_degrees": completed,
                "workbook": str(workbook_path),
                "report": str(report_path),
                "figures": [str(path) for path in figure_paths],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
