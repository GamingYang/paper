import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


RUN_NAME = "rul-quick-v1"
SEED = 42
GLOBAL_MAX_CYCLE = 3000.0
RUL_SCALE = 3000.0
SOH_AUX_WEIGHT = 0.05
MONOTONIC_WEIGHT = 0.10
TARGET_MMD_WEIGHT = 5e-5
LR_DECAY_FACTOR = 0.2


def find_data_dir(root: Path) -> Path:
    candidates = []
    for child in root.iterdir():
        if child.is_dir() and child.name.lower().startswith("python"):
            candidates.append(child / "processed_clean")
    candidates.extend([root / "processed_clean", root / "PYthon_training_code" / "processed_clean"])
    for path in candidates:
        if (path / "clean_source_x.npy").exists():
            return path
    raise FileNotFoundError("Cannot find processed_clean data directory from project root.")


SCRIPT_PATH = Path(__file__).resolve()
RUN_DIR = SCRIPT_PATH.parent
ROOT = RUN_DIR.parents[1]
DATA_DIR = find_data_dir(ROOT)
CHECKPOINT_PATH = RUN_DIR / f"{RUN_NAME}_best_by_target_adapt.pt"
HISTORY_CSV = RUN_DIR / f"{RUN_NAME}_history.csv"
METRICS_JSON = RUN_DIR / f"{RUN_NAME}_metrics.json"
PREDICTIONS_CSV = RUN_DIR / f"{RUN_NAME}_target_test_predictions.csv"
PER_BATTERY_CSV = RUN_DIR / f"{RUN_NAME}_per_battery_metrics.csv"
REPORT_MD = RUN_DIR / f"{RUN_NAME}_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick RUL training on processed_clean battery data.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-steps-per-epoch", type=int, default=60)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--max-runtime-min", type=float, default=25.0)
    return parser.parse_args()


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


def load_split(split_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    x = load_npy(DATA_DIR / f"clean_{split_name}_x.npy").astype(np.float32)
    soh = load_npy(DATA_DIR / f"clean_{split_name}_y_nominal.npy").astype(np.float32)
    rul = load_npy(DATA_DIR / f"clean_{split_name}_rul.npy").astype(np.float32)
    p = load_npy(DATA_DIR / f"clean_{split_name}_p.npy").astype(np.float32)
    bidx = load_npy(DATA_DIR / f"clean_{split_name}_battery_index.npy").astype(np.int64)
    batch = load_npy(DATA_DIR / f"clean_{split_name}_batch.npy").astype(np.int64)
    meta = pd.read_csv(DATA_DIR / f"clean_{split_name}_meta.csv")
    return x, soh, rul, p, bidx, batch, meta


class RULDataset(Dataset):
    def __init__(self, x: np.ndarray, soh: np.ndarray, rul: np.ndarray, p: np.ndarray, bidx: np.ndarray):
        self.x = x.astype(np.float32)
        self.soh = soh.astype(np.float32).reshape(-1, 1)
        self.rul_norm = (rul.astype(np.float32).reshape(-1, 1) / RUL_SCALE)
        self.p_norm = (p.astype(np.float32).reshape(-1, 1) / GLOBAL_MAX_CYCLE)
        self.bidx = bidx.astype(np.int64)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.soh[idx], self.rul_norm[idx], self.p_norm[idx], self.bidx[idx]


class LifetimeProgressMMD(nn.Module):
    def __init__(self, kernel_mul: float = 2.0, kernel_num: int = 5, lambda_scale: float = 2.0):
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
        return sum(torch.exp(-l2_distance / (bandwidth * (self.kernel_mul**i))) for i in range(self.kernel_num))

    def forward(self, source_features: torch.Tensor, target_features: torch.Tensor, target_progress: torch.Tensor) -> torch.Tensor:
        if source_features.size(0) == 0 or target_features.size(0) == 0:
            return torch.tensor(0.0, device=source_features.device)
        batch_size_s = int(source_features.size(0))
        kernels = self.gaussian_kernel(source_features, target_features)
        xx = kernels[:batch_size_s, :batch_size_s]
        yy = kernels[batch_size_s:, batch_size_s:]
        xy = kernels[:batch_size_s, batch_size_s:]
        mmd = torch.mean(xx) + torch.mean(yy) - 2 * torch.mean(xy)
        dynamic_weight = torch.exp(self.lambda_scale * torch.clamp(torch.mean(target_progress), 0.0, 1.0))
        return dynamic_weight * mmd


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, lstm_outputs: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.attention(lstm_outputs), dim=1)
        return torch.sum(weights * lstm_outputs, dim=1)


class ResidualRULNet(nn.Module):
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
        self.shared = nn.Sequential(nn.Linear(72, 48), nn.ReLU(), nn.Dropout(0.15))
        self.rul_head = nn.Linear(48, 1)
        self.soh_head = nn.Linear(48, 1)

    def forward(self, x: torch.Tensor, p_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        a, b, c, d = self.base_coeffs
        base_rul = a * (p_norm**3) + b * (p_norm**2) + c * p_norm + d
        seq = self.cnn(x.unsqueeze(1)).permute(0, 2, 1)
        lstm_out, _ = self.lstm(seq)
        context = self.attention(lstm_out)
        features = torch.cat((context, self.phy(p_norm)), dim=1)
        hidden = self.shared(features)
        raw_rul = base_rul + self.rul_head(hidden)
        pred_rul = torch.clamp(F.softplus(raw_rul, beta=5.0), min=0.0, max=1.2)
        pred_soh = self.soh_head(hidden)
        return pred_rul, pred_soh, hidden, base_rul


def monotonic_decrease_loss(preds: torch.Tensor, progress: torch.Tensor, battery_idx: torch.Tensor) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    pred_flat = preds.reshape(-1)
    progress_flat = progress.reshape(-1)
    batt_flat = battery_idx.reshape(-1)
    for b in torch.unique(batt_flat):
        mask = batt_flat == b
        if int(mask.sum()) < 2:
            continue
        order = torch.argsort(progress_flat[mask])
        local = pred_flat[mask][order]
        increase = local[1:] - local[:-1]
        losses.append(torch.mean(F.relu(increase) ** 2))
    if not losses:
        return torch.tensor(0.0, device=preds.device)
    return torch.stack(losses).mean()


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean((y_true.reshape(-1) - y_pred.reshape(-1)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true.reshape(-1) - y_pred.reshape(-1))))


def macro_rmse_by_battery(y_true: np.ndarray, y_pred: np.ndarray, battery_idx: np.ndarray) -> float:
    vals = []
    for b in np.unique(battery_idx):
        mask = battery_idx == b
        if np.any(mask):
            vals.append(rmse(y_true[mask], y_pred[mask]))
    return float(np.mean(vals)) if vals else float("nan")


def binned_metrics(y_true_cycles: np.ndarray, y_pred_cycles: np.ndarray) -> Dict[str, float]:
    bins = {
        "late_le_300": y_true_cycles.reshape(-1) <= 300,
        "mid_300_1000": (y_true_cycles.reshape(-1) > 300) & (y_true_cycles.reshape(-1) <= 1000),
        "early_gt_1000": y_true_cycles.reshape(-1) > 1000,
    }
    out: Dict[str, float] = {}
    for name, mask in bins.items():
        if np.any(mask):
            out[f"{name}_rmse_cycles"] = rmse(y_true_cycles[mask], y_pred_cycles[mask])
            out[f"{name}_mae_cycles"] = mae(y_true_cycles[mask], y_pred_cycles[mask])
            out[f"{name}_n"] = int(np.sum(mask))
    return out


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[Dict[str, float], pd.DataFrame]:
    model.eval()
    pred_norm_all, true_norm_all, bidx_all = [], [], []
    with torch.no_grad():
        for x, _, rul_norm, p_norm, bidx in loader:
            x = x.float().to(device)
            p_norm = p_norm.float().to(device)
            pred_norm, _, _, _ = model(x, p_norm)
            pred_norm_all.append(pred_norm.cpu().numpy().reshape(-1))
            true_norm_all.append(rul_norm.numpy().reshape(-1))
            bidx_all.append(bidx.numpy().reshape(-1))
    y_pred_norm = np.concatenate(pred_norm_all)
    y_true_norm = np.concatenate(true_norm_all)
    bidx = np.concatenate(bidx_all)
    y_pred_cycles = y_pred_norm * RUL_SCALE
    y_true_cycles = y_true_norm * RUL_SCALE
    metrics = {
        "rmse_cycles": rmse(y_true_cycles, y_pred_cycles),
        "mae_cycles": mae(y_true_cycles, y_pred_cycles),
        "macro_rmse_cycles": macro_rmse_by_battery(y_true_cycles, y_pred_cycles, bidx),
        "rmse_norm": rmse(y_true_norm, y_pred_norm),
    }
    metrics.update(binned_metrics(y_true_cycles, y_pred_cycles))
    rows = []
    for b in np.unique(bidx):
        mask = bidx == b
        rows.append(
            {
                "battery_index": int(b),
                "rmse_cycles": rmse(y_true_cycles[mask], y_pred_cycles[mask]),
                "mae_cycles": mae(y_true_cycles[mask], y_pred_cycles[mask]),
                "n_points": int(np.sum(mask)),
            }
        )
    return metrics, pd.DataFrame(rows)


def predict_target_test(model: nn.Module, loader: DataLoader, meta: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _, _, p_norm, _ in loader:
            pred_norm, _, _, _ = model(x.float().to(device), p_norm.float().to(device))
            preds.append(pred_norm.cpu().numpy().reshape(-1) * RUL_SCALE)
    out = meta.copy()
    out["pred_rul_cycle"] = np.concatenate(preds)
    out["abs_error_cycle"] = np.abs(out["rul_cycle"].to_numpy(dtype=float) - out["pred_rul_cycle"].to_numpy(dtype=float))
    return out


def split_summary(name: str, x: np.ndarray, rul: np.ndarray, p: np.ndarray, bidx: np.ndarray, batch: np.ndarray) -> Dict:
    return {
        "split": name,
        "samples": int(len(x)),
        "cells": int(len(np.unique(bidx))),
        "batches": ",".join(map(str, sorted(np.unique(batch).tolist()))),
        "rul_min": float(np.min(rul)),
        "rul_max": float(np.max(rul)),
        "cycle_min": float(np.min(p)),
        "cycle_max": float(np.max(p)),
    }


def endless(loader: DataLoader) -> Iterable:
    while True:
        for batch in loader:
            yield batch


def write_report(payload: Dict) -> None:
    summary = payload["summary"]
    ta = payload["metrics"]["target_adapt"]
    tt = payload["metrics"]["target_test"]
    lines = [
        "# RUL quick training report",
        "",
        f"- Run: `{summary['run_name']}`",
        f"- Time: {summary['finished_at']}",
        f"- Device: `{summary['device']}`",
        f"- Runtime: {summary['elapsed_seconds']:.1f} s",
        f"- Protocol: source supervised training + target_adapt MMD/model selection + target_test final once",
        f"- Epochs finished: {summary['epochs_finished']} / {summary['epochs_requested']}",
        f"- Best epoch by target_adapt RMSE: {summary['best_epoch']}",
        "",
        "## Main result",
        "",
        "| Split | RMSE (cycles) | MAE (cycles) | Macro RMSE (cycles) |",
        "|---|---:|---:|---:|",
        f"| target_adapt | {ta['rmse_cycles']:.2f} | {ta['mae_cycles']:.2f} | {ta['macro_rmse_cycles']:.2f} |",
        f"| target_test | {tt['rmse_cycles']:.2f} | {tt['mae_cycles']:.2f} | {tt['macro_rmse_cycles']:.2f} |",
        "",
        "## RUL difficulty note",
        "",
        "RUL is usually harder than pointwise SOH regression under a strict blind-test protocol. SOH asks for the current health state, while RUL asks for the future EOL distance. The same SOH error can map to very different RUL errors when degradation rates differ across cells, and early-life RUL is especially ambiguous because the terminal failure point has not yet been revealed.",
        "",
        "## Reviewer-style diagnosis",
        "",
        "- This is only a quick RUL pilot. It is not enough for an Applied Energy-level main claim.",
        "- The useful part is that the code now tests a RUL-specific monotonic constraint instead of merely renaming the SOH model.",
        "- The largest remaining risk is protocol validity: target_test must not be used for epoch selection, scaling choices, model selection, or visual tuning.",
        "- A stronger paper needs multi-seed runs, external baselines, early-cycle RUL experiments, uncertainty calibration, and at least one cross-dataset or cross-condition validation.",
        "",
        "## Output files",
        "",
        f"- Metrics JSON: `{METRICS_JSON}`",
        f"- History CSV: `{HISTORY_CSV}`",
        f"- Target-test predictions: `{PREDICTIONS_CSV}`",
        f"- Per-battery metrics: `{PER_BATTERY_CSV}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    start = time.time()
    max_runtime_seconds = args.max_runtime_min * 60.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"RUN={RUN_NAME}", flush=True)
    print(f"python={sys.executable}", flush=True)
    print(f"device={device}", flush=True)
    print(f"data_dir={DATA_DIR}", flush=True)

    s_x, s_soh, s_rul, s_p, s_bidx, s_batch, _ = load_split("source")
    ta_x, ta_soh, ta_rul, ta_p, ta_bidx, ta_batch, _ = load_split("target_adapt")
    tt_x, tt_soh, tt_rul, tt_p, tt_bidx, tt_batch, tt_meta = load_split("target_test")

    coeffs = np.polyfit((s_p.reshape(-1) / GLOBAL_MAX_CYCLE), (s_rul.reshape(-1) / RUL_SCALE), 3)
    base_coeffs = torch.tensor(coeffs, dtype=torch.float32, device=device)
    print(f"base_rul_coeffs={coeffs}", flush=True)

    ds_s = RULDataset(s_x, s_soh, s_rul, s_p, s_bidx)
    ds_ta = RULDataset(ta_x, ta_soh, ta_rul, ta_p, ta_bidx)
    ds_tt = RULDataset(tt_x, tt_soh, tt_rul, tt_p, tt_bidx)

    generator = torch.Generator()
    generator.manual_seed(SEED)
    loader_s = DataLoader(ds_s, batch_size=args.batch_size, shuffle=True, drop_last=True, generator=generator)
    loader_ta_train = DataLoader(ds_ta, batch_size=args.batch_size, shuffle=True, drop_last=True, generator=generator)
    loader_ta_eval = DataLoader(ds_ta, batch_size=args.eval_batch_size, shuffle=False)
    loader_tt_eval = DataLoader(ds_tt, batch_size=args.eval_batch_size, shuffle=False)

    model = ResidualRULNet(base_coeffs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    mmd_loss = LifetimeProgressMMD().to(device)

    best_rmse = float("inf")
    best_epoch = -1
    best_metrics = None
    epochs_without_improvement = 0
    history_rows: List[Dict] = []
    source_iter = endless(loader_s)
    target_iter = endless(loader_ta_train)
    epochs_finished = 0

    for epoch in range(1, args.epochs + 1):
        if time.time() - start > max_runtime_seconds:
            print("Time budget reached before starting next epoch.", flush=True)
            break
        epochs_finished = epoch
        model.train()
        if epoch == max(2, args.epochs // 2 + 1):
            for group in optimizer.param_groups:
                group["lr"] = args.lr * LR_DECAY_FACTOR
        mmd_weight = 0.0 if epoch <= 3 else TARGET_MMD_WEIGHT
        batch_losses, rul_losses, soh_losses, mono_losses, mmd_losses = [], [], [], [], []

        for _ in range(args.max_steps_per_epoch):
            sx, s_soh_b, s_rul_b, sp, sb = next(source_iter)
            tx, _, _, tp, tb = next(target_iter)
            sx = sx.float().to(device)
            s_soh_b = s_soh_b.float().to(device)
            s_rul_b = s_rul_b.float().to(device)
            sp = sp.float().to(device)
            sb = sb.to(device)
            tx = tx.float().to(device)
            tp = tp.float().to(device)
            tb = tb.to(device)

            optimizer.zero_grad()
            pred_rul_s, pred_soh_s, feat_s, _ = model(sx, sp)
            pred_rul_t, _, feat_t, _ = model(tx, tp)
            loss_rul = mse(pred_rul_s, s_rul_b)
            loss_soh = mse(pred_soh_s, s_soh_b)
            loss_mono = monotonic_decrease_loss(pred_rul_s, sp, sb) + monotonic_decrease_loss(pred_rul_t, tp, tb)
            loss_mmd = mmd_loss(feat_s, feat_t, tp) if mmd_weight > 0 else torch.tensor(0.0, device=device)
            loss = loss_rul + SOH_AUX_WEIGHT * loss_soh + MONOTONIC_WEIGHT * loss_mono + mmd_weight * loss_mmd
            loss.backward()
            optimizer.step()

            batch_losses.append(float(loss.item()))
            rul_losses.append(float(loss_rul.item()))
            soh_losses.append(float(loss_soh.item()))
            mono_losses.append(float(loss_mono.item()))
            mmd_losses.append(float(loss_mmd.item()))

        target_adapt_metrics, _ = evaluate(model, loader_ta_eval, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(batch_losses)),
            "train_rul_mse": float(np.mean(rul_losses)),
            "train_soh_aux_mse": float(np.mean(soh_losses)),
            "train_monotonic": float(np.mean(mono_losses)),
            "train_mmd": float(np.mean(mmd_losses)),
            "mmd_weight": mmd_weight,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "target_adapt_rmse_cycles": target_adapt_metrics["rmse_cycles"],
            "target_adapt_mae_cycles": target_adapt_metrics["mae_cycles"],
        }
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(HISTORY_CSV, index=False, encoding="utf-8-sig")

        improved = target_adapt_metrics["rmse_cycles"] < best_rmse
        if improved:
            best_rmse = target_adapt_metrics["rmse_cycles"]
            best_epoch = epoch
            best_metrics = target_adapt_metrics
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "base_coeffs": coeffs,
                    "target_adapt_metrics": target_adapt_metrics,
                    "config": vars(args),
                },
                CHECKPOINT_PATH,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"loss={row['train_loss']:.6f} "
            f"target_adapt_RMSE={target_adapt_metrics['rmse_cycles']:.2f}cy "
            f"best={best_rmse:.2f}cy@{best_epoch}",
            flush=True,
        )

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after {args.patience} epochs without improvement.", flush=True)
            break

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    target_adapt_metrics, target_adapt_cells = evaluate(model, loader_ta_eval, device)
    target_test_metrics, target_test_cells = evaluate(model, loader_tt_eval, device)
    predictions = predict_target_test(model, loader_tt_eval, tt_meta, device)
    predictions.to_csv(PREDICTIONS_CSV, index=False, encoding="utf-8-sig")

    target_adapt_cells.insert(0, "split", "target_adapt")
    target_test_cells.insert(0, "split", "target_test")
    per_battery = pd.concat([target_adapt_cells, target_test_cells], ignore_index=True)
    per_battery.to_csv(PER_BATTERY_CSV, index=False, encoding="utf-8-sig")

    elapsed = time.time() - start
    summary = {
        "run_name": RUN_NAME,
        "status": "success",
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "python_executable": sys.executable,
        "device": str(device),
        "data_dir": str(DATA_DIR),
        "seed": SEED,
        "epochs_requested": args.epochs,
        "epochs_finished": epochs_finished,
        "max_steps_per_epoch": args.max_steps_per_epoch,
        "best_epoch": best_epoch,
        "best_target_adapt_rmse_cycles": best_metrics["rmse_cycles"] if best_metrics else None,
        "final_target_test_rmse_cycles": target_test_metrics["rmse_cycles"],
        "elapsed_seconds": elapsed,
        "checkpoint_path": str(CHECKPOINT_PATH),
        "history_csv": str(HISTORY_CSV),
        "predictions_csv": str(PREDICTIONS_CSV),
        "per_battery_csv": str(PER_BATTERY_CSV),
        "report_md": str(REPORT_MD),
        "rul_scale": RUL_SCALE,
        "global_max_cycle": GLOBAL_MAX_CYCLE,
        "soh_aux_weight": SOH_AUX_WEIGHT,
        "monotonic_weight": MONOTONIC_WEIGHT,
        "target_mmd_weight": TARGET_MMD_WEIGHT,
        "base_coeffs": [float(x) for x in coeffs],
    }
    payload = {
        "summary": summary,
        "splits": [
            split_summary("source", s_x, s_rul, s_p, s_bidx, s_batch),
            split_summary("target_adapt", ta_x, ta_rul, ta_p, ta_bidx, ta_batch),
            split_summary("target_test", tt_x, tt_rul, tt_p, tt_bidx, tt_batch),
        ],
        "metrics": {
            "target_adapt": target_adapt_metrics,
            "target_test": target_test_metrics,
        },
        "history_tail": history_rows[-5:],
    }
    METRICS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)

    print(
        "FINAL_METRICS "
        + json.dumps(
            {
                "best_epoch": best_epoch,
                "target_adapt_rmse_cycles": target_adapt_metrics["rmse_cycles"],
                "target_test_rmse_cycles": target_test_metrics["rmse_cycles"],
                "target_test_mae_cycles": target_test_metrics["mae_cycles"],
                "elapsed_seconds": elapsed,
                "report_md": str(REPORT_MD),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
