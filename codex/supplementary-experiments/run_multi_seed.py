"""
Multi-seed training for supplementary experiments.
Trains all 4 model variants (final-retained, no-physical-base, no-mmd, no-attention)
with specified seeds. Results saved to seed-specific subdirectories.

Usage: KMP_DUPLICATE_LIB_OK=TRUE python run_multi_seed.py
"""
import json, math, sys, time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

ROOT = Path(r"C:\Users\ROG\Desktop\paper")
CODEX_DIR = ROOT / "codex"
DATA_DIR = ROOT / "PYthon训练代码" / "processed_clean"
OUT_DIR = CODEX_DIR / "supplementary-experiments"

LABEL_MODE = "nominal"
DATA_PROTOCOL = "processed_clean_fixed_split"
EPOCHS = 150
BATCH_SIZE = 128
EVAL_BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
GLOBAL_MAX_CYCLE = 3000.0
TARGET_MMD_WEIGHT = 0.00005
LAMBDA_SMOOTH = 0.0
LR_DECAY_EPOCH = 100
LR_DECAY_FACTOR = 0.1

NEW_SEEDS = [7, 123]

VARIANTS = [
    {
        "model": "final-retained",
        "use_physical_base": True,
        "use_attention": True,
        "use_mmd": True,
        "description": "Final retained model with all modules.",
    },
    {
        "model": "no-physical-base",
        "use_physical_base": False,
        "use_attention": True,
        "use_mmd": True,
        "description": "Removed cubic physical degradation base.",
    },
    {
        "model": "no-mmd",
        "use_physical_base": True,
        "use_attention": True,
        "use_mmd": False,
        "description": "Disabled life-cycle adaptive MMD.",
    },
    {
        "model": "no-attention",
        "use_physical_base": True,
        "use_attention": False,
        "use_mmd": True,
        "description": "Removed Attention, used LSTM last hidden state.",
    },
]


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
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path)


def load_split(split_name: str, label_mode: str) -> Tuple:
    x = load_npy(DATA_DIR / f"clean_{split_name}_x.npy").astype(np.float32)
    p = load_npy(DATA_DIR / f"clean_{split_name}_p.npy").astype(np.float32)
    bidx = load_npy(DATA_DIR / f"clean_{split_name}_battery_index.npy").astype(np.int64)
    batch = load_npy(DATA_DIR / f"clean_{split_name}_batch.npy").astype(np.int64)
    if label_mode == "nominal":
        y = load_npy(DATA_DIR / f"clean_{split_name}_y_nominal.npy").astype(np.float32)
    else:
        y = load_npy(DATA_DIR / f"clean_{split_name}_y_relative.npy").astype(np.float32)
    return x, y, p, bidx, batch


class HybridDataset(Dataset):
    def __init__(self, x, y, p, bidx):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        self.p = (p.astype(np.float32) / GLOBAL_MAX_CYCLE).reshape(-1, 1)
        self.bidx = bidx.astype(np.int64)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.p[idx], self.bidx[idx]


class LifetimeAdaptiveMMD(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, lambda_scale=5.5):
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.lambda_scale = lambda_scale
    def gaussian_kernel(self, source, target):
        n = int(source.size(0)) + int(target.size(0))
        total = torch.cat([source, target], dim=0)
        t0 = total.unsqueeze(0).expand(n, n, total.size(1))
        t1 = total.unsqueeze(1).expand(n, n, total.size(1))
        l2 = ((t0 - t1) ** 2).sum(2)
        bw = torch.sum(l2.detach()) / max(n*n - n, 1)
        bw = torch.clamp(bw, min=1e-6) / (self.kernel_mul ** (self.kernel_num // 2))
        bw_list = [bw * (self.kernel_mul**i) for i in range(self.kernel_num)]
        return sum(torch.exp(-l2 / b) for b in bw_list)
    def compute_mmd(self, source, target):
        if source.size(0) == 0 or target.size(0) == 0:
            return torch.tensor(0.0, device=source.device)
        bs = int(source.size(0))
        K = self.gaussian_kernel(source, target)
        xx = K[:bs, :bs]; yy = K[bs:, bs:]; xy = K[:bs, bs:]
        return torch.mean(xx) + torch.mean(yy) - 2*torch.mean(xy)
    def forward(self, sf, tf, s_base, t_base):
        if sf.size(0) == 0 or tf.size(0) == 0:
            return torch.tensor(0.0, device=sf.device)
        soh = torch.clamp(torch.mean(t_base), max=1.0)
        dw = torch.exp(self.lambda_scale * (1.0 - soh))
        return dw * self.compute_mmd(sf, tf)


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.Tanh(), nn.Linear(hidden_dim//2, 1))
    def forward(self, lstm_out):
        w = F.softmax(self.attn(lstm_out), dim=1)
        return torch.sum(w * lstm_out, dim=1), w


class ResidualCRNN(nn.Module):
    def __init__(self, base_coeffs, use_physical_base=True, use_attention=True):
        super().__init__()
        self.base_coeffs = base_coeffs
        self.use_physical_base = use_physical_base
        self.use_attention = use_attention
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        self.attention = TemporalAttention(64) if use_attention else None
        self.phy = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU())
        self.fc_soh = nn.Sequential(nn.Linear(72, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))
    def forward(self, x, p):
        a, b, c, d = self.base_coeffs
        base_soh = a*(p**3)+b*(p**2)+c*p+d if self.use_physical_base else torch.zeros((x.size(0),1), device=x.device, dtype=x.dtype)
        out = self.cnn(x.unsqueeze(1)).permute(0,2,1)
        lstm_out, _ = self.lstm(out)
        temporal_feat = self.attention(lstm_out)[0] if self.use_attention else lstm_out[:, -1, :]
        fused = torch.cat((temporal_feat, self.phy(p)), dim=1)
        residual = self.fc_soh(fused)
        pred = base_soh + residual if self.use_physical_base else residual
        return pred, fused, base_soh


def rmse(y_true, y_pred):
    return float(math.sqrt(np.mean((y_true.reshape(-1)-y_pred.reshape(-1))**2)))
def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true.reshape(-1)-y_pred.reshape(-1))))
def mape(y_true, y_pred):
    denom = np.clip(np.abs(y_true.reshape(-1)), 1e-8, None)
    return float(np.mean(np.abs((y_true.reshape(-1)-y_pred.reshape(-1))/denom))*100)
def macro_rmse(y_true, y_pred, bidx):
    vals = [rmse(y_true[bidx==b], y_pred[bidx==b]) for b in np.unique(bidx) if np.any(bidx==b)]
    return float(np.mean(vals)) if vals else float("nan")


def evaluate(model, loader, device):
    model.eval()
    preds, trues, bidxs = [], [], []
    with torch.no_grad():
        for x, y, p, b in loader:
            x, p = x.float().to(device), p.float().to(device)
            out, _, _ = model(x, p)
            preds.append(out.cpu().numpy().reshape(-1))
            trues.append(y.numpy().reshape(-1))
            bidxs.append(b.numpy().reshape(-1))
    y_pred, y_true = np.concatenate(preds), np.concatenate(trues)
    bidx = np.concatenate(bidxs)
    metrics = {
        "pooled_rmse": rmse(y_true, y_pred),
        "macro_rmse": macro_rmse(y_true, y_pred, bidx),
        "mae": mae(y_true, y_pred),
        "mape_percent": mape(y_true, y_pred),
    }
    per_battery = pd.DataFrame([
        {"battery_index": int(b), "rmse": rmse(y_true[bidx==b], y_pred[bidx==b]),
         "mae": mae(y_true[bidx==b], y_pred[bidx==b]), "n_points": int(np.sum(bidx==b))}
        for b in np.unique(bidx)
    ])
    return metrics, per_battery


def train_variant(variant: Dict, seed: int, data: Dict, coeffs: np.ndarray, device: torch.device) -> Dict:
    start_time = time.time()
    set_seed(seed)
    model_name = f"{variant['model']}-seed{seed}"
    run_dir = OUT_DIR / model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / f"{model_name}_best.pt"
    best_json = run_dir / f"{model_name}_best_metrics.json"
    history_csv = run_dir / f"{model_name}_history.csv"

    s_x, s_y, s_p, s_bidx, _ = data["source"]
    ta_x, ta_y, ta_p, ta_bidx, _ = data["target_adapt"]
    tt_x, tt_y, tt_p, tt_bidx, _ = data["target_test"]

    ds_s = HybridDataset(s_x, s_y, s_p, s_bidx)
    ds_ta = HybridDataset(ta_x, ta_y, ta_p, ta_bidx)
    ds_tt = HybridDataset(tt_x, tt_y, tt_p, tt_bidx)
    loader_s = DataLoader(ds_s, BATCH_SIZE, shuffle=True, drop_last=True)
    loader_ta = DataLoader(ds_ta, BATCH_SIZE, shuffle=True, drop_last=True)
    loader_ta_eval = DataLoader(ds_ta, EVAL_BATCH_SIZE, shuffle=False)
    loader_tt_eval = DataLoader(ds_tt, EVAL_BATCH_SIZE, shuffle=False)

    base_tensor = torch.tensor(coeffs, dtype=torch.float32, device=device)
    model = ResidualCRNN(base_tensor, use_physical_base=variant["use_physical_base"],
                          use_attention=variant["use_attention"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_mse = nn.MSELoss()
    criterion_mmd = LifetimeAdaptiveMMD().to(device)

    best_rmse, best_epoch = float("inf"), -1
    history_rows = []

    print(f"\n{'='*60}", flush=True)
    print(f"STARTING: {model_name} | seed={seed}", flush=True)
    print(f"{'='*60}", flush=True)

    for epoch in range(1, EPOCHS+1):
        model.train()
        if epoch == LR_DECAY_EPOCH + 1:
            for g in optimizer.param_groups:
                g["lr"] = LEARNING_RATE * LR_DECAY_FACTOR

        if epoch <= 20:
            mmd_w, stage = 0.0, "Warmup"
        elif epoch <= 120:
            mmd_w = TARGET_MMD_WEIGHT if variant["use_mmd"] else 0.0
            stage = "MMD" if variant["use_mmd"] else "NoMMD"
        else:
            mmd_w, stage = 0.0, "Sprint"

        losses, mse_losses, mmd_losses = [], [], []
        for (sx, sy, sp, _), (tx, _, tp, _) in zip(loader_s, loader_ta):
            sx, sy, sp = sx.float().to(device), sy.float().to(device), sp.float().to(device)
            tx, tp = tx.float().to(device), tp.float().to(device)
            optimizer.zero_grad()
            pred_s, feat_s, base_s = model(sx, sp)
            _, feat_t, base_t = model(tx, tp)
            lmse = criterion_mse(pred_s, sy)
            lmmd = criterion_mmd(feat_s, feat_t, base_s, base_t) if mmd_w > 0 else torch.tensor(0.0, device=device)
            loss = lmse + mmd_w * lmmd
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
            mse_losses.append(float(lmse))
            mmd_losses.append(float(lmmd))

        ta_metrics, _ = evaluate(model, loader_ta_eval, device)
        if ta_metrics["pooled_rmse"] < best_rmse:
            best_rmse, best_epoch = ta_metrics["pooled_rmse"], epoch
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "seed": seed,
                "variant": variant, "target_adapt_metrics": ta_metrics,
            }, checkpoint_path)

        history_rows.append({
            "epoch": epoch, "stage": stage,
            "train_loss": float(np.mean(losses)),
            "train_mse": float(np.mean(mse_losses)),
            "train_mmd": float(np.mean(mmd_losses)),
            "mmd_weight": float(mmd_w),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "ta_rmse": ta_metrics["pooled_rmse"],
            "ta_macro_rmse": ta_metrics["macro_rmse"],
            "best_epoch": best_epoch, "best_rmse": best_rmse,
        })
        pd.DataFrame(history_rows).to_csv(history_csv, index=False, encoding="utf-8-sig")
        if epoch % 10 == 0 or epoch == 1:
            print(f"  {model_name} ep={epoch:03d} {stage:6s} loss={np.mean(losses):.6f} ta_rmse={ta_metrics['pooled_rmse']:.5f} best={best_rmse:.5f}@{best_epoch}", flush=True)

    # Final evaluation with best checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    ta_metrics, ta_cells = evaluate(model, loader_ta_eval, device)
    tt_metrics, tt_cells = evaluate(model, loader_tt_eval, device)
    elapsed = time.time() - start_time

    result = {
        "model": model_name, "seed": seed, "variant": variant["model"],
        "best_epoch": best_epoch,
        "target_adapt_rmse": ta_metrics["pooled_rmse"],
        "target_test_rmse": tt_metrics["pooled_rmse"],
        "target_test_macro_rmse": tt_metrics["macro_rmse"],
        "target_test_mae": tt_metrics["mae"],
        "target_test_mape": tt_metrics["mape_percent"],
        "elapsed_min": round(elapsed/60, 1),
    }

    # Save detailed results
    with pd.ExcelWriter(run_dir / f"{model_name}_results.xlsx", engine="openpyxl") as w:
        pd.DataFrame([result]).to_excel(w, index=False, sheet_name="summary")
        tt_cells.to_excel(w, index=False, sheet_name="target_test_cells")
        ta_cells.to_excel(w, index=False, sheet_name="target_adapt_cells")
        pd.DataFrame(history_rows).to_excel(w, index=False, sheet_name="history")

    best_json.write_text(
        json.dumps({"result": result, "target_test_cells": tt_cells.to_dict(orient="records"),
                     "target_adapt_cells": ta_cells.to_dict(orient="records")},
                    ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  DONE: {model_name} | tt_rmse={tt_metrics['pooled_rmse']:.6f} | {elapsed/60:.1f}min", flush=True)

    del model, optimizer, criterion_mmd
    torch.cuda.empty_cache()
    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Output dir: {OUT_DIR}", flush=True)
    print(f"New seeds: {NEW_SEEDS}", flush=True)

    # Load data
    data = {
        "source": load_split("source", LABEL_MODE),
        "target_adapt": load_split("target_adapt", LABEL_MODE),
        "target_test": load_split("target_test", LABEL_MODE),
    }
    s_x, s_y, s_p, _, _ = data["source"]
    coeffs = np.polyfit((s_p.flatten() / GLOBAL_MAX_CYCLE), s_y.flatten(), 3)
    print(f"Base coeffs: {coeffs}", flush=True)

    all_results = []

    for seed in NEW_SEEDS:
        for variant in VARIANTS:
            result = train_variant(variant, seed, data, coeffs, device)
            all_results.append(result)

    # Summary
    summary_df = pd.DataFrame(all_results)
    summary_path = OUT_DIR / "multi_seed_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n{'='*60}", flush=True)
    print("MULTI-SEED SUMMARY", flush=True)
    print(summary_df.to_string(), flush=True)
    print(f"\nSummary saved to: {summary_path}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
