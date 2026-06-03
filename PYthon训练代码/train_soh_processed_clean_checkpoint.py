import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ============================================================
# SOH training on processed_clean dataset (checkpoint edition)
# - Loads clean split: source / target_adapt / target_test
# - Supports nominal or relative SOH labels
# - Smoothness penalty is computed ONLY within each battery
# - Reports both pooled RMSE and macro RMSE
# - Saves full checkpoint, best epoch, and training history CSV
# ============================================================

# ================= 1. Config =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Make runs more reproducible
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Data directory: point this to your processed_clean folder
DATA_DIR = Path("./processed_clean")

# Choose label mode explicitly to avoid confusion
# options: "nominal" or "relative"
LABEL_MODE = "nominal"
DATA_PROTOCOL = "processed_clean_fixed_split"

EPOCHS = 150
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
GLOBAL_MAX_CYCLE = 3000.0
LAMBDA_SMOOTH = 0.01
TARGET_MMD_WEIGHT = 0.00005

SAVE_NAME = f"best_soh_{LABEL_MODE}_processed_clean.pth"
HISTORY_CSV = f"train_history_soh_{LABEL_MODE}_processed_clean.csv"
BEST_JSON = f"best_metrics_soh_{LABEL_MODE}_processed_clean.json"

print(f"🚀 Training SOH model | device={DEVICE}")
print(f"📂 DATA_DIR = {DATA_DIR.resolve()}")
print(f"🏷️ LABEL_MODE = {LABEL_MODE}")
print(f"🧪 DATA_PROTOCOL = {DATA_PROTOCOL}")


# ================= 2. Utilities =================
class HybridDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, p: np.ndarray, bidx: np.ndarray):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        self.p = (p.astype(np.float32) / GLOBAL_MAX_CYCLE)
        self.bidx = bidx.astype(np.int64)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.p[idx], self.bidx[idx]


def load_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"❌ Missing file: {path}")
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


def describe_split(name: str, x: np.ndarray, y: np.ndarray, p: np.ndarray, bidx: np.ndarray, batch: np.ndarray) -> None:
    print(
        f"{name:>12} | samples={len(x):6d} | cells={len(np.unique(bidx)):2d} | "
        f"batches={sorted(np.unique(batch).tolist())} | y_range=({y.min():.4f}, {y.max():.4f}) | "
        f"cycle_range=({p.min():.0f}, {p.max():.0f})"
    )


# ================= 3. Model =================
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
        bandwidth = torch.sum(l2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth = bandwidth / (self.kernel_mul ** (self.kernel_num // 2))
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        kernel_vals = [torch.exp(-l2_distance / bw) for bw in bandwidth_list]
        return sum(kernel_vals)

    def compute_mmd(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if source.size(0) == 0 or target.size(0) == 0:
            return torch.tensor(0.0, device=source.device if source.numel() else target.device)
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
        current_soh = torch.mean(target_base_soh)
        current_soh = torch.clamp(current_soh, max=1.0)
        dynamic_weight = torch.exp(self.lambda_scale * (1.0 - current_soh))
        base_mmd_loss = self.compute_mmd(source_features, target_features)
        return dynamic_weight * base_mmd_loss


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, lstm_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_weights = self.attention(lstm_outputs)
        attn_weights = F.softmax(attn_weights, dim=1)
        context_vector = torch.sum(attn_weights * lstm_outputs, dim=1)
        return context_vector, attn_weights


class ResidualCRNN(nn.Module):
    def __init__(self, base_coeffs: torch.Tensor):
        super().__init__()
        self.base_coeffs = base_coeffs
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        self.attention = TemporalAttention(hidden_dim=64)
        self.phy = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
        )
        self.fc_soh = nn.Sequential(
            nn.Linear(72, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a, b, c, d = self.base_coeffs
        base_soh = a * (p ** 3) + b * (p ** 2) + c * p + d

        x_in = x.unsqueeze(1)
        out = self.cnn(x_in).permute(0, 2, 1)
        lstm_out, _ = self.lstm(out)

        context_feat, _ = self.attention(lstm_out)
        f_phy = self.phy(p)
        fused_features = torch.cat((context_feat, f_phy), dim=1)

        residual_soh = self.fc_soh(fused_features)
        pred_soh = base_soh + residual_soh
        return pred_soh, fused_features, base_soh


# ================= 4. Loss & Metrics =================
def smoothness_loss_within_battery(preds: torch.Tensor, cycles: torch.Tensor, battery_idx: torch.Tensor) -> torch.Tensor:
    """
    Smoothness is computed ONLY within the same battery.
    If a battery contributes fewer than 2 samples in the current batch, it is skipped.
    """
    preds_flat = preds.flatten()
    cycles_flat = cycles.flatten()
    batt_flat = battery_idx.flatten()

    losses: List[torch.Tensor] = []
    for b in torch.unique(batt_flat):
        mask = batt_flat == b
        if int(mask.sum()) < 2:
            continue
        local_cycles = cycles_flat[mask]
        local_preds = preds_flat[mask]
        order = torch.argsort(local_cycles)
        local_preds = local_preds[order]
        diff = local_preds[1:] - local_preds[:-1]
        losses.append(torch.mean(diff ** 2))

    if len(losses) == 0:
        return torch.tensor(0.0, device=preds.device)
    return torch.stack(losses).mean()


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def macro_rmse_by_battery(y_true: np.ndarray, y_pred: np.ndarray, battery_idx: np.ndarray) -> float:
    vals = []
    for b in np.unique(battery_idx):
        mask = battery_idx == b
        vals.append(rmse(y_true[mask], y_pred[mask]))
    return float(np.mean(vals))


# ================= 5. Data =================
s_x, s_y, s_p, s_bidx, s_batch = load_split("source", LABEL_MODE)
ta_x, ta_y, ta_p, ta_bidx, ta_batch = load_split("target_adapt", LABEL_MODE)
tt_x, tt_y, tt_p, tt_bidx, tt_batch = load_split("target_test", LABEL_MODE)

describe_split("source", s_x, s_y, s_p, s_bidx, s_batch)
describe_split("target_adapt", ta_x, ta_y, ta_p, ta_bidx, ta_batch)
describe_split("target_test", tt_x, tt_y, tt_p, tt_bidx, tt_batch)

# Fit cubic physical base from source labels
coeffs = np.polyfit((s_p.flatten() / GLOBAL_MAX_CYCLE), s_y.flatten(), 3)
POLY_COEFFS = torch.tensor(coeffs, dtype=torch.float32, device=DEVICE)
print(f"📈 Base cubic coeffs = {coeffs}")

ds_s = HybridDataset(s_x, s_y, s_p, s_bidx)
ds_ta = HybridDataset(ta_x, ta_y, ta_p, ta_bidx)
ds_tt = HybridDataset(tt_x, tt_y, tt_p, tt_bidx)

loader_s = DataLoader(ds_s, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
loader_ta = DataLoader(ds_ta, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
loader_tt = DataLoader(ds_tt, batch_size=BATCH_SIZE, shuffle=False)


# ================= 6. Train =================
model = ResidualCRNN(POLY_COEFFS).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion_mse = nn.MSELoss()
criterion_mmd = LifetimeAdaptiveMMD().to(DEVICE)

best_pooled_rmse = float("inf")
best_macro_rmse = float("inf")
best_epoch = -1
history: List[Dict] = []

print("🚀 Start training...")
for epoch in range(EPOCHS):
    model.train()

    if epoch == 100:
        for param_group in optimizer.param_groups:
            param_group["lr"] = LEARNING_RATE * 0.1

    if epoch < 20:
        curr_mmd_weight = 0.0
        stage = "MSE Warmup"
    elif epoch < 120:
        curr_mmd_weight = TARGET_MMD_WEIGHT
        stage = "MMD Guide"
    else:
        curr_mmd_weight = 0.0
        stage = "Pure MSE Sprint"

    batch_losses: List[float] = []
    for (sx, sy, sp, sb), (tx, _, tp, tb) in zip(loader_s, loader_ta):
        sx = sx.float().to(DEVICE)
        sy = sy.float().to(DEVICE)
        sp = sp.float().to(DEVICE).view(-1, 1)
        sb = sb.to(DEVICE)

        tx = tx.float().to(DEVICE)
        tp = tp.float().to(DEVICE).view(-1, 1)
        tb = tb.to(DEVICE)

        optimizer.zero_grad()

        pred_s, feat_s, base_s = model(sx, sp)
        loss_mse = criterion_mse(pred_s, sy)

        pred_t, feat_t, base_t = model(tx, tp)
        loss_smooth = smoothness_loss_within_battery(pred_t, tp, tb)

        if curr_mmd_weight > 0:
            loss_mmd = criterion_mmd(feat_s, feat_t, base_s, base_t)
        else:
            loss_mmd = torch.tensor(0.0, device=DEVICE)

        loss = loss_mse + (LAMBDA_SMOOTH * loss_smooth) + (curr_mmd_weight * loss_mmd)
        loss.backward()
        optimizer.step()
        batch_losses.append(float(loss.item()))

    # Validation on target_test
    model.eval()
    preds_all: List[np.ndarray] = []
    trues_all: List[np.ndarray] = []
    bidx_all: List[np.ndarray] = []

    with torch.no_grad():
        for x, y, p, b in loader_tt:
            x = x.float().to(DEVICE)
            y = y.float().to(DEVICE)
            p = p.float().to(DEVICE).view(-1, 1)

            out, _, _ = model(x, p)
            preds_all.append(out.cpu().numpy())
            trues_all.append(y.cpu().numpy())
            bidx_all.append(b.numpy())

    preds_np = np.concatenate(preds_all).flatten()
    trues_np = np.concatenate(trues_all).flatten()
    bidx_np = np.concatenate(bidx_all).flatten()

    pooled = rmse(trues_np, preds_np)
    macro = macro_rmse_by_battery(trues_np, preds_np, bidx_np)
    train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0

    history.append(
        {
            "epoch": epoch + 1,
            "stage": stage,
            "train_loss": train_loss,
            "pooled_rmse": float(pooled),
            "macro_rmse": float(macro),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "mmd_weight": float(curr_mmd_weight),
        }
    )

    if pooled < best_pooled_rmse:
        best_pooled_rmse = pooled
        best_macro_rmse = macro
        best_epoch = epoch + 1

        checkpoint = {
            "epoch": best_epoch,
            "best_pooled_rmse": float(best_pooled_rmse),
            "best_macro_rmse": float(best_macro_rmse),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "label_mode": LABEL_MODE,
            "data_protocol": DATA_PROTOCOL,
            "seed": SEED,
            "base_coeffs": coeffs,
            "config": {
                "epochs": EPOCHS,
                "lr": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "global_max_cycle": GLOBAL_MAX_CYCLE,
                "lambda_smooth": LAMBDA_SMOOTH,
                "target_mmd_weight": TARGET_MMD_WEIGHT,
            },
        }
        torch.save(checkpoint, SAVE_NAME)

        best_json_payload = {
            "best_epoch": int(best_epoch),
            "best_pooled_rmse": float(best_pooled_rmse),
            "best_macro_rmse": float(best_macro_rmse),
            "label_mode": LABEL_MODE,
            "data_protocol": DATA_PROTOCOL,
            "seed": SEED,
            "checkpoint_path": SAVE_NAME,
        }
        with open(BEST_JSON, "w", encoding="utf-8") as f:
            json.dump(best_json_payload, f, ensure_ascii=False, indent=2)

    # Save history CSV every epoch so you don't lose logs if interrupted
    pd.DataFrame(history).to_csv(HISTORY_CSV, index=False, encoding="utf-8-sig")

    print(
        f"\rEpoch {epoch+1:03d} [{stage}] | loss={train_loss:.6f} | "
        f"pooled RMSE={pooled:.5f} | macro RMSE={macro:.5f} | best epoch={best_epoch}",
        end="",
    )

print("\n✅ Training finished")
print(f"🏆 Best pooled RMSE = {best_pooled_rmse:.5f}")
print(f"🏆 Best macro RMSE  = {best_macro_rmse:.5f}")
print(f"🏆 Best epoch       = {best_epoch}")
print(f"💾 Saved checkpoint = {SAVE_NAME}")
print(f"📝 Saved history    = {HISTORY_CSV}")
print(f"🧾 Saved best json  = {BEST_JSON}")
