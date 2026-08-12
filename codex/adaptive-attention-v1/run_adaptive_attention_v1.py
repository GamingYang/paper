"""
Multi-seed training for the degradation-state-guided curve attention experiment.

This script keeps the retained physical-residual SOH protocol unchanged and
only replaces the temporal aggregation module with an adaptive attention block.
Results are saved to seed-specific subdirectories under this experiment folder.
"""
import argparse
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
OUT_DIR = CODEX_DIR / "adaptive-attention-v1"

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

NEW_SEEDS = [42, 7, 123]
LOG_EVERY = 10

VARIANTS = [
    {
        "model": "adaptive-attention-v1",
        "use_physical_base": True,
        "use_mmd": True,
        "description": (
            "Retained physical-residual model with degradation-state-guided "
            "curve attention and life-cycle adaptive MMD."
        ),
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


class DegradationStateGuidedCurveAttention(nn.Module):
    """Adaptive attention conditioned on degradation state and local curve shape."""

    def __init__(self, hidden_dim=64, curve_dim=3, state_dim=3, tau_min=0.35):
        super().__init__()
        self.tau_min = tau_min
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        self.curve_proj = nn.Linear(curve_dim, hidden_dim)
        self.query_proj = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh())
        self.tau_proj = nn.Linear(state_dim, 1)

    @staticmethod
    def curve_tokens(x: torch.Tensor, steps: int) -> torch.Tensor:
        voltage = F.interpolate(
            x.unsqueeze(1),
            size=steps,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        slope = torch.zeros_like(voltage)
        slope[:, 1:, :] = voltage[:, 1:, :] - voltage[:, :-1, :]
        curvature = torch.zeros_like(voltage)
        curvature[:, 1:, :] = slope[:, 1:, :] - slope[:, :-1, :]
        return torch.cat([voltage, slope, curvature], dim=-1)

    def forward(
        self,
        lstm_out: torch.Tensor,
        x: torch.Tensor,
        degradation_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        curve_feat = self.curve_tokens(x, lstm_out.size(1))
        query = self.query_proj(degradation_state)
        score_feat = torch.tanh(
            self.hidden_proj(lstm_out)
            + self.curve_proj(curve_feat)
            + query.unsqueeze(1)
        )
        scores = torch.sum(query.unsqueeze(1) * score_feat, dim=-1)
        temperature = self.tau_min + F.softplus(self.tau_proj(degradation_state))
        attn_weights = F.softmax(scores / temperature, dim=1).unsqueeze(-1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return context, attn_weights, temperature


class ResidualCRNN(nn.Module):
    def __init__(self, base_coeffs, use_physical_base=True):
        super().__init__()
        self.base_coeffs = base_coeffs
        self.use_physical_base = use_physical_base
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        self.attention = DegradationStateGuidedCurveAttention(hidden_dim=64)
        self.phy = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU())
        self.fc_soh = nn.Sequential(nn.Linear(72, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))
    def forward(self, x, p):
        a, b, c, d = self.base_coeffs
        base_soh = a*(p**3)+b*(p**2)+c*p+d if self.use_physical_base else torch.zeros((x.size(0),1), device=x.device, dtype=x.dtype)
        out = self.cnn(x.unsqueeze(1)).permute(0,2,1)
        lstm_out, _ = self.lstm(out)
        degradation_state = torch.cat([p, base_soh, 1.0 - base_soh], dim=1)
        temporal_feat, _, _ = self.attention(lstm_out, x, degradation_state)
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
    preds, trues, bidxs, pvals = [], [], [], []
    with torch.no_grad():
        for x, y, p, b in loader:
            x, p = x.float().to(device), p.float().to(device)
            out, _, _ = model(x, p)
            preds.append(out.cpu().numpy().reshape(-1))
            trues.append(y.numpy().reshape(-1))
            bidxs.append(b.numpy().reshape(-1))
            pvals.append(p.cpu().numpy().reshape(-1))
    y_pred, y_true = np.concatenate(preds), np.concatenate(trues)
    bidx = np.concatenate(bidxs)
    p_norm = np.concatenate(pvals)
    metrics = {
        "pooled_rmse": rmse(y_true, y_pred),
        "macro_rmse": macro_rmse(y_true, y_pred, bidx),
        "mae": mae(y_true, y_pred),
        "mape_percent": mape(y_true, y_pred),
    }
    stage_masks = {
        "early": p_norm < (1.0 / 3.0),
        "mid": (p_norm >= (1.0 / 3.0)) & (p_norm < (2.0 / 3.0)),
        "late": p_norm >= (2.0 / 3.0),
    }
    for stage, mask in stage_masks.items():
        if np.any(mask):
            metrics[f"{stage}_rmse"] = rmse(y_true[mask], y_pred[mask])
            metrics[f"{stage}_mae"] = mae(y_true[mask], y_pred[mask])
            metrics[f"{stage}_n"] = int(np.sum(mask))
        else:
            metrics[f"{stage}_rmse"] = float("nan")
            metrics[f"{stage}_mae"] = float("nan")
            metrics[f"{stage}_n"] = 0
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
    model = ResidualCRNN(base_tensor, use_physical_base=variant["use_physical_base"]).to(device)
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
        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(f"  {model_name} ep={epoch:03d} {stage:6s} loss={np.mean(losses):.6f} ta_rmse={ta_metrics['pooled_rmse']:.5f} best={best_rmse:.5f}@{best_epoch}", flush=True)

    # Final evaluation with best checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    ta_metrics, ta_cells = evaluate(model, loader_ta_eval, device)
    tt_metrics, tt_cells = evaluate(model, loader_tt_eval, device)
    elapsed = time.time() - start_time

    result = {
        "model": model_name, "seed": seed, "variant": variant["model"],
        "description": variant["description"],
        "attention": "degradation-state-guided curve attention",
        "best_epoch": best_epoch,
        "target_adapt_rmse": ta_metrics["pooled_rmse"],
        "target_test_rmse": tt_metrics["pooled_rmse"],
        "target_test_macro_rmse": tt_metrics["macro_rmse"],
        "target_test_mae": tt_metrics["mae"],
        "target_test_mape": tt_metrics["mape_percent"],
        "target_test_early_rmse": tt_metrics["early_rmse"],
        "target_test_mid_rmse": tt_metrics["mid_rmse"],
        "target_test_late_rmse": tt_metrics["late_rmse"],
        "checkpoint_path": str(checkpoint_path),
        "history_csv": str(history_csv),
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run adaptive attention SOH training.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seeds", type=int, nargs="+", default=NEW_SEEDS)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    return parser.parse_args()


def main():
    global EPOCHS, LOG_EVERY
    args = parse_args()
    EPOCHS = args.epochs
    LOG_EVERY = args.log_every
    seeds = args.seeds

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Output dir: {OUT_DIR}", flush=True)
    print(f"Seeds: {seeds}", flush=True)
    print(f"Epochs: {EPOCHS}", flush=True)

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

    for seed in seeds:
        for variant in VARIANTS:
            result = train_variant(variant, seed, data, coeffs, device)
            all_results.append(result)

    # Summary
    summary_df = pd.DataFrame(all_results)
    summary_path = OUT_DIR / "adaptive_attention_v1_multi_seed_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    summary_df.to_excel(OUT_DIR / "adaptive_attention_v1_multi_seed_summary.xlsx", index=False)
    (OUT_DIR / "adaptive_attention_v1_multi_seed_summary.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n{'='*60}", flush=True)
    print("MULTI-SEED SUMMARY", flush=True)
    print(summary_df.to_string(), flush=True)
    print(f"\nSummary saved to: {summary_path}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
