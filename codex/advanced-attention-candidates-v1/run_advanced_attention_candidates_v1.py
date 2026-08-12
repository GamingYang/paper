"""
Three-seed feasibility training for two advanced attention candidates.

The data protocol, physical cubic base, residual head, optimizer, MMD schedule,
and checkpoint-selection rule are kept aligned with adaptive-attention-v1.
Only the attention module changes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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
OUT_DIR = CODEX_DIR / "advanced-attention-candidates-v1"

LABEL_MODE = "nominal"
DATA_PROTOCOL = "processed_clean_fixed_split"
EPOCHS = 150
BATCH_SIZE = 128
EVAL_BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
GLOBAL_MAX_CYCLE = 3000.0
TARGET_MMD_WEIGHT = 0.00005
LR_DECAY_EPOCH = 100
LR_DECAY_FACTOR = 0.1
SEEDS = [42, 7, 123]
LOG_EVERY = 10

VARIANTS = [
    {
        "model": "multi-expert-gated-attention-v1",
        "attention_type": "multi_expert",
        "description": (
            "Physical-residual SOH model with early/middle/late curve-attention "
            "experts and degradation-state soft gating."
        ),
    },
    {
        "model": "stage-aware-branch-attention-v1",
        "attention_type": "stage_branch",
        "description": (
            "Physical-residual SOH model with an explicit degradation-stage "
            "branch that modulates temporal features before attention."
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


class CurveTokenMixin:
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


class MultiExpertGatedCurveAttention(nn.Module, CurveTokenMixin):
    """Early/middle/late attention experts with degradation-state soft gates."""

    def __init__(
        self,
        hidden_dim: int = 64,
        curve_dim: int = 3,
        state_dim: int = 3,
        n_experts: int = 3,
        tau_min: float = 0.35,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.tau_min = tau_min
        self.hidden_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_experts)])
        self.curve_proj = nn.ModuleList([nn.Linear(curve_dim, hidden_dim) for _ in range(n_experts)])
        self.query_proj = nn.ModuleList([nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh()) for _ in range(n_experts)])
        self.tau_proj = nn.ModuleList([nn.Linear(state_dim, 1) for _ in range(n_experts)])
        self.gate = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_experts),
        )

    def forward(
        self,
        lstm_out: torch.Tensor,
        x: torch.Tensor,
        degradation_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        curve_feat = self.curve_tokens(x, lstm_out.size(1))
        gates = F.softmax(self.gate(degradation_state), dim=1)
        contexts, attns, temps = [], [], []
        for k in range(self.n_experts):
            query = self.query_proj[k](degradation_state)
            score_feat = torch.tanh(
                self.hidden_proj[k](lstm_out)
                + self.curve_proj[k](curve_feat)
                + query.unsqueeze(1)
            )
            scores = torch.sum(query.unsqueeze(1) * score_feat, dim=-1)
            temperature = self.tau_min + F.softplus(self.tau_proj[k](degradation_state))
            attn = F.softmax(scores / temperature, dim=1)
            context = torch.sum(attn.unsqueeze(-1) * lstm_out, dim=1)
            contexts.append(context)
            attns.append(attn)
            temps.append(temperature)

        context_stack = torch.stack(contexts, dim=1)
        attn_stack = torch.stack(attns, dim=1)
        temp_stack = torch.cat(temps, dim=1)
        context = torch.sum(gates.unsqueeze(-1) * context_stack, dim=1)
        combined_attn = torch.sum(gates.unsqueeze(-1) * attn_stack, dim=1).unsqueeze(-1)
        combined_temp = torch.sum(gates * temp_stack, dim=1, keepdim=True)
        aux = {
            "gate": gates,
            "expert_attention": attn_stack,
            "expert_temperature": temp_stack,
        }
        return context, combined_attn, combined_temp, aux


class StageAwareBranchAttention(nn.Module, CurveTokenMixin):
    """Explicit stage branch that modulates temporal features before attention."""

    def __init__(
        self,
        hidden_dim: int = 64,
        curve_dim: int = 3,
        state_dim: int = 3,
        branch_dim: int = 64,
        tau_min: float = 0.35,
        rho_gamma: float = 0.5,
        rho_beta: float = 0.1,
    ):
        super().__init__()
        self.tau_min = tau_min
        self.rho_gamma = rho_gamma
        self.rho_beta = rho_beta
        self.stage_branch = nn.Sequential(
            nn.Linear(state_dim + curve_dim, 32),
            nn.ReLU(),
            nn.Linear(32, branch_dim),
            nn.Tanh(),
        )
        self.gamma_proj = nn.Linear(branch_dim, hidden_dim)
        self.beta_proj = nn.Linear(branch_dim, hidden_dim)
        self.query_proj = nn.Sequential(nn.Linear(branch_dim, hidden_dim), nn.Tanh())
        self.tau_proj = nn.Linear(branch_dim, 1)
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        self.curve_proj = nn.Linear(curve_dim, hidden_dim)

    def forward(
        self,
        lstm_out: torch.Tensor,
        x: torch.Tensor,
        degradation_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        curve_feat = self.curve_tokens(x, lstm_out.size(1))
        curve_summary = torch.stack(
            [
                curve_feat[:, :, 0].mean(dim=1),
                curve_feat[:, :, 1].abs().mean(dim=1),
                curve_feat[:, :, 2].abs().mean(dim=1),
            ],
            dim=1,
        )
        branch_input = torch.cat([degradation_state, curve_summary], dim=1)
        stage_embedding = self.stage_branch(branch_input)
        gamma = torch.tanh(self.gamma_proj(stage_embedding))
        beta = torch.tanh(self.beta_proj(stage_embedding))
        modulated = lstm_out * (1.0 + self.rho_gamma * gamma.unsqueeze(1)) + self.rho_beta * beta.unsqueeze(1)

        query = self.query_proj(stage_embedding)
        score_feat = torch.tanh(
            self.hidden_proj(modulated)
            + self.curve_proj(curve_feat)
            + query.unsqueeze(1)
        )
        scores = torch.sum(query.unsqueeze(1) * score_feat, dim=-1)
        temperature = self.tau_min + F.softplus(self.tau_proj(stage_embedding))
        attn = F.softmax(scores / temperature, dim=1).unsqueeze(-1)
        context = torch.sum(attn * modulated, dim=1)
        aux = {
            "stage_embedding": stage_embedding,
            "gamma": gamma,
            "beta": beta,
            "curve_summary": curve_summary,
        }
        return context, attn, temperature, aux


class ResidualCRNN(nn.Module):
    def __init__(self, base_coeffs: torch.Tensor, attention_type: str):
        super().__init__()
        self.base_coeffs = base_coeffs
        self.attention_type = attention_type
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
        if attention_type == "multi_expert":
            self.attention = MultiExpertGatedCurveAttention(hidden_dim=64)
        elif attention_type == "stage_branch":
            self.attention = StageAwareBranchAttention(hidden_dim=64)
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")
        self.phy = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU())
        self.fc_soh = nn.Sequential(nn.Linear(72, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a, b, c, d = self.base_coeffs
        base_soh = a * (p**3) + b * (p**2) + c * p + d
        out = self.cnn(x.unsqueeze(1)).permute(0, 2, 1)
        lstm_out, _ = self.lstm(out)
        degradation_state = torch.cat([p, base_soh, 1.0 - base_soh], dim=1)
        temporal_feat, _, _, _ = self.attention(lstm_out, x, degradation_state)
        fused_features = torch.cat((temporal_feat, self.phy(p)), dim=1)
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


def macro_rmse(y_true: np.ndarray, y_pred: np.ndarray, bidx: np.ndarray) -> float:
    vals = [rmse(y_true[bidx == b], y_pred[bidx == b]) for b in np.unique(bidx) if np.any(bidx == b)]
    return float(np.mean(vals)) if vals else float("nan")


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[Dict[str, float], pd.DataFrame]:
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
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
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

    per_battery = pd.DataFrame(
        [
            {
                "battery_index": int(b),
                "rmse": rmse(y_true[bidx == b], y_pred[bidx == b]),
                "mae": mae(y_true[bidx == b], y_pred[bidx == b]),
                "n_points": int(np.sum(bidx == b)),
            }
            for b in np.unique(bidx)
        ]
    )
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
    model = ResidualCRNN(base_tensor, attention_type=variant["attention_type"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_mse = nn.MSELoss()
    criterion_mmd = LifetimeAdaptiveMMD().to(device)

    best_rmse, best_epoch = float("inf"), -1
    history_rows = []

    print(f"\n{'=' * 72}", flush=True)
    print(f"STARTING: {model_name} | seed={seed} | {variant['attention_type']}", flush=True)
    print(f"{'=' * 72}", flush=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        if epoch == LR_DECAY_EPOCH + 1:
            for group in optimizer.param_groups:
                group["lr"] = LEARNING_RATE * LR_DECAY_FACTOR

        if epoch <= 20:
            mmd_w, phase = 0.0, "Warmup"
        elif epoch <= 120:
            mmd_w, phase = TARGET_MMD_WEIGHT, "MMD"
        else:
            mmd_w, phase = 0.0, "Sprint"

        losses, mse_losses, mmd_losses = [], [], []
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
            loss_mmd = criterion_mmd(feat_s, feat_t, base_s, base_t) if mmd_w > 0 else torch.tensor(0.0, device=device)
            loss = loss_mse + mmd_w * loss_mmd
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            mse_losses.append(float(loss_mse.item()))
            mmd_losses.append(float(loss_mmd.item()))

        ta_metrics, _ = evaluate(model, loader_ta_eval, device)
        if ta_metrics["pooled_rmse"] < best_rmse:
            best_rmse, best_epoch = ta_metrics["pooled_rmse"], epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "seed": seed,
                    "variant": variant,
                    "base_coeffs": coeffs,
                    "target_adapt_metrics": ta_metrics,
                    "data_protocol": DATA_PROTOCOL,
                },
                checkpoint_path,
            )

        history_rows.append(
            {
                "epoch": epoch,
                "stage": phase,
                "train_loss": float(np.mean(losses)),
                "train_mse": float(np.mean(mse_losses)),
                "train_mmd": float(np.mean(mmd_losses)),
                "mmd_weight": float(mmd_w),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "ta_rmse": ta_metrics["pooled_rmse"],
                "ta_macro_rmse": ta_metrics["macro_rmse"],
                "ta_mae": ta_metrics["mae"],
                "best_epoch": best_epoch,
                "best_rmse": best_rmse,
            }
        )
        pd.DataFrame(history_rows).to_csv(history_csv, index=False, encoding="utf-8-sig")
        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(
                f"  {model_name} ep={epoch:03d} {phase:6s} "
                f"loss={np.mean(losses):.6f} ta_rmse={ta_metrics['pooled_rmse']:.5f} "
                f"best={best_rmse:.5f}@{best_epoch}",
                flush=True,
            )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    ta_metrics, ta_cells = evaluate(model, loader_ta_eval, device)
    tt_metrics, tt_cells = evaluate(model, loader_tt_eval, device)
    elapsed = time.time() - start_time

    result = {
        "model": model_name,
        "seed": seed,
        "variant": variant["model"],
        "attention_type": variant["attention_type"],
        "description": variant["description"],
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
        "elapsed_min": round(elapsed / 60, 1),
    }

    with pd.ExcelWriter(run_dir / f"{model_name}_results.xlsx", engine="openpyxl") as writer:
        pd.DataFrame([result]).to_excel(writer, index=False, sheet_name="summary")
        tt_cells.to_excel(writer, index=False, sheet_name="target_test_cells")
        ta_cells.to_excel(writer, index=False, sheet_name="target_adapt_cells")
        pd.DataFrame(history_rows).to_excel(writer, index=False, sheet_name="history")

    best_json.write_text(
        json.dumps(
            {
                "result": result,
                "target_test_cells": tt_cells.to_dict(orient="records"),
                "target_adapt_cells": ta_cells.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"  DONE: {model_name} | tt_rmse={tt_metrics['pooled_rmse']:.6f} | "
        f"late={tt_metrics['late_rmse']:.6f} | {elapsed / 60:.1f}min",
        flush=True,
    )
    del model, optimizer, criterion_mmd
    torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run advanced attention candidate SOH training.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[v["model"] for v in VARIANTS],
        choices=[v["model"] for v in VARIANTS],
    )
    return parser.parse_args()


def main() -> None:
    global EPOCHS, LOG_EVERY
    args = parse_args()
    EPOCHS = args.epochs
    LOG_EVERY = args.log_every
    selected_variants = [v for v in VARIANTS if v["model"] in set(args.variants)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Output dir: {OUT_DIR}", flush=True)
    print(f"Data dir: {DATA_DIR}", flush=True)
    print(f"Seeds: {args.seeds}", flush=True)
    print(f"Epochs: {EPOCHS}", flush=True)
    print(f"Variants: {[v['model'] for v in selected_variants]}", flush=True)

    data = {
        "source": load_split("source", LABEL_MODE),
        "target_adapt": load_split("target_adapt", LABEL_MODE),
        "target_test": load_split("target_test", LABEL_MODE),
    }
    _, s_y, s_p, _, _ = data["source"]
    coeffs = np.polyfit((s_p.flatten() / GLOBAL_MAX_CYCLE), s_y.flatten(), 3)
    print(f"Base coeffs: {coeffs}", flush=True)

    all_results = []
    for seed in args.seeds:
        for variant in selected_variants:
            result = train_variant(variant, seed, data, coeffs, device)
            all_results.append(result)

    summary_df = pd.DataFrame(all_results)
    summary_path = OUT_DIR / "advanced_attention_candidates_v1_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    summary_df.to_excel(OUT_DIR / "advanced_attention_candidates_v1_summary.xlsx", index=False)
    (OUT_DIR / "advanced_attention_candidates_v1_summary.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    aggregate_rows = []
    for variant_name, group in summary_df.groupby("variant"):
        aggregate_rows.append(
            {
                "variant": variant_name,
                "n_seeds": int(group["seed"].nunique()),
                "mean_rmse": float(group["target_test_rmse"].mean()),
                "std_rmse": float(group["target_test_rmse"].std(ddof=1)),
                "mean_macro_rmse": float(group["target_test_macro_rmse"].mean()),
                "mean_mae": float(group["target_test_mae"].mean()),
                "mean_late_rmse": float(group["target_test_late_rmse"].mean()),
            }
        )
    aggregate_df = pd.DataFrame(aggregate_rows)
    aggregate_df.to_csv(OUT_DIR / "advanced_attention_candidates_v1_aggregate.csv", index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 72}", flush=True)
    print("MULTI-SEED SUMMARY", flush=True)
    print(summary_df.to_string(index=False), flush=True)
    print("\nAGGREGATE", flush=True)
    print(aggregate_df.to_string(index=False), flush=True)
    print(f"\nSummary saved to: {summary_path}", flush=True)
    print(f"{'=' * 72}", flush=True)


if __name__ == "__main__":
    main()
