"""
SOH MMD representation/domain diagnostic.

This script compares source Batch 1-8 and Batch 9 target-development fused
features from retained SOH checkpoints. Batch 10 is deliberately not loaded or
used: this is a representation diagnostic, not a model-selection procedure.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


ROOT = Path(r"C:\Users\ROG\Desktop\paper")
CODEX_DIR = ROOT / "codex"
OUT_DIR = CODEX_DIR / "soh-mmd-representation-diagnostic-v1"
FIG_DIR = OUT_DIR / "figures"

RUN_NAME = "soh_mmd_representation_diagnostic_v1"
SUMMARY_XLSX = OUT_DIR / f"{RUN_NAME}_summary.xlsx"
REPORT_MD = OUT_DIR / f"{RUN_NAME}_report.md"

LABEL_MODE = "nominal"
DATA_PROTOCOL = "processed_clean_fixed_split"
GLOBAL_MAX_CYCLE = 3000.0
SEED = 42
BATCH_SIZE = 2048
FEATURE_DISTANCE_PER_DOMAIN = 3000
DOMAIN_CLASSIFIER_PER_DOMAIN = 10000
PCA_PER_DOMAIN = 1500
SOH_BIN_PER_DOMAIN = 1500
MMD_KERNEL_MUL = 2.0
MMD_KERNEL_NUM = 5
MMD_CHUNK = 512


@dataclass(frozen=True)
class ModelSpec:
    model: str
    checkpoint_path: Path
    use_physical_base: bool
    use_attention: bool
    comparison_role: str


MODEL_SPECS = [
    ModelSpec(
        model="final-retained-v1",
        checkpoint_path=CODEX_DIR / "final-retained-v1" / "final-retained-v1_best_by_target_adapt.pt",
        use_physical_base=True,
        use_attention=True,
        comparison_role="required_final_retained",
    ),
    ModelSpec(
        model="no-mmd-v1",
        checkpoint_path=CODEX_DIR / "no-mmd-v1" / "no-mmd-v1_best_by_target_adapt.pt",
        use_physical_base=True,
        use_attention=True,
        comparison_role="required_without_mmd",
    ),
    ModelSpec(
        model="no-physical-base-v1",
        checkpoint_path=CODEX_DIR / "no-physical-base-v1" / "no-physical-base-v1_best_by_target_adapt.pt",
        use_physical_base=False,
        use_attention=True,
        comparison_role="optional_context",
    ),
    ModelSpec(
        model="no-attention-v1",
        checkpoint_path=CODEX_DIR / "no-attention-v1" / "no-attention-v1_best_by_target_adapt.pt",
        use_physical_base=True,
        use_attention=False,
        comparison_role="optional_context",
    ),
]

SOH_BINS = [
    ("soh_ge_0p95", 0.95, math.inf),
    ("soh_0p90_0p95", 0.90, 0.95),
    ("soh_0p85_0p90", 0.85, 0.90),
    ("soh_lt_0p85", -math.inf, 0.85),
]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def find_processed_clean(root: Path) -> Path:
    candidates = []
    for child in root.iterdir():
        candidate = child / "processed_clean"
        if child.is_dir() and child.name.lower().startswith("python"):
            candidates.append(candidate)
    candidates.extend([root / "processed_clean", root / "PYthon_training_code" / "processed_clean"])
    for path in candidates:
        if (path / "clean_source_x.npy").exists() and (path / "clean_target_adapt_x.npy").exists():
            return path
    raise FileNotFoundError("Cannot find processed_clean data directory from project root.")


DATA_DIR = find_processed_clean(ROOT)


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
    def __init__(self, base_coeffs: torch.Tensor, use_physical_base: bool, use_attention: bool):
        super().__init__()
        self.base_coeffs = base_coeffs
        self.use_physical_base = use_physical_base
        self.use_attention = use_attention
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
        self.attention = TemporalAttention(hidden_dim=64) if use_attention else None
        self.phy = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU())
        self.fc_soh = nn.Sequential(nn.Linear(72, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_physical_base:
            a, b, c, d = self.base_coeffs
            base_soh = a * (p**3) + b * (p**2) + c * p + d
        else:
            base_soh = torch.zeros((x.size(0), 1), device=x.device, dtype=x.dtype)
        out = self.cnn(x.unsqueeze(1)).permute(0, 2, 1)
        lstm_out, _ = self.lstm(out)
        if self.use_attention:
            temporal_feat, _ = self.attention(lstm_out)
        else:
            temporal_feat = lstm_out[:, -1, :]
        fused_features = torch.cat((temporal_feat, self.phy(p)), dim=1)
        residual_soh = self.fc_soh(fused_features)
        pred_soh = base_soh + residual_soh if self.use_physical_base else residual_soh
        return pred_soh, fused_features, base_soh


def deterministic_indices(n: int, k: int, seed: int, salt: int = 0) -> np.ndarray:
    if n <= k:
        return np.arange(n)
    rng = np.random.default_rng(seed + salt)
    return np.sort(rng.choice(n, size=k, replace=False))


def deterministic_balanced_sample(
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_soh: np.ndarray,
    target_soh: np.ndarray,
    per_domain: int,
    seed: int,
    salt: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = min(per_domain, len(source_features), len(target_features))
    idx_s = deterministic_indices(len(source_features), n, seed, salt=salt)
    idx_t = deterministic_indices(len(target_features), n, seed, salt=salt + 97)
    return source_features[idx_s], target_features[idx_t], source_soh[idx_s], target_soh[idx_t]


def zscore_pair(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    combined = np.vstack([source, target])
    mean = combined.mean(axis=0, keepdims=True)
    std = combined.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return ((source - mean) / std).astype(np.float32), ((target - mean) / std).astype(np.float32)


def squared_distances_chunk(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cdist(a, b, p=2.0).pow(2)


def mean_offdiag_sqdist(x: np.ndarray, chunk: int = MMD_CHUNK) -> float:
    xt = torch.from_numpy(x.astype(np.float32))
    n = int(xt.shape[0])
    total = 0.0
    for start in range(0, n, chunk):
        dist = squared_distances_chunk(xt[start : start + chunk], xt)
        total += float(dist.sum().item())
    return total / max(n * n - n, 1)


def kernel_mean(a: np.ndarray, b: np.ndarray, bandwidths: Iterable[float], chunk: int = MMD_CHUNK) -> float:
    at = torch.from_numpy(a.astype(np.float32))
    bt = torch.from_numpy(b.astype(np.float32))
    bandwidth_list = [max(float(bw), 1e-8) for bw in bandwidths]
    total = 0.0
    count = int(at.shape[0]) * int(bt.shape[0])
    for start in range(0, int(at.shape[0]), chunk):
        dist = squared_distances_chunk(at[start : start + chunk], bt)
        ksum = torch.zeros_like(dist)
        for bw in bandwidth_list:
            ksum = ksum + torch.exp(-dist / bw)
        total += float(ksum.sum().item())
    return total / max(count, 1)


def multikernel_mmd_biased(
    source: np.ndarray,
    target: np.ndarray,
    kernel_mul: float = MMD_KERNEL_MUL,
    kernel_num: int = MMD_KERNEL_NUM,
) -> Dict[str, float]:
    combined = np.vstack([source, target]).astype(np.float32)
    base_bandwidth = max(mean_offdiag_sqdist(combined), 1e-8)
    first_bandwidth = base_bandwidth / (kernel_mul ** (kernel_num // 2))
    bandwidths = [first_bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kxx = kernel_mean(source, source, bandwidths)
    kyy = kernel_mean(target, target, bandwidths)
    kxy = kernel_mean(source, target, bandwidths)
    return {
        "mmd2": float(kxx + kyy - 2.0 * kxy),
        "base_bandwidth_mean_offdiag_sqdist": float(base_bandwidth),
        "bandwidths": json.dumps([float(x) for x in bandwidths]),
    }


def covariance_frobenius(source: np.ndarray, target: np.ndarray) -> float:
    cs = np.cov(source, rowvar=False)
    ct = np.cov(target, rowvar=False)
    return float(np.linalg.norm(cs - ct, ord="fro"))


def compute_feature_distance_metrics(
    model: str,
    comparison_role: str,
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_soh: np.ndarray,
    target_soh: np.ndarray,
) -> Dict[str, object]:
    src, tgt, _, _ = deterministic_balanced_sample(
        source_features,
        target_features,
        source_soh,
        target_soh,
        FEATURE_DISTANCE_PER_DOMAIN,
        SEED,
        salt=11,
    )
    src_z, tgt_z = zscore_pair(src, tgt)
    raw_mmd = multikernel_mmd_biased(src, tgt)
    z_mmd = multikernel_mmd_biased(src_z, tgt_z)
    mean_gap = src_z.mean(axis=0) - tgt_z.mean(axis=0)
    row = {
        "model": model,
        "comparison_role": comparison_role,
        "source_samples_used": int(len(src)),
        "target_adapt_samples_used": int(len(tgt)),
        "feature_dim": int(src.shape[1]),
        "mmd2_rbf_multikernel_zscore": z_mmd["mmd2"],
        "mmd2_rbf_multikernel_raw": raw_mmd["mmd2"],
        "mmd_kernel_mul": MMD_KERNEL_MUL,
        "mmd_kernel_num": MMD_KERNEL_NUM,
        "mmd_estimator": "biased finite-sample estimator matching training form",
        "mmd_zscore_base_bandwidth": z_mmd["base_bandwidth_mean_offdiag_sqdist"],
        "mmd_zscore_bandwidths": z_mmd["bandwidths"],
        "mmd_raw_base_bandwidth": raw_mmd["base_bandwidth_mean_offdiag_sqdist"],
        "mmd_raw_bandwidths": raw_mmd["bandwidths"],
        "centroid_l2_zscore": float(np.linalg.norm(mean_gap)),
        "mean_abs_feature_mean_gap_zscore": float(np.mean(np.abs(mean_gap))),
        "linear_mmd2_zscore": float(np.sum(mean_gap**2)),
        "coral_cov_frobenius_zscore": covariance_frobenius(src_z, tgt_z),
    }
    return row


def compute_domain_classifier_metrics(
    model: str,
    comparison_role: str,
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_soh: np.ndarray,
    target_soh: np.ndarray,
) -> Dict[str, object]:
    src, tgt, _, _ = deterministic_balanced_sample(
        source_features,
        target_features,
        source_soh,
        target_soh,
        DOMAIN_CLASSIFIER_PER_DOMAIN,
        SEED,
        salt=23,
    )
    x = np.vstack([src, tgt])
    y = np.concatenate([np.zeros(len(src), dtype=np.int64), np.ones(len(tgt), dtype=np.int64)])
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=SEED,
        stratify=y,
    )
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED),
    )
    clf.fit(x_train, y_train)
    val_prob = clf.predict_proba(x_val)[:, 1]
    val_pred = (val_prob >= 0.5).astype(np.int64)
    return {
        "model": model,
        "comparison_role": comparison_role,
        "classifier": "StandardScaler + LogisticRegression(lbfgs)",
        "purpose": "diagnostic only; not used for checkpoint/model selection",
        "source_samples_used": int(len(src)),
        "target_adapt_samples_used": int(len(tgt)),
        "train_fraction": 0.70,
        "validation_fraction": 0.30,
        "validation_accuracy": float(accuracy_score(y_val, val_pred)),
        "validation_balanced_accuracy": float(balanced_accuracy_score(y_val, val_pred)),
        "validation_auc": float(roc_auc_score(y_val, val_prob)),
    }


def compute_soh_stratified_metrics(
    model: str,
    comparison_role: str,
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_soh: np.ndarray,
    target_soh: np.ndarray,
) -> List[Dict[str, object]]:
    rows = []
    for bin_name, low, high in SOH_BINS:
        source_mask = (source_soh >= low) & (source_soh < high)
        target_mask = (target_soh >= low) & (target_soh < high)
        if int(source_mask.sum()) < 20 or int(target_mask.sum()) < 20:
            rows.append(
                {
                    "model": model,
                    "comparison_role": comparison_role,
                    "soh_bin": bin_name,
                    "source_available": int(source_mask.sum()),
                    "target_adapt_available": int(target_mask.sum()),
                    "status": "skipped_insufficient_samples",
                }
            )
            continue
        src = source_features[source_mask]
        tgt = target_features[target_mask]
        s_soh = source_soh[source_mask]
        t_soh = target_soh[target_mask]
        src_sub, tgt_sub, _, _ = deterministic_balanced_sample(
            src,
            tgt,
            s_soh,
            t_soh,
            SOH_BIN_PER_DOMAIN,
            SEED,
            salt=101 + len(rows) * 13,
        )
        src_z, tgt_z = zscore_pair(src_sub, tgt_sub)
        z_mmd = multikernel_mmd_biased(src_z, tgt_z)
        mean_gap = src_z.mean(axis=0) - tgt_z.mean(axis=0)
        rows.append(
            {
                "model": model,
                "comparison_role": comparison_role,
                "soh_bin": bin_name,
                "source_available": int(source_mask.sum()),
                "target_adapt_available": int(target_mask.sum()),
                "source_samples_used": int(len(src_sub)),
                "target_adapt_samples_used": int(len(tgt_sub)),
                "mmd2_rbf_multikernel_zscore": z_mmd["mmd2"],
                "centroid_l2_zscore": float(np.linalg.norm(mean_gap)),
                "linear_mmd2_zscore": float(np.sum(mean_gap**2)),
                "coral_cov_frobenius_zscore": covariance_frobenius(src_z, tgt_z),
                "status": "computed",
            }
        )
    return rows


def load_model(spec: ModelSpec, device: torch.device) -> Tuple[ResidualCRNN, Dict]:
    if not spec.checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {spec.checkpoint_path}")
    checkpoint = torch.load(spec.checkpoint_path, map_location=device, weights_only=False)
    coeffs = checkpoint.get("base_coeffs")
    if coeffs is None:
        raise KeyError(f"Checkpoint has no base_coeffs: {spec.checkpoint_path}")
    base_coeffs = torch.tensor(coeffs, dtype=torch.float32, device=device)
    model = ResidualCRNN(
        base_coeffs=base_coeffs,
        use_physical_base=spec.use_physical_base,
        use_attention=spec.use_attention,
    ).to(device)
    state_dict = checkpoint["model_state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        allowed_missing = set()
        allowed_unexpected = set()
        if not spec.use_attention:
            allowed_missing = {key for key in missing if key.startswith("attention.")}
            allowed_unexpected = {key for key in unexpected if key.startswith("attention.")}
        if set(missing) - allowed_missing or set(unexpected) - allowed_unexpected:
            raise RuntimeError(
                f"State dict mismatch for {spec.model}; missing={missing}; unexpected={unexpected}"
            )
    model.eval()
    return model, checkpoint


def extract_features(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    bidx: np.ndarray,
    batch: np.ndarray,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    dataset = HybridDataset(x, y, p, bidx)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    features, preds, bases = [], [], []
    with torch.no_grad():
        for bx, _, bp, _ in loader:
            bx = bx.float().to(device)
            bp = bp.float().to(device)
            pred, feat, base = model(bx, bp)
            features.append(feat.detach().cpu().numpy().astype(np.float32))
            preds.append(pred.detach().cpu().numpy().reshape(-1).astype(np.float32))
            bases.append(base.detach().cpu().numpy().reshape(-1).astype(np.float32))
    return {
        "features": np.concatenate(features, axis=0),
        "y": y.reshape(-1).astype(np.float32),
        "p": p.reshape(-1).astype(np.float32),
        "battery_index": bidx.reshape(-1).astype(np.int64),
        "batch": batch.reshape(-1).astype(np.int64),
        "pred": np.concatenate(preds),
        "base_soh": np.concatenate(bases),
    }


def plot_bar_metrics(feature_df: pd.DataFrame, clf_df: pd.DataFrame) -> Path:
    required_models = ["final-retained-v1", "no-mmd-v1"]
    fdf = feature_df[feature_df["model"].isin(required_models)].set_index("model").loc[required_models]
    cdf = clf_df[clf_df["model"].isin(required_models)].set_index("model").loc[required_models]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    metrics = [
        ("mmd2_rbf_multikernel_zscore", fdf, "Feature MMD^2, z-scored"),
        ("centroid_l2_zscore", fdf, "Centroid gap, z-scored"),
        ("validation_auc", cdf, "Domain classifier AUC"),
        ("validation_accuracy", cdf, "Domain classifier accuracy"),
    ]
    colors = ["#2563eb", "#f97316"]
    for ax, (metric, df, title) in zip(axes.flat, metrics):
        vals = [float(df.loc[model, metric]) for model in required_models]
        ax.bar(["Final", "w/o MMD"], vals, color=colors, width=0.62)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for i, val in enumerate(vals):
            ax.text(i, val, f"{val:.4g}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Source vs Batch9 Target-Development Feature/Domain Diagnostics", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = FIG_DIR / "mmd_representation_domain_bars.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pca_comparison(extracted: Dict[str, Dict[str, Dict[str, np.ndarray]]]) -> Optional[Path]:
    model_names = [name for name in ["final-retained-v1", "no-mmd-v1"] if name in extracted]
    if len(model_names) != 2:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, model_name in zip(axes, model_names):
        source = extracted[model_name]["source"]
        target = extracted[model_name]["target_adapt"]
        src, tgt, _, _ = deterministic_balanced_sample(
            source["features"],
            target["features"],
            source["y"],
            target["y"],
            PCA_PER_DOMAIN,
            SEED,
            salt=211,
        )
        x = np.vstack(zscore_pair(src, tgt))
        domain = np.array(["Source B1-8"] * len(src) + ["Batch9 target-dev"] * len(tgt))
        emb = PCA(n_components=2, random_state=SEED).fit_transform(x)
        for label, color in [("Source B1-8", "#2563eb"), ("Batch9 target-dev", "#f97316")]:
            mask = domain == label
            ax.scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.35, color=color, label=label)
        ax.set_title(model_name)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.grid(alpha=0.18)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Compact PCA View of Fused SOH Features", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = FIG_DIR / "final_vs_no_mmd_pca_2d.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def split_summary(name: str, x: np.ndarray, y: np.ndarray, p: np.ndarray, bidx: np.ndarray, batch: np.ndarray) -> Dict:
    return {
        "split": name,
        "role": "source_training_domain" if name == "source" else "target_domain_development_adaptation",
        "samples": int(len(x)),
        "cells": int(len(np.unique(bidx))),
        "batches": ",".join(map(str, sorted(np.unique(batch).tolist()))),
        "x_shape": "x".join(map(str, x.shape)),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "cycle_min": float(np.min(p)),
        "cycle_max": float(np.max(p)),
    }


def relative_delta(final_value: float, baseline_value: float) -> float:
    if baseline_value == 0:
        return float("nan")
    return float((final_value - baseline_value) / baseline_value)


def build_report(
    feature_df: pd.DataFrame,
    clf_df: pd.DataFrame,
    config_df: pd.DataFrame,
    figure_paths: List[Path],
) -> None:
    f = feature_df.set_index("model")
    c = clf_df.set_index("model")
    final_mmd = float(f.loc["final-retained-v1", "mmd2_rbf_multikernel_zscore"])
    no_mmd = float(f.loc["no-mmd-v1", "mmd2_rbf_multikernel_zscore"])
    final_auc = float(c.loc["final-retained-v1", "validation_auc"])
    no_mmd_auc = float(c.loc["no-mmd-v1", "validation_auc"])
    final_acc = float(c.loc["final-retained-v1", "validation_accuracy"])
    no_mmd_acc = float(c.loc["no-mmd-v1", "validation_accuracy"])
    if final_mmd < no_mmd or final_auc < no_mmd_auc or final_acc < no_mmd_acc:
        interpretation = (
            "The retained model shows representation-level support for the MMD regularizer on at least one "
            "domain-mixing diagnostic. This should be written as auxiliary representation evidence, not as a "
            "dominant accuracy-driver claim."
        )
    else:
        interpretation = (
            "The MMD representation evidence is weak/mixed under these diagnostics. The paper should avoid "
            "claiming clear domain mixing support from this run."
        )
    figure_lines = "\n".join(f"- `{path}`" for path in figure_paths)
    config_lines = "\n".join(
        f"- {row['item']}: {row['value']}" for row in config_df[["item", "value"]].to_dict(orient="records")
    )
    text = f"""# SOH MMD Representation Diagnostic v1

## Protocol

This package diagnoses fused SOH representations using source Batch 1-8 and Batch 9 target-development/adaptation features only. Batch 10 is not loaded and is treated as an untouched blind-test set outside this diagnostic. The diagnostic is not used for checkpoint selection or hyperparameter tuning.

## Key final vs w/o MMD findings

| Metric | final-retained-v1 | no-mmd-v1 | Direction |
| --- | ---: | ---: | --- |
| RBF multikernel MMD^2, z-scored fused features | {final_mmd:.6f} | {no_mmd:.6f} | {relative_delta(final_mmd, no_mmd):+.1%} vs w/o MMD |
| Domain classifier AUC | {final_auc:.6f} | {no_mmd_auc:.6f} | {relative_delta(final_auc, no_mmd_auc):+.1%} vs w/o MMD |
| Domain classifier accuracy | {final_acc:.6f} | {no_mmd_acc:.6f} | {relative_delta(final_acc, no_mmd_acc):+.1%} vs w/o MMD |

## Interpretation

{interpretation}

Lower feature MMD or lower domain classifier separability suggests stronger source/Batch9 feature mixing. Because the known Batch10 RMSE difference between final retained and w/o MMD is small, this diagnostic should be presented cautiously as representation-level support or as mixed evidence, not as proof that MMD dominates prediction accuracy.

## Configuration

{config_lines}

## Figures

{figure_lines}
"""
    REPORT_MD.write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"RUN={RUN_NAME}", flush=True)
    print(f"python={sys.executable}", flush=True)
    print(f"device={device}", flush=True)
    print(f"data_dir={DATA_DIR}", flush=True)
    print("Batch10 is not loaded or used by this diagnostic.", flush=True)

    source = load_split("source", LABEL_MODE)
    target_adapt = load_split("target_adapt", LABEL_MODE)
    s_x, s_y, s_p, s_bidx, s_batch = source
    t_x, t_y, t_p, t_bidx, t_batch = target_adapt
    if set(np.unique(s_batch).tolist()) - set(range(1, 9)):
        raise RuntimeError(f"Unexpected source batch ids: {sorted(np.unique(s_batch).tolist())}")
    if set(np.unique(t_batch).tolist()) != {9}:
        raise RuntimeError(f"Unexpected target_adapt batch ids: {sorted(np.unique(t_batch).tolist())}")

    extracted: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    feature_rows, classifier_rows, stratified_rows = [], [], []
    checkpoint_rows = []
    for spec in MODEL_SPECS:
        if not spec.checkpoint_path.exists():
            print(f"SKIP missing optional checkpoint: {spec.checkpoint_path}", flush=True)
            continue
        print(f"Extracting {spec.model} from {spec.checkpoint_path}", flush=True)
        model, checkpoint = load_model(spec, device)
        extracted[spec.model] = {
            "source": extract_features(model, s_x, s_y, s_p, s_bidx, s_batch, device),
            "target_adapt": extract_features(model, t_x, t_y, t_p, t_bidx, t_batch, device),
        }
        source_feat = extracted[spec.model]["source"]["features"]
        target_feat = extracted[spec.model]["target_adapt"]["features"]
        source_soh = extracted[spec.model]["source"]["y"]
        target_soh = extracted[spec.model]["target_adapt"]["y"]
        feature_rows.append(
            compute_feature_distance_metrics(
                spec.model,
                spec.comparison_role,
                source_feat,
                target_feat,
                source_soh,
                target_soh,
            )
        )
        classifier_rows.append(
            compute_domain_classifier_metrics(
                spec.model,
                spec.comparison_role,
                source_feat,
                target_feat,
                source_soh,
                target_soh,
            )
        )
        stratified_rows.extend(
            compute_soh_stratified_metrics(
                spec.model,
                spec.comparison_role,
                source_feat,
                target_feat,
                source_soh,
                target_soh,
            )
        )
        checkpoint_rows.append(
            {
                "model": spec.model,
                "comparison_role": spec.comparison_role,
                "checkpoint_path": str(spec.checkpoint_path),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "selection_split": checkpoint.get("selection_split"),
                "use_physical_base": spec.use_physical_base,
                "use_attention": spec.use_attention,
                "feature_dim": int(source_feat.shape[1]),
                "source_feature_rows": int(source_feat.shape[0]),
                "target_adapt_feature_rows": int(target_feat.shape[0]),
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    feature_df = pd.DataFrame(feature_rows)
    classifier_df = pd.DataFrame(classifier_rows)
    stratified_df = pd.DataFrame(stratified_rows)
    counts_df = pd.DataFrame(
        [
            split_summary("source", s_x, s_y, s_p, s_bidx, s_batch),
            split_summary("target_adapt", t_x, t_y, t_p, t_bidx, t_batch),
            {
                "split": "target_test",
                "role": "blind_test_not_loaded_or_used",
                "samples": None,
                "cells": None,
                "batches": "10",
                "x_shape": "not_loaded",
                "y_min": None,
                "y_max": None,
                "cycle_min": None,
                "cycle_max": None,
            },
        ]
    )
    config_df = pd.DataFrame(
        [
            {"item": "run_name", "value": RUN_NAME},
            {"item": "created_at", "value": datetime.now().isoformat(timespec="seconds")},
            {"item": "project_root", "value": str(ROOT)},
            {"item": "output_dir", "value": str(OUT_DIR)},
            {"item": "data_dir", "value": str(DATA_DIR)},
            {"item": "label_mode", "value": LABEL_MODE},
            {"item": "data_protocol", "value": DATA_PROTOCOL},
            {"item": "seed", "value": SEED},
            {"item": "device", "value": str(device)},
            {"item": "batch10_policy", "value": "not loaded; blind-test reporting only outside this diagnostic"},
            {"item": "feature_distance_per_domain", "value": FEATURE_DISTANCE_PER_DOMAIN},
            {"item": "domain_classifier_per_domain", "value": DOMAIN_CLASSIFIER_PER_DOMAIN},
            {"item": "soh_bin_per_domain", "value": SOH_BIN_PER_DOMAIN},
            {
                "item": "mmd_rbf_multikernel_setup",
                "value": (
                    "biased MMD^2; Gaussian bandwidth base is mean off-diagonal squared distance "
                    f"on the deterministic balanced sample; kernel_mul={MMD_KERNEL_MUL}; "
                    f"kernel_num={MMD_KERNEL_NUM}; bandwidth_i=base/kernel_mul^(kernel_num//2)*kernel_mul^i"
                ),
            },
            {
                "item": "domain_classifier_setup",
                "value": "balanced deterministic source/Batch9 sample; 70/30 stratified split; StandardScaler + LogisticRegression",
            },
        ]
    )
    checkpoint_df = pd.DataFrame(checkpoint_rows)

    bar_path = plot_bar_metrics(feature_df, classifier_df)
    figure_paths = [bar_path]
    pca_path = plot_pca_comparison(extracted)
    if pca_path is not None:
        figure_paths.append(pca_path)

    with pd.ExcelWriter(SUMMARY_XLSX, engine="openpyxl") as writer:
        feature_df.to_excel(writer, sheet_name="feature_distance_metrics", index=False)
        classifier_df.to_excel(writer, sheet_name="domain_classifier_metrics", index=False)
        stratified_df.to_excel(writer, sheet_name="optional_soh_stratified_metrics", index=False)
        counts_df.to_excel(writer, sheet_name="sample_counts_and_config", index=False, startrow=0)
        config_df.to_excel(writer, sheet_name="sample_counts_and_config", index=False, startrow=len(counts_df) + 3)
        checkpoint_df.to_excel(writer, sheet_name="checkpoint_inventory", index=False)

    build_report(feature_df, classifier_df, config_df, figure_paths)
    print(f"SUMMARY_XLSX={SUMMARY_XLSX}", flush=True)
    print(f"REPORT_MD={REPORT_MD}", flush=True)
    print("FINAL_REQUIRED_COMPARISON", flush=True)
    required_feature = feature_df[feature_df["model"].isin(["final-retained-v1", "no-mmd-v1"])]
    required_clf = classifier_df[classifier_df["model"].isin(["final-retained-v1", "no-mmd-v1"])]
    print(required_feature.to_json(orient="records"), flush=True)
    print(required_clf.to_json(orient="records"), flush=True)


if __name__ == "__main__":
    main()
