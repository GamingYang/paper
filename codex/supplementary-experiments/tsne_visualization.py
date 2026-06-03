"""
t-SNE feature visualization script.
Extracts features from trained model checkpoint and generates t-SNE plots.
Compares source vs target_adapt distribution and different degradation stages.

Usage: KMP_DUPLICATE_LIB_OK=TRUE python tsne_visualization.py
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(r"C:\Users\ROG\Desktop\paper")
CODEX_DIR = ROOT / "codex"
DATA_DIR = ROOT / "PYthon训练代码" / "processed_clean"
OUT_DIR = CODEX_DIR / "supplementary-experiments" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_MAX_CYCLE = 3000.0

sys.path.insert(0, str(CODEX_DIR))

# Use the same model classes from the training script
from run_ablation_suite_v1 import HybridDataset, load_split, LifetimeAdaptiveMMD

class TemporalAttention(nn.Module):
    """Matches naming used in multi-seed training script (self.attn)."""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, 1),
        )
    def forward(self, lstm_out):
        w = F.softmax(self.attn(lstm_out), dim=1)
        return torch.sum(w * lstm_out, dim=1), w

class ResidualCRNN(nn.Module):
    """Matches the checkpoint naming from run_multi_seed.py."""
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


def extract_features(model, loader, device):
    """Extract fused features from model for t-SNE."""
    model.eval()
    all_features, all_labels, all_domains, all_soh = [], [], [], []
    with torch.no_grad():
        for x, y, p, b in loader:
            x, p = x.float().to(device), p.float().to(device)
            _, features, base_soh = model(x, p)
            all_features.append(features.cpu().numpy())
            all_labels.append(y.numpy().reshape(-1))
            all_soh.append(base_soh.cpu().numpy().reshape(-1))
            all_domains.extend(['source' if np.any(b.numpy() <= 8) else 'target'] * len(y))
    return np.concatenate(all_features), np.concatenate(all_labels), \
           np.array(all_domains), np.concatenate(all_soh)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # We need to use final-retained-seed7 which just finished, or the seed=42 model
    # Since seed=42 is complete, let's use no-physical-base-seed7 for comparison
    model_paths = []

    # Check which checkpoints are available
    for ckpt_dir in sorted(OUT_DIR.parent.glob("*-seed*")):
        ckpt_file = ckpt_dir / f"{ckpt_dir.name}_best.pt"
        if ckpt_file.exists():
            model_paths.append((ckpt_dir.name, ckpt_file))
        else:
            # Try the original seed=42 in codex root
            pass

    # Also check the original codex directories
    orig_checkpoints = {
        'final-retained-seed42': CODEX_DIR / 'final-retained-v1' / 'final-retained-v1_best_by_target_adapt.pt',
        'no-physical-base-seed42': CODEX_DIR / 'no-physical-base-v1' / 'no-physical-base-v1_best_by_target_adapt.pt',
        'no-mmd-seed42': CODEX_DIR / 'no-mmd-v1' / 'no-mmd-v1_best_by_target_adapt.pt',
        'no-attention-seed42': CODEX_DIR / 'no-attention-v1' / 'no-attention-v1_best_by_target_adapt.pt',
    }

    # Also check supplementary-experiments for new seeds
    for name, path in list(orig_checkpoints.items()):
        if path.exists():
            model_paths.append((name, path))

    for ckpt_dir in OUT_DIR.parent.glob("*-seed*"):
        ckpt_file = ckpt_dir / f"{ckpt_dir.name}_best.pt"
        if ckpt_file.exists():
            name = ckpt_dir.name
            if (name, ckpt_file) not in model_paths:
                model_paths.append((name, ckpt_file))

    print(f"Found {len(model_paths)} checkpoints:")
    for name, path in model_paths:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        variant = ckpt.get('variant', {})
        desc = variant.get('model', name)
        print(f"  {name} (epoch={ckpt.get('epoch','?')})")

    # Load data for feature extraction
    s_x, s_y, s_p, s_bidx, s_batch = load_split("source", "nominal")
    ta_x, ta_y, ta_p, ta_bidx, ta_batch = load_split("target_adapt", "nominal")

    # Subsample for t-SNE (feature extraction on full data takes too much memory)
    np.random.seed(42)
    n_source = 5000  # subsample source
    n_target = 5000  # subsample target

    idx_s = np.random.choice(len(s_x), min(n_source, len(s_x)), replace=False)
    idx_t = np.random.choice(len(ta_x), min(n_target, len(ta_x)), replace=False)

    x_sub = np.concatenate([s_x[idx_s], ta_x[idx_t]])
    y_sub = np.concatenate([s_y[idx_s], ta_y[idx_t]])
    p_sub = np.concatenate([s_p[idx_s], ta_p[idx_t]])
    bidx_sub = np.concatenate([s_bidx[idx_s], ta_bidx[idx_t]])
    batch_sub = np.concatenate([s_batch[idx_s], ta_batch[idx_t]])
    domain_colors = ['#2196F3'] * len(idx_s) + ['#FF5722'] * len(idx_t)  # blue=source, red=target

    loader = DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(x_sub), torch.from_numpy(y_sub),
            torch.from_numpy(p_sub), torch.from_numpy(bidx_sub)),
        batch_size=1024, shuffle=False
    )

    # Fit baseline coefficients
    coeffs = np.polyfit((s_p.flatten() / GLOBAL_MAX_CYCLE), s_y.flatten(), 3)
    base_coeffs = torch.tensor(coeffs, dtype=torch.float32, device=device)

    # Feature extraction for each model variant
    for model_name, ckpt_path in model_paths:
        # Determine model config from checkpoint or name
        if 'no-physical-base' in model_name or 'no_physical' in model_name.lower():
            use_phy, use_attn = False, True
        elif 'no-mmd' in model_name or 'no_mmd' in model_name.lower():
            use_phy, use_attn = True, True
        elif 'no-attention' in model_name or 'no_attention' in model_name.lower():
            use_phy, use_attn = True, False
        else:
            use_phy, use_attn = True, True

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = ResidualCRNN(base_coeffs, use_physical_base=use_phy, use_attention=use_attn).to(device)

        # Handle different key naming between training scripts
        sd = ckpt['model_state_dict']
        # Check if there's a key mismatch and fix it
        has_attn_keys = any('attention.attn' in k for k in sd.keys())
        if has_attn_keys:
            # Our TemporalAttention uses self.attn, correct already
            pass
        else:
            # Check if keys use 'attention.attention' (original naming)
            rename_map = {}
            for k in sd.keys():
                if 'attention.attention' in k:
                    rename_map[k] = k.replace('attention.attention', 'attention.attn')
            if rename_map:
                for old_k, new_k in rename_map.items():
                    sd[new_k] = sd.pop(old_k)

        model.load_state_dict(sd, strict=False)

        # Extract features
        print(f"\nExtracting features from {model_name}...")
        features = []
        with torch.no_grad():
            for x, y, p, b in loader:
                x, p = x.float().to(device), p.float().to(device)
                _, feat, _ = model(x, p)
                features.append(feat.cpu().numpy())
        features = np.concatenate(features)
        print(f"  Features shape: {features.shape}")

        # t-SNE
        print(f"  Running t-SNE (n_samples={len(features)})...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=300)
        emb = tsne.fit_transform(features)

        # Plot domain colors
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Left: domain
        ax = axes[0]
        ax.scatter(emb[:len(idx_s), 0], emb[:len(idx_s), 1], c='#2196F3', s=8, alpha=0.5, label='Source (Batch 1-8)')
        ax.scatter(emb[len(idx_s):, 0], emb[len(idx_s):, 1], c='#FF5722', s=8, alpha=0.5, label='Target_adapt (Batch 9)')
        ax.set_title(f'{model_name} | Feature Space (by Domain)', fontsize=13)
        ax.legend(fontsize=10)
        ax.set_xlabel('t-SNE dim 1')
        ax.set_ylabel('t-SNE dim 2')

        # Right: degradation stage
        ax = axes[1]
        # Divide SOH into stages
        soh = y_sub.flatten()
        stages = np.digitize(soh, bins=[0.85, 0.90, 0.95])  # categories: <0.85, 0.85-0.90, 0.90-0.95, >0.95
        stage_colors = ['#4CAF50', '#FFC107', '#FF9800', '#f44336']
        stage_labels = ['SOH>0.95', '0.90<SOH≤0.95', '0.85<SOH≤0.90', 'SOH≤0.85']
        for s in range(4):
            mask = stages == s
            if np.any(mask):
                ax.scatter(emb[mask, 0], emb[mask, 1], c=stage_colors[s], s=8, alpha=0.5, label=stage_labels[s])
        ax.set_title(f'{model_name} | Feature Space (by SOH Level)', fontsize=13)
        ax.legend(fontsize=10)
        ax.set_xlabel('t-SNE dim 1')
        ax.set_ylabel('t-SNE dim 2')

        plt.tight_layout()
        fname = f"tsne_{model_name.replace('-', '_')}.png"
        plt.savefig(OUT_DIR / fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {OUT_DIR / fname}")

        del model, features, emb
        torch.cuda.empty_cache()

    print(f"\nAll t-SNE plots saved to {OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
