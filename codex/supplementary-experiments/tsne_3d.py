"""
Generate 3D t-SNE visualization for 2D vs 3D comparison.
"""
import sys, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ROOT = Path(r"C:\Users\ROG\Desktop\paper")
CODEX_DIR = ROOT / "codex"
DATA_DIR = ROOT / "PYthon训练代码" / "processed_clean"
OUT_DIR = CODEX_DIR / "supplementary-experiments" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
GLOBAL_MAX_CYCLE = 3000.0

sys.path.insert(0, str(CODEX_DIR))
from run_ablation_suite_v1 import HybridDataset, load_split

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2), nn.Tanh(), nn.Linear(hidden_dim//2, 1),
        )
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load the final-retained-seed42 model
    ckpt_path = CODEX_DIR / 'final-retained-v1' / 'final-retained-v1_best_by_target_adapt.pt'
    if not ckpt_path.exists():
        print("Checkpoint not found!")
        return

    # Load data
    s_x, s_y, s_p, s_bidx, s_batch = load_split("source", "nominal")
    ta_x, ta_y, ta_p, ta_bidx, ta_batch = load_split("target_adapt", "nominal")

    np.random.seed(42)
    n_source = 3000
    n_target = 3000
    idx_s = np.random.choice(len(s_x), min(n_source, len(s_x)), replace=False)
    idx_t = np.random.choice(len(ta_x), min(n_target, len(ta_x)), replace=False)

    x_sub = np.concatenate([s_x[idx_s], ta_x[idx_t]])
    y_sub = np.concatenate([s_y[idx_s], ta_y[idx_t]])
    p_sub = np.concatenate([s_p[idx_s], ta_p[idx_t]])

    loader = DataLoader(torch.utils.data.TensorDataset(
        torch.from_numpy(x_sub), torch.from_numpy(y_sub),
        torch.from_numpy(p_sub)), batch_size=1024, shuffle=False)

    # Fit cubic baseline
    coeffs = np.polyfit((s_p.flatten() / GLOBAL_MAX_CYCLE), s_y.flatten(), 3)
    base_coeffs = torch.tensor(coeffs, dtype=torch.float32, device=device)

    # Load model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ResidualCRNN(base_coeffs, use_physical_base=True, use_attention=True).to(device)
    sd = ckpt['model_state_dict']
    rename_map = {}
    for k in sd.keys():
        if 'attention.attention' in k:
            rename_map[k] = k.replace('attention.attention', 'attention.attn')
    for old_k, new_k in rename_map.items():
        sd[new_k] = sd.pop(old_k)
    model.load_state_dict(sd, strict=False)

    # Extract features
    print("Extracting features...")
    features = []
    with torch.no_grad():
        for x, y, p in loader:
            x, p = x.float().to(device), p.float().to(device)
            _, feat, _ = model(x, p)
            features.append(feat.cpu().numpy())
    features = np.concatenate(features)
    print(f"Features shape: {features.shape}")

    # Run both 2D and 3D t-SNE
    print("Running 2D t-SNE...")
    tsne2d = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=300)
    emb2d = tsne2d.fit_transform(features)

    print("Running 3D t-SNE...")
    tsne3d = TSNE(n_components=3, random_state=42, perplexity=30, max_iter=300)
    emb3d = tsne3d.fit_transform(features)

    # ---- 2D comparison plot (final look) ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    # Left: domain
    axes[0].scatter(emb2d[:n_source, 0], emb2d[:n_source, 1], c='#2196F3', s=8, alpha=0.5, label='Source')
    axes[0].scatter(emb2d[n_source:, 0], emb2d[n_source:, 1], c='#FF5722', s=8, alpha=0.5, label='Target_adapt')
    axes[0].set_title('2D t-SNE (by Domain)', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].set_xlabel('t-SNE dim 1'); axes[0].set_ylabel('t-SNE dim 2')
    # Right: SOH
    soh = y_sub.flatten()
    stages = np.digitize(soh, bins=[0.85, 0.90, 0.95])
    colors_soh = ['#4CAF50', '#FFC107', '#FF9800', '#f44336']
    labels_soh = ['SOH>0.95', '0.90<SOH≤0.95', '0.85<SOH≤0.90', 'SOH≤0.85']
    for s in range(4):
        mask = stages == s
        if np.any(mask):
            axes[1].scatter(emb2d[mask, 0], emb2d[mask, 1], c=colors_soh[s], s=8, alpha=0.5, label=labels_soh[s])
    axes[1].set_title('2D t-SNE (by SOH Level)', fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].set_xlabel('t-SNE dim 1'); axes[1].set_ylabel('t-SNE dim 2')
    plt.tight_layout()
    plt.savefig(OUT_DIR / "tsne_2d_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: tsne_2d_comparison.png")

    # ---- 3D plot: domain view (two angles) ----
    fig = plt.figure(figsize=(16, 7))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(emb3d[:n_source, 0], emb3d[:n_source, 1], emb3d[:n_source, 2],
                c='#2196F3', s=6, alpha=0.4, label='Source')
    ax1.scatter(emb3d[n_source:, 0], emb3d[n_source:, 1], emb3d[n_source:, 2],
                c='#FF5722', s=6, alpha=0.4, label='Target_adapt')
    ax1.set_title('3D t-SNE — View 1 (Domain)', fontsize=13)
    ax1.view_init(elev=25, azim=45)
    ax1.legend(fontsize=10)
    ax1.set_xlabel('Dim 1'); ax1.set_ylabel('Dim 2'); ax1.set_zlabel('Dim 3')

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(emb3d[:n_source, 0], emb3d[:n_source, 1], emb3d[:n_source, 2],
                c='#2196F3', s=6, alpha=0.4, label='Source')
    ax2.scatter(emb3d[n_source:, 0], emb3d[n_source:, 1], emb3d[n_source:, 2],
                c='#FF5722', s=6, alpha=0.4, label='Target_adapt')
    ax2.set_title('3D t-SNE — View 2 (Domain)', fontsize=13)
    ax2.view_init(elev=10, azim=-60)
    ax2.legend(fontsize=10)
    ax2.set_xlabel('Dim 1'); ax2.set_ylabel('Dim 2'); ax2.set_zlabel('Dim 3')
    plt.tight_layout()
    plt.savefig(OUT_DIR / "tsne_3d_domain.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: tsne_3d_domain.png")

    # ---- 3D plot: SOH view ----
    fig = plt.figure(figsize=(16, 7))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    for s in range(4):
        mask = stages == s
        if np.any(mask):
            ax1.scatter(emb3d[mask, 0], emb3d[mask, 1], emb3d[mask, 2],
                        c=colors_soh[s], s=6, alpha=0.4, label=labels_soh[s])
    ax1.set_title('3D t-SNE — View 1 (SOH)', fontsize=13)
    ax1.view_init(elev=25, azim=45)
    ax1.legend(fontsize=10)
    ax1.set_xlabel('Dim 1'); ax1.set_ylabel('Dim 2'); ax1.set_zlabel('Dim 3')

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    for s in range(4):
        mask = stages == s
        if np.any(mask):
            ax2.scatter(emb3d[mask, 0], emb3d[mask, 1], emb3d[mask, 2],
                        c=colors_soh[s], s=6, alpha=0.4, label=labels_soh[s])
    ax2.set_title('3D t-SNE — View 2 (SOH)', fontsize=13)
    ax2.view_init(elev=10, azim=-60)
    ax2.legend(fontsize=10)
    ax2.set_xlabel('Dim 1'); ax2.set_ylabel('Dim 2'); ax2.set_zlabel('Dim 3')
    plt.tight_layout()
    plt.savefig(OUT_DIR / "tsne_3d_soh.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: tsne_3d_soh.png")

    print(f"\nAll 3D plots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
