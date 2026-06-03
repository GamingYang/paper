from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ============================================================
# Visual evaluation for SOH model trained on processed_clean
# - 2x4 plots for Batch 10 (target test)
# - Times New Roman style
# - 95% uncertainty band via MC Dropout
# - Absolute error bars on secondary axis
# - Per-battery RMSE table
# - pooled / macro metrics summary
# ============================================================

# ------------------------- Config -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GLOBAL_MAX_CYCLE = 3000.0
DATA_DIR = Path("./processed_clean")
CHECKPOINT_PATH = Path("./best_soh_nominal_processed_clean.pth")
OUTPUT_DIR = Path("./eval_processed_clean")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MC_DROPOUT_SAMPLES = 50   # increase to 100 for smoother uncertainty estimates
BATCH_SIZE = 1024
USE_SAVGOL = False        # keep False unless you explicitly want smoothed predicted curve

# color palette aligned to your uploaded plotting style
COLOR_TRUE_LINE = '#8B0000'   # dark red
COLOR_PRED_LINE = '#2874A6'   # blue
COLOR_CI_FILL = '#D6EAF8'     # light cyan band
COLOR_ERROR_BAR = '#1ABC9C'   # cyan/green error bar
COLOR_GRID = '#BFBFBF'

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.unicode_minus'] = False


# ------------------------- Model -------------------------
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, lstm_outputs: torch.Tensor):
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

    def forward(self, x: torch.Tensor, p: torch.Tensor):
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
        return pred_soh


# ------------------------- Helpers -------------------------
def load_checkpoint(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=DEVICE)
    return ckpt


def load_target_test(label_mode: str):
    x = np.load(DATA_DIR / 'clean_target_test_x.npy').astype(np.float32)
    p = np.load(DATA_DIR / 'clean_target_test_p.npy').astype(np.float32).reshape(-1)
    bidx = np.load(DATA_DIR / 'clean_target_test_battery_index.npy').astype(np.int64).reshape(-1)
    batch = np.load(DATA_DIR / 'clean_target_test_batch.npy').astype(np.int64).reshape(-1)
    if label_mode == 'relative':
        y = np.load(DATA_DIR / 'clean_target_test_y_relative.npy').astype(np.float32).reshape(-1)
    else:
        y = np.load(DATA_DIR / 'clean_target_test_y_nominal.npy').astype(np.float32).reshape(-1)

    meta = pd.read_csv(DATA_DIR / 'clean_target_test_meta.csv')
    return x, y, p, bidx, batch, meta


def load_source_for_base(label_mode: str):
    p = np.load(DATA_DIR / 'clean_source_p.npy').astype(np.float32).reshape(-1)
    if label_mode == 'relative':
        y = np.load(DATA_DIR / 'clean_source_y_relative.npy').astype(np.float32).reshape(-1)
    else:
        y = np.load(DATA_DIR / 'clean_source_y_nominal.npy').astype(np.float32).reshape(-1)
    return y, p


def enable_dropout_only(model: nn.Module):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(np.abs(y_pred - y_true)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    eps = 1e-8
    return float(np.mean(np.abs((y_pred - y_true) / np.maximum(np.abs(y_true), eps))) * 100.0)


def macro_rmse_by_battery(y_true: np.ndarray, y_pred: np.ndarray, battery_idx: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    battery_idx = np.asarray(battery_idx).reshape(-1)
    vals = []
    for b in np.unique(battery_idx):
        mask = battery_idx == b
        vals.append(rmse(y_true[mask], y_pred[mask]))
    return float(np.mean(vals))


def maybe_smooth(y: np.ndarray) -> np.ndarray:
    if not USE_SAVGOL:
        return y
    try:
        from scipy.signal import savgol_filter
        wl = min(51, len(y) // 2 * 2 + 1)
        if wl > 3:
            return savgol_filter(y, window_length=wl, polyorder=3)
    except Exception:
        pass
    return y


def mc_predict(model: nn.Module, x_np: np.ndarray, p_np: np.ndarray, batch_size: int = 1024, n_samples: int = 50):
    x_t = torch.from_numpy(x_np).float()
    p_t = torch.from_numpy((p_np / GLOBAL_MAX_CYCLE).reshape(-1, 1)).float()
    n = len(x_np)
    all_preds = []

    for _ in range(n_samples):
        enable_dropout_only(model)
        preds_this = []
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                xb = x_t[start:end].to(DEVICE)
                pb = p_t[start:end].to(DEVICE)
                out = model(xb, pb)
                preds_this.append(out.cpu().numpy().reshape(-1))
        all_preds.append(np.concatenate(preds_this, axis=0))

    pred_samples = np.stack(all_preds, axis=0)         # [T, N]
    pred_mean = pred_samples.mean(axis=0)
    pred_lower = np.percentile(pred_samples, 2.5, axis=0)
    pred_upper = np.percentile(pred_samples, 97.5, axis=0)
    pred_std = pred_samples.std(axis=0)
    return pred_mean, pred_lower, pred_upper, pred_std, pred_samples


def build_cell_id_map(meta: pd.DataFrame):
    # Use most common cell_id string per battery_index
    tmp = meta[['battery_index', 'cell_id']].drop_duplicates().copy()
    cell_map = dict(zip(tmp['battery_index'].astype(int), tmp['cell_id'].astype(str)))
    return cell_map


def plot_2x4(cycles, y_true, y_pred, y_lo, y_hi, battery_idx, cell_id_map, summary_text, save_path: Path):
    cycles = np.asarray(cycles).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    y_lo = np.asarray(y_lo).reshape(-1)
    y_hi = np.asarray(y_hi).reshape(-1)
    battery_idx = np.asarray(battery_idx).reshape(-1)
    unique_bats = np.unique(battery_idx)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=False)
    axes = axes.flatten()

    # unified legend handles
    legend_handles = [
        Line2D([0], [0], color=COLOR_TRUE_LINE, lw=2.0, label='Measured SOH'),
        Line2D([0], [0], color=COLOR_PRED_LINE, lw=2.0, label='Predicted SOH'),
        Patch(facecolor=COLOR_CI_FILL, edgecolor=COLOR_CI_FILL, alpha=0.7, label='95% uncertainty band'),
        Patch(facecolor=COLOR_ERROR_BAR, edgecolor=COLOR_ERROR_BAR, alpha=0.45, label='Abs. Error'),
    ]

    max_err = float(np.max(np.abs(y_true - y_pred)))
    err_ylim = max(0.05, min(0.20, max_err * 1.25))

    battery_rows = []
    for i, b in enumerate(unique_bats):
        ax = axes[i]
        ax2 = ax.twinx()
        mask = battery_idx == b
        x = cycles[mask]
        yt = y_true[mask]
        yp = maybe_smooth(y_pred[mask])
        ylow = y_lo[mask]
        yhigh = y_hi[mask]
        err = np.abs(yt - y_pred[mask])

        this_rmse = rmse(yt, y_pred[mask])
        this_mae = mae(yt, y_pred[mask])
        cid = cell_id_map.get(int(b), f'Battery-{int(b)}')
        battery_rows.append((cid, int(len(x)), this_rmse, this_mae))

        # error bars on right axis
        width = max(3.0, float(np.median(np.diff(x))) * 0.8) if len(x) > 1 else 5.0
        ax2.bar(x, err, width=width, color=COLOR_ERROR_BAR, alpha=0.45)
        ax2.set_ylim(0, err_ylim)
        ax2.set_ylabel('Abs. Error', fontsize=10)
        ax2.tick_params(axis='y', labelsize=9)
        ax2.spines['right'].set_linewidth(1.0)

        # main axis
        ax.fill_between(x, ylow, yhigh, color=COLOR_CI_FILL, alpha=0.7)
        ax.plot(x, yt, color=COLOR_TRUE_LINE, linewidth=2.0)
        ax.plot(x, yp, color=COLOR_PRED_LINE, linewidth=2.0)
        ax.set_title(f'{cid}\nRMSE = {this_rmse:.4f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Cycle', fontsize=11)
        ax.set_ylabel('SOH', fontsize=11)
        # auto y-range with small padding
        y_min = min(np.min(yt), np.min(ylow), np.min(yp))
        y_max = max(np.max(yt), np.max(yhigh), np.max(yp))
        pad = max(0.03, (y_max - y_min) * 0.10)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.35, color=COLOR_GRID)

    # hide unused panels if any
    for j in range(len(unique_bats), len(axes)):
        axes[j].axis('off')

    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=False, fontsize=12)
    fig.suptitle('Batch 10 SOH Prediction (2 × 4)\nTimes New Roman + 95% uncertainty band', fontsize=18, fontweight='bold', y=1.03)
    fig.text(0.5, 0.005, summary_text, ha='center', va='bottom', fontsize=11)
    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.95])
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return battery_rows


# ------------------------- Main -------------------------
def main():
    print(f'📂 DATA_DIR = {DATA_DIR.resolve()}')
    print(f'📦 CHECKPOINT = {CHECKPOINT_PATH.resolve()}')

    ckpt = load_checkpoint(CHECKPOINT_PATH)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        label_mode = ckpt.get('label_mode', 'nominal')
        base_coeffs = ckpt.get('base_coeffs', None)
        best_pooled_rmse = ckpt.get('best_pooled_rmse', None)
        best_macro_rmse = ckpt.get('best_macro_rmse', None)
        best_epoch = ckpt.get('best_epoch', None)  # not present in your current checkpoint
        train_config = ckpt.get('config', {})
    else:
        state_dict = ckpt
        label_mode = 'nominal'
        base_coeffs = None
        best_pooled_rmse = None
        best_macro_rmse = None
        best_epoch = None
        train_config = {}

    print(f'🏷️ label_mode = {label_mode}')

    # load data
    x, y, p, bidx, batch, meta = load_target_test(label_mode)
    cell_id_map = build_cell_id_map(meta)

    # fit / load base coeffs
    if base_coeffs is None:
        s_y, s_p = load_source_for_base(label_mode)
        coeffs = np.polyfit((s_p.flatten() / GLOBAL_MAX_CYCLE), s_y.flatten(), 3)
    else:
        coeffs = np.array(base_coeffs, dtype=np.float32)
    coeffs_t = torch.tensor(coeffs, dtype=torch.float32, device=DEVICE)
    print(f'📈 Base cubic coeffs = {coeffs}')

    # load model
    model = ResidualCRNN(coeffs_t).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    # MC dropout prediction
    print(f'🔁 Running MC Dropout inference | samples = {MC_DROPOUT_SAMPLES}')
    pred_mean, pred_lo, pred_hi, pred_std, pred_samples = mc_predict(
        model=model,
        x_np=x,
        p_np=p,
        batch_size=BATCH_SIZE,
        n_samples=MC_DROPOUT_SAMPLES,
    )

    pooled = rmse(y, pred_mean)
    macro = macro_rmse_by_battery(y, pred_mean, bidx)
    overall_mae = mae(y, pred_mean)
    overall_mape = mape(y, pred_mean)
    avg_band_width = float(np.mean(pred_hi - pred_lo))

    # per-battery table
    battery_records = []
    for b in np.unique(bidx):
        mask = bidx == b
        cid = cell_id_map.get(int(b), f'Battery-{int(b)}')
        battery_records.append({
            'battery_index': int(b),
            'cell_id': cid,
            'n_points': int(mask.sum()),
            'rmse': rmse(y[mask], pred_mean[mask]),
            'mae': mae(y[mask], pred_mean[mask]),
            'mape_percent': mape(y[mask], pred_mean[mask]),
            'mean_band_width': float(np.mean(pred_hi[mask] - pred_lo[mask])),
        })
    battery_df = pd.DataFrame(battery_records).sort_values('battery_index').reset_index(drop=True)
    battery_df.to_csv(OUTPUT_DIR / 'per_battery_metrics_batch10.csv', index=False)

    # summary json/txt
    summary = {
        'checkpoint_path': str(CHECKPOINT_PATH.resolve()),
        'label_mode': label_mode,
        'best_epoch': int(best_epoch) if best_epoch is not None else None,
        'best_pooled_rmse_from_checkpoint': float(best_pooled_rmse) if best_pooled_rmse is not None else None,
        'best_macro_rmse_from_checkpoint': float(best_macro_rmse) if best_macro_rmse is not None else None,
        'evaluation_pooled_rmse': float(pooled),
        'evaluation_macro_rmse': float(macro),
        'evaluation_mae': float(overall_mae),
        'evaluation_mape_percent': float(overall_mape),
        'average_95_band_width': float(avg_band_width),
        'mc_dropout_samples': int(MC_DROPOUT_SAMPLES),
        'base_cubic_coeffs': [float(v) for v in coeffs],
        'train_config': train_config,
    }
    with open(OUTPUT_DIR / 'summary_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    summary_lines = [
        '=== SOH Evaluation Summary ===',
        f'Checkpoint: {CHECKPOINT_PATH.resolve()}',
        f'Label mode: {label_mode}',
        f'Best epoch: {best_epoch if best_epoch is not None else "not stored in checkpoint"}',
        f'Best pooled RMSE (checkpoint metadata): {best_pooled_rmse if best_pooled_rmse is not None else "not stored"}',
        f'Best macro RMSE (checkpoint metadata): {best_macro_rmse if best_macro_rmse is not None else "not stored"}',
        f'Evaluation pooled RMSE: {pooled:.5f}',
        f'Evaluation macro RMSE: {macro:.5f}',
        f'Evaluation MAE: {overall_mae:.5f}',
        f'Evaluation MAPE (%): {overall_mape:.3f}',
        f'Average 95% band width: {avg_band_width:.5f}',
        '',
        'Per-battery RMSE:',
    ]
    for _, row in battery_df.iterrows():
        summary_lines.append(f"  - {row['cell_id']}: RMSE={row['rmse']:.5f}, MAE={row['mae']:.5f}, n={int(row['n_points'])}")

    with open(OUTPUT_DIR / 'summary_metrics.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))

    # plot
    summary_text = (
        f'Pooled RMSE = {pooled:.5f}    |    Macro RMSE = {macro:.5f}    |    '
        f'MAE = {overall_mae:.5f}    |    95% band avg width = {avg_band_width:.5f}    |    '
        f'Best epoch = {best_epoch if best_epoch is not None else "not stored"}'
    )
    fig_path = OUTPUT_DIR / 'batch10_soh_2x4_with_95CI.png'
    plot_2x4(
        cycles=p,
        y_true=y,
        y_pred=pred_mean,
        y_lo=pred_lo,
        y_hi=pred_hi,
        battery_idx=bidx,
        cell_id_map=cell_id_map,
        summary_text=summary_text,
        save_path=fig_path,
    )

    print('\n✅ Evaluation finished')
    print(f'📊 Figure saved   : {fig_path}')
    print(f'📄 Battery metrics: {OUTPUT_DIR / "per_battery_metrics_batch10.csv"}')
    print(f'📝 Summary txt    : {OUTPUT_DIR / "summary_metrics.txt"}')
    print(f'🧾 Summary json   : {OUTPUT_DIR / "summary_metrics.json"}')
    print('\n=== Final Metrics ===')
    print(f'Pooled RMSE = {pooled:.5f}')
    print(f'Macro RMSE  = {macro:.5f}')
    print(f'MAE         = {overall_mae:.5f}')
    print(f'MAPE (%)    = {overall_mape:.3f}')
    print(f'Best epoch  = {best_epoch if best_epoch is not None else "not stored in checkpoint"}')
    print('\nPer-battery RMSE:')
    print(battery_df[['cell_id', 'rmse', 'mae', 'n_points']].to_string(index=False))


if __name__ == '__main__':
    main()
