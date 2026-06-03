from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GLOBAL_MAX_CYCLE = 3000.0
DATA_DIR = Path('./processed_clean')
CHECKPOINT_PATH = Path('./best_soh_nominal_processed_clean_no_physical_base.pth')
OUTPUT_DIR = Path('./eval_processed_clean_v3-NO-physical')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MC_DROPOUT_SAMPLES = 80
BATCH_SIZE = 1024
LABEL_MODE = 'nominal'
DISPLAY_SMOOTH = True
DISPLAY_SAVGOL_WINDOW = 41
DISPLAY_SAVGOL_POLY = 3
ERROR_STRIDE_MIN = 12
ERROR_STRIDE_MAX_BARS = 80
ERROR_BAR_ALPHA = 0.22
CI_ALPHA = 0.40

COLOR_TRUE_LINE = '#8B0000'
COLOR_PRED_LINE = '#2874A6'
COLOR_CI_FILL = '#D6EAF8'
COLOR_ERROR_BAR = '#1ABC9C'
COLOR_GRID = '#BFBFBF'

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.unicode_minus'] = False


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


class ResidualCRNNNoPhysicalBase(nn.Module):
    """
    Must match the NO-PHYSICAL-BASE training architecture exactly.
    Key point: prediction path DOES NOT add cubic base_soh.
    """
    def __init__(self, base_coeffs: torch.Tensor):
        super().__init__()
        self.base_coeffs = base_coeffs  # kept only for checkpoint compatibility
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        self.attention = TemporalAttention(hidden_dim=64)
        self.phy = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU()
        )
        self.fc_soh = nn.Sequential(
            nn.Linear(72, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor, p: torch.Tensor):
        x_in = x.unsqueeze(1)
        out = self.cnn(x_in).permute(0, 2, 1)
        lstm_out, _ = self.lstm(out)
        context_feat, _ = self.attention(lstm_out)
        f_phy = self.phy(p)
        fused_features = torch.cat((context_feat, f_phy), dim=1)
        pred_soh = self.fc_soh(fused_features)
        return pred_soh


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(np.abs(y_pred - y_true)))


def mape(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    eps = 1e-8
    return float(np.mean(np.abs((y_pred - y_true) / np.maximum(np.abs(y_true), eps))) * 100.0)


def macro_rmse_by_battery(y_true, y_pred, battery_idx):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    battery_idx = np.asarray(battery_idx).reshape(-1)
    vals = []
    for b in np.unique(battery_idx):
        m = battery_idx == b
        vals.append(rmse(y_true[m], y_pred[m]))
    return float(np.mean(vals))


def sg_smooth(y, window=41, poly=3):
    y = np.asarray(y).reshape(-1)
    if not DISPLAY_SMOOTH or len(y) < 7:
        return y.copy()
    try:
        from scipy.signal import savgol_filter
        win = min(window, len(y) if len(y) % 2 == 1 else len(y) - 1)
        if win < 5:
            return y.copy()
        if win <= poly:
            win = poly + 3 if (poly + 3) % 2 == 1 else poly + 4
        win = min(win, len(y) if len(y) % 2 == 1 else len(y) - 1)
        if win < 5:
            return y.copy()
        return savgol_filter(y, window_length=win, polyorder=poly)
    except Exception:
        return y.copy()


def load_target_test(label_mode='nominal'):
    x = np.load(DATA_DIR / 'clean_target_test_x.npy').astype(np.float32)
    p = np.load(DATA_DIR / 'clean_target_test_p.npy').astype(np.float32).reshape(-1)
    bidx = np.load(DATA_DIR / 'clean_target_test_battery_index.npy').astype(np.int64).reshape(-1)
    if label_mode == 'relative':
        y = np.load(DATA_DIR / 'clean_target_test_y_relative.npy').astype(np.float32).reshape(-1)
    else:
        y = np.load(DATA_DIR / 'clean_target_test_y_nominal.npy').astype(np.float32).reshape(-1)
    meta = pd.read_csv(DATA_DIR / 'clean_target_test_meta.csv')
    return x, y, p, bidx, meta


def load_source_for_record(label_mode='nominal'):
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


def mc_predict(model, x_np, p_np, n_samples=80, batch_size=1024):
    x_t = torch.from_numpy(x_np).float()
    p_t = torch.from_numpy((p_np / GLOBAL_MAX_CYCLE).reshape(-1, 1)).float()
    n = len(x_np)
    preds_all = []
    for _ in range(n_samples):
        enable_dropout_only(model)
        one = []
        with torch.no_grad():
            for s in range(0, n, batch_size):
                e = min(s + batch_size, n)
                xb = x_t[s:e].to(DEVICE)
                pb = p_t[s:e].to(DEVICE)
                out = model(xb, pb)
                one.append(out.cpu().numpy().reshape(-1))
        preds_all.append(np.concatenate(one, axis=0))
    ps = np.stack(preds_all, axis=0)
    pm = ps.mean(axis=0)
    lo = np.percentile(ps, 2.5, axis=0)
    hi = np.percentile(ps, 97.5, axis=0)
    return pm, lo, hi, ps


def build_cell_id_map(meta):
    tmp = meta[['battery_index', 'cell_id']].drop_duplicates().copy()
    return dict(zip(tmp['battery_index'].astype(int), tmp['cell_id'].astype(str)))


def infer_best_epoch_from_checkpoint(ckpt):
    if isinstance(ckpt, dict):
        for key in ['best_epoch', 'epoch', 'best_epoch_idx']:
            if key in ckpt and isinstance(ckpt[key], (int, np.integer)):
                return int(ckpt[key])
    return None


def plot_2x4(cycles, y_true, y_pred_raw, y_lo_raw, y_hi_raw, battery_idx, cell_id_map,
             pooled_rmse, macro_rmse, mae_all, best_epoch, save_path):
    cycles = np.asarray(cycles).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)
    y_pred_raw = np.asarray(y_pred_raw).reshape(-1)
    y_lo_raw = np.asarray(y_lo_raw).reshape(-1)
    y_hi_raw = np.asarray(y_hi_raw).reshape(-1)
    battery_idx = np.asarray(battery_idx).reshape(-1)

    unique_bats = np.unique(battery_idx)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=False)
    axes = axes.flatten()

    legend_handles = [
        Line2D([0], [0], color=COLOR_TRUE_LINE, lw=2.0, label='Measured SOH'),
        Line2D([0], [0], color=COLOR_PRED_LINE, lw=1.9, label='Predicted SOH'),
        Patch(facecolor=COLOR_CI_FILL, edgecolor=COLOR_CI_FILL, alpha=CI_ALPHA, label='95% uncertainty band'),
        Patch(facecolor=COLOR_ERROR_BAR, edgecolor=COLOR_ERROR_BAR, alpha=ERROR_BAR_ALPHA, label='Abs. Error'),
    ]

    battery_rows = []
    max_err = float(np.max(np.abs(y_true - y_pred_raw)))
    err_ylim = max(0.04, min(0.16, max_err * 1.25))

    for i, b in enumerate(unique_bats):
        ax = axes[i]
        ax2 = ax.twinx()
        mask = battery_idx == b

        x = cycles[mask]
        yt = y_true[mask]
        yp_raw = y_pred_raw[mask]
        ylo_raw = y_lo_raw[mask]
        yhi_raw = y_hi_raw[mask]

        yp = sg_smooth(yp_raw, DISPLAY_SAVGOL_WINDOW, DISPLAY_SAVGOL_POLY)
        ylo = sg_smooth(ylo_raw, DISPLAY_SAVGOL_WINDOW, DISPLAY_SAVGOL_POLY)
        yhi = sg_smooth(yhi_raw, DISPLAY_SAVGOL_WINDOW, DISPLAY_SAVGOL_POLY)
        err = np.abs(yt - yp_raw)

        stride = max(ERROR_STRIDE_MIN, len(x) // ERROR_STRIDE_MAX_BARS)
        xs = x[::stride]
        es = err[::stride]
        width = max(6.0, float(np.median(np.diff(x))) * stride * 0.8) if len(x) > 1 else 6.0

        ax2.bar(xs, es, width=width, color=COLOR_ERROR_BAR, alpha=ERROR_BAR_ALPHA, linewidth=0)
        ax2.set_ylim(0, err_ylim)
        ax2.set_ylabel('Abs. Error', fontsize=10)
        ax2.tick_params(axis='y', labelsize=9)
        ax2.spines['right'].set_linewidth(1.0)
        ax2.grid(False)

        ax.fill_between(x, ylo, yhi, color=COLOR_CI_FILL, alpha=CI_ALPHA, linewidth=0)
        ax.plot(x, yt, color=COLOR_TRUE_LINE, linewidth=2.0)
        ax.plot(x, yp, color=COLOR_PRED_LINE, linewidth=1.9)

        this_rmse = rmse(yt, yp_raw)
        this_mae = mae(yt, yp_raw)
        cid = cell_id_map.get(int(b), f'Battery-{int(b)}')
        battery_rows.append((cid, this_rmse, this_mae, len(x)))

        ax.set_title(f'{cid}\nRMSE = {this_rmse:.4f}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Cycle', fontsize=11)
        ax.set_ylabel('SOH', fontsize=11)
        y_min = min(float(np.min(yt)), float(np.min(ylo)), float(np.min(yp)))
        y_max = max(float(np.max(yt)), float(np.max(yhi)), float(np.max(yp)))
        pad = 0.03
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.grid(True, linestyle='--', color=COLOR_GRID, alpha=0.5)
        ax.tick_params(axis='both', labelsize=10)

    for j in range(len(unique_bats), 8):
        axes[j].axis('off')

    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.985),
               ncol=4, frameon=False, fontsize=12)
    epoch_text = str(best_epoch) if best_epoch is not None else 'not stored'
    fig.suptitle('Batch 10 SOH Prediction (2 × 4)\nTimes New Roman + 95% uncertainty band', fontsize=20, y=1.03, fontweight='bold')
    fig.text(0.5, 0.02,
             f'Pooled RMSE = {pooled_rmse:.5f}   |   Macro RMSE = {macro_rmse:.5f}   |   MAE = {mae_all:.5f}   |   Best epoch = {epoch_text}',
             ha='center', fontsize=13)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return battery_rows


def main():
    y_src, p_src = load_source_for_record(LABEL_MODE)
    coeffs = np.polyfit((p_src / GLOBAL_MAX_CYCLE).reshape(-1), y_src.reshape(-1), 3)
    base_coeffs = torch.tensor(coeffs, dtype=torch.float32).to(DEVICE)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model = ResidualCRNNNoPhysicalBase(base_coeffs).to(DEVICE)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    best_epoch = infer_best_epoch_from_checkpoint(ckpt)

    x, y, p, bidx, meta = load_target_test(LABEL_MODE)
    pred_mean, pred_lo, pred_hi, _ = mc_predict(model, x, p, n_samples=MC_DROPOUT_SAMPLES, batch_size=BATCH_SIZE)

    pooled = rmse(y, pred_mean)
    macro = macro_rmse_by_battery(y, pred_mean, bidx)
    mae_all = mae(y, pred_mean)
    mape_all = mape(y, pred_mean)
    cell_map = build_cell_id_map(meta)

    fig_path = OUTPUT_DIR / 'batch10_soh_2x4_with_95CI_publication_no_physical.png'
    battery_rows = plot_2x4(p, y, pred_mean, pred_lo, pred_hi, bidx, cell_map,
                            pooled, macro, mae_all, best_epoch, fig_path)

    df = pd.DataFrame(battery_rows, columns=['cell_id', 'rmse', 'mae', 'n_points'])
    metrics_csv = OUTPUT_DIR / 'per_battery_metrics_batch10_no_physical.csv'
    summary_json = OUTPUT_DIR / 'summary_metrics_no_physical.json'
    summary_txt = OUTPUT_DIR / 'summary_metrics_no_physical.txt'
    df.to_csv(metrics_csv, index=False)

    summary = {
        'label_mode': LABEL_MODE,
        'pooled_rmse': pooled,
        'macro_rmse': macro,
        'mae': mae_all,
        'mape_percent': mape_all,
        'best_epoch': best_epoch,
        'mc_dropout_samples': MC_DROPOUT_SAMPLES,
        'display_smooth': DISPLAY_SMOOTH,
        'display_savgol_window': DISPLAY_SAVGOL_WINDOW,
        'display_savgol_poly': DISPLAY_SAVGOL_POLY,
        'figure_path': str(fig_path),
        'checkpoint_path': str(CHECKPOINT_PATH),
    }
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(summary_txt, 'w', encoding='utf-8') as f:
        f.write('=== Final Metrics ===\n')
        f.write(f'Pooled RMSE = {pooled:.5f}\n')
        f.write(f'Macro RMSE  = {macro:.5f}\n')
        f.write(f'MAE         = {mae_all:.5f}\n')
        f.write(f'MAPE (%)    = {mape_all:.3f}\n')
        f.write(f'Best epoch  = {best_epoch if best_epoch is not None else "not stored in checkpoint"}\n')

    print('✅ Evaluation finished (NO-PHYSICAL-BASE)')
    print(f'📊 Figure saved   : {fig_path}')
    print(f'📄 Battery metrics: {metrics_csv}')
    print(f'📝 Summary txt    : {summary_txt}')
    print(f'🧾 Summary json   : {summary_json}')
    print('\n=== Final Metrics ===')
    print(f'Pooled RMSE = {pooled:.5f}')
    print(f'Macro RMSE  = {macro:.5f}')
    print(f'MAE         = {mae_all:.5f}')
    print(f'MAPE (%)    = {mape_all:.3f}')
    print(f'Best epoch  = {best_epoch if best_epoch is not None else "not stored in checkpoint"}')
    print('\nPer-battery RMSE:')
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
