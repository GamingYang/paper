"""
Compile all supplementary experiment results into a single Excel file.
Combines: multi-seed stats, per-battery breakdown, model complexity.
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(r"C:\Users\ROG\Desktop\paper")
CODEX_DIR = ROOT / "codex"
OUT_DIR = CODEX_DIR / "supplementary-experiments"

# ============== 1. Gather multi-seed results ==============
print("Gathering multi-seed results...")
all_results = []

# Seed=42 from original codex directories
seed42_data = {
    'final-retained': {
        'path': CODEX_DIR / 'final-retained-v1' / 'final-retained-v1_best_metrics.json',
        'format': 'summary',
    },
    'no-physical-base': {
        'path': CODEX_DIR / 'no-physical-base-v1' / 'no-physical-base-v1_best_metrics.json',
        'format': 'config',
    },
    'no-mmd': {
        'path': CODEX_DIR / 'no-mmd-v1' / 'no-mmd-v1_best_metrics.json',
        'format': 'config',
    },
    'no-attention': {
        'path': CODEX_DIR / 'no-attention-v1' / 'no-attention-v1_best_metrics.json',
        'format': 'config',
    },
}

for model_name, info in seed42_data.items():
    if info['path'].exists():
        with open(info['path'], 'r', encoding='utf-8') as f:
            d = json.load(f)
        if info['format'] == 'summary':
            s = d['summary']
            m = d.get('metrics', [{}])[-1]
            all_results.append({
                'model': model_name, 'seed': 42,
                'target_test_pooled_rmse': s.get('final_target_test_rmse', ''),
                'target_test_macro_rmse': m.get('macro_rmse', ''),
                'target_test_mae': m.get('mae', ''),
                'target_test_mape': m.get('mape_percent', ''),
                'best_epoch': s.get('best_epoch_selected_by_target_adapt', ''),
            })
        else:
            c = d['config']
            m = d.get('metrics', [{}])[-1]
            all_results.append({
                'model': model_name, 'seed': 42,
                'target_test_pooled_rmse': c.get('final_target_test_rmse', ''),
                'target_test_macro_rmse': m.get('macro_rmse', ''),
                'target_test_mae': m.get('mae', ''),
                'target_test_mape': m.get('mape_percent', ''),
                'best_epoch': c.get('best_epoch_selected_by_target_adapt', ''),
            })

# New seeds (7, 123) from supplementary-experiments
for seed_dir in sorted(OUT_DIR.glob("*-seed*")):
    if not seed_dir.is_dir():
        continue
    name_parts = seed_dir.name.split('-seed')
    variant_name = '-'.join(name_parts[:-1]) if len(name_parts) > 1 else seed_dir.name
    seed = int(name_parts[-1]) if len(name_parts) > 1 else 0

    result_file = seed_dir / f"{seed_dir.name}_best_metrics.json"
    if result_file.exists():
        with open(result_file, 'r', encoding='utf-8') as f:
            d = json.load(f)
        r = d.get('result', d)
        all_results.append({
            'model': variant_name, 'seed': seed,
            'target_test_pooled_rmse': r.get('target_test_rmse', ''),
            'target_test_macro_rmse': r.get('target_test_macro_rmse', ''),
            'target_test_mae': r.get('target_test_mae', ''),
            'target_test_mape': r.get('target_test_mape', ''),
            'best_epoch': r.get('best_epoch', ''),
        })

# ============== 2. Compute mean ± std ==============
print("Computing statistics...")
results_df = pd.DataFrame(all_results)
if not results_df.empty:
    for col in ['target_test_pooled_rmse', 'target_test_macro_rmse', 'target_test_mae', 'target_test_mape']:
        results_df[col] = pd.to_numeric(results_df[col], errors='coerce')

# Per-model statistics
stat_rows = []
for model_name in sorted(results_df['model'].unique()):
    subset = results_df[results_df['model'] == model_name]
    seeds_found = sorted(subset['seed'].unique())
    stat_rows.append({
        'model': model_name,
        'seeds': str(seeds_found),
        'n_seeds': len(seeds_found),
        'pooled_rmse_mean': subset['target_test_pooled_rmse'].mean(),
        'pooled_rmse_std': subset['target_test_pooled_rmse'].std(),
        'pooled_rmse_str': f"{subset['target_test_pooled_rmse'].mean():.6f}±{subset['target_test_pooled_rmse'].std():.6f}" if len(seeds_found) > 1 else f"{subset['target_test_pooled_rmse'].iloc[0]:.6f}",
        'macro_rmse_mean': subset['target_test_macro_rmse'].mean(),
        'macro_rmse_std': subset['target_test_macro_rmse'].std(),
        'macro_rmse_str': f"{subset['target_test_macro_rmse'].mean():.6f}±{subset['target_test_macro_rmse'].std():.6f}" if len(seeds_found) > 1 else f"{subset['target_test_macro_rmse'].iloc[0]:.6f}",
        'mae_mean': subset['target_test_mae'].mean(),
        'mae_std': subset['target_test_mae'].std(),
        'mape_mean': subset['target_test_mape'].mean(),
        'mape_std': subset['target_test_mape'].std(),
    })
stats_df = pd.DataFrame(stat_rows)

# ============== 3. Gather per-battery data ==============
print("Gathering per-battery data...")
all_cells = []
for csv_file in sorted(OUT_DIR.glob("per_battery_*.csv")):
    df = pd.read_csv(csv_file)

    # Parse model name from filename
    fname = csv_file.stem  # e.g. per_battery_final-retained_seed42
    parts = fname.replace('per_battery_', '').rsplit('_seed', 1)
    if len(parts) == 2:
        df['model'] = parts[0]
        df['seed'] = int(parts[1])
    else:
        df['model'] = parts[0]
        df['seed'] = 42
    all_cells.append(df)

# Also try to get from new seed runs
for seed_dir in sorted(OUT_DIR.glob("*-seed*")):
    if not seed_dir.is_dir():
        continue
    xlsx_file = seed_dir / f"{seed_dir.name}_results.xlsx"
    if xlsx_file.exists():
        try:
            cells_df = pd.read_excel(xlsx_file, sheet_name='target_test_cells')
            name_parts = seed_dir.name.split('-seed')
            variant_name = '-'.join(name_parts[:-1]) if len(name_parts) > 1 else seed_dir.name
            seed = int(name_parts[-1]) if len(name_parts) > 1 else 0
            cells_df['model'] = variant_name
            cells_df['seed'] = seed
            all_cells.append(cells_df)
        except Exception as e:
            print(f"  Warning: couldn't read {xlsx_file}: {e}")

per_battery_df = pd.concat(all_cells, ignore_index=True) if all_cells else pd.DataFrame()

# ============== 4. Model complexity ==============
complexity_df = pd.DataFrame([
    {'model': 'final-retained', 'params': 31466, 'inference_ms_per_128': 4.34},
    {'model': 'no-physical-base', 'params': 31466, 'inference_ms_per_128': 3.78},
    {'model': 'no-mmd', 'params': 31466, 'inference_ms_per_128': 4.35},
    {'model': 'no-attention', 'params': 29353, 'inference_ms_per_128': 3.53},
])

# ============== 5. Table 1 format (paper-ready) ==============
print("Creating paper-ready comparison table...")
# Recreate Table 1 with all available data
external_baselines = {
    'Pure cubic physical base': 0.032649,
    'LSTM baseline': 0.032742,
    'GRU baseline': 0.033109,
    'CNN-LSTM baseline': 0.016075,
    'PCA-GPR baseline': 0.023108,
    'PCA-RBF-SVR baseline': 0.023823,
    'PCA-LinearSVR baseline': 0.036993,
}

# Build the main comparison table
table1_rows = []
for _, row in stats_df.sort_values('pooled_rmse_mean').iterrows():
    model = row['model']
    rmse = row['pooled_rmse_str']
    macro = row['macro_rmse_str']
    table1_rows.append({
        'model': model,
        'type': 'Ours (ablation)',
        'pooled_rmse': rmse,
        'macro_rmse': macro,
    })

# Add our best seed=42 single result too
for _, r in results_df[results_df['seed'] == 42].sort_values('target_test_pooled_rmse').iterrows():
    table1_rows.append({
        'model': f"{r['model']} (seed=42)",
        'type': 'Ours (seed=42 detail)',
        'pooled_rmse': f"{r['target_test_pooled_rmse']:.6f}",
        'macro_rmse': f"{r['target_test_macro_rmse']:.6f}",
    })

for name, rmse in external_baselines.items():
    table1_rows.append({
        'model': name,
        'type': 'External baseline',
        'pooled_rmse': f"{rmse:.6f}",
        'macro_rmse': '-',
    })

table1_df = pd.DataFrame(table1_rows)

# ============== 6. Write Excel ==============
print("Writing Excel...")
xlsx_path = OUT_DIR / "supplementary_experiments_summary.xlsx"
with pd.ExcelWriter(xlsx_path, engine='openpyxl') as w:
    results_df.to_excel(w, index=False, sheet_name='all_seeds_raw')
    stats_df.to_excel(w, index=False, sheet_name='multi_seed_stats')
    if not per_battery_df.empty:
        per_battery_df.to_excel(w, index=False, sheet_name='per_battery')
    complexity_df.to_excel(w, index=False, sheet_name='model_complexity')
    table1_df.to_excel(w, index=False, sheet_name='table1_comparison')

print(f"Summary saved to: {xlsx_path}")
print(f"\nMulti-seed stats:")
print(stats_df.to_string(index=False))
print(f"\nTotal results collected: {len(all_results)}")
