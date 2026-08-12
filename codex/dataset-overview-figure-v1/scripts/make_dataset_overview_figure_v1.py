from __future__ import annotations

import pickle
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy import signal, stats


ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = ROOT / "codex" / "dataset-overview-figure-v1"
FIG_DIR = OUT_ROOT / "figures"
TABLE_DIR = OUT_ROOT / "tables"

RAW_DIR = ROOT / "our_data"
PROC_DIR = ROOT / "PYthon训练代码" / "processed_clean"

REP_CELL = "10-1"


COLORS = {
    "source": "#2C7FB8",
    "target_adapt": "#41AB5D",
    "target_test": "#D95F0E",
    "dark": "#1F2933",
    "grid": "#E6E6E6",
    "blue": "#1F77B4",
    "cyan": "#46B7C9",
    "orange": "#E9824A",
    "red": "#C0392B",
    "purple": "#6A51A3",
    "green": "#238B45",
    "gray": "#888888",
}


def natural_key(path: Path) -> tuple[int, int]:
    m = re.match(r"(\d+)-(\d+)", path.stem)
    return (int(m.group(1)), int(m.group(2))) if m else (999, 999)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.labelsize": 9.4,
            "axes.titlesize": 10.2,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axes(ax: plt.Axes, grid: str | None = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(length=2.5, width=0.7, color="#333333")
    if grid:
        ax.grid(True, axis=grid, color=COLORS["grid"], linestyle="--", linewidth=0.55, zorder=0)
        ax.set_axisbelow(True)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def load_cell(path: Path) -> tuple[str, dict]:
    obj = pickle.load(open(path, "rb"))
    cell_id = next(iter(obj))
    return cell_id, obj[cell_id]


def build_raw_cache(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = TABLE_DIR / "raw_cell_capacity_summary.csv"
    traj_path = TABLE_DIR / "raw_capacity_trajectories_sampled.csv"
    if summary_path.exists() and traj_path.exists() and not force:
        return pd.read_csv(summary_path), pd.read_csv(traj_path)

    summary_rows: list[dict] = []
    traj_rows: list[dict] = []
    for path in sorted(RAW_DIR.glob("*.pkl"), key=natural_key):
        cell_id, inner = load_cell(path)
        batch = int(cell_id.split("-")[0])
        cycles = np.array(sorted(inner["dq"].keys()), dtype=int)
        cap_ah = np.array([float(inner["dq"][int(c)]) / 1000.0 for c in cycles])
        eol = int(cycles.max())
        step = max(1, int(np.ceil(len(cycles) / 260)))
        for c, q in zip(cycles[::step], cap_ah[::step]):
            traj_rows.append(
                {
                    "cell_id": cell_id,
                    "batch": batch,
                    "cycle": int(c),
                    "discharge_capacity_ah": float(q),
                    "eol_cycle": eol,
                }
            )
        summary_rows.append(
            {
                "cell_id": cell_id,
                "batch": batch,
                "eol_cycle": eol,
                "initial_capacity_ah": float(cap_ah[0]),
                "max_capacity_ah": float(cap_ah.max()),
                "final_capacity_ah": float(cap_ah[-1]),
                "capacity_drop_ah": float(cap_ah[0] - cap_ah[-1]),
            }
        )

    summary = pd.DataFrame(summary_rows)
    traj = pd.DataFrame(traj_rows)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    traj.to_csv(traj_path, index=False, encoding="utf-8-sig")
    return summary, traj


def load_processed_frames() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    frames = []
    x_arrays: dict[str, np.ndarray] = {}
    for split in ["source", "target_adapt", "target_test"]:
        meta = pd.read_csv(PROC_DIR / f"clean_{split}_meta.csv")
        meta["split"] = split
        frames.append(meta)
        x_arrays[split] = np.load(PROC_DIR / f"clean_{split}_x.npy", mmap_mode="r")
    return pd.concat(frames, ignore_index=True), x_arrays


def representative_raw_data() -> tuple[dict[int, pd.DataFrame], int]:
    path = RAW_DIR / f"{REP_CELL}.pkl"
    _, inner = load_cell(path)
    cycles = sorted(inner["data"].keys())
    eol = max(cycles)
    selected = [1, max(1, eol // 2), max(1, eol - 20)]
    return {c: inner["data"][c].copy() for c in selected}, eol


def charge_segment(df: pd.DataFrame) -> pd.DataFrame:
    seg = df[df["Current (mA)"] > 0].copy()
    if seg.empty:
        return df.copy()
    return seg


def make_voltage_heatmap(meta_all: pd.DataFrame, x_arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    meta = meta_all[(meta_all["split"] == "target_test") & (meta_all["cell_id"] == REP_CELL)].copy()
    if meta.empty:
        meta = meta_all[meta_all["split"] == "target_test"].copy()
    indices = meta.index.to_numpy()
    local_indices = np.arange(len(meta))
    # In the concatenated frame target_test starts after source and target_adapt;
    # use local row indices inside the target_test mmap array.
    sample_pos = np.linspace(0, len(meta) - 1, min(90, len(meta))).astype(int)
    cycles = meta.iloc[sample_pos]["cycle"].to_numpy()
    heat = np.asarray(x_arrays["target_test"][local_indices[sample_pos], :], dtype=float)
    return cycles, heat


def smooth_density(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bins: tuple[int, int],
    sigma: float = 1.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
    z = gaussian_filter(hist.T, sigma=sigma)
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    return xcenters, ycenters, z


def draw_dataset_overview() -> Path:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    summary, traj = build_raw_cache()
    meta_all, x_arrays = load_processed_frames()
    raw_cycles, rep_eol = representative_raw_data()
    heat_cycles, heat = make_voltage_heatmap(meta_all, x_arrays)
    split_summary = pd.read_csv(ROOT / "codex" / "6月29日最新证据链" / "tables" / "fixed_split_summary.csv")

    fig = plt.figure(figsize=(15.8, 12.2), constrained_layout=False)
    gs = fig.add_gridspec(3, 4, left=0.045, right=0.975, bottom=0.055, top=0.935, wspace=0.36, hspace=0.48)
    fig.suptitle(
        "Dataset overview for cross-batch lithium-ion battery SOH prediction",
        fontsize=15.5,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.958,
        "Raw cycling records from 77 cells and processed_clean split: "
        "Batch 1-8 source, Batch 9 target-development, Batch 10 blind test",
        ha="center",
        va="top",
        fontsize=9.3,
        color="#444444",
    )

    # a. Capacity trajectories.
    ax = fig.add_subplot(gs[0, 0:2])
    norm = mpl.colors.Normalize(vmin=summary["eol_cycle"].min(), vmax=summary["eol_cycle"].max())
    cmap = mpl.cm.YlGnBu
    for cell_id, g in traj.groupby("cell_id"):
        eol = float(g["eol_cycle"].iloc[0])
        ax.plot(g["cycle"], g["discharge_capacity_ah"], color=cmap(norm(eol)), lw=0.8, alpha=0.78)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.012, fraction=0.035)
    cbar.set_label("Cycle life")
    ax.set_xlabel("Cycle number")
    ax.set_ylabel("Discharge capacity (Ah)")
    ax.set_title("Cell-level capacity degradation trajectories")
    ax.set_xlim(0, summary["eol_cycle"].max() + 80)
    clean_axes(ax)
    add_panel_label(ax, "a")

    # b. Cycle-SOH density.
    ax = fig.add_subplot(gs[0, 2])
    xx, yy, zz = smooth_density(
        meta_all["cycle"].to_numpy(),
        meta_all["y_nominal"].to_numpy(),
        bins=(140, 95),
        sigma=1.45,
    )
    levels = np.linspace(max(1e-9, zz[zz > 0].min()), zz.max(), 13)
    cf = ax.contourf(xx, yy, zz, levels=levels, cmap="Blues", antialiased=True)
    ax.contour(xx, yy, zz, levels=levels[::3], colors="#2B6C9B", linewidths=0.25, alpha=0.45)
    fig.colorbar(cf, ax=ax, pad=0.012, fraction=0.05).set_label("Smoothed samples")
    ax.axhline(0.8, color=COLORS["red"], lw=1.0, ls="--")
    ax.set_xlabel("Cycle number")
    ax.set_ylabel("Nominal SOH")
    ax.set_title("Cycle-SOH sample density")
    clean_axes(ax)
    add_panel_label(ax, "b")

    # c. Life distribution by batch.
    ax = fig.add_subplot(gs[0, 3])
    batches = sorted(summary["batch"].unique())
    data = [summary.loc[summary["batch"] == b, "eol_cycle"].to_numpy() for b in batches]
    parts = ax.violinplot(data, positions=batches, widths=0.75, showmeans=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#9ECAE1")
        body.set_edgecolor("#2B6C9B")
        body.set_alpha(0.75)
    parts["cmeans"].set_color(COLORS["red"])
    parts["cmeans"].set_linewidth(1.1)
    ax.scatter(summary["batch"], summary["eol_cycle"], s=8, color=COLORS["dark"], alpha=0.45, zorder=3)
    ax.axvspan(0.5, 8.5, color=COLORS["source"], alpha=0.06)
    ax.axvspan(8.5, 9.5, color=COLORS["target_adapt"], alpha=0.08)
    ax.axvspan(9.5, 10.5, color=COLORS["target_test"], alpha=0.08)
    ax.set_xticks(batches)
    ax.set_xlabel("Batch")
    ax.set_ylabel("Cycle life")
    ax.set_title("Cycle-life distribution across batches")
    clean_axes(ax, grid="y")
    add_panel_label(ax, "c")

    # d. Initial capacity vs life.
    ax = fig.add_subplot(gs[1, 0])
    colors = [COLORS["source"] if b <= 8 else COLORS["target_adapt"] if b == 9 else COLORS["target_test"] for b in summary["batch"]]
    ax.scatter(summary["initial_capacity_ah"], summary["eol_cycle"], c=colors, s=28, edgecolor="white", linewidth=0.35)
    rho, pval = stats.spearmanr(summary["initial_capacity_ah"], summary["eol_cycle"])
    ax.text(0.04, 0.94, rf"$\rho$={rho:.3f}", transform=ax.transAxes, ha="left", va="top", fontsize=10.5)
    handles = [
        mpl.lines.Line2D([0], [0], marker="o", linestyle="", color=COLORS["source"], label="Source"),
        mpl.lines.Line2D([0], [0], marker="o", linestyle="", color=COLORS["target_adapt"], label="Target-dev"),
        mpl.lines.Line2D([0], [0], marker="o", linestyle="", color=COLORS["target_test"], label="Blind test"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right", handlelength=0.8)
    ax.set_xlabel("Initial capacity (Ah)")
    ax.set_ylabel("Cycle life")
    ax.set_title("Initial capacity vs. cycle life")
    clean_axes(ax)
    add_panel_label(ax, "d")

    # e. Raw voltage/current time profile.
    ax = fig.add_subplot(gs[1, 1])
    mid_cycle = sorted(raw_cycles.keys())[1]
    df_mid = raw_cycles[mid_cycle]
    t_min = df_mid["Time (s)"].to_numpy() / 60.0
    ax.plot(t_min, df_mid["Voltage (V)"], color=COLORS["orange"], lw=1.3, label="Voltage")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Voltage (V)", color=COLORS["orange"])
    ax.tick_params(axis="y", labelcolor=COLORS["orange"])
    ax2 = ax.twinx()
    ax2.plot(t_min, df_mid["Current (mA)"].to_numpy() / 1000.0, color=COLORS["blue"], lw=0.85, alpha=0.82, label="Current")
    ax2.set_ylabel("Current (A)", color=COLORS["blue"])
    ax2.tick_params(axis="y", labelcolor=COLORS["blue"], labelsize=8.0)
    ax.set_title(f"Raw voltage/current profile ({REP_CELL}, cycle {mid_cycle})")
    clean_axes(ax)
    ax2.spines["top"].set_visible(False)
    add_panel_label(ax, "e")

    # f. Charge voltage curves at early/mid/late cycles.
    ax = fig.add_subplot(gs[1, 2])
    cycle_colors = [COLORS["green"], COLORS["blue"], COLORS["red"]]
    for color, (cycle, df) in zip(cycle_colors, raw_cycles.items()):
        ch = charge_segment(df)
        cap = ch["Capacity (mAh)"].to_numpy(dtype=float)
        if cap.max() > cap.min():
            x = 100 * (cap - cap.min()) / (cap.max() - cap.min())
        else:
            x = np.linspace(0, 100, len(cap))
        ax.plot(x, ch["Voltage (V)"], lw=1.5, color=color, label=f"Cycle {cycle}")
    ax.set_xlabel("Normalized charge capacity (%)")
    ax.set_ylabel("Charge voltage (V)")
    ax.set_title("Charge voltage evolution")
    ax.legend(frameon=False, loc="lower right")
    clean_axes(ax)
    add_panel_label(ax, "f")

    # g. Processed 100-point voltage heatmap.
    ax = fig.add_subplot(gs[1, 3])
    heat_smooth = gaussian_filter(heat, sigma=(0.75, 0.45))
    im = ax.imshow(
        heat_smooth,
        aspect="auto",
        origin="lower",
        cmap="turbo",
        interpolation="bicubic",
        resample=True,
        extent=[1, 100, float(heat_cycles.min()), float(heat_cycles.max())],
    )
    fig.colorbar(im, ax=ax, pad=0.012, fraction=0.05).set_label("Voltage (V)")
    ax.set_xlabel("Interpolated point index")
    ax.set_ylabel("Cycle number")
    ax.set_title(f"Processed charge-voltage map ({REP_CELL})")
    clean_axes(ax, grid=None)
    add_panel_label(ax, "g")

    # h. SOH distributions by split.
    ax = fig.add_subplot(gs[2, 0])
    bins = np.linspace(0.79, 1.13, 42)
    split_labels = {"source": "Source (B1-8)", "target_adapt": "Target-dev (B9)", "target_test": "Blind test (B10)"}
    for split, color in [("source", COLORS["source"]), ("target_adapt", COLORS["target_adapt"]), ("target_test", COLORS["target_test"])]:
        vals = meta_all.loc[meta_all["split"] == split, "y_nominal"].to_numpy()
        ax.hist(vals, bins=bins, density=True, histtype="step", lw=1.55, color=color, label=split_labels[split])
    ax.set_xlabel("Nominal SOH")
    ax.set_ylabel("Density")
    ax.set_title("SOH label distribution by split")
    ax.legend(frameon=False)
    clean_axes(ax, grid="y")
    add_panel_label(ax, "h")

    # i. RUL-SOH density.
    ax = fig.add_subplot(gs[2, 1])
    sample = meta_all.sample(n=min(70000, len(meta_all)), random_state=20260710)
    xx, yy, zz = smooth_density(
        sample["rul_cycle"].to_numpy(),
        sample["y_nominal"].to_numpy(),
        bins=(125, 90),
        sigma=1.45,
    )
    levels = np.linspace(max(1e-9, zz[zz > 0].min()), zz.max(), 13)
    cf = ax.contourf(xx, yy, zz, levels=levels, cmap="YlGnBu", antialiased=True)
    ax.contour(xx, yy, zz, levels=levels[::3], colors="#2A788E", linewidths=0.25, alpha=0.45)
    fig.colorbar(cf, ax=ax, pad=0.012, fraction=0.05).set_label("Smoothed samples")
    ax.set_xlabel("RUL label (cycles)")
    ax.set_ylabel("Nominal SOH")
    ax.set_title("RUL-SOH label coupling")
    clean_axes(ax)
    add_panel_label(ax, "i")

    # j. Current-voltage phase portrait.
    ax = fig.add_subplot(gs[2, 2])
    raw = df_mid.iloc[:: max(1, len(df_mid) // 520), :].copy()
    sc = ax.scatter(
        raw["Voltage (V)"],
        raw["Current (mA)"] / 1000.0,
        c=raw["Time (s)"] / 60.0,
        cmap="viridis",
        s=10,
        alpha=0.85,
        linewidths=0,
    )
    fig.colorbar(sc, ax=ax, pad=0.012, fraction=0.05).set_label("Time (min)")
    ax.axhline(0, color="#555555", lw=0.6)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (A)")
    ax.set_title("Voltage-current phase view")
    clean_axes(ax)
    add_panel_label(ax, "j")

    # k. Spectrogram of the representative voltage signal.
    ax = fig.add_subplot(gs[2, 3])
    v = df_mid["Voltage (V)"].to_numpy(dtype=float)
    t = df_mid["Time (s)"].to_numpy(dtype=float)
    if len(t) > 1:
        dt = np.median(np.diff(t))
        fs = 1.0 / max(dt, 1e-6)
    else:
        fs = 1.0
    nperseg = min(96, max(16, len(v) // 6))
    f, tt, sxx = signal.spectrogram(v - np.mean(v), fs=fs, nperseg=nperseg, noverlap=nperseg // 2, scaling="spectrum")
    sxx_db = gaussian_filter(10 * np.log10(sxx + 1e-12), sigma=(0.65, 0.55))
    im = ax.pcolormesh(tt / 60.0, f, sxx_db, cmap="magma", shading="gouraud")
    fig.colorbar(im, ax=ax, pad=0.012, fraction=0.05).set_label("Power (dB)")
    ax.set_ylim(0, min(0.08, f.max() if len(f) else 0.08))
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Voltage time-frequency view")
    clean_axes(ax, grid=None)
    add_panel_label(ax, "k")

    out_png = FIG_DIR / "Figure_dataset_overview_multi_panel_v2_smooth.png"
    out_pdf = FIG_DIR / "Figure_dataset_overview_multi_panel_v2_smooth.pdf"
    out_svg = FIG_DIR / "Figure_dataset_overview_multi_panel_v2_smooth.svg"
    for out in [out_png, out_pdf, out_svg]:
        fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_png


def main() -> None:
    out = draw_dataset_overview()
    print(out)


if __name__ == "__main__":
    main()
