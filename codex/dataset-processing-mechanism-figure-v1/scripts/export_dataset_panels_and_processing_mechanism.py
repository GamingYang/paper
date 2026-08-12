from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = ROOT / "codex" / "dataset-processing-mechanism-figure-v1"
FIG_DIR = RUN_DIR / "figures"
PANEL_DIR = FIG_DIR / "dataset_overview_split_panels"
PROCESSING_PANEL_DIR = FIG_DIR / "processing_display_panels"
TABLE_DIR = RUN_DIR / "tables"
SOURCE_SCRIPT = ROOT / "codex" / "dataset-overview-figure-v1" / "scripts" / "make_dataset_overview_figure_v1.py"


def load_overview_module():
    spec = importlib.util.spec_from_file_location("dataset_overview_source", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import source script: {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["dataset_overview_source"] = module
    spec.loader.exec_module(module)
    return module


ov = load_overview_module()


PANEL_NAMES = {
    "a": "capacity_degradation_trajectories",
    "b": "cycle_soh_sample_density",
    "c": "cycle_life_distribution_by_batch",
    "d": "initial_capacity_vs_cycle_life",
    "e": "raw_voltage_current_profile",
    "f": "charge_voltage_evolution",
    "g": "processed_charge_voltage_map",
    "h": "soh_distribution_by_split",
    "i": "rul_soh_label_coupling",
    "j": "voltage_current_phase_view",
    "k": "voltage_time_frequency_view",
}


def setup_publication_style() -> None:
    ov.setup_style()
    mpl.rcParams.update(
        {
            "axes.labelsize": 11.0,
            "axes.titlesize": 12.0,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.0,
            "figure.dpi": 140,
            "savefig.dpi": 360,
        }
    )


def load_context() -> dict[str, object]:
    summary, traj = ov.build_raw_cache()
    meta_all, x_arrays = ov.load_processed_frames()
    raw_cycles, rep_eol = ov.representative_raw_data()
    heat_cycles, heat = ov.make_voltage_heatmap(meta_all, x_arrays)
    target_meta = pd.read_csv(ov.PROC_DIR / "clean_target_test_meta.csv")
    return {
        "summary": summary,
        "traj": traj,
        "meta_all": meta_all,
        "x_arrays": x_arrays,
        "raw_cycles": raw_cycles,
        "rep_eol": rep_eol,
        "heat_cycles": heat_cycles,
        "heat": heat,
        "target_meta": target_meta,
    }


def save_figure(fig: plt.Figure, out_base: Path, *, dpi: int = 360) -> list[Path]:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    paths = [out_base.with_suffix(ext) for ext in [".png", ".pdf", ".svg"]]
    fig.savefig(paths[0], dpi=dpi, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    fig.savefig(paths[1], bbox_inches="tight", pad_inches=0.06, facecolor="white")
    fig.savefig(paths[2], bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(fig)
    return paths


def new_panel(label: str, *, figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    ov.add_panel_label(ax, label)
    return fig, ax


def panel_a(ctx: dict[str, object]) -> plt.Figure:
    summary: pd.DataFrame = ctx["summary"]  # type: ignore[assignment]
    traj: pd.DataFrame = ctx["traj"]  # type: ignore[assignment]
    fig, ax = new_panel("a", figsize=(7.2, 4.25))
    norm = mpl.colors.Normalize(vmin=summary["eol_cycle"].min(), vmax=summary["eol_cycle"].max())
    cmap = mpl.cm.YlGnBu
    for _, g in traj.groupby("cell_id"):
        eol = float(g["eol_cycle"].iloc[0])
        ax.plot(g["cycle"], g["discharge_capacity_ah"], color=cmap(norm(eol)), lw=0.95, alpha=0.82)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.012, fraction=0.04)
    cbar.set_label("Cycle life", fontweight="bold")
    ax.set_xlabel("Cycle number")
    ax.set_ylabel("Discharge capacity (Ah)")
    ax.set_title("Cell-level capacity degradation trajectories", fontweight="bold")
    ax.set_xlim(0, summary["eol_cycle"].max() + 80)
    ov.clean_axes(ax)
    return fig


def panel_b(ctx: dict[str, object]) -> plt.Figure:
    meta_all: pd.DataFrame = ctx["meta_all"]  # type: ignore[assignment]
    fig, ax = new_panel("b", figsize=(4.35, 4.1))
    xx, yy, zz = ov.smooth_density(meta_all["cycle"].to_numpy(), meta_all["y_nominal"].to_numpy(), bins=(140, 95), sigma=1.45)
    levels = np.linspace(max(1e-9, zz[zz > 0].min()), zz.max(), 13)
    cf = ax.contourf(xx, yy, zz, levels=levels, cmap="Blues", antialiased=True)
    ax.contour(xx, yy, zz, levels=levels[::3], colors="#2B6C9B", linewidths=0.35, alpha=0.45)
    cbar = fig.colorbar(cf, ax=ax, pad=0.012, fraction=0.055)
    cbar.set_label("Smoothed samples", fontweight="bold")
    ax.axhline(0.8, color=ov.COLORS["red"], lw=1.1, ls="--")
    ax.set_xlabel("Cycle number")
    ax.set_ylabel("Nominal SOH")
    ax.set_title("Cycle-SOH sample density", fontweight="bold")
    ov.clean_axes(ax)
    return fig


def panel_c(ctx: dict[str, object]) -> plt.Figure:
    summary: pd.DataFrame = ctx["summary"]  # type: ignore[assignment]
    fig, ax = new_panel("c", figsize=(4.35, 4.1))
    batches = sorted(summary["batch"].unique())
    data = [summary.loc[summary["batch"] == b, "eol_cycle"].to_numpy() for b in batches]
    parts = ax.violinplot(data, positions=batches, widths=0.75, showmeans=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#9ECAE1")
        body.set_edgecolor("#2B6C9B")
        body.set_alpha(0.78)
    parts["cmeans"].set_color(ov.COLORS["red"])
    parts["cmeans"].set_linewidth(1.2)
    ax.scatter(summary["batch"], summary["eol_cycle"], s=13, color=ov.COLORS["dark"], alpha=0.5, zorder=3)
    ax.axvspan(0.5, 8.5, color=ov.COLORS["source"], alpha=0.06)
    ax.axvspan(8.5, 9.5, color=ov.COLORS["target_adapt"], alpha=0.08)
    ax.axvspan(9.5, 10.5, color=ov.COLORS["target_test"], alpha=0.08)
    ax.set_xticks(batches)
    ax.set_xlabel("Batch")
    ax.set_ylabel("Cycle life")
    ax.set_title("Cycle-life distribution across batches", fontweight="bold")
    ov.clean_axes(ax, grid="y")
    return fig


def panel_d(ctx: dict[str, object]) -> plt.Figure:
    summary: pd.DataFrame = ctx["summary"]  # type: ignore[assignment]
    fig, ax = new_panel("d", figsize=(4.35, 4.1))
    colors = [
        ov.COLORS["source"] if b <= 8 else ov.COLORS["target_adapt"] if b == 9 else ov.COLORS["target_test"]
        for b in summary["batch"]
    ]
    ax.scatter(summary["initial_capacity_ah"], summary["eol_cycle"], c=colors, s=34, edgecolor="white", linewidth=0.45)
    rho, _ = stats.spearmanr(summary["initial_capacity_ah"], summary["eol_cycle"])
    ax.text(0.04, 0.94, rf"$\rho$={rho:.3f}", transform=ax.transAxes, ha="left", va="top", fontsize=12.5)
    handles = [
        mpl.lines.Line2D([0], [0], marker="o", linestyle="", color=ov.COLORS["source"], label="Source"),
        mpl.lines.Line2D([0], [0], marker="o", linestyle="", color=ov.COLORS["target_adapt"], label="Target-dev"),
        mpl.lines.Line2D([0], [0], marker="o", linestyle="", color=ov.COLORS["target_test"], label="Blind test"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right", handlelength=0.8)
    ax.set_xlabel("Initial capacity (Ah)")
    ax.set_ylabel("Cycle life")
    ax.set_title("Initial capacity vs. cycle life", fontweight="bold")
    ov.clean_axes(ax)
    return fig


def panel_e(ctx: dict[str, object]) -> plt.Figure:
    raw_cycles: dict[int, pd.DataFrame] = ctx["raw_cycles"]  # type: ignore[assignment]
    fig, ax = new_panel("e", figsize=(4.55, 4.1))
    mid_cycle = sorted(raw_cycles.keys())[1]
    df_mid = raw_cycles[mid_cycle]
    t_min = df_mid["Time (s)"].to_numpy() / 60.0
    ax.plot(t_min, df_mid["Voltage (V)"], color=ov.COLORS["orange"], lw=1.45, label="Voltage")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Voltage (V)", color=ov.COLORS["orange"])
    ax.tick_params(axis="y", labelcolor=ov.COLORS["orange"])
    ax2 = ax.twinx()
    ax2.plot(t_min, df_mid["Current (mA)"].to_numpy() / 1000.0, color=ov.COLORS["blue"], lw=0.95, alpha=0.82, label="Current")
    ax2.set_ylabel("Current (A)", color=ov.COLORS["blue"])
    ax2.tick_params(axis="y", labelcolor=ov.COLORS["blue"], labelsize=9.5)
    ax.set_title(f"Raw voltage/current profile ({ov.REP_CELL}, cycle {mid_cycle})", fontweight="bold")
    ov.clean_axes(ax)
    ax2.spines["top"].set_visible(False)
    return fig


def panel_f(ctx: dict[str, object]) -> plt.Figure:
    raw_cycles: dict[int, pd.DataFrame] = ctx["raw_cycles"]  # type: ignore[assignment]
    fig, ax = new_panel("f", figsize=(4.35, 4.1))
    cycle_colors = [ov.COLORS["green"], ov.COLORS["blue"], ov.COLORS["red"]]
    for color, (cycle, df) in zip(cycle_colors, raw_cycles.items()):
        ch = ov.charge_segment(df)
        cap = ch["Capacity (mAh)"].to_numpy(dtype=float)
        x = 100 * (cap - cap.min()) / (cap.max() - cap.min()) if cap.max() > cap.min() else np.linspace(0, 100, len(cap))
        ax.plot(x, ch["Voltage (V)"], lw=1.75, color=color, label=f"Cycle {cycle}")
    ax.set_xlabel("Normalized charge capacity (%)")
    ax.set_ylabel("Charge voltage (V)")
    ax.set_title("Charge voltage evolution", fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    ov.clean_axes(ax)
    return fig


def panel_g(ctx: dict[str, object]) -> plt.Figure:
    heat_cycles: np.ndarray = ctx["heat_cycles"]  # type: ignore[assignment]
    heat: np.ndarray = ctx["heat"]  # type: ignore[assignment]
    fig, ax = new_panel("g", figsize=(4.45, 4.1))
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
    cbar = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.055)
    cbar.set_label("Voltage (V)", fontweight="bold")
    ax.set_xlabel("Interpolated point index")
    ax.set_ylabel("Cycle number")
    ax.set_title(f"Processed charge-voltage map ({ov.REP_CELL})", fontweight="bold")
    ov.clean_axes(ax, grid=None)
    return fig


def panel_h(ctx: dict[str, object]) -> plt.Figure:
    meta_all: pd.DataFrame = ctx["meta_all"]  # type: ignore[assignment]
    fig, ax = new_panel("h", figsize=(4.35, 4.1))
    bins = np.linspace(0.79, 1.13, 42)
    split_labels = {"source": "Source (B1-8)", "target_adapt": "Target-dev (B9)", "target_test": "Blind test (B10)"}
    for split, color in [("source", ov.COLORS["source"]), ("target_adapt", ov.COLORS["target_adapt"]), ("target_test", ov.COLORS["target_test"])]:
        vals = meta_all.loc[meta_all["split"] == split, "y_nominal"].to_numpy()
        ax.hist(vals, bins=bins, density=True, histtype="step", lw=1.75, color=color, label=split_labels[split])
    ax.set_xlabel("Nominal SOH")
    ax.set_ylabel("Density")
    ax.set_title("SOH label distribution by split", fontweight="bold")
    ax.legend(frameon=False)
    ov.clean_axes(ax, grid="y")
    return fig


def panel_i(ctx: dict[str, object]) -> plt.Figure:
    meta_all: pd.DataFrame = ctx["meta_all"]  # type: ignore[assignment]
    fig, ax = new_panel("i", figsize=(4.45, 4.1))
    sample = meta_all.sample(n=min(70000, len(meta_all)), random_state=20260710)
    xx, yy, zz = ov.smooth_density(sample["rul_cycle"].to_numpy(), sample["y_nominal"].to_numpy(), bins=(125, 90), sigma=1.45)
    levels = np.linspace(max(1e-9, zz[zz > 0].min()), zz.max(), 13)
    cf = ax.contourf(xx, yy, zz, levels=levels, cmap="YlGnBu", antialiased=True)
    ax.contour(xx, yy, zz, levels=levels[::3], colors="#2A788E", linewidths=0.35, alpha=0.45)
    cbar = fig.colorbar(cf, ax=ax, pad=0.012, fraction=0.055)
    cbar.set_label("Smoothed samples", fontweight="bold")
    ax.set_xlabel("RUL label (cycles)")
    ax.set_ylabel("Nominal SOH")
    ax.set_title("RUL-SOH label coupling", fontweight="bold")
    ov.clean_axes(ax)
    return fig


def panel_j(ctx: dict[str, object]) -> plt.Figure:
    raw_cycles: dict[int, pd.DataFrame] = ctx["raw_cycles"]  # type: ignore[assignment]
    fig, ax = new_panel("j", figsize=(4.35, 4.1))
    mid_cycle = sorted(raw_cycles.keys())[1]
    df_mid = raw_cycles[mid_cycle]
    raw = df_mid.iloc[:: max(1, len(df_mid) // 520), :].copy()
    sc = ax.scatter(raw["Voltage (V)"], raw["Current (mA)"] / 1000.0, c=raw["Time (s)"] / 60.0, cmap="viridis", s=12, alpha=0.85, linewidths=0)
    cbar = fig.colorbar(sc, ax=ax, pad=0.012, fraction=0.055)
    cbar.set_label("Time (min)", fontweight="bold")
    ax.axhline(0, color="#555555", lw=0.7)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (A)")
    ax.set_title("Voltage-current phase view", fontweight="bold")
    ov.clean_axes(ax)
    return fig


def panel_k(ctx: dict[str, object]) -> plt.Figure:
    raw_cycles: dict[int, pd.DataFrame] = ctx["raw_cycles"]  # type: ignore[assignment]
    fig, ax = new_panel("k", figsize=(4.45, 4.1))
    mid_cycle = sorted(raw_cycles.keys())[1]
    df_mid = raw_cycles[mid_cycle]
    v = df_mid["Voltage (V)"].to_numpy(dtype=float)
    t = df_mid["Time (s)"].to_numpy(dtype=float)
    fs = 1.0 / max(np.median(np.diff(t)), 1e-6) if len(t) > 1 else 1.0
    nperseg = min(96, max(16, len(v) // 6))
    f, tt, sxx = ov.signal.spectrogram(v - np.mean(v), fs=fs, nperseg=nperseg, noverlap=nperseg // 2, scaling="spectrum")
    sxx_db = gaussian_filter(10 * np.log10(sxx + 1e-12), sigma=(0.65, 0.55))
    im = ax.pcolormesh(tt / 60.0, f, sxx_db, cmap="magma", shading="gouraud")
    cbar = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.055)
    cbar.set_label("Power (dB)", fontweight="bold")
    ax.set_ylim(0, min(0.08, f.max() if len(f) else 0.08))
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Voltage time-frequency view", fontweight="bold")
    ov.clean_axes(ax, grid=None)
    return fig


PANEL_BUILDERS: dict[str, Callable[[dict[str, object]], plt.Figure]] = {
    "a": panel_a,
    "b": panel_b,
    "c": panel_c,
    "d": panel_d,
    "e": panel_e,
    "f": panel_f,
    "g": panel_g,
    "h": panel_h,
    "i": panel_i,
    "j": panel_j,
    "k": panel_k,
}


def export_split_panels(ctx: dict[str, object]) -> list[Path]:
    exported: list[Path] = []
    for label in list("abcdefghijk"):
        fig = PANEL_BUILDERS[label](ctx)
        stem = PANEL_DIR / f"Figure_dataset_overview_panel_{label}_{PANEL_NAMES[label]}"
        paths = save_figure(fig, stem, dpi=360)
        exported.append(paths[0])
    return exported


def positive_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def normalized_charge_curve(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    charge = ov.charge_segment(df)
    cap = charge["Capacity (mAh)"].to_numpy(dtype=float)
    voltage = charge["Voltage (V)"].to_numpy(dtype=float)
    if cap.max() > cap.min():
        cap_norm = 100.0 * (cap - cap.min()) / (cap.max() - cap.min())
    else:
        cap_norm = np.linspace(0.0, 100.0, len(cap))
    order = np.argsort(cap_norm)
    x_sorted = cap_norm[order]
    v_sorted = voltage[order]
    unique_x, unique_index = np.unique(x_sorted, return_index=True)
    unique_v = v_sorted[unique_index]
    xi = np.linspace(0.0, 100.0, 100)
    vi = np.interp(xi, unique_x, unique_v)
    return cap_norm, voltage, xi, vi


def flow_box(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=color,
        edgecolor="#263238",
        linewidth=1.0,
        alpha=0.95,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=11.0, fontweight="bold", color="#111827")
    ax.text(x + w / 2, y + h * 0.32, body, ha="center", va="center", fontsize=8.4, color="#263238")


def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.2,
        color="#3A3A3A",
        shrinkA=3,
        shrinkB=3,
    )
    ax.add_patch(arrow)


def export_processing_mechanism(ctx: dict[str, object]) -> Path:
    setup_publication_style()
    mpl.rcParams.update(
        {
            "axes.labelsize": 10.8,
            "axes.titlesize": 11.8,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "legend.fontsize": 8.8,
        }
    )

    meta_all: pd.DataFrame = ctx["meta_all"]  # type: ignore[assignment]
    x_arrays: dict[str, np.ndarray] = ctx["x_arrays"]  # type: ignore[assignment]
    raw_cycles: dict[int, pd.DataFrame] = ctx["raw_cycles"]  # type: ignore[assignment]
    target_meta: pd.DataFrame = ctx["target_meta"]  # type: ignore[assignment]

    rep_cycle = sorted(raw_cycles.keys())[1]
    df_mid = raw_cycles[rep_cycle]
    t_min = df_mid["Time (s)"].to_numpy(dtype=float) / 60.0
    voltage = df_mid["Voltage (V)"].to_numpy(dtype=float)
    current_a = df_mid["Current (mA)"].to_numpy(dtype=float) / 1000.0
    cap_norm, v_raw_charge, xi, vi = normalized_charge_curve(df_mid)

    target_cell_mask = target_meta["cell_id"].eq(ov.REP_CELL)
    target_positions = np.flatnonzero(target_cell_mask.to_numpy())
    target_cell = target_meta.loc[target_cell_mask].reset_index(drop=True)
    sample_pos = np.linspace(0, len(target_positions) - 1, min(160, len(target_positions))).astype(int)
    matrix_cycles = target_cell.iloc[sample_pos]["cycle"].to_numpy(dtype=float)
    matrix = np.asarray(x_arrays["target_test"][target_positions[sample_pos], :], dtype=float)
    matrix = gaussian_filter(matrix, sigma=(0.65, 0.35))

    final_blocks = []
    final_labels = []
    for split in ["source", "target_adapt", "target_test"]:
        arr = np.asarray(x_arrays[split], dtype=np.float32)
        take = np.linspace(0, arr.shape[0] - 1, 34).astype(int)
        final_blocks.append(np.asarray(arr[take, :], dtype=float))
        final_labels.append(split)
    final_matrix = gaussian_filter(np.vstack(final_blocks), sigma=(0.45, 0.35))

    split_counts = meta_all.groupby("split").size().reindex(["source", "target_adapt", "target_test"])
    split_colors = [ov.COLORS["source"], ov.COLORS["target_adapt"], ov.COLORS["target_test"]]

    fig = plt.figure(figsize=(17.4, 9.25), dpi=320)
    fig.suptitle("Data processing pipeline for cross-batch battery SOH samples", fontsize=16.5, fontweight="bold", y=0.985)

    flow = fig.add_axes([0.045, 0.835, 0.91, 0.115])
    flow.axis("off")
    boxes = [
        ("Raw cycling record", "V(t), I(t), Q(t)", "#EAF4FB"),
        ("Charge segment cut", "positive current region", "#FFF3D7"),
        ("Capacity normalization", "Qch -> 0-100%", "#EBF7EE"),
        ("Fixed-length interpolation", "100 voltage points", "#F1ECFA"),
        ("Sample tensor assembly", "X in R^(N x 100)", "#FBEDEA"),
        ("SOH/RUL labels and split", "B1-8 / B9 / B10", "#EEF2FF"),
    ]
    w = 0.142
    gap = 0.022
    y = 0.18
    h = 0.66
    xs = [0.0 + i * (w + gap) for i in range(len(boxes))]
    for x, (title, body, color) in zip(xs, boxes):
        flow_box(flow, x, y, w, h, title, body, color)
    for i in range(len(xs) - 1):
        draw_arrow(flow, (xs[i] + w, y + h / 2), (xs[i + 1], y + h / 2))
    flow.set_xlim(-0.01, 1.0)
    flow.set_ylim(0, 1)

    gs = fig.add_gridspec(2, 5, left=0.052, right=0.985, bottom=0.075, top=0.78, wspace=0.68, hspace=0.55)

    ax = fig.add_subplot(gs[0, 0:2])
    ax.plot(t_min, voltage, color=ov.COLORS["orange"], lw=1.55, label="Voltage")
    for start, end in positive_segments(current_a > 0):
        ax.axvspan(t_min[start], t_min[end], color="#FFE2A7", alpha=0.48, lw=0)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Voltage (V)", color=ov.COLORS["orange"])
    ax.tick_params(axis="y", labelcolor=ov.COLORS["orange"])
    ax2 = ax.twinx()
    ax2.plot(t_min, current_a, color=ov.COLORS["blue"], lw=0.95, alpha=0.82, label="Current")
    ax2.set_ylabel("Current (A)", color=ov.COLORS["blue"])
    ax2.tick_params(axis="y", labelcolor=ov.COLORS["blue"], labelsize=9.2)
    ax.text(0.02, 0.92, "charge segment selected by I > 0", transform=ax.transAxes, fontsize=9.2, fontweight="bold", color="#9A5B00")
    ax.set_title(f"a  Raw voltage/current cycle record ({ov.REP_CELL}, cycle {rep_cycle})", fontweight="bold", loc="left")
    ov.clean_axes(ax)
    ax2.spines["top"].set_visible(False)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(cap_norm, v_raw_charge, color="#8A8F98", lw=0.9, alpha=0.65, label="raw charge samples")
    ax.plot(xi, vi, color=ov.COLORS["red"], lw=1.8, label="100-point interpolation")
    ax.scatter(xi[::10], vi[::10], s=16, color=ov.COLORS["red"], edgecolor="white", linewidth=0.35, zorder=4)
    ax.set_xlabel("Normalized charge capacity (%)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("b  Charge curve normalization", fontweight="bold", loc="left")
    ax.legend(frameon=False, loc="lower right")
    ov.clean_axes(ax)

    ax = fig.add_subplot(gs[0, 3:5])
    for color, (cycle, df) in zip([ov.COLORS["green"], ov.COLORS["blue"], ov.COLORS["red"]], raw_cycles.items()):
        cap_n, vv, _, _ = normalized_charge_curve(df)
        ax.plot(cap_n, vv, lw=1.55, color=color, label=f"Cycle {cycle}")
    ax.set_xlabel("Normalized charge capacity (%)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("c  Cycle-wise slicing along the aging path", fontweight="bold", loc="left")
    ax.legend(frameon=False, loc="lower right")
    ov.clean_axes(ax)

    ax = fig.add_subplot(gs[1, 0:2])
    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        cmap="turbo",
        interpolation="bicubic",
        extent=[1, 100, float(matrix_cycles.min()), float(matrix_cycles.max())],
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.035)
    cbar.set_label("Voltage (V)", fontweight="bold")
    ax.set_xlabel("Interpolated point index")
    ax.set_ylabel("Cycle number")
    ax.set_title(f"d  Processed charge-voltage matrix for cell {ov.REP_CELL}", fontweight="bold", loc="left")
    ov.clean_axes(ax, grid=None)

    ax = fig.add_subplot(gs[1, 2])
    im2 = ax.imshow(final_matrix, aspect="auto", origin="lower", cmap="viridis", interpolation="nearest")
    boundaries = np.cumsum([block.shape[0] for block in final_blocks])
    for boundary in boundaries[:-1]:
        ax.axhline(boundary - 0.5, color="white", lw=1.0)
    ax.set_yticks([17, 51, 85])
    ax.set_yticklabels(["B1-8\nsource", "B9\ntarget-dev", "B10\nblind-test"], fontweight="bold")
    ax.set_xticks([0, 49, 99])
    ax.set_xticklabels(["1", "50", "100"])
    ax.set_xlabel("Voltage point")
    ax.set_title("e  Model-ready voltage tensor", fontweight="bold", loc="left")
    fig.colorbar(im2, ax=ax, pad=0.015, fraction=0.055).set_label("Voltage (V)", fontweight="bold")
    ov.clean_axes(ax, grid=None)

    ax = fig.add_subplot(gs[1, 3])
    label_sample = meta_all.sample(n=min(45000, len(meta_all)), random_state=20260731)
    sc = ax.scatter(
        label_sample["rul_cycle"],
        label_sample["y_nominal"],
        c=label_sample["cycle"],
        cmap="YlGnBu",
        s=3,
        alpha=0.18,
        linewidths=0,
    )
    ax.scatter(target_cell["rul_cycle"].iloc[::25], target_cell["y_nominal"].iloc[::25], s=8, color=ov.COLORS["target_test"], alpha=0.75, label=ov.REP_CELL)
    ax.set_xlabel("RUL label (cycles)")
    ax.set_ylabel("Nominal SOH")
    ax.set_title("f  SOH/RUL label construction", fontweight="bold", loc="left")
    ax.legend(frameon=False, loc="lower right")
    fig.colorbar(sc, ax=ax, pad=0.015, fraction=0.055).set_label("Cycle", fontweight="bold")
    ov.clean_axes(ax)

    ax = fig.add_subplot(gs[1, 4])
    labels = ["Source\nB1-8", "Target-dev\nB9", "Blind test\nB10"]
    bars = ax.bar(labels, split_counts.to_numpy(), color=split_colors, edgecolor="white", linewidth=0.8)
    for bar, count in zip(bars, split_counts.to_numpy()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(count):,}", ha="center", va="bottom", fontsize=9.0, fontweight="bold")
    total = int(split_counts.sum())
    ax.text(
        0.04,
        0.92,
        f"Total samples: {total:,}\nFeature length: 100\nLabels: SOH, RUL, batch",
        transform=ax.transAxes,
        fontsize=9.2,
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#C7CDD6", linewidth=0.8),
    )
    ax.set_ylabel("Samples")
    ax.set_title("g  Cross-batch split protocol", fontweight="bold", loc="left")
    ov.clean_axes(ax, grid="y")

    out = FIG_DIR / "Figure_data_processing_mechanism_pipeline"
    paths = save_figure(fig, out, dpi=320)
    return paths[0]


def export_processing_display_panels(ctx: dict[str, object]) -> list[Path]:
    setup_publication_style()
    mpl.rcParams.update(
        {
            "axes.labelsize": 11.2,
            "axes.titlesize": 12.4,
            "xtick.labelsize": 9.6,
            "ytick.labelsize": 9.6,
            "legend.fontsize": 9.0,
        }
    )

    meta_all: pd.DataFrame = ctx["meta_all"]  # type: ignore[assignment]
    x_arrays: dict[str, np.ndarray] = ctx["x_arrays"]  # type: ignore[assignment]
    raw_cycles: dict[int, pd.DataFrame] = ctx["raw_cycles"]  # type: ignore[assignment]
    target_meta: pd.DataFrame = ctx["target_meta"]  # type: ignore[assignment]

    rep_cycle = sorted(raw_cycles.keys())[1]
    df_mid = raw_cycles[rep_cycle]
    t_min = df_mid["Time (s)"].to_numpy(dtype=float) / 60.0
    voltage = df_mid["Voltage (V)"].to_numpy(dtype=float)
    current_a = df_mid["Current (mA)"].to_numpy(dtype=float) / 1000.0
    cap_norm, v_raw_charge, xi, vi = normalized_charge_curve(df_mid)

    target_cell_mask = target_meta["cell_id"].eq(ov.REP_CELL)
    target_positions = np.flatnonzero(target_cell_mask.to_numpy())
    target_cell = target_meta.loc[target_cell_mask].reset_index(drop=True)
    sample_pos = np.linspace(0, len(target_positions) - 1, min(160, len(target_positions))).astype(int)
    matrix_cycles = target_cell.iloc[sample_pos]["cycle"].to_numpy(dtype=float)
    matrix = np.asarray(x_arrays["target_test"][target_positions[sample_pos], :], dtype=float)
    matrix = gaussian_filter(matrix, sigma=(0.65, 0.35))

    final_blocks = []
    for split in ["source", "target_adapt", "target_test"]:
        arr = np.asarray(x_arrays[split], dtype=np.float32)
        take = np.linspace(0, arr.shape[0] - 1, 34).astype(int)
        final_blocks.append(np.asarray(arr[take, :], dtype=float))
    final_matrix = gaussian_filter(np.vstack(final_blocks), sigma=(0.45, 0.35))

    split_counts = meta_all.groupby("split").size().reindex(["source", "target_adapt", "target_test"])
    split_colors = [ov.COLORS["source"], ov.COLORS["target_adapt"], ov.COLORS["target_test"]]

    exported: list[Path] = []

    fig, ax = new_panel("a", figsize=(7.4, 4.2))
    ax.plot(t_min, voltage, color=ov.COLORS["orange"], lw=1.65, label="Voltage")
    for start, end in positive_segments(current_a > 0):
        ax.axvspan(t_min[start], t_min[end], color="#FFE2A7", alpha=0.48, lw=0)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Voltage (V)", color=ov.COLORS["orange"])
    ax.tick_params(axis="y", labelcolor=ov.COLORS["orange"])
    ax2 = ax.twinx()
    ax2.plot(t_min, current_a, color=ov.COLORS["blue"], lw=1.0, alpha=0.82, label="Current")
    ax2.set_ylabel("Current (A)", color=ov.COLORS["blue"])
    ax2.tick_params(axis="y", labelcolor=ov.COLORS["blue"], labelsize=9.6)
    ax.text(0.02, 0.92, "charge segment selected by I > 0", transform=ax.transAxes, fontsize=9.4, fontweight="bold", color="#9A5B00")
    ax.set_title(f"Raw voltage/current cycle record ({ov.REP_CELL}, cycle {rep_cycle})", fontweight="bold")
    ov.clean_axes(ax)
    ax2.spines["top"].set_visible(False)
    exported.append(save_figure(fig, PROCESSING_PANEL_DIR / "Figure_processing_panel_a_raw_voltage_current_cycle_record", dpi=360)[0])

    fig, ax = new_panel("b", figsize=(4.45, 4.1))
    ax.plot(cap_norm, v_raw_charge, color="#8A8F98", lw=0.95, alpha=0.65, label="raw charge samples")
    ax.plot(xi, vi, color=ov.COLORS["red"], lw=1.9, label="100-point interpolation")
    ax.scatter(xi[::10], vi[::10], s=18, color=ov.COLORS["red"], edgecolor="white", linewidth=0.35, zorder=4)
    ax.set_xlabel("Normalized charge capacity (%)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Charge curve normalization", fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    ov.clean_axes(ax)
    exported.append(save_figure(fig, PROCESSING_PANEL_DIR / "Figure_processing_panel_b_charge_curve_normalization", dpi=360)[0])

    fig, ax = new_panel("c", figsize=(6.2, 4.1))
    for color, (cycle, df) in zip([ov.COLORS["green"], ov.COLORS["blue"], ov.COLORS["red"]], raw_cycles.items()):
        cap_n, vv, _, _ = normalized_charge_curve(df)
        ax.plot(cap_n, vv, lw=1.7, color=color, label=f"Cycle {cycle}")
    ax.set_xlabel("Normalized charge capacity (%)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Cycle-wise slicing along the aging path", fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    ov.clean_axes(ax)
    exported.append(save_figure(fig, PROCESSING_PANEL_DIR / "Figure_processing_panel_c_cycle_wise_slicing", dpi=360)[0])

    fig, ax = new_panel("d", figsize=(6.5, 4.25))
    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        cmap="turbo",
        interpolation="bicubic",
        extent=[1, 100, float(matrix_cycles.min()), float(matrix_cycles.max())],
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.04)
    cbar.set_label("Voltage (V)", fontweight="bold")
    ax.set_xlabel("Interpolated point index")
    ax.set_ylabel("Cycle number")
    ax.set_title(f"Processed charge-voltage matrix for cell {ov.REP_CELL}", fontweight="bold")
    ov.clean_axes(ax, grid=None)
    exported.append(save_figure(fig, PROCESSING_PANEL_DIR / "Figure_processing_panel_d_processed_charge_voltage_matrix", dpi=360)[0])

    fig, ax = new_panel("e", figsize=(4.55, 4.25))
    im2 = ax.imshow(final_matrix, aspect="auto", origin="lower", cmap="viridis", interpolation="nearest")
    boundaries = np.cumsum([block.shape[0] for block in final_blocks])
    for boundary in boundaries[:-1]:
        ax.axhline(boundary - 0.5, color="white", lw=1.0)
    ax.set_yticks([17, 51, 85])
    ax.set_yticklabels(["B1-8\nsource", "B9\ntarget-dev", "B10\nblind-test"], fontweight="bold")
    ax.set_xticks([0, 49, 99])
    ax.set_xticklabels(["1", "50", "100"])
    ax.set_xlabel("Voltage point")
    ax.set_title("Model-ready voltage tensor", fontweight="bold")
    cbar = fig.colorbar(im2, ax=ax, pad=0.015, fraction=0.055)
    cbar.set_label("Voltage (V)", fontweight="bold")
    ov.clean_axes(ax, grid=None)
    exported.append(save_figure(fig, PROCESSING_PANEL_DIR / "Figure_processing_panel_e_model_ready_voltage_tensor", dpi=360)[0])

    fig, ax = new_panel("f", figsize=(4.65, 4.25))
    label_sample = meta_all.sample(n=min(45000, len(meta_all)), random_state=20260731)
    sc = ax.scatter(
        label_sample["rul_cycle"],
        label_sample["y_nominal"],
        c=label_sample["cycle"],
        cmap="YlGnBu",
        s=3,
        alpha=0.18,
        linewidths=0,
    )
    ax.scatter(target_cell["rul_cycle"].iloc[::25], target_cell["y_nominal"].iloc[::25], s=9, color=ov.COLORS["target_test"], alpha=0.78, label=ov.REP_CELL)
    ax.set_xlabel("RUL label (cycles)")
    ax.set_ylabel("Nominal SOH")
    ax.set_title("SOH/RUL label construction", fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    cbar = fig.colorbar(sc, ax=ax, pad=0.015, fraction=0.055)
    cbar.set_label("Cycle", fontweight="bold")
    ov.clean_axes(ax)
    exported.append(save_figure(fig, PROCESSING_PANEL_DIR / "Figure_processing_panel_f_soh_rul_label_construction", dpi=360)[0])

    fig, ax = new_panel("g", figsize=(4.65, 4.25))
    labels = ["Source\nB1-8", "Target-dev\nB9", "Blind test\nB10"]
    bars = ax.bar(labels, split_counts.to_numpy(), color=split_colors, edgecolor="white", linewidth=0.8)
    for bar, count in zip(bars, split_counts.to_numpy()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(count):,}", ha="center", va="bottom", fontsize=9.4, fontweight="bold")
    total = int(split_counts.sum())
    ax.text(
        0.04,
        0.92,
        f"Total samples: {total:,}\nFeature length: 100\nLabels: SOH, RUL, batch",
        transform=ax.transAxes,
        fontsize=9.4,
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#C7CDD6", linewidth=0.8),
    )
    ax.set_ylabel("Samples")
    ax.set_title("Cross-batch split protocol", fontweight="bold")
    ov.clean_axes(ax, grid="y")
    exported.append(save_figure(fig, PROCESSING_PANEL_DIR / "Figure_processing_panel_g_cross_batch_split_protocol", dpi=360)[0])

    return exported


def write_manifest(panel_paths: list[Path], mechanism_path: Path, processing_panel_paths: list[Path]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = TABLE_DIR / "dataset_panel_and_mechanism_exports.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "panel", "description", "png_path"])
        for label, path in zip(list("abcdefghijk"), panel_paths):
            writer.writerow(["split_panel", label, PANEL_NAMES[label], str(path)])
        writer.writerow(["mechanism", "", "data_processing_pipeline", str(mechanism_path)])
        for label, path in zip(list("abcdefg"), processing_panel_paths):
            writer.writerow(["processing_display_panel", label, path.stem.replace(f"Figure_processing_panel_{label}_", ""), str(path)])


def main() -> None:
    setup_publication_style()
    ctx = load_context()
    panel_paths = export_split_panels(ctx)
    mechanism_path = export_processing_mechanism(ctx)
    processing_panel_paths = export_processing_display_panels(ctx)
    write_manifest(panel_paths, mechanism_path, processing_panel_paths)
    print("Split panels:")
    for path in panel_paths:
        print(path)
    print("Mechanism:")
    print(mechanism_path)
    print("Processing display panels:")
    for path in processing_panel_paths:
        print(path)


if __name__ == "__main__":
    main()
