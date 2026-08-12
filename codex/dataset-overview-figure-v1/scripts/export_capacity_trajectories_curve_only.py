"""Export transparent curve-only capacity degradation trajectories."""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "our_data"
OUT_DIR = ROOT / "codex" / "dataset-overview-figure-v1"
FIG_DIR = OUT_DIR / "figures"


def natural_key(path: Path) -> tuple[int, int]:
    m = re.match(r"(\d+)-(\d+)", path.stem)
    return (int(m.group(1)), int(m.group(2))) if m else (999, 999)


def load_cell(path: Path) -> tuple[str, np.ndarray, np.ndarray, int]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    cell_id = next(iter(obj))
    dq = obj[cell_id]["dq"]
    cycles = np.array(sorted(dq.keys()), dtype=float)
    capacity_ah = np.array([float(dq[int(c)]) / 1000.0 for c in cycles], dtype=float)
    return cell_id, cycles, capacity_ah, int(cycles.max())


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    records = [load_cell(path) for path in sorted(RAW_DIR.glob("*.pkl"), key=natural_key)]
    eol_values = np.array([record[3] for record in records], dtype=float)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.unicode_minus": False,
            "path.simplify": True,
            "path.simplify_threshold": 0.35,
        }
    )

    cmap = mpl.cm.YlGnBu
    norm = mpl.colors.Normalize(vmin=float(eol_values.min()), vmax=float(eol_values.max()))

    fig, ax = plt.subplots(figsize=(7.2, 4.25))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    for _, cycles, capacity_ah, eol in records:
        ax.plot(
            cycles,
            capacity_ah,
            color=cmap(norm(float(eol))),
            lw=2.25,
            alpha=0.72,
            solid_capstyle="round",
            solid_joinstyle="round",
        )

    ax.set_xlim(-15, float(eol_values.max()) + 90)
    ax.set_ylim(0.865, 1.235)
    ax.axis("off")

    png_path = FIG_DIR / "capacity_degradation_trajectories_curve_only_transparent.png"
    svg_path = FIG_DIR / "capacity_degradation_trajectories_curve_only_transparent.svg"
    fig.savefig(png_path, dpi=600, transparent=True, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(svg_path, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"saved: {png_path}")
    print(f"saved: {svg_path}")


if __name__ == "__main__":
    main()
