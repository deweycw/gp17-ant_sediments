"""
Supplemental Figures: Pixel-level XRF element correlations.
Generates one figure per element pair (Fe vs Ca, Fe vs Ti, Fe vs Mn, Fe vs K),
each showing density scatter plots for all 16 maps in a 4×4 grid.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import pearsonr
from pathlib import Path
import h5py

# ---------- Config ----------
DPI = 300
TARGET_WIDTH_MM = 180
TARGET_WIDTH_IN = TARGET_WIDTH_MM / 25.4

MAP_DIR = Path("maps")
OUT_DIR = Path("supplemental_figures")
OUT_DIR.mkdir(exist_ok=True)

ELEMENTS = ["Fe Ka", "Ca Ka", "K Ka", "Ti Ka", "Mn Ka"]
SHORT = {e: e.replace(" Ka", "") for e in ELEMENTS}

PAIRS = [("Fe", "Ca"), ("Fe", "Ti"), ("Fe", "Mn"), ("Fe", "K")]


def get_roi_map(f, roi_name):
    names = [n.decode() if isinstance(n, bytes) else n for n in f["xrmmap/roimap/sum_name"][:]]
    if roi_name in names:
        idx = names.index(roi_name)
        return f["xrmmap/roimap/sum_cor"][:, 1:-1, idx].astype(float)
    return None


# ---------- Discover maps (exclude _002 rescans) ----------
all_h5 = sorted([
    p for p in MAP_DIR.glob("*.h5")
    if "test_map" not in p.name and not p.stem.endswith("_002")
    and "elongated_particle" not in p.name
])
map_names = [p.stem for p in all_h5]
print(f"Found {len(all_h5)} maps")

# ---------- Pre-load element maps ----------
cached_maps = {}
for p in all_h5:
    with h5py.File(p, "r") as f:
        maps = {}
        for elem in ELEMENTS:
            m = get_roi_map(f, elem)
            if m is not None:
                maps[SHORT[elem]] = m
    cached_maps[p.stem] = maps
print("Maps loaded")


# ---------- Map label dictionary (filename stem → display label) ----------
MAP_LABELS = {name: f"Map {i+1}" for i, name in enumerate(map_names)}

# Print mapping for reference
print("\nMap label assignments:")
for name, label in MAP_LABELS.items():
    short = (name.replace("_001", "").replace("2x2_10um_", "")
             .replace("1x1_10um_", "").replace("_", " "))
    print(f"  {label}: {short}")


# ---------- Generate figures ----------
from matplotlib.gridspec import GridSpec

n_maps = len(map_names)
ncols = 4
nrows = int(np.ceil(n_maps / ncols))
last_row_count = n_maps - (nrows - 1) * ncols  # panels in last row
panel_w = TARGET_WIDTH_IN / ncols
panel_h = panel_w * 0.9
fig_h = panel_h * nrows + 0.3

for x_elem, y_elem in PAIRS:
    fig = plt.figure(figsize=(TARGET_WIDTH_IN, fig_h))
    gs = GridSpec(nrows, 2 * ncols, figure=fig, hspace=0.5, wspace=1.0)

    ax_list = []
    for i in range(n_maps):
        row = i // ncols
        col = i % ncols
        if row < nrows - 1:
            # Full rows: each panel spans 2 gridspec columns
            ax = fig.add_subplot(gs[row, col * 2:(col + 1) * 2])
        else:
            # Last row: center by offsetting into the 2*ncols grid
            offset = ncols - last_row_count  # offset in half-column units
            ax = fig.add_subplot(gs[row, offset + col * 2:offset + (col + 1) * 2])
        ax_list.append(ax)

    for i, name in enumerate(map_names):
        ax = ax_list[i]
        maps = cached_maps[name]

        label = MAP_LABELS[name]

        if x_elem not in maps or y_elem not in maps:
            ax.set_title(f"{label}\n(missing)", fontsize=5)
            ax.axis("off")
            continue

        xd = maps[x_elem].ravel()
        yd = maps[y_elem].ravel()
        mask = (xd > 0) & (yd > 0)

        if mask.sum() < 10:
            ax.set_title(f"{label}\n(too few px)", fontsize=5)
            ax.axis("off")
            continue

        r, _ = pearsonr(xd[mask], yd[mask])
        ax.hist2d(xd[mask], yd[mask], bins=80, cmap="inferno",
                  norm=LogNorm(), rasterized=True)
        ax.set_title(label, fontsize=5, pad=2)
        ax.text(0.95, 0.95, f"r = {r:.2f}", transform=ax.transAxes,
                fontsize=4.5, ha="right", va="top",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))
        ax.tick_params(labelsize=4)
        ax.ticklabel_format(style="scientific", axis="both", scilimits=(0, 0))
        ax.xaxis.get_offset_text().set_fontsize(4)
        ax.yaxis.get_offset_text().set_fontsize(4)
        ax.set_xlabel(f"{x_elem} K\u03b1", fontsize=6, labelpad=1)
        ax.set_ylabel(f"{y_elem} K\u03b1", fontsize=6, labelpad=1)

    stem = f"supp_xrf_{x_elem}_vs_{y_elem}"
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=DPI,
                bbox_inches="tight", pad_inches=0.05, facecolor="white")
    fig.savefig(OUT_DIR / f"{stem}.pdf", dpi=DPI,
                bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"Saved {stem}.png / .pdf")

print("Done.")
