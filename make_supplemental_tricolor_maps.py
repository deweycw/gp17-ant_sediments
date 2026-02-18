"""
Supplemental Figures: Individual tricolor µ-XRF maps.
Renders one figure per map as RGB composite with XANES point locations
colored by cluster, scale bar, cluster legend, and tricolor legend.

Usage:
    python make_supplemental_tricolor_maps.py                     # default Fe/Ca/K
    python make_supplemental_tricolor_maps.py --red Fe --green Ca --blue K
    python make_supplemental_tricolor_maps.py --red Ti --green Fe --blue Mn
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from pathlib import Path
import h5py

# ---------- CLI arguments ----------
parser = argparse.ArgumentParser(description="Generate supplemental tricolor maps.")
parser.add_argument("--red", default="Fe",
                    help="Element for red channel (default: Fe)")
parser.add_argument("--green", default="Ca",
                    help="Element for green channel (default: Ca)")
parser.add_argument("--blue", default="K",
                    help="Element for blue channel (default: K)")
args = parser.parse_args()

# ---------- Config ----------
DPI = 300

MAP_DIR = Path("maps")
PCA_DIR = Path("pca_results")
OUT_DIR = Path("supplemental_figures")
OUT_DIR.mkdir(exist_ok=True)

# Map short element name to ROI name (element Ka)
R_ROI = f"{args.red} Ka"
G_ROI = f"{args.green} Ka"
B_ROI = f"{args.blue} Ka"

print(f"RGB channels: R={args.red}, G={args.green}, B={args.blue}")

# Map labels (consistent across all figures)
MAP_LABELS = {
    "1x1_10um_flaky_dark_gt15_001": "Map 1",
    "1x1_10um_flaky_gray_mix_gt15_001": "Map 2",
    "1x1_10um_rectangles_flakes_gt15_2_001": "Map 3",
    "2x2_10um_concentric_gray_1_001": "Map 4",
    "2x2_10um_concentric_gray_3_001": "Map 5",
    "2x2_10um_flaky_1_001": "Map 6",
    "2x2_10um_flaky_2_001": "Map 7",
    "2x2_10um_flaky_nodule_001": "Map 8",
    "2x2_10um_flaky_smooth_2_001": "Map 9",
    "2x2_10um_rectangles_gt15_1_001": "Map 10",
    "2x2_10um_striated_gt15_2_001": "Map 11",
    "2x2_10um_super_dark_gt15_4_001": "Map 12",
    "2x2_10um_white_band_001": "Map 13",
}

CLUSTER_STYLE = {
    1:    {"marker": "o", "label": "Group 1"},
    2:    {"marker": "s", "label": "Group 2"},
    "3a": {"marker": "^", "label": "Group 3a"},
    "3b": {"marker": "v", "label": "Group 3b"},
    4:    {"marker": "D", "label": "Group 4"},
    5:    {"marker": "p", "label": "Group 5"},
}


# ---------- HDF5 helpers ----------
def area_to_spectrum(area_name):
    return f"FeXANES_{area_name}.001"


def get_roi_map(f, roi_name):
    for path in ["xrmmap/roimap/sum_cor", "xrmmap/roimap/sum_raw"]:
        if path in f:
            names = [n.decode() if isinstance(n, bytes) else n
                     for n in f["xrmmap/roimap/sum_name"][:]]
            if roi_name in names:
                idx = names.index(roi_name)
                return f[path][:, 1:-1, idx].astype(float)
    return None


def make_rgb(f):
    channels = []
    for name in [R_ROI, G_ROI, B_ROI]:
        ch = get_roi_map(f, name)
        if ch is None:
            ch = np.zeros((1, 1))
        vmin = np.percentile(ch, 1)
        vmax = np.percentile(ch, 99.5)
        if vmax > vmin:
            ch = np.clip((ch - vmin) / (vmax - vmin), 0, 1)
        else:
            ch = np.zeros_like(ch)
        channels.append(ch)
    return np.stack(channels, axis=-1)


def get_area_centroids(f):
    centroids = {}
    areas_grp = f.get("xrmmap/areas")
    if areas_grp is None:
        return centroids
    for area_name in areas_grp:
        mask = areas_grp[area_name][:]
        if mask.any():
            rows, cols = np.where(mask)
            centroids[area_name] = (rows.mean(), cols.mean())
    return centroids


def find_clear_corner(extent, xanes_points, bar_length_mm):
    """Find a corner for the scale bar that doesn't overlap XANES markers.
    Returns (x_frac, corner_name) where x_frac is the axes-fraction x position
    for the right end of the bar. Tries bottom-right, bottom-left, top-right, top-left."""
    x_range = extent[1] - extent[0]
    y_range = extent[3] - extent[2]
    bar_frac = bar_length_mm / x_range
    # Margin from edge in data coords
    mx = x_range * 0.05
    my = y_range * 0.08

    # Collect all XANES point positions
    all_pts = []
    for sk in xanes_points:
        xs, ys = xanes_points[sk]
        for x, y in zip(xs, ys):
            all_pts.append((x, y))

    corners = [
        ("bottom-right", 0.97, 0.05),
        ("bottom-left",  bar_frac + 0.03, 0.05),
        ("top-right",    0.97, 0.93),
        ("top-left",     bar_frac + 0.03, 0.93),
    ]

    for name, x_end_frac, y_frac in corners:
        # Convert bar region to data coordinates
        x_start_data = extent[0] + (x_end_frac - bar_frac) * x_range
        x_end_data = extent[0] + x_end_frac * x_range
        y_data = extent[2] + y_frac * y_range

        conflict = False
        for px, py in all_pts:
            if (x_start_data - mx <= px <= x_end_data + mx and
                    y_data - my <= py <= y_data + my):
                conflict = True
                break
        if not conflict:
            return x_end_frac, y_frac, name

    # Default to bottom-right if all corners have conflicts
    return 0.97, 0.05, "bottom-right"


def add_scale_bar(ax, extent, xanes_points, bar_length_mm=0.5):
    x_range = extent[1] - extent[0]
    bar_frac = bar_length_mm / x_range
    x_end_frac, y_frac, corner = find_clear_corner(
        extent, xanes_points, bar_length_mm)
    x_start_frac = x_end_frac - bar_frac

    # Draw bar
    ax.plot([x_start_frac, x_end_frac], [y_frac, y_frac], color="white",
            linewidth=3, solid_capstyle="butt", zorder=10,
            clip_on=False, transform=ax.transAxes)

    # Label above or below depending on corner
    if "bottom" in corner:
        text_y = y_frac + 0.03
        va = "bottom"
    else:
        text_y = y_frac - 0.03
        va = "top"

    label = f"{bar_length_mm:.1f} mm" if bar_length_mm < 1 else f"{bar_length_mm:.0f} mm"
    ax.text((x_start_frac + x_end_frac) / 2, text_y, label,
            color="white", fontsize=9, ha="center", va=va,
            fontweight="bold", zorder=10, clip_on=False,
            transform=ax.transAxes)
    return corner


# Map scale bar corner → best tricolor legend location (opposite corner)
_LEGEND_LOC = {
    "bottom-right": "upper left",
    "bottom-left":  "upper right",
    "top-right":    "lower left",
    "top-left":     "lower right",
}


# ---------- Load cluster assignments + sub-clusters ----------
print("Loading cluster assignments...")
cluster_df = pd.read_csv(PCA_DIR / "cluster_assignments.csv")
cluster_lookup = dict(zip(cluster_df["spectrum"], cluster_df["cluster"]))

c3_df = cluster_df[cluster_df["cluster"] == 3].copy()
pc_cols = [c for c in cluster_df.columns if c.startswith("PC")]
km = KMeans(n_clusters=2, random_state=42, n_init=10)
c3_labels = km.fit_predict(c3_df[pc_cols].values)

lcf_df = pd.read_csv(PCA_DIR / "lcf_individual.csv")
c3_lcf = lcf_df[lcf_df["cluster"] == 3].copy()
c3_lcf["sub_label"] = c3_labels
sub0_pyrr = c3_lcf[c3_lcf["sub_label"] == 0]["Pyrrhotite"].mean()
sub1_pyrr = c3_lcf[c3_lcf["sub_label"] == 1]["Pyrrhotite"].mean()
label_map = {0: "3a", 1: "3b"} if sub0_pyrr >= sub1_pyrr else {1: "3a", 0: "3b"}

sub3_lookup = {}
for i, (_, row) in enumerate(c3_df.iterrows()):
    sub3_lookup[row["spectrum"]] = label_map[c3_labels[i]]


def get_style_key(spec_name, cluster_id):
    if cluster_id == 3:
        return sub3_lookup.get(spec_name, "3a")
    return cluster_id


# ---------- Build legend handles ----------
cluster_handles = []
for sk, style in CLUSTER_STYLE.items():
    cluster_handles.append(Line2D(
        [0], [0], marker=style["marker"], color="none",
        markerfacecolor="none", markeredgecolor="white",
        markersize=7, markeredgewidth=1.2, label=style["label"],
    ))

tricolor_handles = [
    Line2D([0], [0], marker="s", color="none",
           markerfacecolor="red", markersize=8, label=f"{args.red} K\u03b1"),
    Line2D([0], [0], marker="s", color="none",
           markerfacecolor="lime", markersize=8, label=f"{args.green} K\u03b1"),
    Line2D([0], [0], marker="s", color="none",
           markerfacecolor="blue", markersize=8, label=f"{args.blue} K\u03b1"),
]


# ---------- Discover maps ----------
h5_files = sorted([
    p for p in MAP_DIR.glob("*.h5")
    if "test_map" not in p.name and not p.stem.endswith("_002")
    and "elongated_particle" not in p.name
])
print(f"Found {len(h5_files)} maps")


# ---------- Generate one figure per map ----------
for h5_path in h5_files:
    label = MAP_LABELS.get(h5_path.stem, h5_path.stem)

    with h5py.File(h5_path, "r") as f:
        rgb = make_rgb(f)
        centroids = get_area_centroids(f)
        pos = f["xrmmap/positions/pos"]
        ny, nx_full = pos.shape[:2]
        x_pos = pos[:, 1:-1, 0][:]
        y_pos = pos[:, 1:-1, 1][:]
        nx = nx_full - 2
        extent = [x_pos.min(), x_pos.max(), y_pos.min(), y_pos.max()]

    x_range = extent[1] - extent[0]
    y_range = extent[3] - extent[2]
    aspect = y_range / x_range if x_range > 0 else 1.0

    fig_w = 7.0
    fig_h = fig_w * aspect + 0.5  # extra for title and legends
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.imshow(rgb, extent=extent, aspect="equal",
              interpolation="nearest", origin="lower")

    # Plot XANES points
    style_points = {k: ([], []) for k in CLUSTER_STYLE}
    for area_name, (row_c, col_c) in centroids.items():
        spec_name = area_to_spectrum(area_name)
        cluster_id = cluster_lookup.get(spec_name)
        if cluster_id is None:
            continue
        sk = get_style_key(spec_name, cluster_id)
        if sk not in style_points:
            continue
        col_adj = col_c - 1
        if col_adj < 0 or col_adj >= nx:
            continue
        x_disp = np.interp(col_adj, [0, nx - 1], [extent[0], extent[1]])
        y_disp = np.interp(row_c, [0, ny - 1], [extent[2], extent[3]])
        style_points[sk][0].append(x_disp)
        style_points[sk][1].append(y_disp)

    for sk, style in CLUSTER_STYLE.items():
        xs, ys = style_points[sk]
        if xs:
            ax.scatter(xs, ys, marker=style["marker"], facecolors="none",
                       edgecolors="white", s=100, linewidths=1.2, zorder=5)

    ax.set_title(label, fontsize=14, fontweight="bold", pad=6)
    ax.set_xticks([])
    ax.set_yticks([])

    # Scale bar avoiding XANES markers (draw first to know which corner)
    bar_mm = 0.5 if x_range > 1.2 else 0.2
    bar_corner = add_scale_bar(ax, extent, style_points, bar_length_mm=bar_mm)

    # Tricolor legend (placed in opposite corner from scale bar)
    legend_loc = _LEGEND_LOC.get(bar_corner, "upper left")
    tri_leg = ax.legend(handles=tricolor_handles, loc=legend_loc,
                        fontsize=8, frameon=True, fancybox=False,
                        edgecolor="gray", facecolor="black", labelcolor="white",
                        handletextpad=0.3, borderpad=0.4)
    tri_leg.set_zorder(20)
    ax.add_artist(tri_leg)

    # Cluster legend (below the map)
    # Only include groups that appear in this map
    present_handles = []
    for sk, style in CLUSTER_STYLE.items():
        xs, ys = style_points[sk]
        if xs:
            present_handles.append(Line2D(
                [0], [0], marker=style["marker"], color="none",
                markerfacecolor="none", markeredgecolor="black",
                markersize=7, markeredgewidth=1.2, label=style["label"],
            ))

    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.05)

    if present_handles:
        fig.legend(handles=present_handles, loc="lower center",
                   ncol=len(present_handles), fontsize=8,
                   frameon=True, fancybox=False, edgecolor="gray",
                   handletextpad=0.3, columnspacing=1.5,
                   bbox_to_anchor=(0.5, 0.005))

    num = label.replace("Map ", "")
    rgb_tag = f"{args.red}_{args.green}_{args.blue}"
    stem = f"supp_tricolor_{rgb_tag}_map{num}"
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=DPI,
                bbox_inches="tight", pad_inches=0.1, facecolor="white")
    fig.savefig(OUT_DIR / f"{stem}.pdf", dpi=DPI,
                bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)
    print(f"Saved {stem} ({label})")

print("Done.")
