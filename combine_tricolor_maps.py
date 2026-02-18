"""
Combine two tricolor renderings of the same map side by side.
Generates both color schemes and composites them per map.

Usage:
    python combine_tricolor_maps.py                    # all maps
    python combine_tricolor_maps.py --maps 1 5 9       # specific maps
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

# ---------- CLI ----------
parser = argparse.ArgumentParser(
    description="Side-by-side tricolor maps: Fe/Ca/K and Fe/Ca/Ti.")
parser.add_argument("--maps", type=int, nargs="*", default=None,
                    help="Map numbers to generate (default: all)")
args = parser.parse_args()

# ---------- Config ----------
DPI = 300
MAP_DIR = Path("maps")
PCA_DIR = Path("pca_results")
OUT_DIR = Path("supplemental_figures")
OUT_DIR.mkdir(exist_ok=True)

# Two color schemes: left and right panels
SCHEMES = [
    {"red": "Fe Ka", "green": "K Ka", "blue": "Ca Ka",
     "label_r": "Fe", "label_g": "K", "label_b": "Ca"},
    {"red": "Fe Ka", "green": "Ti Ka", "blue": "Ca Ka",
     "label_r": "Fe", "label_g": "Ti", "label_b": "Ca"},
]

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
MAP_LABELS_REV = {v: k for k, v in MAP_LABELS.items()}

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


def make_rgb(f, r_roi, g_roi, b_roi):
    channels = []
    for name in [r_roi, g_roi, b_roi]:
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
    x_range = extent[1] - extent[0]
    y_range = extent[3] - extent[2]
    bar_frac = bar_length_mm / x_range
    mx = x_range * 0.05
    my = y_range * 0.08
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
    return 0.97, 0.05, "bottom-right"


def add_scale_bar(ax, extent, xanes_points, bar_length_mm=0.5):
    """Draw scale bar, returns the corner name used."""
    x_range = extent[1] - extent[0]
    bar_frac = bar_length_mm / x_range
    x_end_frac, y_frac, corner = find_clear_corner(
        extent, xanes_points, bar_length_mm)
    x_start_frac = x_end_frac - bar_frac
    ax.plot([x_start_frac, x_end_frac], [y_frac, y_frac], color="white",
            linewidth=3, solid_capstyle="butt", zorder=10,
            clip_on=False, transform=ax.transAxes)
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


# ---------- Discover maps ----------
h5_files = sorted([
    p for p in MAP_DIR.glob("*.h5")
    if "test_map" not in p.name and not p.stem.endswith("_002")
    and "elongated_particle" not in p.name
])
h5_lookup = {MAP_LABELS.get(p.stem): p for p in h5_files if p.stem in MAP_LABELS}

# Filter to requested maps
if args.maps:
    selected = [f"Map {n}" for n in args.maps]
else:
    selected = sorted(h5_lookup.keys(), key=lambda x: int(x.replace("Map ", "")))

print(f"Generating {len(selected)} combined figures")


# ---------- Render one map ----------
def render_panel(ax, h5_path, scheme, extent_info=None):
    """Render a single tricolor panel onto ax. Returns extent info for reuse."""
    with h5py.File(h5_path, "r") as f:
        rgb = make_rgb(f, scheme["red"], scheme["green"], scheme["blue"])
        centroids = get_area_centroids(f)
        pos = f["xrmmap/positions/pos"]
        ny, nx_full = pos.shape[:2]
        x_pos = pos[:, 1:-1, 0][:]
        y_pos = pos[:, 1:-1, 1][:]
        nx = nx_full - 2
        extent = [x_pos.min(), x_pos.max(), y_pos.min(), y_pos.max()]

    ax.imshow(rgb, extent=extent, aspect="equal",
              interpolation="nearest", origin="lower")

    # XANES points
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
                       edgecolors="white", s=80, linewidths=1.0, zorder=5)

    # Scale bar (draw first to know which corner it uses)
    x_range = extent[1] - extent[0]
    bar_mm = 0.5 if x_range > 1.2 else 0.2
    bar_corner = add_scale_bar(ax, extent, style_points, bar_length_mm=bar_mm)

    # Tricolor legend (placed in opposite corner from scale bar)
    legend_loc = _LEGEND_LOC.get(bar_corner, "upper left")
    tri_handles = [
        Line2D([0], [0], marker="s", color="none",
               markerfacecolor="red", markersize=7,
               label=f"{scheme['label_r']} K\u03b1"),
        Line2D([0], [0], marker="s", color="none",
               markerfacecolor="lime", markersize=7,
               label=f"{scheme['label_g']} K\u03b1"),
        Line2D([0], [0], marker="s", color="none",
               markerfacecolor="blue", markersize=7,
               label=f"{scheme['label_b']} K\u03b1"),
    ]
    tri_leg = ax.legend(handles=tri_handles, loc=legend_loc,
                        fontsize=7, frameon=True, fancybox=False,
                        edgecolor="gray", facecolor="black", labelcolor="white",
                        handletextpad=0.3, borderpad=0.3)
    tri_leg.set_zorder(20)

    ax.set_xticks([])
    ax.set_yticks([])

    return style_points


# ---------- Generate figures ----------
for map_label in selected:
    h5_path = h5_lookup.get(map_label)
    if h5_path is None:
        print(f"  Skipping {map_label}: not found")
        continue

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 7))

    sp_l = render_panel(ax_l, h5_path, SCHEMES[0])
    sp_r = render_panel(ax_r, h5_path, SCHEMES[1])

    left_tag = f"{SCHEMES[0]['label_r']}/{SCHEMES[0]['label_g']}/{SCHEMES[0]['label_b']}"
    right_tag = f"{SCHEMES[1]['label_r']}/{SCHEMES[1]['label_g']}/{SCHEMES[1]['label_b']}"
    ax_l.set_title(f"{map_label} — {left_tag}", fontsize=11, fontweight="bold", pad=4)
    ax_r.set_title(f"{map_label} — {right_tag}", fontsize=11, fontweight="bold", pad=4)

    # Cluster legend below — combine present groups from both panels
    present_keys = set()
    for sp in [sp_l, sp_r]:
        for sk in sp:
            if sp[sk][0]:
                present_keys.add(sk)

    if present_keys:
        present_handles = []
        for sk, style in CLUSTER_STYLE.items():
            if sk in present_keys:
                present_handles.append(Line2D(
                    [0], [0], marker=style["marker"], color="none",
                    markerfacecolor="none", markeredgecolor="black",
                    markersize=7, markeredgewidth=1.2, label=style["label"],
                ))
        fig.legend(handles=present_handles, loc="lower center",
                   ncol=len(present_handles), fontsize=8,
                   frameon=True, fancybox=False, edgecolor="gray",
                   handletextpad=0.3, columnspacing=1.5,
                   bbox_to_anchor=(0.5, 0.005))

    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.05, wspace=0.05)

    num = map_label.replace("Map ", "")
    stem = f"supp_tricolor_combined_map{num}"
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=DPI,
                bbox_inches="tight", pad_inches=0.1, facecolor="white")
    fig.savefig(OUT_DIR / f"{stem}.pdf", dpi=DPI,
                bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)
    print(f"Saved {stem} ({map_label})")

print("Done.")
