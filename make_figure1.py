"""
Figure 1: XFM tricolor maps (2×2 panel) for Maps 6, 9, 10, 11.
RGB = Fe (red), K (green), Ca (blue).
All XANES locations shown as white circles; selected spectra annotated
with Figure 2 panel labels.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from sklearn.cluster import KMeans
import h5py

# ---------- Config ----------
DPI = 300
TARGET_WIDTH_MM = 180
TARGET_WIDTH_IN = TARGET_WIDTH_MM / 25.4

MAP_DIR = Path("maps")
PCA_DIR = Path("pca_results")
OUT_DIR = Path(".")

# Panel definitions: key -> (h5 filename, scale bar mm)
PANEL_FILES = {
    "a": ("2x2_10um_flaky_2_001.h5",            0.5),   # Map 7
    "b": ("2x2_10um_flaky_smooth_2_001.h5",     0.5),   # Map 9
    "c": ("2x2_10um_rectangles_gt15_1_001.h5",  0.5),   # Map 10
    "d": ("2x2_10um_striated_gt15_2_001.h5",    0.5),   # Map 11
}
PANEL_MAP_LABELS = {"a": "Map 7", "b": "Map 9", "c": "Map 10", "d": "Map 11"}

# RGB channel ROI names
R_ROI, G_ROI, B_ROI = "Fe Ka", "K Ka", "Ca Ka"

# Spectrum prefixes that belong to each panel (for Figure 2 cross-ref)
PANEL_PREFIXES = {
    "a": ["FeXANES_GT5_flaky2_FeXRD_", "FeXANES_GT5_flaky2_Fe_"],
    "b": ["FeXANES_GT5_flakysmooth2_Fe_"],
    "c": ["FeXANES_GT15_rectangles_Fe_", "FeXANES_GT15_Fe_"],
    "d": ["FeXANES_GT15_FeTiXRD_striated2_", "FeXANES_GT15_Fe_striated2_"],
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


GREEN_THRESHOLD = 0.15  # zero out green below this normalized intensity
GREEN_SCALE = 0.8       # scale down green brightness


def make_rgb(f, green_target=None):
    channels = []
    green_mean = None
    for i, name in enumerate([R_ROI, G_ROI, B_ROI]):
        ch = get_roi_map(f, name)
        if ch is None:
            ch = np.zeros((1, 1))
        vmin = np.percentile(ch, 1)
        vmax = np.percentile(ch, 99.5)
        if vmax > vmin:
            ch = np.clip((ch - vmin) / (vmax - vmin), 0, 1)
        else:
            ch = np.zeros_like(ch)
        if i == 1:  # green channel
            ch[ch < GREEN_THRESHOLD] = 0
            nonzero = ch[ch > 0]
            if len(nonzero) > 0:
                green_mean = nonzero.mean()
                if green_target is not None and green_mean > 0:
                    ch[ch > 0] *= (green_target / green_mean)
                    ch = np.clip(ch, 0, 1)
            ch = ch * GREEN_SCALE
        channels.append(ch)
    return np.stack(channels, axis=-1), green_mean


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


# ---------- Load cluster assignments + sub-clusters ----------
print("Loading cluster assignments...")
cluster_df = pd.read_csv(PCA_DIR / "cluster_assignments.csv")
cluster_lookup = dict(zip(cluster_df["spectrum"], cluster_df["cluster"]))

# Sub-cluster cluster 3
c3_mask = cluster_df["cluster"] == 3
c3_df = cluster_df[c3_mask].copy()
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


# Build spectrum -> group key lookup
spec_to_group = {}
for _, row in cluster_df.iterrows():
    spec_to_group[row["spectrum"]] = get_style_key(row["spectrum"], row["cluster"])

# ---------- Select representative spectra for Figure 2 ----------
# Pick one spectrum per group, spreading across panels so each panel
# has at least one selected point.
print("Selecting representative spectra for Figure 2...")

# Assign each group to a preferred panel to ensure coverage
# (a) Map 7:  Groups 1, 2, 3b, 4, 5
# (b) Map 9:  Groups 1, 3a, 3b, 4, 5
# (c) Map 10: Groups 1, 3a, 4, 5
# (d) Map 11: Groups 3a, 4, 5
GROUP_PANEL = {1: "c", 2: "a", "3a": "b", "3b": "a", 4: "d", 5: "b"}

GROUP_ORDER = [1, 2, "3a", "3b", 4, 5]
SELECTED_SPECTRA = {}  # group_key -> spectrum name

for grp in GROUP_ORDER:
    panel = GROUP_PANEL[grp]
    prefixes = PANEL_PREFIXES[panel]

    # Filter to spectra from the assigned panel
    candidates = lcf_df[lcf_df["spectrum"].apply(
        lambda s, px=prefixes: any(s.startswith(p) for p in px))].copy()
    candidates["group"] = candidates["spectrum"].map(spec_to_group)
    candidates = candidates[candidates["group"] == grp]

    if candidates.empty:
        print(f"  Group {grp}: NO CANDIDATES in panel ({panel})")
        continue

    # Prefer 3+ component fits with good R-factor
    multi = candidates[candidates["n_refs"] >= 3]
    if len(multi) >= 1:
        candidates = multi
    good = candidates[candidates["r_factor"] <= 0.025]
    if len(good) >= 1:
        candidates = good

    # Pick closest to median R-factor
    median_r = candidates["r_factor"].median()
    idx = (candidates["r_factor"] - median_r).abs().idxmin()
    row = candidates.loc[idx]
    SELECTED_SPECTRA[grp] = row["spectrum"]
    print(f"  Group {grp}: {row['spectrum']} (R={row['r_factor']:.4f}) — panel ({panel})")

# Figure 2 panel labels for annotation
FIG2_PANEL = {}
for i, grp in enumerate(GROUP_ORDER):
    if grp in SELECTED_SPECTRA:
        FIG2_PANEL[grp] = f"2{chr(ord('a') + i)}"

# Build reverse lookup: area_name -> group key (for selected spectra only)
selected_areas = {}
for grp, spec_name in SELECTED_SPECTRA.items():
    area_name = spec_name.replace("FeXANES_", "").replace(".001", "")
    selected_areas[area_name] = grp


# ---------- Scale bar helper ----------
def find_clear_corner(extent, all_points, bar_length_mm):
    """Find a corner that doesn't overlap with XANES points."""
    x_range = extent[1] - extent[0]
    y_range = extent[3] - extent[2]
    bar_frac = bar_length_mm / x_range
    mx = x_range * 0.05
    my = y_range * 0.08

    # top-left excluded: reserved for panel label
    corners = [
        ("bottom-left",  bar_frac + 0.03, 0.05),
        ("bottom-right", 0.97, 0.05),
        ("top-right",    0.97, 0.93),
    ]
    for name, x_end_frac, y_frac in corners:
        x_start_data = extent[0] + (x_end_frac - bar_frac) * x_range
        x_end_data = extent[0] + x_end_frac * x_range
        y_data = extent[2] + y_frac * y_range
        conflict = False
        for px, py in all_points:
            if (x_start_data - mx <= px <= x_end_data + mx and
                    y_data - my <= py <= y_data + my):
                conflict = True
                break
        if not conflict:
            return x_end_frac, y_frac, name
    return 0.97, 0.05, "bottom-right"


# Tricolor legend location opposite to scale bar
_LEGEND_LOC = {
    "bottom-right": "upper left",
    "bottom-left":  "upper right",
    "top-right":    "lower left",
    "top-left":     "lower right",
}


# ---------- Render each panel ----------
print("Rendering maps from HDF5...")

# Pass 1: compute green channel means for each map
green_means = {}
for panel_key, (h5_name, bar_mm) in PANEL_FILES.items():
    h5_path = MAP_DIR / h5_name
    with h5py.File(h5_path, "r") as f:
        _, gm = make_rgb(f, green_target=None)
    green_means[panel_key] = gm
    print(f"  Panel ({panel_key}) green mean: {gm:.3f}" if gm else
          f"  Panel ({panel_key}) green mean: N/A")

green_target = np.median([v for v in green_means.values() if v is not None])
print(f"  Green target (median): {green_target:.3f}")

# Pass 2: render with equalized green
panel_data = {}  # key -> (rgb, extent, all_points, sel_points, bar_mm)

for panel_key, (h5_name, bar_mm) in PANEL_FILES.items():
    h5_path = MAP_DIR / h5_name
    with h5py.File(h5_path, "r") as f:
        rgb, _ = make_rgb(f, green_target=green_target)
        centroids = get_area_centroids(f)
        pos = f["xrmmap/positions/pos"]
        ny, nx_full = pos.shape[:2]
        x_pos = pos[:, 1:-1, 0][:]
        y_pos = pos[:, 1:-1, 1][:]
        nx = nx_full - 2
        extent = [float(x_pos.min()), float(x_pos.max()),
                  float(y_pos.min()), float(y_pos.max())]

    # Map centroids to display coordinates
    all_points = []   # (x, y) for all XANES points
    sel_points = []   # (x, y, group_key) for selected spectra

    for area_name, (row_c, col_c) in centroids.items():
        spec_name = area_to_spectrum(area_name)
        cluster_id = cluster_lookup.get(spec_name)
        if cluster_id is None:
            continue
        col_adj = col_c - 1
        if col_adj < 0 or col_adj >= nx:
            continue
        x_disp = np.interp(col_adj, [0, nx - 1], [extent[0], extent[1]])
        y_disp = np.interp(row_c, [0, ny - 1], [extent[2], extent[3]])
        all_points.append((x_disp, y_disp))

        if area_name in selected_areas:
            sel_points.append((x_disp, y_disp, selected_areas[area_name]))

    panel_data[panel_key] = (rgb, extent, all_points, sel_points, bar_mm)
    print(f"  Panel ({panel_key}) {PANEL_MAP_LABELS[panel_key]}: "
          f"{rgb.shape[1]}x{rgb.shape[0]}, "
          f"{len(all_points)} XANES points, {len(sel_points)} selected")


# ---------- Build the figure ----------
print("Assembling figure...")

fig_w = TARGET_WIDTH_IN
panel_w_in = fig_w / 2.06
aspect = panel_data["a"][0].shape[0] / panel_data["a"][0].shape[1]
panel_h_in = panel_w_in * aspect
gap = 0.04
legend_h_in = 0.4
total_h_in = 2 * panel_h_in + gap + legend_h_in + 0.1

fig = plt.figure(figsize=(fig_w, total_h_in), dpi=DPI)


def to_fig(x, y, w, h):
    return [x / fig_w, y / total_h_in, w / fig_w, h / total_h_in]


x_start = (fig_w - 2 * panel_w_in - gap) / 2
y_grid_bottom = legend_h_in + 0.05

panel_layout = [["a", "b"], ["c", "d"]]
panel_labels = {"a": "(a)", "b": "(b)", "c": "(c)", "d": "(d)"}

panel_axes = {}
for row_idx, row_keys in enumerate(panel_layout):
    for col_idx, key in enumerate(row_keys):
        x = x_start + col_idx * (panel_w_in + gap)
        y = y_grid_bottom + (1 - row_idx) * (panel_h_in + gap)
        rect = to_fig(x, y, panel_w_in, panel_h_in)
        ax = fig.add_axes(rect)
        panel_axes[key] = ax

        rgb, extent, all_points, sel_points, bar_mm = panel_data[key]

        ax.imshow(rgb, extent=extent, aspect="equal", interpolation="nearest",
                  origin="lower")

        # Plot only selected XANES locations and label with Figure 2 ref
        for x_disp, y_disp, grp in sel_points:
            ax.scatter(x_disp, y_disp, marker="o", facecolors="none",
                       edgecolors="white", s=80, linewidths=1.0, zorder=5)

        for x_disp, y_disp, grp in sel_points:
            fig2_label = FIG2_PANEL.get(grp)
            if fig2_label is None:
                continue
            ax.annotate(
                fig2_label,
                xy=(x_disp, y_disp),
                xytext=(14, 14), textcoords="offset points",
                fontsize=8, fontweight="bold", color="yellow",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black",
                          alpha=0.8, edgecolor="yellow", lw=0.8),
                arrowprops=dict(arrowstyle="-|>", color="yellow",
                                lw=1.2, mutation_scale=8,
                                shrinkA=0, shrinkB=3),
                zorder=10,
            )

        # Panel label + map label
        ax.text(0.04, 0.96, panel_labels[key],
                transform=ax.transAxes,
                fontsize=9, fontweight="bold", color="white", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black",
                          alpha=0.5, edgecolor="none"))

        # Scale bar
        x_range = extent[1] - extent[0]
        bar_frac = bar_mm / x_range
        bar_x_end, bar_y, bar_corner = find_clear_corner(
            extent, all_points, bar_mm)
        bar_x_start = bar_x_end - bar_frac
        ax.plot([bar_x_start, bar_x_end], [bar_y, bar_y], color="white",
                linewidth=3, solid_capstyle="butt", zorder=10, clip_on=False,
                transform=ax.transAxes)
        if "bottom" in bar_corner:
            text_y = bar_y + 0.03
            va = "bottom"
        else:
            text_y = bar_y - 0.03
            va = "top"
        label = f"{bar_mm:.1f} mm" if bar_mm < 1 else f"{bar_mm:.0f} mm"
        ax.text((bar_x_start + bar_x_end) / 2, text_y, label, color="white",
                fontsize=7, ha="center", va=va, fontweight="bold",
                zorder=10, clip_on=False, transform=ax.transAxes)

        ax.set_xticks([])
        ax.set_yticks([])

# ---------- Shared tricolor legend below panels ----------
tri_handles = [
    Line2D([0], [0], marker="s", color="none",
           markerfacecolor="red", markersize=8, label="Fe K\u03b1"),
    Line2D([0], [0], marker="s", color="none",
           markerfacecolor="lime", markersize=8, label="K K\u03b1"),
    Line2D([0], [0], marker="s", color="none",
           markerfacecolor="blue", markersize=8, label="Ca K\u03b1"),
]
fig.legend(handles=tri_handles, loc="lower center",
           ncol=3, fontsize=8, frameon=True, fancybox=False,
           edgecolor="gray", facecolor="black", labelcolor="white",
           handletextpad=0.3, columnspacing=1.5, borderpad=0.4,
           bbox_to_anchor=(0.5, 0.005))

# Save
fig.savefig(OUT_DIR / "figure_xfm_maps.png", dpi=DPI,
            bbox_inches="tight", facecolor="white", pad_inches=0.05)
fig.savefig(OUT_DIR / "figure_xfm_maps.pdf", dpi=DPI,
            bbox_inches="tight", facecolor="white", pad_inches=0.05)
plt.close(fig)
print("Figure 1 saved: figure_xfm_maps.png and .pdf")

# Print selected spectra for Figure 2 reference
print("\nSelected spectra for Figure 2:")
for grp in GROUP_ORDER:
    if grp in SELECTED_SPECTRA:
        spec = SELECTED_SPECTRA[grp]
        panel = FIG2_PANEL[grp]
        print(f"  Fig. {panel} — Group {grp}: {spec}")
