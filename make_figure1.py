"""
Figure 1: XFM Maps 2x2 panel with shared legend strip.
Renders µ-XRF tricolor maps from HDF5 files with cluster-assigned point
symbols, and labels the 6 representative spectra shown in Figure 2.
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
    "a": ("2x2_10um_striated_gt15_2_001.h5",   0.5),
    "b": ("2x2_10um_rectangles_gt15_1_001.h5",  0.5),
    "c": ("2x2_10um_flaky_nodule_001.h5",       0.5),
    "d": ("1x1_10um_flaky_dark_gt15_001.h5",    0.2),
}

# RGB channel ROI names
R_ROI, G_ROI, B_ROI = "Fe Ka", "Ca Ka", "K Ka"

# Cluster styles
CLUSTER_STYLE = {
    1:    {"marker": "o", "label": "Group 1"},
    2:    {"marker": "s", "label": "Group 2"},
    "3a": {"marker": "^", "label": "Group 3a"},
    "3b": {"marker": "v", "label": "Group 3b"},
    4:    {"marker": "D", "label": "Group 4"},
    5:    {"marker": "p", "label": "Group 5"},
}

# Phase grouping: mineral name -> phase group label
PHASE_GROUPS = {
    "6L-Fhy": "Fe(III) oxyhydroxide", "2L-Fhy": "Fe(III) oxyhydroxide",
    "2L-Fhy on sand": "Fe(III) oxyhydroxide", "Goethite": "Fe(III) oxyhydroxide",
    "Goethite on sand": "Fe(III) oxyhydroxide", "Lepidocrocite": "Fe(III) oxyhydroxide",
    "Schwertmannite": "Fe(III) oxyhydroxide",
    "Hematite": "Fe(III) oxide", "Hematite on sand": "Fe(III) oxide",
    "Maghemite": "Fe(III) oxide",
    "Ferrosmectite": "Fe(III) phyllosilicate", "Nontronite": "Fe(III) phyllosilicate",
    "Biotite": "Fe(II) phyllosilicate",
    "Hornblende": "Fe(II) silicate", "Augite": "Fe(II) silicate",
    "Mackinawite (aged)": "Fe sulfide", "Mackinawite": "Fe sulfide",
    "Pyrrhotite": "Fe sulfide", "Pyrite": "Fe sulfide", "FeS": "Fe sulfide",
    "Siderite-s": "Fe(II) carbonate", "Siderite-n": "Fe(II) carbonate",
    "Ilmenite": "Fe-Ti oxide",
    "Vivianite": "Fe(II) phosphate",
    "Jarosite": "Fe(III) sulfate",
    "Green Rust - Carbonate": "Green rust", "Green Rust - Chloride": "Green rust",
    "Green Rust - Sulfate": "Green rust",
}

# Base group definitions (labels will be filled from LCF results)
GROUPS = [
    {"key": "1",  "marker": "o"},
    {"key": "2",  "marker": "s"},
    {"key": "3a", "marker": "^"},
    {"key": "3b", "marker": "v"},
    {"key": "4",  "marker": "D"},
    {"key": "5",  "marker": "p"},
]

# Selected spectra from Figure 2 (group -> spectrum name)
SELECTED_SPECTRA = {
    1:    "FeXANES_GT15_rectangles_Fe_6.001",
    2:    "FeXANES_GT5_flaky_nodule_Fe_8.001",
    "3a": "FeXANES_GT15_rectangles_Fe_3.001",
    "3b": "FeXANES_GT15_flakydark_Fe_1.001",
    4:    "FeXANES_GT15_FeTiXRD_striated2_2.001",
    5:    "FeXANES_GT5_flaky_nodule_Fe_10.001",
}

# Figure 2 panel labels for annotation
FIG2_PANEL = {
    1:    "2a",
    2:    "2b",
    "3a": "2c",
    "3b": "2d",
    4:    "2e",
    5:    "2f",
}


# ---------- HDF5 helpers (from notebook) ----------
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

# Determine 3a vs 3b: need LCF data to check pyrrhotite content
lcf_df = pd.read_csv(PCA_DIR / "lcf_individual.csv")
c3_lcf = lcf_df[lcf_df["cluster"] == 3].copy()
c3_lcf["sub_label"] = c3_labels
sub0_pyrr = c3_lcf[c3_lcf["sub_label"] == 0]["Pyrrhotite"].mean()
sub1_pyrr = c3_lcf[c3_lcf["sub_label"] == 1]["Pyrrhotite"].mean()
if sub0_pyrr >= sub1_pyrr:
    label_map = {0: "3a", 1: "3b"}
else:
    label_map = {1: "3a", 0: "3b"}

sub3_lookup = {}
for i, (_, row) in enumerate(c3_df.iterrows()):
    sub3_lookup[row["spectrum"]] = label_map[c3_labels[i]]

def get_style_key(spec_name, cluster_id):
    if cluster_id == 3:
        return sub3_lookup.get(spec_name, "3a")
    return cluster_id

# Build reverse lookup: spectrum -> group key
spec_to_group = {}
for _, row in cluster_df.iterrows():
    spec_to_group[row["spectrum"]] = get_style_key(row["spectrum"], row["cluster"])

# Build set of selected area names for quick lookup
selected_areas = {}  # area_name -> group key
for grp, spec_name in SELECTED_SPECTRA.items():
    area_name = spec_name.replace("FeXANES_", "").replace(".001", "")
    selected_areas[area_name] = grp

# ---------- Build legend labels from individual LCF results ----------
# Get reference mineral columns from lcf_df
ref_columns = [c for c in lcf_df.columns if c not in
               ["spectrum", "cluster", "r_factor", "chi_sq", "weight_sum", "n_refs"]]

for grp_info in GROUPS:
    grp_key = grp_info["key"]
    # Map string keys to the right type for SELECTED_SPECTRA lookup
    if grp_key in SELECTED_SPECTRA:
        sk = grp_key
    elif grp_key.isdigit() and int(grp_key) in SELECTED_SPECTRA:
        sk = int(grp_key)
    else:
        grp_info["label"] = ""
        continue

    spec_name = SELECTED_SPECTRA[sk]
    row = lcf_df[lcf_df["spectrum"] == spec_name]
    if row.empty:
        grp_info["label"] = ""
        continue
    row = row.iloc[0]
    weight_sum = row["weight_sum"]

    # Aggregate mineral weights into phase groups
    phase_pcts = {}
    for ref_name in ref_columns:
        w = row[ref_name]
        if w > 0.005:
            pg = PHASE_GROUPS.get(ref_name, ref_name)
            pct = w / weight_sum * 100 if weight_sum > 0 else 0
            phase_pcts[pg] = phase_pcts.get(pg, 0) + pct

    # Sort by descending percentage
    sorted_phases = sorted(phase_pcts.items(), key=lambda x: x[1], reverse=True)
    parts = [f"{name} {pct:.0f}%" for name, pct in sorted_phases if pct >= 1]
    grp_info["label"] = ", ".join(parts)

print("Legend labels (from individual LCF):")
for g in GROUPS:
    print(f"  Group {g['key']}: {g['label']}")


# ---------- RGB triangle helper ----------
def make_rgb_triangle(ax):
    """Draw Fe(red)-K(blue)-Ca(green) tricolor triangle."""
    top_left = np.array([0.2, 0.95])
    top_right = np.array([0.8, 0.95])
    bottom = np.array([0.5, 0.95 - 0.6 * np.sqrt(3) / 2])

    n = 200
    tri_img = np.ones((n, n, 3))
    for iy in range(n):
        for ix in range(n):
            px = ix / (n - 1)
            py = 1.0 - iy / (n - 1)
            p = np.array([px, py])
            v0 = top_right - top_left
            v1 = bottom - top_left
            v2 = p - top_left
            d00 = np.dot(v0, v0)
            d01 = np.dot(v0, v1)
            d11 = np.dot(v1, v1)
            d20 = np.dot(v2, v0)
            d21 = np.dot(v2, v1)
            denom = d00 * d11 - d01 * d01
            if abs(denom) < 1e-10:
                continue
            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            u = 1.0 - v - w
            if u >= -0.01 and v >= -0.01 and w >= -0.01:
                u = max(u, 0); v = max(v, 0); w = max(w, 0)
                s = u + v + w
                if s > 0:
                    u /= s; v /= s; w /= s
                tri_img[iy, ix] = [u, w, v]
            else:
                tri_img[iy, ix] = [1, 1, 1]

    ax.imshow(tri_img, extent=[0, 1, 0, 1], aspect='auto')
    tri_coords = [top_left, top_right, bottom, top_left]
    ax.plot([c[0] for c in tri_coords], [c[1] for c in tri_coords], 'k-', lw=1.0)
    ax.text(top_left[0] - 0.03, top_left[1] + 0.03, "Fe", fontsize=8,
            fontweight='bold', color='red', ha='center', va='bottom')
    ax.text(top_right[0] + 0.03, top_right[1] + 0.03, "K", fontsize=8,
            fontweight='bold', color='blue', ha='center', va='bottom')
    ax.text(bottom[0], bottom[1] - 0.05, "Ca", fontsize=8,
            fontweight='bold', color='green', ha='center', va='top')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')


# ---------- Render each panel ----------
print("Rendering maps from HDF5...")

panel_data = {}  # key -> (rgb, extent, style_points, selected_points, bar_mm)

for panel_key, (h5_name, bar_mm) in PANEL_FILES.items():
    h5_path = MAP_DIR / h5_name
    with h5py.File(h5_path, "r") as f:
        rgb = make_rgb(f)
        centroids = get_area_centroids(f)
        pos = f["xrmmap/positions/pos"]
        ny, nx_full = pos.shape[:2]
        x_pos = pos[:, 1:-1, 0][:]
        y_pos = pos[:, 1:-1, 1][:]
        nx = nx_full - 2
        extent = [float(x_pos.min()), float(x_pos.max()),
                  float(y_pos.min()), float(y_pos.max())]

    # Map centroids to display coordinates, tracking selected spectra
    style_points = {k: ([], []) for k in CLUSTER_STYLE}
    sel_points = []  # list of (x_disp, y_disp, group_key) for selected spectra

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

        # Check if this is a selected spectrum
        if area_name in selected_areas:
            sel_points.append((x_disp, y_disp, selected_areas[area_name]))

    panel_data[panel_key] = (rgb, extent, style_points, sel_points, bar_mm)
    print(f"  Panel ({panel_key}): {rgb.shape[1]}x{rgb.shape[0]}, "
          f"{sum(len(v[0]) for v in style_points.values())} points, "
          f"{len(sel_points)} selected")


# ---------- Build the figure ----------
print("Assembling figure...")

fig_w = TARGET_WIDTH_IN
panel_w_in = fig_w / 2.06  # slight margin
# Use aspect of first panel
aspect = panel_data["a"][0].shape[0] / panel_data["a"][0].shape[1]
panel_h_in = panel_w_in * aspect
gap = 0.04  # inches between panels
legend_h_in = 1.8
total_h_in = 2 * panel_h_in + gap + legend_h_in + 0.15

fig = plt.figure(figsize=(fig_w, total_h_in), dpi=DPI)

def to_fig(x, y, w, h):
    return [x / fig_w, y / total_h_in, w / fig_w, h / total_h_in]

x_start = (fig_w - 2 * panel_w_in - gap) / 2
y_grid_bottom = legend_h_in + 0.15

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

        rgb, extent, style_points, sel_points, bar_mm = panel_data[key]

        ax.imshow(rgb, extent=extent, aspect="equal", interpolation="nearest",
                  origin="lower")

        # Plot all cluster symbols
        for sk, style in CLUSTER_STYLE.items():
            xs, ys = style_points[sk]
            if xs:
                ax.scatter(xs, ys, marker=style["marker"], facecolors="none",
                           edgecolors="white", s=80, linewidths=1.0, zorder=5)

        # Label selected spectra with Figure 2 panel reference
        for x_disp, y_disp, grp in sel_points:
            fig2_label = FIG2_PANEL[grp]
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

        # Panel label
        ax.text(0.04, 0.96, panel_labels[key], transform=ax.transAxes,
                fontsize=11, fontweight='bold', color='white', va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='black',
                          alpha=0.5, edgecolor='none'))

        # Scale bar (inside plot, bottom left, white on dark background)
        x_range = extent[1] - extent[0]
        bar_frac = bar_mm / x_range
        x_sb = 0.03
        x_end = x_sb + bar_frac
        y_sb = 0.04
        ax.plot([x_sb, x_end], [y_sb, y_sb], color="white", linewidth=3,
                solid_capstyle="butt", zorder=10, clip_on=False,
                transform=ax.transAxes)
        label = f"{bar_mm:.1f} mm" if bar_mm < 1 else f"{bar_mm:.0f} mm"
        ax.text((x_sb + x_end) / 2, y_sb + 0.03, label, color="white",
                fontsize=7, ha="center", va="bottom", fontweight="bold",
                zorder=10, clip_on=False, transform=ax.transAxes)

        ax.set_xticks([]); ax.set_yticks([])

# ---------- Legend strip ----------
legend_y = 0.0
legend_x = x_start

# RGB triangle (left side, vertically centered)
tri_w_in = 1.0
tri_h_in = 1.0
tri_y = legend_y + (legend_h_in - tri_h_in) / 2
ax_tri = fig.add_axes(to_fig(legend_x, tri_y, tri_w_in, tri_h_in))
make_rgb_triangle(ax_tri)

# Group legend: single column of 6 entries, using width to the right of triangle
legend_start_x = legend_x + tri_w_in + 0.08
entry_w = fig_w - legend_start_x - 0.05
entry_h = legend_h_in / 6

for i, grp in enumerate(GROUPS):
    x = legend_start_x
    y = legend_y + legend_h_in - (i + 1) * entry_h
    rect = to_fig(x, y, entry_w, entry_h)
    ax_entry = fig.add_axes(rect)
    ax_entry.axis('off')

    # Marker + group name on one line, composition after
    ax_entry.scatter([0.02], [0.5], marker=grp["marker"], s=50,
                     facecolors='none', edgecolors='black', linewidths=1.0,
                     transform=ax_entry.transAxes, clip_on=False, zorder=5)
    ax_entry.text(0.05, 0.5,
                  f"Group {grp['key']}:  {grp['label']}",
                  transform=ax_entry.transAxes, fontsize=5.5,
                  va='center', ha='left')

# Save
fig.savefig(OUT_DIR / "figure_xfm_maps.png", dpi=DPI,
            bbox_inches='tight', facecolor='white', pad_inches=0.05)
fig.savefig(OUT_DIR / "figure_xfm_maps.pdf", dpi=DPI,
            bbox_inches='tight', facecolor='white', pad_inches=0.05)
plt.close(fig)
print("Figure 1 saved: figure_xfm_maps.png and .pdf")
