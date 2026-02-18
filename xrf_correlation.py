"""
XRF Element Correlation Analysis
Pixel-level and cluster-specific correlations from µ-XRF maps.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from pathlib import Path
import h5py, warnings
warnings.filterwarnings("ignore")

# ---------- Config ----------
MAP_DIR = Path("maps")
PCA_DIR = Path("pca_results")
OUT_DIR = Path("xrf_correlation")
OUT_DIR.mkdir(exist_ok=True)
DPI = 300

# Elements to analyze (must match ROI names in HDF5)
ELEMENTS = ["Fe Ka", "Ca Ka", "K Ka", "Ti Ka", "Mn Ka", "S Ka", "Si Ka", "P Ka"]
# Short names for display
SHORT = {e: e.replace(" Ka", "") for e in ELEMENTS}

# The 4 Figure 1 map files (most scientifically relevant)
FIG1_MAPS = {
    "striated_gt15_2":    "2x2_10um_striated_gt15_2_001.h5",
    "rectangles_gt15_1":  "2x2_10um_rectangles_gt15_1_001.h5",
    "flaky_nodule":       "2x2_10um_flaky_nodule_001.h5",
    "flaky_dark_gt15":    "1x1_10um_flaky_dark_gt15_001.h5",
}

# Map labels: filename stem → display label (consistent across all figures)
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

# Reverse lookup: FIG1_MAPS short key → Map label
_FIG1_LABELS = {}
for _k, _v in FIG1_MAPS.items():
    _stem = _v.replace(".h5", "")
    _FIG1_LABELS[_k] = MAP_LABELS.get(_stem, _k)

# Cluster colors and styles
CLUSTER_COLORS = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728', 5: '#9467bd'}
CLUSTER_LABELS = {1: 'Grp 1', 2: 'Grp 2', 3: 'Grp 3', 4: 'Grp 4', 5: 'Grp 5'}


# ---------- HDF5 helpers ----------
def get_roi_map(f, roi_name):
    names = [n.decode() if isinstance(n, bytes) else n for n in f["xrmmap/roimap/sum_name"][:]]
    if roi_name in names:
        idx = names.index(roi_name)
        return f["xrmmap/roimap/sum_cor"][:, 1:-1, idx].astype(float)
    return None

def get_area_centroids_and_cluster(f, cluster_lookup):
    """Return list of (area_name, row, col, cluster) for XANES areas."""
    results = []
    areas = f.get("xrmmap/areas")
    if areas is None:
        return results
    for area_name in areas:
        spec_name = f"FeXANES_{area_name}.001"
        cluster_id = cluster_lookup.get(spec_name)
        if cluster_id is None:
            continue
        mask = areas[area_name][:]
        if mask.any():
            rows, cols = np.where(mask)
            results.append((area_name, rows.mean(), cols.mean(), cluster_id))
    return results


# ---------- Load cluster assignments ----------
clusters_df = pd.read_csv(PCA_DIR / "cluster_assignments.csv")
cluster_lookup = dict(zip(clusters_df["spectrum"], clusters_df["cluster"]))

# Sub-cluster cluster 3
c3_mask = clusters_df["cluster"] == 3
c3_df = clusters_df[c3_mask].copy()
pc_cols = [c for c in clusters_df.columns if c.startswith("PC")]
if len(c3_df) > 1:
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


# ==========================================================
# 1. Extract element maps and XANES point data from all maps
# ==========================================================
print("Extracting element maps from all HDF5 files...")

all_h5 = sorted([p for p in MAP_DIR.glob("*.h5") if "test_map" not in p.name and "elongated_particle" not in p.name])
all_point_data = []  # per-XANES-point element intensities

for h5_path in all_h5:
    with h5py.File(h5_path, "r") as f:
        # Load element maps
        maps = {}
        for elem in ELEMENTS:
            m = get_roi_map(f, elem)
            if m is not None:
                maps[elem] = m

        if "Fe Ka" not in maps:
            continue

        # Get XANES point locations and extract element intensities
        areas = f.get("xrmmap/areas")
        if areas is None:
            continue

        for area_name in areas:
            spec_name = f"FeXANES_{area_name}.001"
            cluster_id = cluster_lookup.get(spec_name)
            if cluster_id is None:
                continue

            mask = areas[area_name][:]
            if not mask.any():
                continue

            rows, cols = np.where(mask)
            cols_adj = cols - 1  # account for 1:-1 slicing
            valid = (cols_adj >= 0) & (cols_adj < maps["Fe Ka"].shape[1])
            rows_v, cols_v = rows[valid], cols_adj[valid]
            if len(rows_v) == 0:
                continue

            point = {
                "spectrum": spec_name,
                "cluster": cluster_id,
                "map_file": h5_path.name,
            }
            for elem in ELEMENTS:
                if elem in maps:
                    point[SHORT[elem]] = maps[elem][rows_v, cols_v].mean()
            all_point_data.append(point)

points_df = pd.DataFrame(all_point_data)
points_df.to_csv(OUT_DIR / "xanes_point_elements.csv", index=False)
print(f"  Extracted element intensities for {len(points_df)} XANES points")


# ==========================================================
# 2. Whole-map pixel-level correlation matrices
# ==========================================================
print("\nComputing pixel-level correlations across all maps...")

# Aggregate all pixels from all maps
all_pixels = {SHORT[e]: [] for e in ELEMENTS}

for h5_path in all_h5:
    with h5py.File(h5_path, "r") as f:
        maps = {}
        for elem in ELEMENTS:
            m = get_roi_map(f, elem)
            if m is not None:
                maps[elem] = m.ravel()

        if "Fe Ka" not in maps:
            continue

        n_pix = len(maps["Fe Ka"])
        for elem in ELEMENTS:
            if elem in maps:
                all_pixels[SHORT[elem]].append(maps[elem])
            else:
                all_pixels[SHORT[elem]].append(np.full(n_pix, np.nan))

for key in all_pixels:
    all_pixels[key] = np.concatenate(all_pixels[key])

pixel_df = pd.DataFrame(all_pixels)
print(f"  Total pixels: {len(pixel_df)}")

# Correlation matrix (Pearson)
elem_short = [SHORT[e] for e in ELEMENTS]
corr_matrix = pixel_df[elem_short].corr(method='pearson')
corr_matrix.to_csv(OUT_DIR / "pixel_correlation_matrix.csv")
print(f"  Pixel-level Pearson correlation with Fe:")
for e in elem_short:
    if e != "Fe":
        r = corr_matrix.loc["Fe", e]
        print(f"    Fe vs {e}: r = {r:.3f}")


# ==========================================================
# 3. XANES-point correlation by cluster
# ==========================================================
print("\nXANES-point element correlations by cluster...")

cluster_corr_results = []
for c in sorted(points_df["cluster"].unique()):
    sub = points_df[points_df["cluster"] == c]
    if len(sub) < 5:
        continue
    for e in elem_short:
        if e == "Fe" or e not in sub.columns:
            continue
        valid = sub[["Fe", e]].dropna()
        if len(valid) < 5:
            continue
        r_pearson, p_pearson = pearsonr(valid["Fe"], valid[e])
        r_spearman, p_spearman = spearmanr(valid["Fe"], valid[e])
        cluster_corr_results.append({
            "cluster": c,
            "element": e,
            "n": len(valid),
            "r_pearson": r_pearson,
            "p_pearson": p_pearson,
            "r_spearman": r_spearman,
            "p_spearman": p_spearman,
        })

cluster_corr_df = pd.DataFrame(cluster_corr_results)
cluster_corr_df.to_csv(OUT_DIR / "cluster_element_correlations.csv", index=False)

print("  Fe correlations by cluster (Pearson r, * = p<0.05):")
for c in sorted(cluster_corr_df["cluster"].unique()):
    sub = cluster_corr_df[cluster_corr_df["cluster"] == c]
    parts = []
    for _, row in sub.iterrows():
        sig = "*" if row["p_pearson"] < 0.05 else ""
        parts.append(f"{row['element']}={row['r_pearson']:+.2f}{sig}")
    print(f"    Cluster {c}: {', '.join(parts)}")


# ==========================================================
# 4. Per-map pixel correlations (Figure 1 maps only)
# ==========================================================
print("\nPer-map pixel correlations for Figure 1 maps...")

map_corr_results = []
map_pixel_data = {}  # store for plotting

for map_label, h5_name in FIG1_MAPS.items():
    h5_path = MAP_DIR / h5_name
    with h5py.File(h5_path, "r") as f:
        maps = {}
        for elem in ELEMENTS:
            m = get_roi_map(f, elem)
            if m is not None:
                maps[elem] = m

    if "Fe Ka" not in maps:
        continue

    fe = maps["Fe Ka"].ravel()
    map_pixel_data[map_label] = {"Fe": fe}

    for elem in ELEMENTS:
        if elem == "Fe Ka" or elem not in maps:
            continue
        other = maps[elem].ravel()
        r, p = pearsonr(fe, other)
        map_corr_results.append({
            "map": map_label,
            "element": SHORT[elem],
            "r_pearson": r,
            "p_pearson": p,
        })
        map_pixel_data[map_label][SHORT[elem]] = other

map_corr_df = pd.DataFrame(map_corr_results)
map_corr_df.to_csv(OUT_DIR / "per_map_correlations.csv", index=False)

for ml in FIG1_MAPS:
    sub = map_corr_df[map_corr_df["map"] == ml]
    parts = [f"{r['element']}={r['r_pearson']:+.3f}" for _, r in sub.iterrows()]
    print(f"  {ml}: {', '.join(parts)}")


# ==========================================================
# 5. Diagnostic plots
# ==========================================================
print("\nCreating plots...")

# --- Plot 1: Global pixel correlation matrix heatmap ---
fig, ax = plt.subplots(figsize=(6, 5), dpi=DPI)
im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(elem_short)))
ax.set_xticklabels(elem_short, fontsize=8, rotation=45, ha="right")
ax.set_yticks(range(len(elem_short)))
ax.set_yticklabels(elem_short, fontsize=8)
for i in range(len(elem_short)):
    for j in range(len(elem_short)):
        val = corr_matrix.values[i, j]
        color = "white" if abs(val) > 0.6 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5, color=color)
plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
ax.set_title("Pixel-Level Element Correlations (All Maps)", fontsize=10)
fig.tight_layout()
fig.savefig(OUT_DIR / "plot1_correlation_matrix.png", dpi=DPI)
plt.close(fig)
print("  Saved plot1_correlation_matrix.png")

# --- Plot 2: Fe vs key elements, scatter by cluster (XANES points) ---
key_elements = ["Ca", "K", "Ti", "Mn", "S", "P"]
key_elements = [e for e in key_elements if e in points_df.columns]

fig, axes = plt.subplots(2, 3, figsize=(10, 6.5), dpi=DPI)
axes = axes.flatten()

for i, elem in enumerate(key_elements):
    ax = axes[i]
    for c in sorted(points_df["cluster"].unique()):
        sub = points_df[points_df["cluster"] == c]
        ax.scatter(sub["Fe"], sub[elem], c=CLUSTER_COLORS[c],
                   label=CLUSTER_LABELS[c], s=20, alpha=0.7, edgecolors="none")

    ax.set_xlabel("Fe Kα", fontsize=8)
    ax.set_ylabel(f"{elem} Kα", fontsize=8)
    ax.tick_params(labelsize=7)

    # Overall correlation
    valid = points_df[["Fe", elem]].dropna()
    if len(valid) > 5:
        r, p = pearsonr(valid["Fe"], valid[elem])
        ax.set_title(f"Fe vs {elem}  (r={r:.2f})", fontsize=9)
    else:
        ax.set_title(f"Fe vs {elem}", fontsize=9)

    if i == 0:
        ax.legend(fontsize=6, loc="best")

for i in range(len(key_elements), 6):
    axes[i].axis("off")

fig.suptitle("XANES Point Element Correlations by Cluster", fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "plot2_fe_vs_elements_by_cluster.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("  Saved plot2_fe_vs_elements_by_cluster.png")

# --- Plot 3: Per-map Fe vs Ca and Fe vs K density scatter (Figure 1 maps) ---
fig, axes = plt.subplots(2, 4, figsize=(12, 6), dpi=DPI)

for col_idx, (map_label, h5_name) in enumerate(FIG1_MAPS.items()):
    if map_label not in map_pixel_data:
        continue
    pdata = map_pixel_data[map_label]
    fe = pdata["Fe"]

    for row_idx, elem in enumerate(["Ca", "K"]):
        ax = axes[row_idx, col_idx]
        if elem not in pdata:
            ax.axis("off")
            continue

        other = pdata[elem]
        # 2D histogram for density
        mask = (fe > 0) & (other > 0)
        ax.hist2d(fe[mask], other[mask], bins=80, cmap="inferno",
                  norm=LogNorm(), rasterized=True)

        r, _ = pearsonr(fe[mask], other[mask])
        fig1_label = _FIG1_LABELS.get(map_label, map_label)
        ax.set_title(f"{fig1_label}\nr={r:.2f}", fontsize=7.5)
        ax.tick_params(labelsize=6)
        ax.ticklabel_format(style='scientific', axis='both', scilimits=(0, 0))
        ax.xaxis.get_offset_text().set_fontsize(5)
        ax.yaxis.get_offset_text().set_fontsize(5)

        if col_idx == 0:
            ax.set_ylabel(f"{elem} Kα counts", fontsize=8)
        if row_idx == 1:
            ax.set_xlabel("Fe Kα counts", fontsize=8)

fig.suptitle("Pixel-Level Fe Correlations on Figure 1 Maps", fontsize=11)
fig.tight_layout()
fig.savefig(OUT_DIR / "plot3_per_map_fe_scatter.png", dpi=DPI)
plt.close(fig)
print("  Saved plot3_per_map_fe_scatter.png")

# --- Plot 4: Cluster-specific correlation bar chart ---
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=DPI)

bar_elements = [e for e in ["Ca", "K", "Ti", "Mn", "S"] if e in cluster_corr_df["element"].values]
x = np.arange(len(bar_elements))
n_clusters = len(cluster_corr_df["cluster"].unique())
width = 0.8 / n_clusters

for ci, c in enumerate(sorted(cluster_corr_df["cluster"].unique())):
    sub = cluster_corr_df[cluster_corr_df["cluster"] == c]
    vals = []
    for e in bar_elements:
        row = sub[sub["element"] == e]
        vals.append(row["r_pearson"].values[0] if len(row) > 0 else 0)
    offset = (ci - n_clusters / 2 + 0.5) * width
    bars = ax.bar(x + offset, vals, width, color=CLUSTER_COLORS[c],
                  label=CLUSTER_LABELS[c], edgecolor="white", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels([f"Fe vs {e}" for e in bar_elements], fontsize=9)
ax.set_ylabel("Pearson r", fontsize=10)
ax.set_title("Fe Correlation with Other Elements by Cluster (XANES Points)", fontsize=11)
ax.axhline(y=0, color="black", lw=0.5)
ax.legend(fontsize=8)
ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "plot4_cluster_correlation_bars.png", dpi=DPI)
plt.close(fig)
print("  Saved plot4_cluster_correlation_bars.png")

# --- Plot 5: Fe-Ca-K ternary-style ratio plot by cluster ---
fig, ax = plt.subplots(figsize=(6, 5), dpi=DPI)
for c in sorted(points_df["cluster"].unique()):
    sub = points_df[points_df["cluster"] == c]
    if "Ca" in sub.columns and "K" in sub.columns:
        total = sub["Fe"] + sub["Ca"] + sub["K"]
        fe_frac = sub["Fe"] / total
        ca_frac = sub["Ca"] / total
        ax.scatter(fe_frac, ca_frac, c=CLUSTER_COLORS[c],
                   label=CLUSTER_LABELS[c], s=25, alpha=0.7, edgecolors="none")

ax.set_xlabel("Fe / (Fe + Ca + K)", fontsize=10)
ax.set_ylabel("Ca / (Fe + Ca + K)", fontsize=10)
ax.set_title("Fe–Ca–K Composition at XANES Points", fontsize=11)
ax.legend(fontsize=8)
ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "plot5_fe_ca_k_ternary.png", dpi=DPI)
plt.close(fig)
print("  Saved plot5_fe_ca_k_ternary.png")

# --- Plot 6: Per-map correlation heatmaps ---
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), dpi=DPI)
corr_elements = ["Fe", "Ca", "K", "Ti", "Mn", "S"]

for idx, (map_label, h5_name) in enumerate(FIG1_MAPS.items()):
    ax = axes[idx]
    h5_path = MAP_DIR / h5_name
    with h5py.File(h5_path, "r") as f:
        map_data = {}
        for elem in ELEMENTS:
            m = get_roi_map(f, elem)
            if m is not None:
                map_data[SHORT[elem]] = m.ravel()

    avail = [e for e in corr_elements if e in map_data]
    df_local = pd.DataFrame({e: map_data[e] for e in avail})
    corr_local = df_local.corr()

    im = ax.imshow(corr_local.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(avail)))
    ax.set_xticklabels(avail, fontsize=6, rotation=45, ha="right")
    ax.set_yticks(range(len(avail)))
    ax.set_yticklabels(avail, fontsize=6)
    for i in range(len(avail)):
        for j in range(len(avail)):
            val = corr_local.values[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5, color=color)

    heatmap_label = _FIG1_LABELS.get(map_label, map_label)
    ax.set_title(heatmap_label, fontsize=7.5)

plt.colorbar(im, ax=axes[-1], label="r", shrink=0.8)
fig.suptitle("Per-Map Element Correlation Matrices", fontsize=11)
fig.tight_layout()
fig.savefig(OUT_DIR / "plot6_per_map_corr_matrices.png", dpi=DPI)
plt.close(fig)
print("  Saved plot6_per_map_corr_matrices.png")


# ==========================================================
# 6. Summary
# ==========================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"\nPixel-level Fe correlations (all maps combined):")
for e in elem_short:
    if e != "Fe":
        r = corr_matrix.loc["Fe", e]
        strength = "strong" if abs(r) > 0.5 else "moderate" if abs(r) > 0.3 else "weak"
        direction = "positive" if r > 0 else "negative"
        print(f"  Fe vs {e:3s}: r = {r:+.3f}  ({strength} {direction})")

print(f"\nCluster-specific Fe–Ca correlation (XANES points):")
for c in sorted(points_df["cluster"].unique()):
    sub = points_df[points_df["cluster"] == c]
    if "Ca" in sub.columns and len(sub) > 5:
        valid = sub[["Fe", "Ca"]].dropna()
        r, p = pearsonr(valid["Fe"], valid["Ca"])
        print(f"  Cluster {c} (n={len(valid)}): r = {r:+.3f}, p = {p:.4f}")

print(f"\nOutput files saved to {OUT_DIR}/")
print("Done.")
